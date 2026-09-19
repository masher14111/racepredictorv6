"""Three selection rules scored on the SAME races — the model's honest comparison set.

WHY this module exists
----------------------
Stage 4 (``reports/calibration_audit_20260727.md``) found the independent
price-free line *losing* to the de-vigged pre-off market by 0.186 race log-loss
on 904 untouched races. A model in that state can still show a flattering ROI,
and the classic way it does so is an **unmatched race set**: the model's ROI is
computed over the handful of races its gate liked, while the benchmark's ROI is
computed over everything. Comparing those two numbers is not a comparison.

So every baseline here is a *one-bet-per-race selection rule* over an identical
race set, and :func:`build_baseline_ledgers` refuses to return until
:func:`assert_same_races` agrees. The three rules differ only in **which runner**
they back:

``model_only``
    The independent model probability, gated by ``cfg.gates``. This is the thing
    under test.
``devigged_market``
    The de-vigged reference-book probability used *as* the probability. This is
    the **null hypothesis**: it contains no model information whatsoever, its
    only possible edge is the gap between the consensus line and the best
    executable price. **If the model cannot beat this, there is no edge** — the
    model is at best an expensive way of reading a price back to itself (Stage 4
    measured the market-adjusted line at 0.925 correlation with ``1/price``).
``favourite``
    The shortest-priced runner. Zero information, zero skill, and historically
    hard to beat after margin. A model that does not clear it is not a model.

Scope: analysis, not issuance
-----------------------------
These ledgers answer "what would each rule have done?" over a historical scored
frame. They are **not** the live issuance path — that is ``execution/gates.py``,
which fails closed on every ``cfg.gates`` condition including source health and
the Stage-4 model verdict. Two gates are deliberately *not* applied here:

* ``require_model_validation`` — the whole purpose of this comparison is to
  produce the evidence a model verdict is made from. Requiring a GO to run it
  would make the verdict unfalsifiable.
* ``require_source_health`` — a live-feed property with no meaning over a
  settled historical window.

Every other gate is enforced when the frame carries the evidence to enforce it;
:func:`eligible_race_ids` documents each one. Where a frame simply lacks the
column (e.g. no runner-history flag on an old backtest parquet) the condition is
logged as unverified rather than silently passed, and the resulting numbers must
be read as an upper bound on what the live gate would have let through.

What is deliberately NOT here
-----------------------------
* **No frictions.** Latency, rejections, suspensions, commission and Rule 4 live
  in ``execution/frictions.py`` and ``execution/settlement.py``; applying them
  here as well would double-count them. These ledgers are pre-friction
  *selection* benchmarks and must be labelled as such wherever they are shown.
* **No staking model.** Every bet is a flat 1.0 unit. Kelly sizing would confound
  selection quality with sizing luck, and the two must be judged separately.
* **No closing price anywhere in a decision.** ``closing_odds`` is carried into
  the ledger for CLV only; selection reads the executable and reference prices,
  both of which are as-of the decision.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import numpy as np
import pandas as pd

from backtest import metrics as bt_metrics
from models.value import devig_field
from utils.logger import get_logger

logger = get_logger(__name__)

BASELINES: dict[str, str] = {
    "model_only": (
        "Back the best-EV runner at the executable price using the independent "
        "model probability, subject to the execution gates (min EV, min edge)."
    ),
    "devigged_market": (
        "Back the best-EV runner using the de-vigged reference book AS the "
        "probability — the null hypothesis: no model input, edge only from the "
        "gap between the consensus line and the best executable price."
    ),
    "favourite": (
        "Back the shortest-priced runner in the race — the zero-information "
        "benchmark a model must clear before it is worth anything."
    ),
}

# Exactly the columns execution.evaluation.evaluate_ledger consumes.
LEDGER_COLUMNS: tuple[str, ...] = (
    "race_uid",
    "race_date",
    "horse_key",
    "stake",
    "decimal_odds",
    "model_prob",
    "won",
    "returns",
    "profit",
    "closing_odds",
    "voided",
)

# Flat unit stake — selection quality only; see the module docstring.
FLAT_STAKE = 1.0

# Accepted aliases when normalising a scored frame. Canonical name first.
_RACE_KEYS = ("race_uid", "race_id")
_DATE_KEYS = ("race_date", "race_time", "date")
_RUNNER_KEYS = ("horse_key", "horse_id", "horse_name", "runner")
_PROB_KEYS = ("model_prob", "value_win_prob_independent", "model_win_prob",
              "value_win_prob", "norm_prob", "prob")
# The price actually takeable (best board price preferred).
_EXEC_KEYS = ("decimal_odds", "best_odds", "bet_price")
# A single coherent consensus line, used ONLY to de-vig. Never the best-price
# overlay — de-vigging a synthetic "best price per runner" field mixes books and
# understates the margin (models/value.py makes the same distinction).
_REF_KEYS = ("reference_odds", "consensus_odds", "bet_price")
_CLOSE_KEYS = ("closing_odds", "close_price", "odds_finish")
# Optional gate evidence. Absent ⇒ the condition is logged as unverified (see
# the module docstring), never silently treated as satisfied.
_SUPPORT_KEYS = ("value_supported", "supported", "has_history")
_FIELD_SIZE_KEYS = ("field_size", "declared_runners", "n_declared")

_NORMALISED_COLUMNS: tuple[str, ...] = (
    "race_uid", "race_date", "horse_key", "model_prob", "decimal_odds",
    "reference_odds", "won", "closing_odds", "ev_eligible", "supported",
    "field_size",
)


def _pick(frame: pd.DataFrame, keys: tuple[str, ...]) -> Optional[str]:
    return next((k for k in keys if k in frame.columns), None)


def _normalise(scored: pd.DataFrame) -> pd.DataFrame:
    """Canonicalise a scored runner frame to the columns the baselines need.

    One row per runner with ``race_uid, race_date, horse_key, model_prob,
    decimal_odds`` (executable), ``reference_odds`` (consensus; equal to the
    executable price only when the frame carries no consensus column at all),
    ``won``, ``closing_odds``, ``ev_eligible``.

    ``ev_eligible`` is the race-level decision already taken and persisted by
    :func:`models.predictor._race_ev_gate`. When the frame carries it we obey it
    — that decision is never re-derived or overridden here.
    """
    if scored is None or len(scored) == 0:
        return pd.DataFrame(columns=list(_NORMALISED_COLUMNS))

    race_col = _pick(scored, _RACE_KEYS)
    if race_col is None:
        raise ValueError(f"scored frame needs one of {_RACE_KEYS}; got {list(scored.columns)}")
    exec_col = _pick(scored, _EXEC_KEYS)
    if exec_col is None:
        raise ValueError(f"scored frame needs an executable price column {_EXEC_KEYS}")

    date_col = _pick(scored, _DATE_KEYS)
    runner_col = _pick(scored, _RUNNER_KEYS)
    prob_col = _pick(scored, _PROB_KEYS)
    ref_col = _pick(scored, _REF_KEYS)
    close_col = _pick(scored, _CLOSE_KEYS)
    support_col = _pick(scored, _SUPPORT_KEYS)
    field_col = _pick(scored, _FIELD_SIZE_KEYS)

    n = len(scored)
    executable = pd.to_numeric(scored[exec_col], errors="coerce")
    if ref_col is None:
        # No consensus column at all: a lone-price field is still one coherent
        # book, so the executable price IS the reference.
        reference = executable.copy()
    else:
        # A reference column that exists but is blank for some runner means the
        # consensus book is genuinely INCOMPLETE. Backfilling it from the
        # executable price would manufacture a fair line out of a partial field
        # (the exact fail-open models.value.find_value_bets refuses); leave the
        # NaN so eligible_race_ids can PASS the race.
        reference = pd.to_numeric(scored[ref_col], errors="coerce")

    if "ev_eligible" in scored.columns:
        ev_eligible = pd.Series(scored["ev_eligible"]).astype("boolean").fillna(False)
    else:
        # No persisted claim ⇒ no claim made here either; the structural gates
        # in eligible_race_ids still apply.
        ev_eligible = pd.Series(True, index=scored.index, dtype="boolean")

    out = pd.DataFrame({
        "race_uid": scored[race_col].astype(str).to_numpy(),
        "race_date": (pd.to_datetime(scored[date_col], errors="coerce").to_numpy()
                      if date_col is not None
                      else np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")),
        "horse_key": (scored[runner_col].astype(str).to_numpy()
                      if runner_col is not None else np.arange(n).astype(str)),
        "model_prob": (pd.to_numeric(scored[prob_col], errors="coerce").to_numpy()
                       if prob_col is not None else np.full(n, np.nan)),
        "decimal_odds": executable.to_numpy(dtype=float),
        "reference_odds": reference.to_numpy(dtype=float),
        "won": (pd.to_numeric(scored["won"], errors="coerce").to_numpy()
                if "won" in scored.columns else np.full(n, np.nan)),
        "closing_odds": (pd.to_numeric(scored[close_col], errors="coerce").to_numpy()
                         if close_col is not None else np.full(n, np.nan)),
        "ev_eligible": ev_eligible.to_numpy(dtype=bool),
        # NaN/NA = "no evidence either way"; the gate reports it as unverified.
        "supported": (pd.Series(scored[support_col]).astype("boolean").to_numpy(dtype=object)
                      if support_col is not None else np.full(n, pd.NA, dtype=object)),
        "field_size": (pd.to_numeric(scored[field_col], errors="coerce").to_numpy()
                       if field_col is not None else np.full(n, np.nan)),
    })
    return out.reset_index(drop=True)


# ── eligibility ──────────────────────────────────────────────────────────────


def eligible_race_ids(scored: pd.DataFrame, *, cfg) -> set[str]:
    """Races every strategy is allowed to act in — fail-closed, structural only.

    A race qualifies only when it affirmatively satisfies each applicable
    ``cfg.gates`` condition. Missing data is never treated as a pass:

    * ``ev_eligible`` is False for any runner → excluded (obey the predictor's
      persisted per-race decision, never re-derive a friendlier one).
    * Always: every runner carries a settled ``won``. An unsettled race cannot
      be scored at all, so this holds regardless of config.
    * ``require_complete_card`` → when the frame records a declared
      ``field_size``, the number of rows must match it: a race whose card
      arrived short is a different race from the one the market priced.
    * ``require_runner_history`` → when the frame records runner support, every
      runner must have prior in-window form.
    * ``min_field_size`` → fewer runners than this and de-vig plus Rule 4
      dominate whatever signal exists.
    * ``require_reference_market`` → *every* runner must carry a valid reference
      price. A partial book de-vigs to a fair line that is simply wrong.
    * ``max_overround`` → ``sum(1/reference_odds)`` above the ceiling means the
      book is too loose for its de-vig to mean anything (config value 1.25 is the
      book total, not the excess).
    * ``require_executable_price`` → every runner must be backable, otherwise the
      "best EV runner in the field" is chosen from a silently partial field.
    * ``require_calibrated_probability`` → every runner carries a finite model
      probability. This is applied to ALL three strategies even though only
      ``model_only`` reads it, because dropping the requirement for the
      benchmarks would give them a larger race set and break the match.

    Deliberately *not* here: whether any runner clears the EV/edge thresholds.
    That is a strategy decision, not a race property — folding it in would let
    the model define its own universe.
    """
    frame = _normalise(scored)
    if frame.empty:
        return set()

    gates = cfg.gates
    eligible: set[str] = set()
    for race_uid, grp in frame.groupby("race_uid", sort=False):
        reasons = _race_gate_reasons(grp, gates)
        if reasons:
            logger.debug("baselines: race %s ineligible (%s)", race_uid, "; ".join(reasons))
            continue
        eligible.add(str(race_uid))
    logger.info(
        "baselines: %d/%d races eligible under the execution gates",
        len(eligible), int(frame["race_uid"].nunique()),
    )
    return eligible


def _race_gate_reasons(grp: pd.DataFrame, gates) -> list[str]:
    """Explicit PASS reasons for one race (empty list ⇒ eligible)."""
    reasons: list[str] = []
    n = len(grp)

    if not bool(grp["ev_eligible"].all()):
        reasons.append("ev_ineligible")

    if n < int(gates.min_field_size):
        reasons.append(f"field_too_small:{n}<{int(gates.min_field_size)}")

    # Unconditional: a race with no result cannot be scored by anything.
    won = grp["won"].to_numpy(dtype=float)
    if not bool(np.isfinite(won).all()):
        reasons.append(f"unsettled_runners:{int((~np.isfinite(won)).sum())}")

    if gates.require_complete_card:
        declared = pd.to_numeric(grp["field_size"], errors="coerce").dropna().unique()
        if declared.size and int(declared[0]) != n:
            reasons.append(f"incomplete_card:{n}/{int(declared[0])}")

    if gates.require_runner_history:
        support = grp["supported"]
        if support.notna().any() and not bool(support.fillna(False).astype(bool).all()):
            n_unsupported = int((~support.fillna(False).astype(bool)).sum())
            reasons.append(f"unsupported_runners:{n_unsupported}")

    ref = grp["reference_odds"].to_numpy(dtype=float)
    ref_ok = np.isfinite(ref) & (ref > 1.0)
    if gates.require_reference_market and not bool(ref_ok.all()):
        reasons.append(f"incomplete_reference_book:{int(ref_ok.sum())}/{n}")

    if ref_ok.any():
        book = float(np.sum(1.0 / ref[ref_ok]))
        # Only meaningful when the book is complete; a partial book always
        # understates the total, so guard on ref_ok.all().
        if bool(ref_ok.all()) and book > float(gates.max_overround):
            reasons.append(f"overround_too_loose:{book:.3f}>{float(gates.max_overround):.3f}")

    ex = grp["decimal_odds"].to_numpy(dtype=float)
    ex_ok = np.isfinite(ex) & (ex > 1.0)
    if gates.require_executable_price and not bool(ex_ok.all()):
        reasons.append(f"unpriced_runners:{int((~ex_ok).sum())}")

    prob = grp["model_prob"].to_numpy(dtype=float)
    if gates.require_calibrated_probability and not bool(np.isfinite(prob).all()):
        reasons.append(f"uncalibrated_runners:{int((~np.isfinite(prob)).sum())}")

    return reasons


# ── shared selection machinery ───────────────────────────────────────────────


def _empty_ledger() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in LEDGER_COLUMNS})


def _fair_probs(grp: pd.DataFrame) -> np.ndarray:
    """De-vigged reference probabilities for one race's field (sums to 1).

    Straight through :func:`models.value.devig_field` so the baselines and the
    live value layer can never disagree about what "the market's fair line" is.
    """
    fair, _ = devig_field(grp["reference_odds"].to_numpy(dtype=float))
    return np.atleast_1d(fair).astype(float)


def _ledger_from_picks(picks: list[dict]) -> pd.DataFrame:
    """Settle a list of one-per-race picks into the evaluate_ledger contract."""
    if not picks:
        return _empty_ledger()
    df = pd.DataFrame(picks)
    stake = df["stake"].to_numpy(dtype=float)
    odds = df["decimal_odds"].to_numpy(dtype=float)
    won = df["won"].to_numpy(dtype=float)
    # Flat back-bet settlement, no commission (frictions live elsewhere).
    profit = np.atleast_1d(bt_metrics.settle(stake, odds, won)).astype(float)
    df["profit"] = profit
    df["returns"] = stake + profit
    df["voided"] = False
    return df.reindex(columns=list(LEDGER_COLUMNS)).reset_index(drop=True)


def _pick_row(grp: pd.DataFrame, i: int, prob: float) -> dict:
    row = grp.iloc[i]
    return {
        "race_uid": str(row["race_uid"]),
        "race_date": row["race_date"],
        "horse_key": str(row["horse_key"]),
        "stake": FLAT_STAKE,
        "decimal_odds": float(row["decimal_odds"]),
        "model_prob": float(prob),
        "won": float(row["won"]) if np.isfinite(row["won"]) else 0.0,
        "closing_odds": float(row["closing_odds"]),
    }


def _restricted(scored: pd.DataFrame, cfg) -> pd.DataFrame:
    """Normalised frame restricted to the structurally eligible races."""
    frame = _normalise(scored)
    if frame.empty:
        return frame
    eligible = eligible_race_ids(frame, cfg=cfg)
    if not eligible:
        return frame.iloc[0:0]
    return frame[frame["race_uid"].isin(eligible)].reset_index(drop=True)


# ── the three strategies ─────────────────────────────────────────────────────


def _band_mask(odds: np.ndarray, band: Optional[tuple]) -> np.ndarray:
    """Which runners a price band permits **backing**. See :func:`model_only_selection`."""
    keep = np.ones(odds.shape, dtype=bool)
    if not band:
        return keep
    lo, hi = (list(band) + [None, None])[:2]
    if lo is not None:
        keep &= odds >= float(lo)
    if hi is not None:
        keep &= odds < float(hi)
    return keep


def model_only_selection(
    scored: pd.DataFrame, *, cfg, odds_band: Optional[tuple] = None
) -> pd.DataFrame:
    """Back the best-EV runner per race on the independent model probability.

    A runner qualifies only when BOTH ``cfg.gates`` thresholds are met at the
    executable price:

    * ``EV = model_prob * decimal_odds - 1 >= gates.min_expected_value``
    * ``edge = model_prob - fair_prob >= gates.min_edge`` where ``fair_prob`` is
      the de-vigged *reference* book (config's own definition of min_edge).

    The highest-EV qualifier is backed; a race with no qualifier produces **no
    row at all**, which is the correct default answer under the Stage-4 NO-GO.

    ``odds_band`` — ``(min, max)`` on the executable price, ``max`` exclusive —
    restricts which runner may be **backed**. It deliberately does *not* filter
    the frame, because two things downstream read the whole field and are wrong
    without it:

    * ``_fair_probs`` de-vigs by normalising the field to sum 1. Drop the
      favourites and the survivors' fair probabilities are renormalised upward
      against a book that no bookmaker ever offered, so ``edge`` is measured
      against a fiction.
    * :func:`eligible_race_ids` checks card completeness and the book total. A
      banded field fails both — a short-price-only subset sums to well under
      1.0 — so pre-filtering silently changes the *race set* per strategy, and
      strategies scored on different race sets cannot be compared at all.

    Both defects were live: banding the frame made the 1.0–4.0 band report 0 of
    3817 races eligible, which reads as "this strategy never fires" when the
    truth was "this strategy was never scored".
    """
    frame = _restricted(scored, cfg)
    if frame.empty:
        return _empty_ledger()

    gates = cfg.gates
    picks: list[dict] = []
    for _, grp in frame.groupby("race_uid", sort=False):
        p = grp["model_prob"].to_numpy(dtype=float)
        d = grp["decimal_odds"].to_numpy(dtype=float)
        fair = _fair_probs(grp)
        ev = np.atleast_1d(bt_metrics.expected_value(p, d)).astype(float)
        edge = p - fair

        keep = (np.isfinite(ev) & np.isfinite(edge)
                & (ev >= float(gates.min_expected_value))
                & (edge >= float(gates.min_edge))
                & _band_mask(d, odds_band))
        if not keep.any():
            continue
        best = int(np.flatnonzero(keep)[np.argmax(ev[keep])])
        picks.append(_pick_row(grp, best, p[best]))
    return _ledger_from_picks(picks)


def devigged_market_selection(scored: pd.DataFrame, *, cfg) -> pd.DataFrame:
    """The null hypothesis: the de-vigged market probability IS the probability.

    No model input at all. The only edge available to this rule is the gap
    between the consensus reference line and the best executable price, so on a
    single-book field where ``decimal_odds == reference_odds`` its EV is
    ``1/sum(1/d) - 1 < 0`` for every runner and it correctly bets nothing.
    **If the model line cannot beat this, the model has no edge** — it is reading
    the price back to itself.

    Gate asymmetry, stated plainly: this rule is held to
    ``gates.min_expected_value`` but **not** to ``gates.min_edge``. The config
    defines edge as ``prob - fair_implied``, which is identically zero when the
    probability *is* the fair implied one; applying it would silently return an
    empty baseline and let the model win the comparison by default. The
    asymmetry therefore makes the model's job harder, never easier.
    """
    frame = _restricted(scored, cfg)
    if frame.empty:
        return _empty_ledger()

    gates = cfg.gates
    picks: list[dict] = []
    for _, grp in frame.groupby("race_uid", sort=False):
        d = grp["decimal_odds"].to_numpy(dtype=float)
        fair = _fair_probs(grp)
        ev = np.atleast_1d(bt_metrics.expected_value(fair, d)).astype(float)

        keep = np.isfinite(ev) & (ev >= float(gates.min_expected_value))
        if not keep.any():
            continue
        best = int(np.flatnonzero(keep)[np.argmax(ev[keep])])
        picks.append(_pick_row(grp, best, fair[best]))
    return _ledger_from_picks(picks)


def favourite_selection(scored: pd.DataFrame, *, cfg) -> pd.DataFrame:
    """Back the shortest-priced runner in every eligible race. No gates.

    The favourite is identified by the **reference** (consensus) line — that is
    what "favourite" means — and backed at the **executable** price, which is
    what you would actually be paid. Ties go to the first runner in frame order.

    No EV or edge threshold is applied: this is a fixed zero-information rule,
    and gating it would turn it into a different strategy rather than a
    benchmark. Its recorded ``model_prob`` is the runner's de-vigged market
    probability, so A/E and the calibration error remain defined.
    """
    frame = _restricted(scored, cfg)
    if frame.empty:
        return _empty_ledger()

    picks: list[dict] = []
    for _, grp in frame.groupby("race_uid", sort=False):
        ref = grp["reference_odds"].to_numpy(dtype=float)
        d = grp["decimal_odds"].to_numpy(dtype=float)
        usable = np.isfinite(ref) & (ref > 1.0) & np.isfinite(d) & (d > 1.0)
        if not usable.any():
            continue
        candidates = np.flatnonzero(usable)
        best = int(candidates[np.argmin(ref[candidates])])
        fair = _fair_probs(grp)
        picks.append(_pick_row(grp, best, fair[best]))
    return _ledger_from_picks(picks)


_STRATEGIES = {
    "model_only": model_only_selection,
    "devigged_market": devigged_market_selection,
    "favourite": favourite_selection,
}


# ── matching ─────────────────────────────────────────────────────────────────


def assert_same_races(ledgers: Mapping[str, pd.DataFrame]) -> None:
    """Raise ``ValueError`` unless every ledger covers exactly the same races.

    This is the invariant that makes a baseline table a comparison rather than
    an anecdote. It is checked, not assumed, because the failure mode is silent:
    two ROI numbers over different denominators still render as two ROI numbers.
    """
    if not ledgers:
        return
    sets: dict[str, set[str]] = {}
    for name, df in ledgers.items():
        if df is None:
            raise ValueError(f"ledger {name!r} is None")
        if len(df) and "race_uid" not in df.columns:
            raise ValueError(f"ledger {name!r} has no 'race_uid' column")
        sets[name] = set(df["race_uid"].astype(str)) if len(df) else set()

    ref_name, ref = next(iter(sets.items()))
    for name, s in sets.items():
        if s == ref:
            continue
        missing = sorted(ref - s)[:5]
        extra = sorted(s - ref)[:5]
        raise ValueError(
            f"baseline race sets differ: {name!r} has {len(s)} races vs "
            f"{ref_name!r} {len(ref)}; missing={missing} extra={extra}. "
            "An unmatched race set makes the comparison meaningless."
        )


def build_baseline_ledgers(
    scored: pd.DataFrame, *, cfg, eligible: Optional[set[str]] = None
) -> dict[str, pd.DataFrame]:
    """Build all three baselines over one matched race set.

    ``eligible`` defaults to :func:`eligible_race_ids`. Every strategy is
    restricted to it, and the returned ledgers are then narrowed to the
    **intersection** of the races each strategy actually bet in, so the three
    ROI/CLV numbers share a denominator. :func:`assert_same_races` verifies this
    before returning.

    Read the result with its caveat attached: the matched set is whittled down by
    the *model's* gate, so ``favourite``'s number here is "backing the favourite
    in the races the model chose to act in", not a standalone favourite-backing
    result. It answers the only question that matters for deployment — on the
    races this system would bet, does the model add anything? — and nothing else.

    An empty eligible set returns three empty ledgers, not an error: "no bet
    anywhere" is a valid and expected outcome.
    """
    frame = _normalise(scored)
    if eligible is None:
        eligible = eligible_race_ids(frame, cfg=cfg)
    eligible = {str(r) for r in eligible}

    frame = (frame[frame["race_uid"].isin(eligible)].reset_index(drop=True)
             if len(frame) else frame)

    ledgers = {name: fn(frame, cfg=cfg) for name, fn in _STRATEGIES.items()}

    race_sets = [set(df["race_uid"].astype(str)) if len(df) else set()
                 for df in ledgers.values()]
    matched: set[str] = set.intersection(*race_sets) if race_sets else set()

    for name, df in ledgers.items():
        acted = set(df["race_uid"].astype(str)) if len(df) else set()
        if acted - matched:
            logger.info(
                "baselines: %s bet in %d race(s) the matched set drops (another "
                "strategy passed there)", name, len(acted - matched),
            )
    logger.info(
        "baselines: %d eligible race(s) → %d matched race(s) scored by all three",
        len(eligible), len(matched),
    )

    ledgers = {
        name: (df[df["race_uid"].astype(str).isin(matched)].reset_index(drop=True)
               if len(df) else df)
        for name, df in ledgers.items()
    }
    assert_same_races(ledgers)
    return ledgers
