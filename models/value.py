"""Value-detection layer: turn calibrated model probabilities + market prices
into ranked **value bets**.

A value bet is one where the model's (price-free) win probability exceeds the
market's *fair* win probability — i.e. the price is bigger than it should be.
The pipeline, per race:

1. **Strip the over-round.** Bookmaker decimal odds imply ``1/d`` per runner, but
   a real book's per-runner implied probabilities sum to >1 (the margin / vig).
   We normalise the field so they sum to 1, giving each runner's *fair* implied
   probability. Comparing the model to the **fair** prob — not the raw, margin-
   inflated one — is the apples-to-apples comparison.
2. **Edge & EV.** ``edge = model_prob - fair_prob`` (how much more likely the
   model thinks the runner is than the de-vigged market). ``expected_value =
   model_prob * d - 1`` uses the **raw** price (that's what you actually get paid),
   so it already accounts for the margin: EV>0 ⇒ +EV at the available price.
3. **Gates** (all config-driven): minimum EV, an odds band that gates both odds-on
   shots and longshots, optional probability-edge thresholds, a minimum model
   probability, a minimum confidence score, and a "must have prior form" support
   gate. See :class:`ValueConfig`.
4. **Stake.** Fractional-Kelly on the *raw* price, capped — the same maths the
   backtester and bet-tracker use.

Public API
----------
    from models.value import find_value_bets, ValueConfig
    picks = find_value_bets(race)          # race = predictor race dict / runner list / DataFrame
    # -> ranked list of dicts: edge, edge_pct, expected_value, kelly_fraction,
    #    suggested_stake, confidence, fair_prob, model_prob, decimal_odds, ...

``evaluate_filter(scored, config)`` replays the same gates over a backtester
out-of-sample frame so a threshold set can be validated on **CLV and A/E**, not
just backtested ROI (see memory/model-16-value-detection.md).

Validated thresholds (saved run ``data/backtests/20260616_195957``, 77k OOS
runners; re-tuned post-recalibration in [[calib-fl-04-backtest]]). The price-free
win prob is favourite-longshot **recalibrated** in the predictor *before* this
layer sees it ([[calib-fl-03-integrate]]), which flattens A/E across every odds
band (was 2.80 odds-on … 0.17 longshot; now ~0.94…0.78). With A/E flat the band is
now tuned on **CLV**: closing-line value is least-negative on the shortest-priced
favourites and worsens as longer prices are admitted, so the band is tightened to
``[2.0, 4.0]`` on EV alone. The probability-edge gates
(``min_edge_pct``/``min_abs_edge``) default OFF — post-recalibration they are no
longer *harmful* but they do not lift CLV. **CLV stays negative (best ~-4% on the
favourite band) in every gate**, so the layer is profitable in backtest at the
price taken but **not yet proven to beat the closing line** — treat picks as
provisional / paper. See [[calib-fl-04-backtest]].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from backtest import metrics
from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger(__name__)

RaceInput = Union[dict, list, pd.DataFrame]

# Column-name fallbacks when extracting a runner frame from arbitrary input.
_PROB_KEYS = ("value_win_prob", "model_prob", "prob")
# Two distinct prices per runner (audit req 5/6):
#   * EXECUTABLE — the price you would actually take. Best board price across
#     bookmakers preferred; drives EV and the odds-band gate.
#   * REFERENCE  — a single coherent market/consensus line used ONLY to de-vig
#     into a fair probability. It must NOT be the synthetic "best price for each
#     runner" overlay (de-vigging that mixes bookmakers and understates the
#     margin), so best_odds is deliberately excluded from the reference keys.
#     Falls back to implied_prob (the fused consensus) and, only when no
#     consensus line exists at all, to the executable price (a single-book field).
_EXEC_ODDS_KEYS = ("best_odds", "decimal_odds", "bet_price")
_REF_ODDS_KEYS = ("reference_odds", "decimal_odds", "bet_price")
# Back-compat alias (external callers may still import this).
_ODDS_KEYS = _EXEC_ODDS_KEYS
_NAME_KEYS = ("horse_name", "name", "runner")
_ID_KEYS = ("horse_id", "id", "selection_id")


@dataclass
class ValueConfig:
    """Config-driven value-bet criteria + staking knobs.

    Every gate is ANDed; set a gate to its no-op value (0.0 / wide band) to
    disable it. Defaults mirror the validated config.yaml ``value:`` block.
    """

    enabled: bool = True
    model_tag: str = "v3nf"
    # ── selection gates ──────────────────────────────────────────────────────
    min_ev: float = 0.05          # back iff model_prob*d - 1 >= this
    min_odds: float = 2.0         # odds-on gate (skip prices below this)
    max_odds: float = 4.0         # longshot gate; CLV-tuned, see calib-fl-04
    min_edge_pct: float = 0.0     # back iff model_prob >= fair*(1+this); 0 = off
    min_abs_edge: float = 0.0     # back iff model_prob - fair >= this; 0 = off
    min_prob: float = 0.0         # minimum model win probability
    min_confidence: float = 0.40  # minimum confidence score [0,1]
    require_support: bool = True  # require the runner to have prior form
    devig: bool = True            # strip the book over-round before edge
    # ── staleness gate ───────────────────────────────────────────────────────
    # Belt-and-suspenders vs features/fuse.py's _drop_stale_live_rows: a caller
    # that hands find_value_bets() a runner frame bypassing fuse.py still can't
    # get a stale EV pick. 0 disables (never used in practice; config-driven).
    max_stale_seconds: float = 900.0
    # ── staking ──────────────────────────────────────────────────────────────
    kelly_fraction: float = 0.25  # fraction of full Kelly to stake
    kelly_cap: float = 0.05       # max single stake as a fraction of bankroll
    bankroll: float = 1000.0      # bankroll the suggested stake is sized against
    # ── confidence shaping ───────────────────────────────────────────────────
    reliable_max_odds: float = 8.0  # odds at/below which odds-reliability is full

    @classmethod
    def from_config(cls, raw: Optional[dict] = None) -> "ValueConfig":
        """Build from the merged config.yaml (``value:`` + ``bet_tracker:``)."""
        raw = raw if raw is not None else get_config()
        v = raw.get("value") or {}
        bt = raw.get("bet_tracker") or {}
        staleness = raw.get("staleness") or {}
        return cls(
            enabled=bool(v.get("enabled", True)),
            model_tag=str(v.get("model_tag", "v3nf")),
            min_ev=float(v.get("min_expected_value", 0.05)),
            min_odds=float(v.get("min_odds", 2.0)),
            max_odds=float(v.get("max_odds", 4.0)),
            min_edge_pct=float(v.get("min_edge_pct", 0.0)),
            min_abs_edge=float(v.get("min_abs_edge", 0.0)),
            min_prob=float(v.get("min_prob", 0.0)),
            min_confidence=float(v.get("min_confidence", 0.40)),
            require_support=bool(v.get("require_support", True)),
            devig=bool(v.get("devig", True)),
            max_stale_seconds=float(staleness.get("max_age_seconds", 900.0)),
            reliable_max_odds=float(v.get("reliable_max_odds", 8.0)),
            kelly_fraction=float(bt.get("kelly_fraction", 0.25)),
            kelly_cap=float(v.get("kelly_cap", 0.05)),
            bankroll=float(bt.get("initial_bankroll", 1000.0)),
        )


# ── shared value-bet gate ────────────────────────────────────────────────────

def passes_core_gate(expected_value, executable_odds, supported,
                     cfg: "ValueConfig") -> np.ndarray:
    """The core value-bet decision shared by the predictor's per-runner
    ``value_bet`` flag (:meth:`models.predictor.Predictor._build_race`) and
    :func:`find_value_bets`.

    A runner clears the core gate iff it has a valid executable price inside the
    configured odds band, positive-enough EV at that price, and (when required)
    prior form support. Defining it once here is what stops the UI's ``value_bet``
    flag and the suggestion engine from ever disagreeing (audit req 7).
    ``find_value_bets`` layers its curation gates (min_prob, probability-edge,
    confidence, de-vig completeness, staleness) on top of this shared core.

    Vectorised and scalar-safe; returns a boolean ``np.ndarray``.
    """
    ev = np.atleast_1d(np.asarray(expected_value, dtype=float))
    d = np.atleast_1d(np.asarray(executable_odds, dtype=float))
    sup = np.atleast_1d(np.asarray(supported, dtype=bool))
    keep = np.isfinite(d) & (d > 1.0) & np.isfinite(ev)
    keep &= ev >= cfg.min_ev
    keep &= d >= cfg.min_odds
    keep &= d <= cfg.max_odds
    if cfg.require_support:
        keep &= sup
    return keep


# ── de-vig ───────────────────────────────────────────────────────────────────

def devig_field(decimal_odds) -> tuple[np.ndarray, float]:
    """Fair (over-round-stripped) implied probabilities for one race's field.

    Returns ``(fair_probs, overround)``. ``fair_i = (1/d_i) / sum_j(1/d_j)`` so
    the field sums to 1; ``overround = sum(1/d) - 1`` (0 for a fair book, ~0.1–0.3
    for a real one). Runners with no valid price get a NaN fair prob and are
    excluded from the normaliser. Only meaningful over a single race's runners —
    that is exactly what :func:`find_value_bets` receives.
    """
    raw = metrics.implied_prob(np.asarray(decimal_odds, dtype=float))
    raw = np.atleast_1d(raw).astype(float)
    fair, overround = metrics.devig(raw)
    return np.atleast_1d(fair).astype(float), float(overround)


# ── confidence ─────────────────────────────────────────────────────────────--

def _confidence(decimal_odds: np.ndarray, n_priced: int, supported: np.ndarray,
                cfg: ValueConfig) -> np.ndarray:
    """A [0,1] trust score for each runner's value signal.

    Product of three documented, monotone factors:

    * **support** — 1.0 if the runner has prior form, else 0.0. A formless runner
      (no in-window history) scores ~base-rate, and base-rate × long odds is the
      longshot trap, so we trust it not at all.
    * **field** — ``clip(n_priced/8, 0.3, 1)``. De-vig (and therefore the fair
      prob / edge) is only reliable over a full field; a 2-runner book barely
      normalises.
    * **odds reliability** — 1.0 at/below ``reliable_max_odds`` (the band where the
      backtest A/E is ~1), decaying linearly to 0.2 by 3× that, then 0. Encodes
      the validated finding that the model is trustworthy on favourites/mid-prices
      and noisy on longshots.
    """
    d = np.asarray(decimal_odds, dtype=float)
    f_support = np.where(supported.astype(bool), 1.0, 0.0)
    f_field = float(np.clip(n_priced / 8.0, 0.3, 1.0))

    hi = max(cfg.reliable_max_odds, cfg.min_odds + 1e-9)
    far = 3.0 * hi
    # 1.0 for d<=hi, linear 1.0->0.2 across (hi, far], 0 beyond.
    f_odds = np.where(
        d <= hi, 1.0,
        np.where(d <= far, 1.0 - 0.8 * (d - hi) / (far - hi), 0.0),
    )
    f_odds = np.where(np.isfinite(d), f_odds, 0.0)
    return np.clip(f_support * f_field * f_odds, 0.0, 1.0)


# ── runner extraction ──────────────────────────────────────────────────────--

def _first_key(d: dict, keys: tuple) -> object:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _reference_odds(r: dict, executable: float) -> float:
    """Reference/consensus decimal odds for de-vig — never the best-price overlay.

    Prefers an explicit single-book/consensus decimal price, then the fused
    ``implied_prob`` (the consensus line), and only when neither exists falls back
    to the executable price (a lone-price synthetic field is itself one coherent
    book). See ``_REF_ODDS_KEYS``.
    """
    ref = _first_key(r, _REF_ODDS_KEYS)
    if ref is not None:
        return _num(ref)
    ip = _num(r.get("implied_prob"))
    if np.isfinite(ip) and ip > 0:
        return 1.0 / ip
    return executable


def _runners_from_dicts(runners: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runners:
        if not isinstance(r, dict):
            continue
        supported = r.get("value_supported")
        if supported is None:
            # Distinguish a present-but-null place rate (no prior form -> NOT
            # supported) from an absent key (unknown -> assume supported, so mock
            # rows / pre-value caches behave as before).
            if "historical_place_rate" in r:
                supported = bool(pd.notna(r["historical_place_rate"]))
            else:
                supported = True
        executable = _num(_first_key(r, _EXEC_ODDS_KEYS))
        rows.append({
            "horse_id": _str(_first_key(r, _ID_KEYS)),
            "horse_name": _str(_first_key(r, _NAME_KEYS)),
            "model_prob": _num(_first_key(r, _PROB_KEYS)),
            # ``decimal_odds`` is the EXECUTABLE price (best board price preferred),
            # kept under this name for back-compat; ``reference_odds`` is the
            # reference line used only to de-vig.
            "decimal_odds": executable,
            "reference_odds": _reference_odds(r, executable),
            "supported": bool(supported),
            "stale": bool(r.get("stale", False)),
            "fetched_at": r.get("fetched_at"),
            # provenance diagnostics carried into picks (audit req 6/8)
            "executable_source": r.get("best_book"),
            "reference_source": r.get("reference_source"),
            "empirical_win_rate": _num(r.get("empirical_win_rate")),
            # Stage-4: the price-free prob BEFORE the F-L market adjustment
            # (None on pre-Stage-4 caches / mock rows).
            "model_prob_independent": _num(r.get("value_win_prob_independent")),
        })
    return pd.DataFrame(rows)


def _extract_runners(race: RaceInput) -> pd.DataFrame:
    """Normalise any supported input to columns
    ``[horse_id, horse_name, model_prob, decimal_odds, supported]``."""
    if isinstance(race, dict):
        # Prefer the complete internal field (audit req 2): ``runners`` carries
        # every valid runner, whereas ``selections`` is only the display top-N and
        # ``excluded_low_odds`` only the short-priced tail — using those would hand
        # the value layer a partial field (the top-three partial-field bug).
        if race.get("runners"):
            runners = list(race.get("runners", []))
        elif "selections" in race or "excluded_low_odds" in race:
            runners = list(race.get("selections", [])) + list(race.get("excluded_low_odds", []))
        else:  # a single runner dict
            runners = [race]
        return _runners_from_dicts(runners)

    if isinstance(race, list):
        return _runners_from_dicts(race)

    if isinstance(race, pd.DataFrame):
        df = race
        prob_col = next((c for c in _PROB_KEYS if c in df.columns), None)
        exec_col = next((c for c in _EXEC_ODDS_KEYS if c in df.columns), None)
        ref_col = next((c for c in _REF_ODDS_KEYS if c in df.columns), None)
        name_col = next((c for c in _NAME_KEYS if c in df.columns), None)
        id_col = next((c for c in _ID_KEYS if c in df.columns), None)
        if prob_col is None or exec_col is None:
            raise ValueError(
                f"DataFrame needs a probability column {_PROB_KEYS} and an odds "
                f"column {_EXEC_ODDS_KEYS}; got {list(df.columns)}")
        if "value_supported" in df.columns:
            supported = df["value_supported"].astype(bool)
        elif "historical_place_rate" in df.columns:
            supported = pd.to_numeric(df["historical_place_rate"], errors="coerce").notna()
        else:
            supported = pd.Series(True, index=df.index)
        stale = (df["stale"].fillna(False).astype(bool)
                 if "stale" in df.columns else pd.Series(False, index=df.index))
        fetched_at = df["fetched_at"] if "fetched_at" in df.columns else pd.Series(pd.NaT, index=df.index)
        executable = pd.to_numeric(df[exec_col], errors="coerce")
        # Reference/consensus line for de-vig (never the best-price overlay).
        if ref_col is not None:
            reference = pd.to_numeric(df[ref_col], errors="coerce")
        elif "implied_prob" in df.columns:
            ip = pd.to_numeric(df["implied_prob"], errors="coerce")
            reference = 1.0 / ip.where(ip > 0)
        else:
            reference = executable.copy()
        # Where no consensus price exists for a runner, fall back to its executable
        # price (a lone-price field is one coherent book).
        reference = reference.where(reference.notna(), executable)
        return pd.DataFrame({
            "horse_id": df[id_col].map(_str) if id_col else "",
            "horse_name": df[name_col].map(_str) if name_col else "",
            "model_prob": pd.to_numeric(df[prob_col], errors="coerce"),
            "decimal_odds": executable,
            "reference_odds": reference,
            "supported": supported.to_numpy(dtype=bool),
            "stale": stale.to_numpy(dtype=bool),
            "fetched_at": fetched_at.to_numpy(),
            "executable_source": (df["best_book"] if "best_book" in df.columns
                                  else pd.Series(None, index=df.index, dtype=object)),
            "reference_source": (df["reference_source"] if "reference_source" in df.columns
                                 else pd.Series(None, index=df.index, dtype=object)),
            "empirical_win_rate": (pd.to_numeric(df["empirical_win_rate"], errors="coerce")
                                   if "empirical_win_rate" in df.columns
                                   else pd.Series(np.nan, index=df.index)),
            "model_prob_independent": (
                pd.to_numeric(df["value_win_prob_independent"], errors="coerce")
                if "value_win_prob_independent" in df.columns
                else pd.Series(np.nan, index=df.index)),
        })

    raise TypeError(f"unsupported race input type: {type(race)!r}")


# ── public API ─────────────────────────────────────────────────────────────--

def find_value_bets(race: RaceInput, config: Optional[ValueConfig] = None,
                    bankroll: Optional[float] = None) -> list[dict]:
    """Ranked value picks for one race.

    ``race`` may be a predictor race dict (``selections``/``excluded_low_odds``),
    a list of runner dicts, a single runner dict, or a DataFrame of runners. Each
    runner must carry a price-free model win probability and a decimal price.

    Returns a list of pick dicts (highest edge first) for the runners that clear
    every configured gate. Each pick has: ``model_prob``, ``fair_prob`` (de-vigged
    market), ``market_implied_prob`` (raw), ``overround``, ``edge``, ``edge_pct``,
    ``expected_value``, ``kelly_fraction``, ``suggested_stake``, ``confidence``.
    An empty list means no runner qualified.
    """
    cfg = config or ValueConfig.from_config()
    if not cfg.enabled:
        return []
    # Honour the race-level EV gate persisted by the predictor (audit req 9 /
    # issue #6): a race stamped ineligible has already PASSed — its card was
    # stale, partial, or mismatched at build time, and no surface may rebuild an
    # EV from it. Absent key (legacy cache / plain runner lists) ⇒ no claim.
    if isinstance(race, dict) and race.get("ev_eligible") is False:
        logger.debug(
            "find_value_bets: race marked EV-ineligible (%s) — PASS",
            "; ".join((race.get("ev_gate") or {}).get("reasons", []) or ["unspecified"]),
        )
        return []
    bank = float(bankroll) if bankroll is not None else cfg.bankroll

    runners = _extract_runners(race)
    if runners.empty:
        return []

    p = runners["model_prob"].to_numpy(dtype=float)
    d = runners["decimal_odds"].to_numpy(dtype=float)        # EXECUTABLE price (EV)
    ref = runners["reference_odds"].to_numpy(dtype=float)     # consensus (de-vig)
    supported = runners["supported"].to_numpy(dtype=bool)
    stale_flag = runners["stale"].to_numpy(dtype=bool)
    fetched_at = pd.to_datetime(runners["fetched_at"], utc=True, errors="coerce")
    age_seconds = (pd.Timestamp.now(tz="UTC") - fetched_at).dt.total_seconds().to_numpy()
    too_stale = stale_flag.copy()
    if cfg.max_stale_seconds > 0:
        has_fetched_at = ~pd.isna(fetched_at).to_numpy()
        too_stale |= has_fetched_at & (age_seconds > cfg.max_stale_seconds)

    # Reference-book completeness (audit req 4): a fair line can only come from a
    # COMPLETE reference book. If de-vig is requested and any runner lacks a valid
    # reference price, the book is partial → fair_prob / edge / confidence are
    # undefined for the whole race → the race PASSes (no picks) rather than being
    # scored off a misleading partial de-vig.
    ref_valid = np.isfinite(ref) & (ref > 1.0)
    if cfg.devig and not bool(ref_valid.all()):
        logger.debug(
            "find_value_bets: incomplete reference book (%d/%d priced) — PASS",
            int(ref_valid.sum()), int(ref_valid.size),
        )
        return []

    # Raw + fair market probabilities come from the REFERENCE book, never the
    # executable best-price overlay (audit req 5).
    raw_implied = np.atleast_1d(metrics.implied_prob(ref)).astype(float)
    if cfg.devig:
        fair, overround = devig_field(ref)
    else:
        fair, overround = raw_implied, float(np.nansum(raw_implied) - 1.0)

    edge = p - fair
    # EV / Kelly are on the EXECUTABLE price — the return you would actually get.
    ev = np.atleast_1d(metrics.expected_value(p, d)).astype(float)
    full_kelly = np.atleast_1d(metrics.kelly_fraction(p, d)).astype(float)
    full_kelly = np.where(np.isfinite(full_kelly), full_kelly, 0.0)
    stake_frac = np.clip(cfg.kelly_fraction * full_kelly, 0.0, cfg.kelly_cap)
    suggested_stake = stake_frac * max(bank, 0.0)

    n_priced = int(np.isfinite(d).sum())
    confidence = _confidence(d, n_priced, supported, cfg)

    # relative edge vs fair; NaN-safe (fair>0 inside a real field)
    with np.errstate(divide="ignore", invalid="ignore"):
        edge_pct = np.where(fair > 0, p / fair - 1.0, np.nan)

    # Core EV/odds-band/support gate — the single shared rule (audit req 7).
    keep = passes_core_gate(ev, d, supported, cfg)
    keep &= np.isfinite(p)
    keep &= p >= cfg.min_prob
    if cfg.min_edge_pct > 0:
        keep &= p >= fair * (1.0 + cfg.min_edge_pct)
    if cfg.min_abs_edge > 0:
        keep &= edge >= cfg.min_abs_edge
    keep &= confidence >= cfg.min_confidence
    keep &= ~too_stale
    keep = np.where(np.isfinite(keep), keep, False).astype(bool)

    picks = []
    computed_at = pd.Timestamp.now(tz="UTC").isoformat()
    for i in np.flatnonzero(keep):
        fair_i = fair[i]
        fair_odds = (_round(1.0 / fair_i, 3)
                     if np.isfinite(fair_i) and fair_i > 0 else None)
        age_i = age_seconds[i] if np.isfinite(age_seconds[i]) else None
        picks.append({
            "horse_id": runners["horse_id"].iat[i],
            "horse_name": runners["horse_name"].iat[i],
            # ── prices ──
            "decimal_odds": _round(d[i], 3),        # executable (legacy name)
            "executable_odds": _round(d[i], 3),
            "reference_odds": _round(ref[i], 3),
            "fair_decimal_odds": fair_odds,
            # ── probabilities (clearly distinguished — audit req 8) ──
            # model_prob is the value layer's decision prob: MARKET-ADJUSTED
            # (F-L recalibrated against the effective price) when the
            # recalibrator is enabled — NOT price-free. The price-free line is
            # model_win_prob_independent (None on pre-Stage-4 caches).
            "model_prob": _round(p[i], 4),
            "model_win_prob": _round(p[i], 4),
            "model_win_prob_independent": _round(
                _num(runners["model_prob_independent"].iat[i]), 4),
            "empirical_win_rate": _round(_num(runners["empirical_win_rate"].iat[i]), 4),
            "fair_prob": _round(fair_i, 4),         # de-vigged market prob
            "devigged_market_prob": _round(fair_i, 4),
            "market_implied_prob": _round(raw_implied[i], 4),   # raw (with vig)
            "raw_implied_prob": _round(raw_implied[i], 4),
            "overround": _round(overround, 4),
            # ── edges ──
            "edge": _round(edge[i], 4),             # probability points (fraction)
            "edge_pp": _round(edge[i] * 100.0, 2),  # same, in percentage points
            "edge_pct": _round(edge_pct[i], 4),     # relative edge model/fair - 1
            "relative_edge": _round(edge_pct[i], 4),
            # ── EV / staking ──
            "expected_value": _round(ev[i], 4),
            "EV": _round(ev[i], 4),
            "kelly_fraction": _round(full_kelly[i], 4),
            "suggested_stake": _round(suggested_stake[i], 2),
            "confidence": _round(confidence[i], 3),
            # ── provenance diagnostics (audit req 6/8) ──
            "executable_source": _src(runners["executable_source"].iat[i]),
            "reference_source": _src(runners["reference_source"].iat[i]),
            "price_age_seconds": _round(age_i, 1) if age_i is not None else None,
            "computed_at": computed_at,
        })
    # Rank by edge (the value signal), tie-broken by EV.
    picks.sort(key=lambda r: (r["edge"] if r["edge"] is not None else -np.inf,
                              r["expected_value"] if r["expected_value"] is not None else -np.inf),
               reverse=True)
    for rank, pick in enumerate(picks, start=1):
        pick["rank"] = rank
    return picks


def evaluate_filter(scored: pd.DataFrame, config: Optional[ValueConfig] = None) -> dict:
    """Replay the value gates over a backtester OOS frame → validation metrics.

    ``scored`` is a :class:`backtest.engine.BacktestRun.scored` frame (or any frame
    with ``prob``/``bet_price``/``won`` and, for CLV, ``close_price``). It has no
    per-race key, so de-vig is unavailable here (audit C3) and the edge gate uses
    the **raw** implied prob — strictly conservative (raw >= fair) and exactly the
    quantity that drives EV. Returns yield, CLV, beat-close rate, hit rate, A/E by
    odds band, and flat / fractional-Kelly bankroll outcomes — the numbers a
    threshold set must be judged on (CLV + A/E over raw ROI).
    """
    cfg = config or ValueConfig.from_config()
    df = scored.copy()
    p = pd.to_numeric(df["prob"], errors="coerce").to_numpy(dtype=float)
    d = pd.to_numeric(df["bet_price"], errors="coerce").to_numpy(dtype=float)
    won = pd.to_numeric(df["won"], errors="coerce").to_numpy(dtype=float)
    close = (pd.to_numeric(df["close_price"], errors="coerce").to_numpy(dtype=float)
             if "close_price" in df.columns else np.full(len(df), np.nan))

    raw_implied = np.atleast_1d(metrics.implied_prob(d)).astype(float)
    ev = np.atleast_1d(metrics.expected_value(p, d)).astype(float)
    edge = p - raw_implied

    keep = np.isfinite(d) & (d > 1.0) & np.isfinite(p)
    keep &= ev >= cfg.min_ev
    keep &= d >= cfg.min_odds
    keep &= d <= cfg.max_odds
    keep &= p >= cfg.min_prob
    if cfg.min_edge_pct > 0:
        keep &= p >= raw_implied * (1.0 + cfg.min_edge_pct)
    if cfg.min_abs_edge > 0:
        keep &= edge >= cfg.min_abs_edge
    keep = np.where(np.isfinite(keep), keep, False).astype(bool)

    n = int(keep.sum())
    if n == 0:
        return {"n_bets": 0, "yield_pct": None, "clv_pct_mean": None,
                "beat_close_rate": None, "hit_rate": None, "avg_odds": None,
                "ae_by_odds": [], "flat_final_bankroll": cfg.bankroll,
                "kelly_final_bankroll": cfg.bankroll}

    ds, ps, ys, cs = d[keep], p[keep], won[keep], close[keep]
    profit = np.atleast_1d(metrics.settle(1.0, ds, ys)).astype(float)  # flat 1u
    clv = np.atleast_1d(metrics.clv_pct(ds, cs)).astype(float)

    # chronological bankroll walks (flat unit + fractional Kelly)
    order = np.argsort(pd.to_datetime(df.loc[keep, "race_date"]).to_numpy()) \
        if "race_date" in df.columns else np.arange(n)
    flat_unit = cfg.bankroll * 0.01  # 1% flat
    flat_bank = cfg.bankroll
    kelly_bank = cfg.bankroll
    for i in order:
        flat_bank += float(metrics.settle(min(flat_unit, max(flat_bank, 0.0)), ds[i], ys[i]))
        kf = metrics.kelly_fraction(ps[i], ds[i])
        kf = float(kf) if np.isfinite(kf) else 0.0
        stake = min(cfg.kelly_fraction * kf, cfg.kelly_cap) * max(kelly_bank, 0.0)
        kelly_bank += float(metrics.settle(stake, ds[i], ys[i]))

    return {
        "n_bets": n,
        "yield_pct": round(100.0 * float(np.nansum(profit)) / n, 3),
        "clv_pct_mean": round(100.0 * float(np.nanmean(clv)), 3) if np.isfinite(clv).any() else None,
        "beat_close_rate": round(float(np.nanmean((ds > cs).astype(float))), 4),
        "hit_rate": round(float(np.nanmean(ys)), 4),
        "avg_odds": round(float(np.nanmean(ds)), 3),
        "ae_by_odds": metrics.ae_table(ps, ys, group_values=ds, bands=metrics._ODDS_BANDS),
        "flat_final_bankroll": round(flat_bank, 2),
        "kelly_final_bankroll": round(kelly_bank, 2),
    }


# ── small helpers ──────────────────────────────────────────────────────────--

def _num(val) -> float:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return np.nan
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def _str(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return val.isoformat() if hasattr(val, "isoformat") else str(val)


def _round(val, nd: int) -> Optional[float]:
    if val is None or not np.isfinite(val):
        return None
    return round(float(val), nd)


def _src(val) -> Optional[str]:
    """A non-empty source label, or None (NaN/None/'' all collapse to None)."""
    return val if isinstance(val, str) and val else None
