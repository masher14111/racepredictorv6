"""Bet-suggestion engine — "what should I bet on today?".

This is the curation layer that sits on top of the value-detection layer
(:mod:`models.value`), the per-prediction SHAP explainer (:mod:`models.explain`)
and the price-free calibrated model. It turns a day's worth of scored races into
a short, ranked list of **actionable, explainable** suggestions, each carrying:

* the runner, market, model probability, fair vs offered odds, edge %, EV and a
  fractional-Kelly suggested stake (all from :func:`models.value.find_value_bets`);
* a **confidence tier** — ``Strong`` / ``Lean`` (a non-qualifying runner is an
  implicit ``Pass`` and is never surfaced) — derived from edge size, the value
  layer's confidence score, and data completeness;
* a short, human-readable **rationale** assembled from the model's own SHAP
  drivers ("Backed by strong recent speed figures, an in-form trainer …") with an
  honest value-only fallback when no feature row exists for the runner.

Design goals (from the Prompt-22 brief):

* **Curate, don't spam.** At most ``max_per_race`` suggestions per race (default 1)
  and an optional global cap, ranked Strong-before-Lean then by edge. A race with
  no qualifying runner contributes nothing — the engine reports the honest count
  of races scanned vs races with value rather than inventing a pick everywhere.
* **Gated to the trustworthy regime.** All longshot / odds-on / EV / support gates
  come straight from :class:`models.value.ValueConfig` (config.yaml ``value:``), so
  the suggestion set is exactly the backtester-validated favourite band.
* **Reproducible & explainable.** Everything is a deterministic function of the
  cached probabilities + config + exact tree SHAP — no sampling, no hidden state.

Public API
----------
    from models.suggestions import suggest_bets, SuggestionConfig
    book = suggest_bets(races)            # races = predictions.json "races" list
    book.suggestions                      # ranked list[Suggestion]
    book.n_races_with_value               # honest curation counts

``suggest_from_cache()`` is the zero-argument convenience wrapper the UI uses: it
reads ``data/predictions.json``, wires the SHAP explainer + feature store, and
returns the same :class:`SuggestionBook`.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from models.explain import label_for
from models.value import ValueConfig, find_value_bets
from utils.logger import get_logger

logger = get_logger(__name__)

_BASE = Path(__file__).resolve().parent.parent
_CACHE_PATH = _BASE / "data" / "predictions.json"
_FEATURES_PATH = _BASE / "data" / "features.parquet"

# Tier labels (Pass is the implicit state of a runner that clears no value gate).
TIER_STRONG = "Strong"
TIER_LEAN = "Lean"
TIER_PASS = "Pass"
_TIER_RANK = {TIER_STRONG: 0, TIER_LEAN: 1, TIER_PASS: 2}

# Positive-driver → plain-English phrasing for the rationale. A SHAP driver that
# pushed this runner's win chance *up* maps to the phrase below; anything absent
# falls back to its humanised feature label, so a new feature still reads sanely.
_DRIVER_PHRASES = {
    "horse_speed": "strong recent speed figures",
    "horse_speed_rank": "a top speed rating in the field",
    "speed_trend": "improving speed form",
    "jockey_win_rate": "an in-form jockey",
    "trainer_win_rate": "an in-form trainer",
    "jt_combo_win_rate": "a proven jockey/trainer combination",
    "jt_combo_runs": "an established jockey/trainer pairing",
    "course_win_rate": "a winning course record",
    "course_place_rate": "consistent course form",
    "course_runs": "course experience",
    "distance_win_rate": "winning form at this trip",
    "distance_place_rate": "consistent form at this trip",
    "distance_runs": "experience at this trip",
    "going_pref_win_rate": "proven on today's going",
    "going_pref_place_rate": "well-suited to the going",
    "historical_win_rate": "a strong recent strike-rate",
    "historical_place_rate": "consistent recent form",
    "field_size": "a favourable field size",
    "race_complexity": "a race shape that suits",
    "distance_furlongs": "a suitable trip",
    "days_since_last_run": "ideal freshness",
    "horse_career_runs": "proven experience",
}


@dataclass
class SuggestionConfig:
    """Tier thresholds + curation knobs for the suggestion engine.

    The *selection* gates (odds band, EV, support, confidence floor, devig) all
    live in :class:`models.value.ValueConfig`; this config only governs how the
    runners that already cleared those gates are **tiered and curated**.
    """

    # ── Strong-tier bars (all three must hold; otherwise the pick is a Lean) ──
    strong_edge_pct: float = 0.15      # model_prob ≥ fair*(1+this) — a clear price edge
    strong_confidence: float = 0.60    # the value layer's [0,1] confidence score
    strong_completeness: float = 0.55  # fraction of signal features populated
    # ── curation ──
    max_per_race: int = 1              # cap suggestions per race (anti-spam)
    max_suggestions: Optional[int] = None  # optional global cap (None = all)
    rationale_top_k: int = 3           # SHAP drivers folded into the rationale

    @classmethod
    def from_config(cls, raw: Optional[dict] = None) -> "SuggestionConfig":
        """Build from the optional ``suggestions:`` config.yaml block (all keys
        optional; absent ⇒ the validated defaults above)."""
        from utils.config_loader import get_config
        raw = raw if raw is not None else get_config()
        s = raw.get("suggestions") or {}
        return cls(
            strong_edge_pct=float(s.get("strong_edge_pct", 0.15)),
            strong_confidence=float(s.get("strong_confidence", 0.60)),
            strong_completeness=float(s.get("strong_completeness", 0.55)),
            max_per_race=int(s.get("max_per_race", 1)),
            max_suggestions=(int(s["max_suggestions"])
                             if s.get("max_suggestions") is not None else None),
            rationale_top_k=int(s.get("rationale_top_k", 3)),
        )


@dataclass
class Suggestion:
    """One actionable, explained bet recommendation.

    Carries everything the UI needs to render a suggestion card and pre-fill the
    one-click paper-bet form, plus the provenance (drivers, tier reasons) that
    keeps it explainable rather than a black box.
    """

    # ── identity ──
    venue: str
    race_time: str
    race_slug: str
    horse_id: str
    horse_name: str
    jockey: str
    trainer: str
    # ── market & price ──
    market: str                 # "Win" (the market value is computed on)
    each_way_available: bool
    offered_odds: Optional[float]   # the price you'd actually take (best board price)
    best_book: Optional[str]
    fair_odds: Optional[float]      # 1 / de-vigged fair prob — what it "should" be
    # ── value maths (from models.value) ──
    model_prob: Optional[float]
    fair_prob: Optional[float]
    market_implied_prob: Optional[float]
    overround: Optional[float]
    edge: Optional[float]           # model_prob − fair_prob (probability points)
    edge_pct: Optional[float]       # model_prob / fair_prob − 1 (relative)
    expected_value: Optional[float]  # model_prob·odds − 1
    kelly_fraction: Optional[float]
    suggested_stake: Optional[float]
    # ── confidence & tier ──
    value_confidence: Optional[float]   # value layer's [0,1] score
    data_confidence: Optional[str]      # predictor's high|med|low
    data_completeness: Optional[float]
    first_time_runner: bool
    tier: str                       # "Strong" | "Lean"
    tier_reasons: list              # human strings explaining the tier
    # ── explanation ──
    rationale: str                  # one-line, human-readable
    rationale_drivers: list         # [{feature, label, value, contribution}, ...]
    # ── ranking ──
    rank: int = 0                   # global rank across the whole suggestion book

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SuggestionBook:
    """The curated set of suggestions for a day, with honest coverage counts."""

    suggestions: list = field(default_factory=list)  # list[Suggestion]
    n_races: int = 0
    n_runners: int = 0
    n_races_with_value: int = 0
    n_strong: int = 0
    n_lean: int = 0
    bankroll: float = 0.0
    generated_at: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["suggestions"] = [s.to_dict() if isinstance(s, Suggestion) else s
                            for s in self.suggestions]
        return d


# ── rationale ────────────────────────────────────────────────────────────────

def _driver_phrase(feature: str) -> str:
    return _DRIVER_PHRASES.get(feature) or label_for(feature).lower()


def _join_phrases(phrases: list) -> str:
    """Join 1-3 phrases into natural English (a, b and c)."""
    phrases = [p for p in phrases if p]
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def build_rationale(
    explanation: Optional[dict],
    *,
    model_prob: Optional[float],
    fair_prob: Optional[float],
    edge: Optional[float],
    offered_odds: Optional[float],
    expected_value: Optional[float],
    top_k: int = 3,
) -> tuple[str, list]:
    """Assemble a one-line rationale + the structured drivers behind it.

    The "backed by …" clause names the model's strongest *positive* SHAP drivers
    in plain English; the value clause states the model-vs-market disagreement and
    the price. When no SHAP explanation exists for the runner (the live feature
    store has no row for it — a known, honest gap) the rationale degrades to the
    value clause alone rather than fabricating form it can't see.
    """
    drivers = []
    phrases: list = []
    if explanation:
        seen: set = set()
        for d in (explanation.get("top_positive") or []):
            if float(d.get("contribution", 0.0)) <= 0:
                continue
            phrase = _driver_phrase(d.get("feature", ""))
            if phrase in seen:
                continue
            seen.add(phrase)
            phrases.append(phrase)
            drivers.append(d)
            if len(phrases) >= top_k:
                break

    # value clause — always present, always honest:
    #   "model rates it 40% vs the market's 27% — a +13pp edge at 3.50 (EV +40%)"
    head = ""
    if model_prob is not None and fair_prob is not None:
        head = (f"model rates it {model_prob * 100:.0f}% vs the market's "
                f"{fair_prob * 100:.0f}%")
    tail_bits = []
    if edge is not None:
        tail_bits.append(f"a {edge * 100:+.0f}pp edge")
    if offered_odds is not None:
        price = f"at {offered_odds:.2f}"
        if expected_value is not None:
            price += f" (EV {expected_value * 100:+.0f}%)"
        tail_bits.append(price)
    tail = " ".join(tail_bits)
    value_clause = " — ".join(filter(None, [head, tail]))

    if phrases:
        backed = f"Backed by {_join_phrases(phrases)}."
        rationale = (f"{backed} {value_clause[0].upper()}{value_clause[1:]}."
                     if value_clause else backed)
    else:
        rationale = (f"{value_clause[0].upper()}{value_clause[1:]}."
                     if value_clause else "No drivers available.")
    return rationale.strip(), drivers


# ── tiering ──────────────────────────────────────────────────────────────────

def assign_tier(
    *,
    edge_pct: Optional[float],
    value_confidence: Optional[float],
    completeness: Optional[float],
    data_confidence: Optional[str],
    first_time: bool,
    cfg: SuggestionConfig,
) -> tuple[str, list]:
    """Tier a runner that has already cleared the value gates.

    ``Strong`` requires **all three** of: a clear relative edge, a high value-layer
    confidence score, and complete-enough, non-debutant data. Anything that clears
    the gates but misses a Strong bar is a ``Lean``. Returns ``(tier, reasons)``
    where ``reasons`` are short human strings (shown in the UI tooltip) explaining
    exactly which bars were met or missed — so the tier is never opaque.
    """
    reasons: list = []
    strong = True

    # edge
    bar = cfg.strong_edge_pct
    if edge_pct is not None and edge_pct >= bar:
        reasons.append(f"edge {edge_pct * 100:.0f}% ≥ {bar * 100:.0f}% bar")
    else:
        strong = False
        ep = "n/a" if edge_pct is None else f"{edge_pct * 100:.0f}%"
        reasons.append(f"edge {ep} below {bar * 100:.0f}% strong bar")

    # value-layer confidence
    vc = value_confidence if value_confidence is not None else 0.0
    if vc >= cfg.strong_confidence:
        reasons.append(f"confidence {vc:.2f} ≥ {cfg.strong_confidence:.2f}")
    else:
        strong = False
        reasons.append(f"confidence {vc:.2f} below {cfg.strong_confidence:.2f}")

    # data completeness (a debutant or unknown completeness is never Strong —
    # Strong implies the form features the model leans on are genuinely present)
    if first_time:
        strong = False
        reasons.append("first-time runner — limited form")
    elif (data_confidence or "").lower() == "low":
        strong = False
        reasons.append("low data confidence")
    elif completeness is None:
        strong = False
        reasons.append("data completeness unknown")
    elif completeness >= cfg.strong_completeness:
        reasons.append(f"data {completeness * 100:.0f}% complete")
    else:
        strong = False
        reasons.append(f"data {completeness * 100:.0f}% below "
                       f"{cfg.strong_completeness * 100:.0f}% bar")

    return (TIER_STRONG if strong else TIER_LEAN), reasons


# ── helpers ──────────────────────────────────────────────────────────────────

def _race_slug(venue: str, race_time: str) -> str:
    """Stable race id matching ui._components.race_slug (venue + ISO post-time)."""
    v = (venue or "race").strip().lower()
    v = "".join(c if c.isalnum() else "-" for c in v).strip("-")
    t = (race_time or "").replace(":", "").replace("-", "")[:13]
    return f"{v}-{t}" if t else v


def _runner_index(race: dict) -> dict:
    """Map horse_id → full runner dict so a value pick can be re-joined to its
    connections / data-quality fields.

    Audit req 2: index the COMPLETE declared field (``runners``) first, so a value
    pick anywhere in the field — not just the display top three — resolves to its
    runner dict. ``selections`` + ``excluded_low_odds`` are folded in for
    backward-compatibility with older caches that predate the ``runners`` key."""
    out: dict = {}
    sources = (
        (race.get("runners") or [])
        + (race.get("selections") or [])
        + (race.get("excluded_low_odds") or [])
    )
    for r in sources:
        hid = str(r.get("horse_id") or "")
        if hid:
            out.setdefault(hid, r)
    return out


def _explain_for(horse_id: str, feature_rows: Optional[dict],
                 explainer) -> Optional[dict]:
    """Best-effort SHAP explanation for one runner (None when unavailable)."""
    if not feature_rows or explainer is None:
        return None
    row = feature_rows.get(str(horse_id))
    if not row:
        return None
    try:
        return explainer.explain(row, top_k=6)
    except Exception as exc:  # noqa: BLE001
        logger.debug("suggestions: explain failed for %s: %s", horse_id, exc)
        return None


# ── public API ───────────────────────────────────────────────────────────────

def suggest_bets(
    races: list,
    *,
    value_config: Optional[ValueConfig] = None,
    config: Optional[SuggestionConfig] = None,
    bankroll: Optional[float] = None,
    feature_rows: Optional[dict] = None,
    explainer=None,
    generated_at: Optional[str] = None,
) -> SuggestionBook:
    """Curate a ranked, explained suggestion book from a day's scored races.

    ``races`` is the ``predictions.json`` ``"races"`` list (each a predictor race
    dict with ``selections`` / ``excluded_low_odds``). For each race we run the
    field through :func:`models.value.find_value_bets` (de-vig + every config gate),
    keep the top ``max_per_race`` qualifying runners, tier and explain each, then
    rank the whole set Strong-before-Lean and by edge.

    ``feature_rows`` (``horse_id`` → feature-row dict) and ``explainer`` (a
    :class:`models.explain.Explainer` for the ``won`` target) are optional; supply
    them to enrich the rationale with SHAP drivers. Without them the rationale is
    the honest value-only clause.
    """
    vcfg = value_config or ValueConfig.from_config()
    scfg = config or SuggestionConfig.from_config()
    bank = float(bankroll) if bankroll is not None else vcfg.bankroll

    book = SuggestionBook(bankroll=bank, generated_at=generated_at)
    candidates: list = []

    for race in races or []:
        venue = str(race.get("venue") or "")
        race_time = str(race.get("race_time") or "")
        ew_available = bool(race.get("each_way_available", False))
        index = _runner_index(race)
        book.n_races += 1
        book.n_runners += len(index)

        picks = find_value_bets(race, config=vcfg, bankroll=bank)
        if not picks:
            continue
        book.n_races_with_value += 1

        for pick in picks[: max(scfg.max_per_race, 0)]:
            runner = index.get(str(pick.get("horse_id") or ""), {})
            completeness = runner.get("data_completeness")
            data_conf = runner.get("confidence")
            first_time = bool(runner.get("first_time_runner", False))

            tier, tier_reasons = assign_tier(
                edge_pct=pick.get("edge_pct"),
                value_confidence=pick.get("confidence"),
                completeness=completeness,
                data_confidence=data_conf if isinstance(data_conf, str) else None,
                first_time=first_time,
                cfg=scfg,
            )

            explanation = _explain_for(pick.get("horse_id"), feature_rows, explainer)
            rationale, drivers = build_rationale(
                explanation,
                model_prob=pick.get("model_prob"),
                fair_prob=pick.get("fair_prob"),
                edge=pick.get("edge"),
                offered_odds=pick.get("decimal_odds"),
                expected_value=pick.get("expected_value"),
                top_k=scfg.rationale_top_k,
            )

            fair_prob = pick.get("fair_prob")
            fair_odds = (round(1.0 / fair_prob, 2)
                         if fair_prob not in (None, 0) else None)

            candidates.append(Suggestion(
                venue=venue,
                race_time=race_time,
                race_slug=_race_slug(venue, race_time),
                horse_id=str(pick.get("horse_id") or ""),
                horse_name=str(pick.get("horse_name") or runner.get("horse_name") or ""),
                jockey=str(runner.get("jockey") or ""),
                trainer=str(runner.get("trainer") or ""),
                market="Win",
                each_way_available=ew_available,
                offered_odds=pick.get("decimal_odds"),
                best_book=(runner.get("best_book")
                           if isinstance(runner.get("best_book"), str) else None),
                fair_odds=fair_odds,
                model_prob=pick.get("model_prob"),
                fair_prob=fair_prob,
                market_implied_prob=pick.get("market_implied_prob"),
                overround=pick.get("overround"),
                edge=pick.get("edge"),
                edge_pct=pick.get("edge_pct"),
                expected_value=pick.get("expected_value"),
                kelly_fraction=pick.get("kelly_fraction"),
                suggested_stake=pick.get("suggested_stake"),
                value_confidence=pick.get("confidence"),
                data_confidence=data_conf if isinstance(data_conf, str) else None,
                data_completeness=completeness,
                first_time_runner=first_time,
                tier=tier,
                tier_reasons=tier_reasons,
                rationale=rationale,
                rationale_drivers=drivers,
            ))

    # Rank: Strong before Lean, then by edge, then by EV (all descending value).
    candidates.sort(key=lambda s: (
        _TIER_RANK.get(s.tier, 9),
        -(s.edge if s.edge is not None else -np.inf),
        -(s.expected_value if s.expected_value is not None else -np.inf),
    ))
    if scfg.max_suggestions is not None:
        candidates = candidates[: scfg.max_suggestions]
    for i, s in enumerate(candidates, start=1):
        s.rank = i

    book.suggestions = candidates
    book.n_strong = sum(1 for s in candidates if s.tier == TIER_STRONG)
    book.n_lean = sum(1 for s in candidates if s.tier == TIER_LEAN)
    logger.info(
        "suggestions: %d suggestion(s) (%d strong / %d lean) from %d races "
        "(%d with value)",
        len(candidates), book.n_strong, book.n_lean, book.n_races,
        book.n_races_with_value,
    )
    return book


# ── cache / explainer wiring (used by the UI) ────────────────────────────────

def _load_cache() -> dict:
    try:
        with open(_CACHE_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def _load_feature_rows() -> dict:
    """horse_id → feature-row dict from the feature store (best-effort, {} if absent)."""
    if not _FEATURES_PATH.exists():
        return {}
    try:
        df = pd.read_parquet(_FEATURES_PATH)
        if "horse_id" not in df.columns:
            return {}
        return {str(r["horse_id"]): r.to_dict() for _, r in df.iterrows()}
    except Exception as exc:  # noqa: BLE001
        logger.debug("suggestions: feature store unreadable: %s", exc)
        return {}


def _load_explainer():
    """Cached price-free ``won`` explainer, or None when the model is unavailable."""
    try:
        from models.explain import get_explainer
        return get_explainer(target="won", version_tag="v3nf")
    except Exception as exc:  # noqa: BLE001
        logger.debug("suggestions: explainer unavailable: %s", exc)
        return None


def suggest_from_cache(
    *,
    value_config: Optional[ValueConfig] = None,
    config: Optional[SuggestionConfig] = None,
    bankroll: Optional[float] = None,
    with_explanations: bool = True,
) -> SuggestionBook:
    """Convenience wrapper: read ``data/predictions.json``, wire the SHAP explainer
    + feature store, and return a :class:`SuggestionBook`. The UI's entry point."""
    cache = _load_cache()
    races = cache.get("races") or []
    feature_rows = _load_feature_rows() if with_explanations else None
    explainer = _load_explainer() if with_explanations else None
    return suggest_bets(
        races,
        value_config=value_config,
        config=config,
        bankroll=bankroll,
        feature_rows=feature_rows,
        explainer=explainer,
        generated_at=cache.get("generated_at"),
    )
