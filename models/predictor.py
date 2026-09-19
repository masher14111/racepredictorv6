"""Real-time race prediction engine.

Loads trained CatBoost models, runs feature engineering over live data,
and emits top-3 selections per race with probability scores, each-way flags,
and low-odds filtering.

Usage (programmatic):
    from models.predictor import predict
    races = predict()          # list[dict], one per upcoming race

    p = Predictor()            # reuse a single Predictor across polling cycles
    p.load()
    races = p.refresh()

Usage (CLI):
    python -m models.predictor [--model-dir PATH] [--min-odds 2.5]
"""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from features.builder import _live_mask, build_inference_matrix
from models.calibration import normalize_within_race
from models.features import (
    EMPIRICALLY_DEAD_COLS,
    FEATURE_COLS,
    PRICE_FREE_FEATURE_COLS,
)
from utils import alert_dedupe
from utils.config_loader import get_config
from utils.logger import get_logger
from utils.notifications import get_notifier
from utils.timezone import now
from utils.market_validation import VALID

logger = get_logger(__name__)

_BASE = Path(__file__).resolve().parent.parent
_CACHE_PATH = _BASE / "data" / "predictions.json"
# Stage 20 (B7): the "saved manifest" a ticket's provenance hashes must resolve
# to — a latest-state snapshot (like data/source_health.json), not an
# append-only audit log; overwritten every cycle on purpose.
_MODEL_MANIFEST_PATH = _BASE / "data" / "execution" / "served_model_manifest.json"

_TARGET_ORDER = ["won", "placed_2", "showed"]
# Composite score weights — win probability drives ranking, place/show refine it.
_COMPOSITE_W = {"won": 0.50, "placed_2": 0.30, "showed": 0.20}
_TOP_N = 3
# A race is a (venue, post-time) pair. Grouping by venue-day instead collapses
# every race at a track into one card — see predict() (with a race_date fallback
# for any row missing race_time).
_RACE_KEY = ["venue", "race_time"]
# Live bookmakers whose per-runner board prices we surface individually (so the
# user can back the best). betsp/timeform are historical SP / ratings, not a
# price you could take, so they never appear in the odds-by-book map.
_BOOKMAKER_SOURCES = ("livescorebet", "paddy_power", "boylesports")
# Runners flagged as non-runners are dropped entirely before ranking — the
# unified schema has no dedicated flag, so we detect them via jockey_name.
_NON_RUNNER_JOCKEY = "non runner"

# ── data-completeness / confidence thresholds ──────────────────────────────────
# A runner's data_completeness is the fraction of *signal-bearing* features
# (FEATURE_COLS minus the empirically-dead, always-null ones) that are populated.
# Confidence buckets it for the UI; a first-time runner is always "low" no matter
# how complete its market/context features are, because the form features the
# model leans on are necessarily absent.
_SIGNAL_FEATURE_COLS = [c for c in FEATURE_COLS if c not in EMPIRICALLY_DEAD_COLS]
_CONF_HIGH_MIN = 0.80
_CONF_LOW_MAX = 0.40


@dataclass
class RunnerPrediction:
    """One scored runner — the typed unit the UI consumes.

    ``won_prob_normalized`` is the **headline** win probability the UI/API
    present: each race's field is rescaled to sum to ~1 (exactly one winner) and,
    out-of-sample, it is well-calibrated (AUC 0.78 / ECE 0.03 on the last settled
    week). ``won_prob`` is the raw per-runner calibrated *marginal* — retained as
    a clearly-named debugging / EV-reference column but NOT shown as the headline:
    its v3 isotonic curve was fit on a narrow historical slice and saturates badly
    OOS (the live field piles up near a 0.73 ceiling, mean 0.33 vs a true 0.11 win
    rate, ECE 0.22). Place/show stay coherent (``placed_2_prob`` ≥ ``won_prob``,
    ``showed_prob`` ≥ ``placed_2_prob``). ``confidence`` / ``data_completeness`` /
    ``first_time_runner`` flag how much real signal backed the score, so a
    debutant's base-rate guess is never mistaken for a strong read. (See
    calib-fl-01.)
    """

    rank: Optional[int]
    horse_id: str
    horse_name: str
    jockey: str
    trainer: str
    decimal_odds: Optional[float]
    odds_by_book: dict
    best_odds: Optional[float]
    best_book: Optional[str]
    implied_prob: float
    won_prob: Optional[float]
    won_prob_normalized: Optional[float]
    placed_2_prob: Optional[float]
    showed_prob: Optional[float]
    composite_score: float
    each_way_value: bool
    low_odds: bool
    value_win_prob: Optional[float]
    value_edge: Optional[float]
    expected_value: Optional[float]
    value_bet: bool
    value_supported: bool
    # ── data-quality layer ──
    data_completeness: Optional[float]
    confidence: str
    first_time_runner: bool
    # ── audit req 8: de-vig provenance + empirical rate, typed & persisted ──
    # reference_odds/_source: the price (and book) the fair line de-vigs — kept
    # strictly separate from best_odds (executable, drives EV). empirical_win_rate
    # is the horse's OWN trailing historical win rate (historical_win_rate
    # feature) — never to be conflated with a model probability.
    reference_odds: Optional[float] = None
    reference_source: Optional[str] = None
    empirical_win_rate: Optional[float] = None
    # Stage-4 audit (live-repair-04): the price-free model's calibrated
    # probability BEFORE the favourite-longshot market adjustment.
    # ``value_win_prob`` is this value remapped against the effective decimal
    # price (market-ADJUSTED); this field is the only persisted value-layer
    # probability the offered price never touched. None on pre-Stage-4 caches.
    value_win_prob_independent: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunnerPrediction":
        """Rehydrate from a cached predictions.json runner dict (tolerates older
        caches missing the newer keys)."""
        names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: d.get(k) for k in names})


@dataclass
class RacePrediction:
    """One race: the full validated field, the top-N selections, the low-odds
    runners, and the race-level EV eligibility decision."""

    venue: str
    race_time: str
    field_size: int
    each_way_available: bool
    selections: list  # list[RunnerPrediction] — display-only top N
    excluded_low_odds: list  # list[RunnerPrediction]
    # Complete validated field (audit req 2) — every valid runner, ranked.
    runners: list = field(default_factory=list)  # list[RunnerPrediction]
    # Race-level EV gate (audit req 9): eligible flag + full diagnostics dict
    # (PASS reasons, reference source/overround, price ages, computed_at).
    ev_eligible: Optional[bool] = None
    ev_gate: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "venue": self.venue,
            "race_time": self.race_time,
            "field_size": self.field_size,
            "each_way_available": self.each_way_available,
            "ev_eligible": self.ev_eligible,
            "ev_gate": self.ev_gate,
            "selections": [r.to_dict() for r in self.selections],
            "excluded_low_odds": [r.to_dict() for r in self.excluded_low_odds],
            "runners": [r.to_dict() for r in self.runners],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RacePrediction":
        return cls(
            venue=d.get("venue", ""),
            race_time=d.get("race_time", ""),
            field_size=int(d.get("field_size", 0)),
            each_way_available=bool(d.get("each_way_available", False)),
            selections=[RunnerPrediction.from_dict(r) for r in d.get("selections", [])],
            excluded_low_odds=[
                RunnerPrediction.from_dict(r) for r in d.get("excluded_low_odds", [])
            ],
            runners=[RunnerPrediction.from_dict(r) for r in d.get("runners", [])],
            ev_eligible=d.get("ev_eligible"),
            ev_gate=d.get("ev_gate"),
        )


def _race_time_key(df: pd.DataFrame) -> pd.Series:
    """Per-race time component: race_time when present and non-null, else
    race_date. A race is (venue, this) — see _RACE_KEY."""
    rd = df.get("race_date")
    if "race_time" in df.columns:
        return df["race_time"].where(df["race_time"].notna(), rd)
    return rd


def _race_group_ids(df: pd.DataFrame) -> list:
    """(venue, race-time) tuples identifying each runner's race — used to group
    a field for within-race win-probability normalization."""
    tk = _race_time_key(df)
    return list(zip(df["venue"].astype(str), pd.Series(tk).astype(str)))


def _is_non_runner(grp: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows that are non-runners/withdrawn (excluded from output).

    Two independent signals, because they come from different sources and
    either alone leaves a withdrawn horse in the field:
      * ``runner_status`` — the racecard's own declaration, carried from
        scraper/timeform/parser.py through the normalizer (step 05). Until the
        step 09 audit nothing read it, so a declared non-runner still reached
        the book unless the odds feed also blanked its jockey.
      * ``jockey_name == "non runner"`` — the marker an odds feed writes when it
        has no racecard status column at all.
    """
    mask = pd.Series(False, index=grp.index)
    status = grp.get("runner_status")
    if status is not None:
        # An all-null column (e.g. runner_status when no declared card has
        # arrived) is inferred as pyarrow's null[pyarrow] dtype; fillna("")
        # on that dtype raises ArrowInvalid, so normalize through object first.
        mask |= status.astype(object).fillna("").astype(str).str.strip().str.upper() == "NON_RUNNER"
    jockey = grp.get("jockey_name")
    if jockey is not None:
        mask |= (jockey.astype(object).fillna("").astype(str).str.strip().str.lower()
                 == _NON_RUNNER_JOCKEY)
    return mask


def _load_cfg() -> dict:
    raw = get_config()
    m = raw.get("model") or {}
    v = raw.get("value") or {}
    return {
        "each_way_threshold": float(raw.get("each_way_threshold", 8.0)),
        "model_dir": str(m.get("model_dir", "models")),
        "targets": list(m.get("targets", _TARGET_ORDER)),
        "version_tag": str(m.get("version_tag", "v3")),
        # ── value-betting layer (price-free model vs market price) ──
        "value_enabled": bool(v.get("enabled", False)),
        "value_model_tag": str(v.get("model_tag", "v3nf")),
        "value_min_ev": float(v.get("min_expected_value", 0.05)),
        "value_min_odds": float(v.get("min_odds", 2.0)),
        "value_max_odds": float(v.get("max_odds", 4.0)),
        # Favourite-longshot recalibration of the price-free win prob (default on).
        "value_fl_recalibration": bool(v.get("fl_recalibration", True)),
    }


class Predictor:
    """Load CatBoost models once; score live runners on demand.

    Typical usage in a polling loop::

        p = Predictor()
        p.load()               # reads .bin files from disk once
        while True:
            races = p.refresh()   # rebuild inference matrix + re-score
            time.sleep(60)
    """

    # Minimum decimal odds for a runner to appear in the top-3 selections.
    # Shorter-priced horses are surfaced separately as excluded_low_odds so the
    # UI can still display them — they just don't count as model selections.
    DEFAULT_MIN_ODDS: float = 2.50

    def __init__(
        self,
        model_dir: Optional[str] = None,
        min_selection_odds: Optional[float] = None,
    ) -> None:
        cfg = _load_cfg()
        self._dir = Path(model_dir) if model_dir else _BASE / cfg["model_dir"]
        self._min_odds: float = (
            min_selection_odds if min_selection_odds is not None else self.DEFAULT_MIN_ODDS
        )
        # each_way_threshold from config: minimum decimal odds for EW recommendation.
        self._ew_threshold: float = cfg["each_way_threshold"]
        self._targets: list[str] = cfg["targets"]
        self._version_tag: str = cfg["version_tag"]
        self._models: dict[str, CatBoostClassifier] = {}
        # feature_cols may be overridden by meta.json (reflects columns actually trained on)
        self._feature_cols: list[str] = list(FEATURE_COLS)

        # ── probability calibrators (isotonic), one per target ──
        # Applied in _score() when present; absent → raw probs pass through.
        # MUST be initialized here: tests inject _models directly without load().
        self._calibrators: dict[str, object] = {}

        # ── value-betting layer (price-free model vs market price) ──
        self._value_enabled: bool = cfg["value_enabled"]
        self._value_model_tag: str = cfg["value_model_tag"]
        self._value_min_ev: float = cfg["value_min_ev"]
        self._value_min_odds: float = cfg["value_min_odds"]
        self._value_max_odds: float = cfg["value_max_odds"]
        self._value_model: Optional[CatBoostClassifier] = None
        self._value_calibrator: Optional[object] = None
        self._value_feature_cols: list[str] = list(PRICE_FREE_FEATURE_COLS)
        # ── favourite-longshot recalibrator (OddsBandCalibrator) ──
        # Corrects the price-free prob's odds-conditional bias against the market
        # price; loaded in _load_value_model when enabled + artifact present, else
        # None (value_win_prob then keeps its market-independent calibration).
        self._value_fl_recalibration: bool = cfg["value_fl_recalibration"]
        self._value_fl_calibrator: Optional[object] = None

    # ── model I/O ─────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """Load all .bin models and meta from disk.  Returns True if ≥1 loaded."""
        meta_path = self._dir / f"catboost_{self._version_tag}_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            fc = meta.get("feature_cols")
            if fc:
                self._feature_cols = fc
            logger.info("predictor: meta loaded — %d feature cols", len(self._feature_cols))

        loaded = 0
        for target in self._targets:
            path = self._dir / f"catboost_{target}_{self._version_tag}.bin"
            if not path.exists():
                logger.warning("predictor: %s not found — target '%s' skipped", path.name, target)
                continue
            model = CatBoostClassifier()
            model.load_model(str(path))
            self._models[target] = model
            loaded += 1
            logger.info("predictor: loaded %s", path.name)

            # Optional isotonic calibrator for this target — raw probs pass
            # through unchanged when absent.
            calib = _load_calibrator(
                self._dir / f"catboost_{target}_{self._version_tag}_calib.pkl"
            )
            if calib is not None:
                self._calibrators[target] = calib
                logger.info("predictor: loaded calibrator for '%s'", target)

        if not loaded:
            logger.error(
                "predictor: no trained models found in %s — "
                "run `python -m models.train` first",
                self._dir,
            )
            return False

        if self._value_enabled:
            self._load_value_model()

        return True

    def _load_value_model(self) -> None:
        """Load the price-free win model + calibrator + feature cols (all optional).

        The value layer needs a probability that is *independent of market price*
        so that edge-vs-market isn't circular. Any missing artefact disables the
        layer gracefully (value fields then default to None / False downstream).
        """
        tag = self._value_model_tag
        path = self._dir / f"catboost_won_{tag}.bin"
        if not path.exists():
            logger.warning(
                "predictor: value model %s not found — value layer disabled",
                path.name,
            )
            return

        model = CatBoostClassifier()
        model.load_model(str(path))
        self._value_model = model
        logger.info("predictor: loaded value model %s", path.name)

        meta_path = self._dir / f"catboost_{tag}_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    fc = json.load(fh).get("feature_cols")
                if fc:
                    self._value_feature_cols = fc
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("predictor: value meta read failed: %s", exc)

        self._value_calibrator = _load_calibrator(
            self._dir / f"catboost_won_{tag}_calib.pkl"
        )
        if self._value_calibrator is not None:
            logger.info("predictor: loaded value calibrator")
        else:
            logger.warning(
                "predictor: no value calibrator on disk — value win-prob is RAW "
                "(likely inflated); run `python -m models.train --reuse-params` to fit it"
            )

        # Favourite-longshot recalibrator (OddsBandCalibrator). Optional + config-
        # gated: corrects the price-free prob's odds-conditional bias (under-rates
        # favourites, over-rates longshots) by remapping it against the pre-off
        # market price. A missing artefact or a disabled flag is a graceful no-op —
        # value_win_prob then stays the market-independent calibrated prob.
        if self._value_fl_recalibration:
            self._value_fl_calibrator = _load_calibrator(
                self._dir / f"fl_oddsband_{tag}_calib.pkl"
            )
            if self._value_fl_calibrator is not None:
                logger.info(
                    "predictor: loaded F-L recalibrator fl_oddsband_%s_calib.pkl", tag
                )
            else:
                logger.info(
                    "predictor: no F-L recalibrator on disk — value win-prob keeps its "
                    "market-independent calibration (run "
                    "`python -m docs.calibration.fit_fl_recalibrator` to build it)"
                )
        else:
            logger.info("predictor: F-L recalibration disabled by config")

    # ── scoring ───────────────────────────────────────────────────────────────

    def _feature_frame(self, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        """Build the model input matrix with EXACTLY ``cols``, in order.

        Any feature absent from the live matrix is inserted as an all-NaN column so
        the matrix always has the width *and column order* CatBoost trained on —
        dropping/realigning columns silently would feed the model the wrong feature
        in each slot and produce garbage probabilities.

        Missing values are left as NaN rather than filled with 0/mean: CatBoost
        learns a dedicated 'missing' split direction at train time, so NaN is the
        imputation that matches training. First-time runners (no form), unknown
        jockey/trainer, and no-recent-form rows therefore route down the branch the
        model already calibrated for absent data instead of being pushed to a
        fabricated value that looks like real signal.
        """
        X = pd.DataFrame(index=df.index)
        for c in cols:
            X[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
        return X.to_numpy(dtype=float)

    def _score(self, live: pd.DataFrame) -> pd.DataFrame:
        """Apply each loaded model; append *_prob columns + composite_score."""
        df = live.copy()
        # Audit req 1: strip non-runners BEFORE any within-race normalization,
        # market de-vigging, or field-size calculation. A withdrawn horse left in
        # the frame would (a) steal probability mass in the sum-to-1 normalization,
        # (b) corrupt the de-vigged market book, and (c) inflate field size. Drop
        # them once here so every downstream field-level computation sees only the
        # real, declared field. _build_race keeps a defensive re-drop.
        non_runner = _is_non_runner(df)
        if bool(non_runner.any()):
            logger.debug("predictor: dropping %d non-runner row(s) before scoring",
                         int(non_runner.sum()))
            df = df[~non_runner].copy().reset_index(drop=True)
        missing = [c for c in self._feature_cols if c not in df.columns]
        if missing:
            logger.debug("predictor: %d feature cols absent from live matrix (NaN-filled): %s",
                         len(missing), sorted(missing))

        X = self._feature_frame(df, self._feature_cols)

        for target, model in self._models.items():
            col = f"{target}_prob"
            try:
                raw = model.predict_proba(X)[:, 1]
            except Exception as exc:  # noqa: BLE001
                logger.warning("predictor: predict_proba('%s') failed: %s", target, exc)
                df[col] = np.nan
                continue
            # Isotonic calibration is monotonic → ranking preserved, only the
            # probability value is corrected. Absent calibrator → raw passes through.
            calib = self._calibrators.get(target)
            df[col] = calib.predict(raw) if calib is not None else raw

        # Coherence: winning ⊆ placing (top-2) ⊆ showing (top-3), so the true
        # probabilities must satisfy P(won) ≤ P(placed_2) ≤ P(showed). The three
        # binary models are fit independently and can violate this; clip each
        # downstream target up to its predecessor so place/show never undercut a
        # higher win probability. NaN (a failed predict) skips cleanly.
        order = [t for t in _TARGET_ORDER if f"{t}_prob" in df.columns]
        for prev, cur in zip(order, order[1:]):
            a, b = f"{prev}_prob", f"{cur}_prob"
            df[b] = df[[a, b]].max(axis=1)

        # weighted composite — only over targets that were actually scored
        composite = np.zeros(len(df))
        wsum = 0.0
        for t, w in _COMPOSITE_W.items():
            col = f"{t}_prob"
            if col in df.columns:
                composite += w * pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
                wsum += w
        df["composite_score"] = composite / wsum if wsum else 0.0

        # Within-race normalization: exactly one runner wins, so a coherent win
        # market must have the field's win probabilities sum to 1. Per-runner
        # calibration enforces no such constraint, leaving the field sum drifting
        # off 1. We compute the normalized win prob as a SEPARATE column and keep
        # won_prob (the raw calibrated marginal) untouched — but the normalized
        # column is the HEADLINE the UI/API present.
        #
        # Why normalized is the headline (was: "won_prob is marginally excellent,
        # keep as headline" — false OOS): the v3 `won` isotonic calibrator was fit
        # on a narrow historical slice, and out-of-sample the model's raw output
        # compresses into a tiny band [~0.28, 0.51] that the steep isotonic curve
        # maps onto y∈[0.007, 0.734]. So the live field saturates near the 0.734
        # ceiling — mean 0.33 vs a true 0.11 win rate, ECE 0.22 over 2026-06-06..12.
        # Within-race normalization re-anchors the field to base rate (sum-to-1 ⇒
        # field mean = base rate) and on the same week scores AUC 0.778 / ECE 0.029,
        # strictly better than the marginal. won_prob is therefore retained only as
        # a clearly-named raw/debug column (and the EV layer's reference). See
        # calib-fl-01. place/show are not mutually exclusive and are not normalized.
        group_ids = _race_group_ids(df) if "venue" in df.columns and len(df) else None
        if "won_prob" in df.columns and group_ids is not None:
            df["won_prob_normalized"] = normalize_within_race(df["won_prob"], group_ids)

        self._score_value(df)
        self._add_data_quality(df)
        # Additive multi-line enrichment: LightGBM win line, de-vigged market prob,
        # per-line EV. Best-effort — never breaks the CatBoost line (see _enrich_unified).
        if group_ids is not None:
            self._enrich_unified(df, group_ids)
        return df

    def _enrich_unified(self, df: pd.DataFrame, group_ids: list) -> None:
        """Attach LightGBM / market / EV columns via :mod:`models.predict_unified`.

        Fully guarded: an import error (e.g. lightgbm absent) or any enrichment
        failure leaves the CatBoost line untouched. The executable board price
        (best book price else fused odds) is the price both the EV and the
        de-vigged market line are computed against.
        """
        try:
            from models import predict_unified
        except Exception as exc:  # noqa: BLE001
            logger.warning("predictor: unified enrichment unavailable: %s", exc)
            return
        # Audit req 5/6: keep the two prices distinct.
        #  • reference (one complete bookmaker board, else the fused consensus) →
        #    de-vigged `market_prob`. De-vigging the best-price overlay would treat
        #    a synthetic "best of every book" book as if one bookmaker quoted it,
        #    understating the overround.
        #  • executable (best board price) → EV, the return you would actually get.
        reference, ref_source = _reference_decimals(df, group_ids)
        executable = df.apply(_effective_decimal, axis=1)
        # Persist the de-vig inputs on the frame so _runner_dict can cache them —
        # every fair-probability input must be auditable from the cache alone.
        df["reference_odds"] = pd.to_numeric(reference, errors="coerce")
        df["reference_source"] = ref_source
        predict_unified.enrich(df, group_ids=group_ids,
                               reference_decimals=reference,
                               executable_decimals=executable,
                               model_dir=self._dir)

    def _add_data_quality(self, df: pd.DataFrame) -> None:
        """Append data_completeness / first_time_runner / confidence in place.

        These let the UI distinguish a confident, form-backed read from a debutant
        whose probability is essentially the model's base rate. ``data_completeness``
        is the fraction of signal-bearing features populated for the runner;
        ``first_time_runner`` marks rows with no prior in-window form (the empirically
        dead, always-null columns are excluded from the denominator so they don't
        drag every runner's completeness down uniformly).
        """
        present = [c for c in _SIGNAL_FEATURE_COLS if c in df.columns]
        denom = len(_SIGNAL_FEATURE_COLS)
        if present and denom:
            filled = df[present].notna().sum(axis=1)
            df["data_completeness"] = (filled / denom).round(4)
        else:
            df["data_completeness"] = np.nan

        career = _num(df, "horse_career_runs")
        hist_place = _num(df, "historical_place_rate")
        # A runner is "first-time" if it has no prior career runs OR no trailing
        # place rate (defined iff ≥1 prior run) — i.e. the model has no form signal.
        first_time = (career.fillna(0) <= 0) | hist_place.isna()
        df["first_time_runner"] = first_time

        comp = pd.to_numeric(df["data_completeness"], errors="coerce")
        confidence = np.where(
            first_time.to_numpy() | (comp < _CONF_LOW_MAX).to_numpy(),
            "low",
            np.where((comp >= _CONF_HIGH_MIN).to_numpy(), "high", "medium"),
        )
        df["confidence"] = confidence

    def _score_value(self, df: pd.DataFrame) -> None:
        """Append value-layer probability columns in place (no-op when disabled).

        Two probabilities are persisted, and they are NOT the same thing
        (Stage-4 calibration audit):

        * ``value_win_prob_independent`` — the calibrated output of the
          price-free model. No market price enters its computation, so
          comparing it to the market is not circular.
        * ``value_win_prob`` — the F-L **market-adjusted** probability: the
          independent prob remapped by the OddsBandCalibrator against the SAME
          effective decimal the EV uses. It is a function of the offered price
          and must never be described as price-free. When the recalibrator is
          absent/disabled the two columns are identical.

        Derived quantities use the market-adjusted prob::

            value_edge      = value_win_prob - implied_prob   (probability edge)
            expected_value  = value_win_prob * decimal_odds - 1   (EV per unit stake)
        """
        if self._value_model is None:
            return

        if not any(c in df.columns for c in self._value_feature_cols):
            logger.warning("predictor: no price-free feature cols in live matrix — value skipped")
            return

        # Same column-aligned, NaN-imputed matrix the main models get.
        Xv = self._feature_frame(df, self._value_feature_cols)
        try:
            raw = self._value_model.predict_proba(Xv)[:, 1]
        except Exception as exc:  # noqa: BLE001
            logger.warning("predictor: value predict_proba failed: %s", exc)
            return

        ind = (
            self._value_calibrator.predict(raw)
            if self._value_calibrator is not None
            else raw
        )
        # The INDEPENDENT price-free probability, quantised to its persisted 4dp
        # before anything downstream consumes it, so the market-adjusted prob is
        # exactly reproducible from the persisted pair:
        #   value_win_prob == round(OBC.predict(value_win_prob_independent, best_odds), 4)
        ind = np.round(np.asarray(ind, dtype=float), 4)
        df["value_win_prob_independent"] = ind

        # Favourite-longshot recalibration. The price-free prob is calibrated
        # *marginally* (A/E≈1 by its own prob bucket) but biased *conditional on
        # the market price* — it under-rates favourites and over-rates longshots.
        # Remap it against the SAME effective decimal the EV uses (best board price
        # else fused odds), so value_win_prob and everything derived from it
        # (value_edge / expected_value / value_bet / models.value) share one
        # corrected probability. THE RESULT IS MARKET-ADJUSTED, NOT PRICE-FREE.
        # Rows with a missing/invalid price keep the independent prob (see
        # _apply_fl_recalibration). No-op when the recalibrator is absent /
        # disabled. See calib-fl-02 / calib-fl-03 / live-repair-04.
        dec = df.apply(_effective_decimal, axis=1)
        vw = self._apply_fl_recalibration(ind, dec)
        # Quantise the probability to the SAME 4dp precision it is persisted at
        # (see _runner_dict/_prob) BEFORE any derived quantity is computed, so the
        # cached edge/EV are exactly reproducible from the cached inputs:
        #   expected_value == round(value_win_prob * best_odds - 1, 4)
        #   value_edge     == round(value_win_prob - implied_prob, 4)
        # ``dec`` is already 3dp (via _effective_decimal) — the exact price cached
        # as best_odds. (Audit req 8/10: every EV recomputable from persisted inputs.)
        vw = np.round(np.asarray(vw, dtype=float), 4)
        df["value_win_prob"] = vw

        # A value bet must be INFORMED. A runner with no prior in-window form has
        # every horse-level price-free feature null, so the model returns ~base rate;
        # base-rate × long odds then masquerades as positive EV (the longshot trap —
        # e.g. unraced/foreign runners at 20/1). historical_place_rate is defined iff
        # the horse has ≥1 prior run, so it marks the runners we actually have signal
        # on. Gate consumed in _build_race; absent column there ⇒ treated as supported
        # (keeps mock-row unit tests and pre-value caches behaving unchanged).
        hp = pd.to_numeric(df.get("historical_place_rate"), errors="coerce")
        df["value_supported"] = hp.notna()

        implied = np.round(pd.to_numeric(df.get("implied_prob"), errors="coerce"), 4)
        df["value_edge"] = vw - implied
        # EV is the return on the price we would actually take — the best board
        # price across bookmakers when known, else the fused market odds (``dec``,
        # the same price fed to the F-L recalibrator above).
        df["expected_value"] = vw * pd.to_numeric(dec, errors="coerce") - 1.0

    def _apply_fl_recalibration(self, prob: np.ndarray, dec) -> np.ndarray:
        """Remap the price-free win prob through the F-L recalibrator.

        Returns ``prob`` unchanged when no recalibrator is loaded. Where one is
        present, each row's probability is corrected by the
        :class:`~models.calibration.OddsBandCalibrator` using its effective
        decimal price ``dec``. Rows whose price is missing/≤1 — or whose
        recalibrated value comes back non-finite — keep the original
        market-independent probability, so an odds-less runner never loses its
        ``value_win_prob`` to a NaN.
        """
        out = np.array(prob, dtype=float, copy=True)
        if self._value_fl_calibrator is None:
            return out
        d = pd.to_numeric(pd.Series(dec), errors="coerce").to_numpy(dtype=float)
        usable = np.isfinite(out) & np.isfinite(d) & (d > 1.0)
        if not usable.any():
            return out
        recal = np.asarray(
            self._value_fl_calibrator.predict(out[usable], d[usable]), dtype=float
        )
        good = np.isfinite(recal)
        out[np.flatnonzero(usable)[good]] = recal[good]
        return out

    # ── per-race selection ────────────────────────────────────────────────────

    def _build_race(self, grp: pd.DataFrame) -> dict:
        # Drop non-runners outright: they must never appear as selections or
        # in excluded_low_odds, and must not count toward field size.
        grp = grp[~_is_non_runner(grp)].copy().reset_index(drop=True)
        field_size = len(grp)

        # _dec_odds: np.nan when price unknown; pandas comparisons with NaN → False
        # so unknown-priced runners neither get excluded nor flagged for EW.
        grp["_dec_odds"] = grp.apply(_decimal_odds, axis=1)
        # The price we'd actually back: best board price across bookmakers when
        # known, else the fused market odds. Drives the value-bet gate / EV.
        grp["_best_dec"] = grp.apply(_effective_decimal, axis=1)
        grp["low_odds"] = grp["_dec_odds"] < self._min_odds
        grp["each_way_value"] = (
            (field_size >= 8)
            & (grp["_dec_odds"] >= self._ew_threshold)
            & grp["_dec_odds"].notna()
        )

        # Value bet: positive expected value within the configured odds band.
        # Independent of the top-3 / min-odds selection gate — a value bet can be
        # a short-priced favourite the model rates higher than the market does.
        #
        # Audit req 7: this MUST be the identical gate that models.value.find_value_bets
        # applies, so the UI value flag can never disagree with the suggestion engine.
        # Both now funnel through the single shared `passes_core_gate` predicate,
        # evaluated on the EXECUTABLE board price (_best_dec) and the same EV column.
        if "expected_value" in grp.columns:
            from models.value import passes_core_gate
            from types import SimpleNamespace
            gate_cfg = SimpleNamespace(
                min_ev=self._value_min_ev,
                min_odds=self._value_min_odds,
                max_odds=self._value_max_odds,
                require_support=True,
            )
            ev = pd.to_numeric(grp["expected_value"], errors="coerce").to_numpy(float)
            exec_odds = pd.to_numeric(grp["_best_dec"], errors="coerce").to_numpy(float)
            # Only flag runners the model is actually informed about (own prior form).
            # Absent column ⇒ supported, so mock-row tests / pre-value caches are unchanged.
            supported = (
                grp["value_supported"].astype(bool).to_numpy()
                if "value_supported" in grp.columns
                else np.ones(len(grp), dtype=bool)
            )
            grp["value_bet"] = passes_core_gate(ev, exec_odds, supported, gate_cfg)
        else:
            grp["value_bet"] = False

        eligible = grp[~grp["low_odds"]].sort_values("composite_score", ascending=False)
        excluded = grp[grp["low_odds"]].sort_values("composite_score", ascending=False)

        first = grp.iloc[0]
        # prefer a dedicated race_time column; fall back to race_date
        race_time_raw = first.get("race_time") or first.get("race_date")

        # Audit req 2: `selections` stays the display-only top three, but the value
        # layer must see the COMPLETE declared field (every valid runner, ranked by
        # composite score). `runners` is that full collection — models.value and the
        # suggestion engine consume it so a value pick outside the top three (a
        # short-priced favourite, say) is never silently dropped.
        full_field = grp.sort_values("composite_score", ascending=False)
        # Audit req 9 / issue #6: ONE race-level eligibility decision, computed on
        # the exact field being cached and persisted beside it. When the race is
        # ineligible every EV/market-relative field is withdrawn from every runner
        # (PASS), so no surface can rebuild an EV from a partial or stale card.
        gate = _race_ev_gate(grp)
        race = {
            "venue": _str(first.get("venue", "")),
            "race_time": _str(race_time_raw),
            "field_size": field_size,
            "each_way_available": field_size >= 8,
            "ev_eligible": gate["eligible"],
            "ev_gate": gate,
            "selections": [
                _runner_dict(row, rank=i + 1)
                for i, (_, row) in enumerate(eligible.head(_TOP_N).iterrows())
            ],
            "excluded_low_odds": [
                _runner_dict(row, rank=None) for _, row in excluded.iterrows()
            ],
            "runners": [
                _runner_dict(row, rank=None) for _, row in full_field.iterrows()
            ],
        }
        if not gate["eligible"]:
            logger.info(
                "predictor: race %s %s EV-ineligible (PASS): %s",
                race["venue"], race["race_time"], "; ".join(gate["reasons"]),
            )
            for r in race["selections"] + race["excluded_low_odds"] + race["runners"]:
                _suppress_ev(r)
        return race

    # ── upcoming-races guard ──────────────────────────────────────────────────

    def _guard_upcoming(self, live: pd.DataFrame) -> pd.DataFrame:
        """Defence-in-depth: drop any row that is not a genuinely upcoming runner.

        ``build_inference_matrix`` already restricts the matrix to rows with no
        finishing position whose race is today-or-later (``_live_mask``). We
        re-assert the *same* predicate here so a regression in the builder — or a
        hand-built frame passed straight to ``predict()`` — can never resurrect the
        historical-join-miss bug, where past races whose results failed to join
        (null position) got scored as tens of thousands of phantom 'live' runners.
        Any row carrying a finishing position is also dropped: a result-bearing row
        is history, never a live bet.
        """
        if live.empty or "race_date" not in live.columns:
            return live
        position = live["position"] if "position" in live.columns else pd.Series(
            pd.NA, index=live.index
        )
        keep = _live_mask(live["race_date"], position)
        if (
            "validation_status" in live.columns
            and live["validation_status"].notna().any()
        ):
            invalid = ~live["validation_status"].astype("string").eq(VALID)
            if invalid.any():
                time_key = _race_time_key(live)
                bad_races = set(zip(
                    live.loc[invalid, "venue"].astype(str),
                    pd.Series(time_key, index=live.index).loc[invalid].astype(str),
                ))
                same_bad_race = pd.Series(
                    [
                        (str(venue), str(race_time)) in bad_races
                        for venue, race_time in zip(live["venue"], time_key)
                    ],
                    index=live.index,
                )
                keep &= ~same_bad_race
                logger.error(
                    "predictor: fail-closed %d contaminated live race(s)",
                    len(bad_races),
                )
        dropped = int((~keep).sum())
        if dropped:
            logger.error(
                "predictor: %d non-upcoming row(s) reached scoring — dropped "
                "(historical-join-miss guard)", dropped,
            )
        return live[keep].reset_index(drop=True)

    # ── public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        unified: Optional[pd.DataFrame] = None,
        live: Optional[pd.DataFrame] = None,
    ) -> list[dict]:
        """Build inference matrix → score → top-3 per race.

        Writes data/predictions.json and returns the list of race dicts.

        ``live`` lets a caller that already built the inference matrix pass it in
        so the (expensive) derive runs once, not twice — the un-fused per-book
        odds map is still taken from ``unified``/disk, independent of ``live``."""
        if not self._models and not self.load():
            return []

        # Snapshot existing predictions before overwriting — used for change alerts
        old_cache = _load_cache()
        old_top_by_race: dict[str, str] = {}
        old_odds_by_horse: dict[str, float] = {}
        for r in old_cache.get("races", []):
            sels = r.get("selections", [])
            race_key = f"{r['venue']}|{r['race_time']}"
            if sels:
                old_top_by_race[race_key] = sels[0]["horse_name"]
            for runner in sels + r.get("excluded_low_odds", []):
                if runner.get("horse_name") and runner.get("decimal_odds") is not None:
                    # Keyed by race AND horse. A bare horse-name key collided
                    # across races and days, so the same name at a different
                    # meeting read as a price move that never happened.
                    key = (race_key, runner["horse_name"])
                    old_odds_by_horse[key] = runner["decimal_odds"]

        # Load the un-fused unified rows once: build_inference_matrix fuses cross-
        # source rows into one price per runner, so we capture each bookmaker's own
        # board price here (before fusion) for the odds-by-book map.
        raw = unified if unified is not None else _read_unified()
        odds_by_book = _odds_by_book_map(raw)

        if live is None:
            live = build_inference_matrix(unified=raw)
        if live.empty:
            logger.warning("predictor: inference matrix is empty — no live races")
            return []

        live = self._guard_upcoming(live)
        if live.empty:
            logger.warning("predictor: no upcoming runners survived the guard — no live races")
            return []

        live = _attach_odds_by_book(live, odds_by_book)
        scored = self._score(live)
        # Persist the per-runner feature rows so the UI's SHAP "why" breakdown can
        # explain *today's* runners exactly (keyed on the same horse_id). Best-effort:
        # a failure here must never block writing predictions.
        _write_inference_features(scored, self._feature_cols, self._value_feature_cols)

        races: list[dict] = []
        # Group into individual races by (venue, post-time). A stray row with no
        # race_time falls back to race_date so grouping stays defined (it just
        # degenerates to venue-day for that row).
        time_key = _race_time_key(scored)
        for _, grp in scored.groupby([scored["venue"], time_key], sort=False):
            races.append(self._build_race(grp))
        races.sort(key=lambda r: r["race_time"])

        # Audit req 9: assert the honesty invariants on the exact output we are about
        # to persist. Violations are logged (never silently swallowed) but do not
        # block the write — the UI honesty layer surfaces them and they are covered
        # by regression tests. A clean field emits nothing.
        violations = check_output_invariants(races)
        if violations:
            logger.error("predictor: %d output invariant violation(s): %s",
                         len(violations), "; ".join(violations[:10]))

        _fire_alerts(races, old_top_by_race, old_odds_by_horse)

        generated_at = now().isoformat()
        manifest = self._model_manifest()
        cycle_id = f"{generated_at}-{manifest.get('model_content_hash') or 'nohash'}"[:80]
        _write_manifest(manifest)

        payload = {
            "generated_at": generated_at,
            "model_targets": list(self._models.keys()),
            "total_races": len(races),
            "total_runners": len(scored),
            # Latest holdout GO/NO-GO verdict per model line (honesty stamp for the
            # UI). Best-effort: absent on read failure rather than blocking the write.
            "verdict": self._load_verdicts(),
            # Stage 20 (B7): what produced this cycle, so a fresh ticket built from
            # a race below can be traced back to an exact served bundle, feature
            # schema and cycle. ``execution.config``'s own fingerprint is added
            # later by the caller that actually loads ExecutionConfig (predictor.py
            # deliberately does not import execution.* to avoid a new dependency
            # edge) — see scripts/daily_paper_loop.py.
            "provenance": {
                "model_content_hash": manifest.get("model_content_hash"),
                "feature_schema_version": manifest.get("feature_schema_version"),
                "prediction_cycle_id": cycle_id,
                "decision_time": generated_at,
                "manifest_path": _manifest_path_for_display(),
            },
            "races": races,
        }
        _write_cache(payload)
        logger.info(
            "predictor: %d races / %d runners scored — cache written",
            len(races), len(scored),
        )
        return races

    def _model_manifest(self) -> dict:
        """Content-hash the exact artifact files this instance actually loaded.

        Stage 20 (B7): a prediction/ticket must be traceable back to a concrete
        served bundle, not merely a version-tag string that could quietly point
        at a different retrain later. Hashes only files that exist (an absent
        calibrator is a genuine, disclosed absence, not a hashing error) and
        combines them into one ``model_content_hash`` — any single differing
        byte anywhere in the bundle changes it. ``feature_schema_version`` is a
        content hash of the actual trained-on column LIST (``self._feature_cols``,
        loaded from the champion's own meta.json), so a schema change is
        detected even if the version tag string does not change.
        """
        import hashlib

        def _hash_file(path: Path) -> Optional[str]:
            if not path.exists():
                return None
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 16), b""):
                    h.update(chunk)
            return h.hexdigest()

        candidates: list[Path] = [self._dir / f"catboost_{self._version_tag}_meta.json"]
        for target in self._targets:
            candidates.append(self._dir / f"catboost_{target}_{self._version_tag}.bin")
            candidates.append(self._dir / f"catboost_{target}_{self._version_tag}_calib.pkl")
        if self._value_enabled:
            tag = self._value_model_tag
            candidates.extend(
                [
                    self._dir / f"catboost_won_{tag}.bin",
                    self._dir / f"catboost_won_{tag}_calib.pkl",
                    self._dir / f"fl_oddsband_{tag}_calib.pkl",
                ]
            )

        per_file: dict[str, str] = {}
        for path in candidates:
            digest = _hash_file(path)
            if digest is not None:
                per_file[path.name] = digest

        combined = (
            hashlib.sha256(
                "|".join(f"{name}:{per_file[name]}" for name in sorted(per_file)).encode(
                    "utf-8"
                )
            ).hexdigest()
            if per_file
            else None
        )
        schema = (
            hashlib.sha256("|".join(sorted(self._feature_cols)).encode("utf-8")).hexdigest()[:16]
            if self._feature_cols
            else None
        )
        return {
            "version_tag": self._version_tag,
            "value_model_tag": self._value_model_tag if self._value_enabled else None,
            "files": per_file,
            "model_content_hash": combined,
            "feature_schema_version": schema,
            "n_feature_cols": len(self._feature_cols),
            "generated_at": now().isoformat(),
        }

    def _load_verdicts(self) -> dict:
        """Latest GO/NO-GO verdict per model line (empty dict on any failure)."""
        try:
            from models import predict_unified
            return predict_unified.load_verdicts(base=_BASE, model_dir=self._dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("predictor: verdict load skipped: %s", exc)
            return {}

    def refresh(self) -> list[dict]:
        """Re-fetch live data and re-score. Suitable for polling loops."""
        return self.predict(unified=None)

    def predict_typed(
        self, unified: Optional[pd.DataFrame] = None
    ) -> list["RacePrediction"]:
        """Same as :meth:`predict` but returns typed :class:`RacePrediction` objects
        (each holding :class:`RunnerPrediction`s) instead of plain dicts. The JSON
        cache is still written; this is purely a typed view for in-process callers."""
        return [RacePrediction.from_dict(r) for r in self.predict(unified=unified)]


# ── module-level helpers ───────────────────────────────────────────────────────

def _decimal_odds(row: pd.Series) -> float:
    """Derive decimal odds from implied_prob, then decimal_odds column.

    The implied probability is quantised to the SAME 4dp precision it is
    persisted at before inversion, so the persisted pair satisfies
    ``decimal_odds == round(1 / implied_prob, 3)`` exactly — every derived price
    is reproducible from the cached inputs (audit req 8/10).

    Returns np.nan when neither is available — callers rely on NaN-safe comparisons."""
    ip = row.get("implied_prob")
    if ip is not None and pd.notna(ip):
        ip4 = round(float(ip), 4)
        if ip4 > 0:
            return round(1.0 / ip4, 3)
    do = row.get("decimal_odds")
    if do is not None and pd.notna(do) and float(do) >= 1.0:
        return round(float(do), 3)
    return np.nan


def _clean_odds(val) -> Optional[float]:
    """Round a decimal-odds value to 3dp, or None when missing/invalid (<1.0)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return round(f, 3) if f >= 1.0 else None


def _effective_decimal(row: pd.Series) -> float:
    """The price we would actually back: best board price across bookmakers when
    known, else the fused market odds (``_decimal_odds``). NaN when neither."""
    best = _clean_odds(row.get("best_odds"))
    if best is not None:
        return best
    return _decimal_odds(row)


# Reference-book policy (audit req 5). The fair line is only ever de-vigged from
# one of these, in order of preference:
#   1. a single bookmaker's COMPLETE board for the race — the first source in
#      _BOOKMAKER_SOURCES whose odds_by_book entry prices every runner (per-source
#      books are already validity/freshness gated by _odds_by_book_map);
#   2. the fused consensus line (_decimal_odds): per-runner highest-priority
#      fresh VALID source, the documented consensus fallback (features/fuse.py).
# NEVER the per-runner best-price overlay — mixing the highest price from
# different books builds a book nobody quotes and understates the overround.
_FUSED_CONSENSUS = "fused_consensus"


def _reference_decimals(df: pd.DataFrame, group_ids: list) -> tuple[pd.Series, pd.Series]:
    """Per-runner reference decimal odds + per-runner reference source label.

    Returns ``(reference_odds, reference_source)`` aligned to ``df``. Within one
    race every runner shares one source: a named bookmaker when that book prices
    the entire field, else ``fused_consensus``. Runners with no price anywhere
    stay NaN/None (the race then fails the completeness gate — see _race_ev_gate).
    """
    fused = df.apply(_decimal_odds, axis=1) if len(df) else pd.Series(dtype=float)
    fnum = pd.to_numeric(fused, errors="coerce")
    source = pd.Series(
        [_FUSED_CONSENSUS if np.isfinite(v) else None for v in fnum],
        index=df.index, dtype=object,
    )
    if df.empty or "odds_by_book" not in df.columns:
        return fnum, source

    ref = fnum.copy()
    keys = pd.Series(
        ["\x1f".join(map(str, k)) if isinstance(k, (tuple, list)) else str(k)
         for k in group_ids],
        index=df.index,
    )
    for _, idx in keys.groupby(keys).groups.items():
        books = df.loc[idx, "odds_by_book"]
        for src in _BOOKMAKER_SOURCES:
            prices = [
                _clean_odds(b.get(src)) if isinstance(b, dict) else None
                for b in books
            ]
            if prices and all(p is not None and p > 1.0 for p in prices):
                ref.loc[idx] = prices
                source.loc[idx] = src
                break
    return ref, source


def _race_ev_gate(grp: pd.DataFrame) -> dict:
    """Single race-level EV eligibility decision (audit req 9 / handoff issue #6).

    Computed once per race on the exact field the cache will carry and persisted
    with it (``ev_eligible`` / ``ev_gate``), so every EV surface — the cached
    ``expected_value`` / ``ev_catboost`` / ``ev_lgbm`` / ``value_bet``,
    :func:`models.value.find_value_bets`, the suggestion engine, and the UI
    badges — is gated by ONE shared decision instead of each re-deriving its own.

    ``reasons`` are the race's explicit PASS reasons:

    * ``incomplete_reference_book:M/N`` — only M of N runners carry a reference
      price, so the fair line is undefined (req 4: never de-vig a partial field);
    * ``unpriced_runners:N`` — N runners have no executable price, so an EV board
      would silently cover only part of the field;
    * ``stale_price_rows:N`` / ``odds_age_exceeds_ttl:..`` — provenance on the
      fused rows is beyond the freshness TTL. Upstream gates (features/fuse.py,
      _odds_by_book_map) should already have failed these closed; this is the
      last line of defence at the cache boundary;
    * ``unreadable_price_timestamp:N`` — N rows carry a ``fetched_at`` that is
      present but unparseable: an age that cannot be computed is never fresh.

    Absent provenance columns make no freshness claim (hand-built frames / unit
    fixtures); on the live path fuse guarantees they are present and fresh.
    """
    n = len(grp)
    ref = grp["reference_odds"] if "reference_odds" in grp.columns else grp["_dec_odds"]
    ref_v = pd.to_numeric(ref, errors="coerce").to_numpy(dtype=float)
    ref_ok = np.isfinite(ref_v) & (ref_v > 1.0)
    exec_v = pd.to_numeric(grp["_best_dec"], errors="coerce").to_numpy(dtype=float)
    exec_ok = np.isfinite(exec_v) & (exec_v > 1.0)

    reasons: list = []
    if not bool(ref_ok.all()):
        reasons.append(f"incomplete_reference_book:{int(ref_ok.sum())}/{n}")
    if not bool(exec_ok.all()):
        reasons.append(f"unpriced_runners:{int((~exec_ok).sum())}")

    if "stale" in grp.columns:
        n_stale = int(grp["stale"].fillna(False).astype(bool).sum())
        if n_stale:
            reasons.append(f"stale_price_rows:{n_stale}")
    max_age = None
    if "fetched_at" in grp.columns:
        # format="mixed" for the reason features/fuse.py documents: without it pandas
        # infers ONE format from the first value and silently emits NaT for every
        # differently-shaped timestamp (whole-second vs sub-second), and ``ages.max()``
        # below skips NaT — so a stale row would be invisible and this gate fail OPEN.
        fetched = pd.to_datetime(grp["fetched_at"], utc=True, errors="coerce", format="mixed")
        # A timestamp that is PRESENT but unreadable is an age we cannot compute, and an
        # age we cannot compute is one we must not trust (same rule as execution.gates).
        n_unreadable = int((grp["fetched_at"].notna() & fetched.isna()).sum())
        if n_unreadable:
            reasons.append(f"unreadable_price_timestamp:{n_unreadable}")
        ages = (pd.Timestamp.now(tz="UTC") - fetched).dt.total_seconds()
        if ages.notna().any():
            max_age = float(ages.max())
            ttl = float(get_config().get("staleness", {}).get("max_age_seconds", 900))
            if max_age > ttl:
                reasons.append(f"odds_age_exceeds_ttl:{int(max_age)}s>{int(ttl)}s")

    if "reference_source" in grp.columns:
        srcs = [s for s in grp["reference_source"].dropna().unique().tolist() if s]
        ref_source = srcs[0] if len(srcs) == 1 else (_FUSED_CONSENSUS if srcs else None)
    else:
        ref_source = _FUSED_CONSENSUS if bool(ref_ok.all()) else None
    overround = (
        round(float(np.sum(1.0 / ref_v[ref_ok]) - 1.0), 4)
        if bool(ref_ok.all()) and n else None
    )

    price_sources: set = set()
    if "odds_by_book" in grp.columns:
        for b in grp["odds_by_book"]:
            if isinstance(b, dict):
                price_sources.update(str(k) for k in b)

    return {
        "eligible": not reasons,
        "reasons": reasons,
        "reference_source": ref_source,
        "reference_book_complete": bool(ref_ok.all()),
        "reference_overround": overround,
        "odds_max_age_seconds": round(max_age, 1) if max_age is not None else None,
        "price_sources": sorted(price_sources),
        "computed_at": now().isoformat(),
    }


# Every market-relative / EV field withdrawn from a runner when its race fails
# the EV gate. The model's own probabilities stay — they are price-free.
_EV_SUPPRESSED_KEYS = ("expected_value", "value_edge", "ev_catboost", "ev_lgbm",
                       "market_prob")


def _suppress_ev(runner: dict) -> None:
    """Blank EV/market fields on a runner of an EV-ineligible race (audit req 9).

    An ineligible race PASSes: it must never carry a value flag, an EV, or a
    fair-line comparison built from a stale, partial, or mismatched card."""
    runner["value_bet"] = False
    for k in _EV_SUPPRESSED_KEYS:
        if k in runner:
            runner[k] = None


def _date_key(value) -> str:
    """Normalise a race_date (Timestamp or 'YYYY-MM-DD' string) to a join key."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)[:10]


def _read_unified() -> Optional[pd.DataFrame]:
    """Load the un-fused unified parquet, or None when absent."""
    path = _BASE / "data" / "unified_races.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("predictor: could not read unified dataset: %s", exc)
        return None


def _race_uid(race_id, race_time) -> str:
    """Robust race-identity component for odds-by-book keys (audit req 3).

    Prefers the exact ``race_time`` — the one race identifier that is STABLE ACROSS
    bookmaker sources. Each source mints its own ``race_id`` for the same physical
    race (e.g. livescorebet ``SBTE_2_1028491168`` vs boylesports ``45869743.10``),
    so keying on ``race_id`` would fragment the cross-book map and collapse every
    runner to a single book — defeating best-price aggregation. ``race_id`` is only
    a last-resort fallback when ``race_time`` is absent. This still disambiguates two
    races at the same venue on the same day (their off-times differ), which a bare
    venue-day key cannot. Empty string when neither is available — the
    (date, venue, horse) key then still applies, so single-meeting cards are fine."""
    if race_time is not None and not (isinstance(race_time, float) and pd.isna(race_time)):
        s = _str(race_time)
        if s:
            return s
    if race_id is not None and not (isinstance(race_id, float) and pd.isna(race_id)):
        s = str(race_id).strip()
        if s and s.lower() != "nan":
            return s
    return ""


def _odds_by_book_map(raw: Optional[pd.DataFrame]) -> dict:
    """Map (race_date, venue, race_uid, horse_id) -> {source: best_decimal_odds}.

    Built from the un-fused unified rows so each bookmaker's own WIN-market board
    price is preserved (fusion collapses them to a single price). Only the live
    bookmaker sources are included; rows without a valid (>1.0) price are skipped.
    ``race_uid`` (race_id else race_time) makes the identity robust so a runner's
    price is never merged across two races at the same venue on the same day.
    """
    if raw is None or getattr(raw, "empty", True):
        return {}
    needed = {"race_date", "venue", "horse_id", "source", "odds_decimal"}
    if not needed.issubset(raw.columns):
        return {}
    df = raw[raw["source"].isin(_BOOKMAKER_SOURCES)].copy()
    if "market_type" in df.columns:
        df = df[df["market_type"].astype("string").eq("WIN")]
    if df.empty:
        return {}

    # The best-price overlay bypasses feature fusion, so it must repeat fusion's
    # fail-closed integrity/freshness gates here. Otherwise an invalid or stale raw
    # quote can outrank the fresh fused price and become the executable EV price.
    # Missing validation/freshness metadata is untrusted, not a legacy pass-through.
    if "validation_status" not in df.columns or "fetched_at" not in df.columns:
        return {}
    fetched = pd.to_datetime(df["fetched_at"], utc=True, errors="coerce")
    max_age = float(get_config().get("staleness", {}).get("max_age_seconds", 900))
    age_seconds = (pd.Timestamp.now(tz="UTC") - fetched).dt.total_seconds()
    stale = (
        df["stale"].fillna(False).astype(bool)
        if "stale" in df.columns
        else pd.Series(False, index=df.index)
    )
    bad = (
        ~df["validation_status"].astype("string").eq(VALID)
        | stale
        | fetched.isna()
        | (age_seconds > max_age)
    )

    # Reject the entire source race when any of its primary-WIN rows is bad. Row
    # salvage would turn a contaminated/partial source card into an apparently
    # clean executable book.
    race_component = pd.Series("", index=df.index, dtype="string")
    if "race_id" in df.columns:
        rid = df["race_id"].astype("string")
        race_component = race_component.where(rid.isna() | rid.eq(""), "id:" + rid)
    if "race_time" in df.columns:
        rt = df["race_time"].astype("string")
        race_component = race_component.where(
            ~race_component.eq("") | rt.isna() | rt.eq(""), "time:" + rt
        )
    fallback = (
        "day:" + df["race_date"].map(_date_key).astype("string")
        + "|venue:" + df["venue"].astype("string")
    )
    race_component = race_component.where(~race_component.eq(""), fallback)
    bad_group = bad.groupby([df["source"].astype("string"), race_component]).transform("any")
    df = df.loc[~bad_group].copy()
    if df.empty:
        return {}

    n = len(df)
    rid_col = df["race_id"] if "race_id" in df.columns else pd.Series([None] * n, index=df.index)
    rt_col = df["race_time"] if "race_time" in df.columns else pd.Series([None] * n, index=df.index)
    out: dict = {}
    for rd, venue, rid, rtime, hid, src, odd in zip(
        df["race_date"], df["venue"], rid_col, rt_col,
        df["horse_id"], df["source"], df["odds_decimal"]
    ):
        price = _clean_odds(odd)
        if price is None or hid is None or (isinstance(hid, float) and pd.isna(hid)):
            continue
        book = out.setdefault((_date_key(rd), venue, _race_uid(rid, rtime), hid), {})
        # Keep the best (highest) price if a source lists the runner more than once.
        if src not in book or price > book[src]:
            book[src] = price
    return out


def _best_of_books(books: dict) -> tuple[Optional[str], Optional[float]]:
    """Return (best_book, best_odds) — the bookmaker offering the highest price."""
    if not books:
        return None, None
    best_book = max(books, key=lambda k: books[k])
    return best_book, _clean_odds(books[best_book])


def _attach_odds_by_book(live: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Attach odds_by_book / best_book / best_odds columns to the live matrix."""
    out = live.copy()
    if not mapping or out.empty:
        out["odds_by_book"] = [dict() for _ in range(len(out))]
        out["best_book"] = None
        out["best_odds"] = np.nan
        return out
    n = len(out)
    rid_col = out["race_id"] if "race_id" in out.columns else pd.Series([None] * n, index=out.index)
    rt_col = out["race_time"] if "race_time" in out.columns else pd.Series([None] * n, index=out.index)
    books, best_book, best_odds = [], [], []
    for rd, venue, rid, rtime, hid in zip(
        out["race_date"], out["venue"], rid_col, rt_col, out["horse_id"]
    ):
        d = mapping.get((_date_key(rd), venue, _race_uid(rid, rtime), hid), {})
        bk, bo = _best_of_books(d)
        books.append(d)
        best_book.append(bk)
        best_odds.append(bo if bo is not None else np.nan)
    out["odds_by_book"] = books
    out["best_book"] = best_book
    out["best_odds"] = best_odds
    return out


def _str(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return val.isoformat() if hasattr(val, "isoformat") else str(val)


def _prob(val) -> Optional[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return round(float(val), 4)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric Series for ``col`` aligned to ``df.index``; all-NaN when absent."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def check_output_invariants(races: list, *, tol: float = 0.02) -> list:
    """Return a list of human-readable invariant violations over built race dicts.

    Audit req 9 — the honesty guardrails, checked on the exact output the UI/API
    will consume. Empty list ⇒ every race is internally consistent. Each invariant:

    1. **Displayed win probabilities sum to 1** — the full declared field's
       ``won_prob_normalized`` sums to 1±tol (within-race normalization must hold
       once non-runners are stripped; a violation means mass leaked to a phantom /
       withdrawn runner or normalization ran over the wrong field).
    2. **Complete reference book de-vigs to 1** — when every runner carries a
       ``market_prob`` (a complete de-vigged book), they sum to 1±tol.
    3. **No invalid odds ≤ 1** — no ``decimal_odds`` / ``best_odds`` at or below 1.0.
    4. **No phantom runners** — every runner has a non-empty horse identity.
    5. **No EV from an incomplete card** — a flagged ``value_bet`` must carry a
       finite ``expected_value`` and a valid (>1) executable price.
    6. **PASS races carry no EV** — a race stamped ``ev_eligible: false`` must
       have every runner's ``value_bet`` false and every EV/market field null;
       an eligible race with a partially-populated market line (some runners
       carrying ``market_prob``, some not) is a partial de-vig and violates.
    """
    violations: list = []

    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    for race in races or []:
        rid = f"{race.get('venue', '?')} {race.get('race_time', '?')}"
        field = race.get("runners")
        if not field:  # older cache without the full field — fall back to display set
            field = (race.get("selections") or []) + (race.get("excluded_low_odds") or [])

        # (4) phantom runners
        for r in field:
            if not (str(r.get("horse_id") or "").strip() or str(r.get("horse_name") or "").strip()):
                violations.append(f"{rid}: phantom runner with no horse identity")

        # (1) displayed win probabilities sum to 1
        norm = [_f(r.get("won_prob_normalized")) for r in field]
        norm = [x for x in norm if x is not None]
        if len(norm) >= 2:
            s = sum(norm)
            if abs(s - 1.0) > tol:
                violations.append(
                    f"{rid}: normalized win probs sum to {s:.4f} (≠1±{tol})")

        # (2) complete reference book de-vigs to 1
        mkt = [_f(r.get("market_prob")) for r in field]
        if field and all(x is not None for x in mkt) and len(mkt) >= 2:
            s = sum(mkt)
            if abs(s - 1.0) > tol:
                violations.append(
                    f"{rid}: complete market book sums to {s:.4f} (≠1±{tol})")

        for r in field:
            # (3) invalid odds ≤ 1
            for key in ("decimal_odds", "best_odds"):
                v = _f(r.get(key))
                if v is not None and v <= 1.0:
                    violations.append(
                        f"{rid}: {r.get('horse_name', '?')} has {key}={v} (≤1)")
            # (5) EV only from a valid, complete price
            if bool(r.get("value_bet")):
                ev = _f(r.get("expected_value"))
                exe = _f(r.get("best_odds")) or _f(r.get("decimal_odds"))
                if ev is None or exe is None or exe <= 1.0:
                    violations.append(
                        f"{rid}: {r.get('horse_name', '?')} flagged value_bet "
                        f"without a finite EV/valid price (ev={ev}, odds={exe})")

        # (6) the race-level EV gate is authoritative in the cache
        if race.get("ev_eligible") is False:
            for r in field:
                leaked = [k for k in ("expected_value", "value_edge", "ev_catboost",
                                      "ev_lgbm", "market_prob") if _f(r.get(k)) is not None]
                if bool(r.get("value_bet")) or leaked:
                    violations.append(
                        f"{rid}: {r.get('horse_name', '?')} carries EV/value fields "
                        f"({', '.join(leaked) or 'value_bet'}) on an EV-ineligible race")
        elif race.get("ev_eligible") is True:
            present = [_f(r.get("market_prob")) is not None for r in field]
            if any(present) and not all(present):
                violations.append(
                    f"{rid}: eligible race has a partially-populated market line "
                    f"({sum(present)}/{len(present)} runners) — partial de-vig")

    return violations


def _runner_dict(row: pd.Series, rank: Optional[int]) -> dict:
    dec = row.get("_dec_odds")
    dec_clean = None if (dec is None or (isinstance(dec, float) and pd.isna(dec))) else dec

    # Per-bookmaker board prices + the best price to take. Default safely so older
    # caches / mock rows (no odds-by-book columns) still produce valid runners.
    obb = row.get("odds_by_book")
    obb = {k: _clean_odds(v) for k, v in obb.items()} if isinstance(obb, dict) else {}
    obb = {k: v for k, v in obb.items() if v is not None}
    best_book = row.get("best_book")
    best_book = best_book if isinstance(best_book, str) and best_book else None
    best_odds = _clean_odds(row.get("best_odds"))
    if best_odds is None and obb:
        best_book, best_odds = _best_of_books(obb)
    if best_odds is None:  # no per-book prices → fall back to the fused odds
        best_odds = dec_clean

    # Data-quality layer. Absent on mock rows / older caches → safe defaults
    # (completeness unknown, confidence "unknown", not first-time).
    conf = row.get("confidence")
    conf = conf if isinstance(conf, str) and conf else "unknown"

    # Reference (de-vig) price + source. When enrichment didn't run (mock rows)
    # the reference IS the fused consensus line the row already carries.
    ref_odds = _clean_odds(row.get("reference_odds"))
    if ref_odds is None:
        ref_odds = dec_clean
    ref_source = row.get("reference_source")
    if not (isinstance(ref_source, str) and ref_source):
        ref_source = _FUSED_CONSENSUS if ref_odds is not None else None

    d = RunnerPrediction(
        rank=rank,
        horse_id=_str(row.get("horse_id", "")),
        horse_name=_str(row.get("horse_name", "")),
        jockey=_str(row.get("jockey_name", "")),
        trainer=_str(row.get("trainer_name", "")),
        decimal_odds=dec_clean,
        odds_by_book=obb,
        best_odds=best_odds,
        best_book=best_book,
        implied_prob=round(float(row.get("implied_prob") or 0), 4),
        # Raw per-runner calibrated marginal — kept for debugging / EV reference,
        # NOT the headline (it saturates OOS); see the RunnerPrediction docstring.
        won_prob=_prob(row.get("won_prob")),
        # HEADLINE win prob: field-coherent (race sums to ~1), well-calibrated OOS.
        # None when not normalized (single-runner row / pre-normalization cache).
        won_prob_normalized=_prob(row.get("won_prob_normalized")),
        placed_2_prob=_prob(row.get("placed_2_prob")),
        showed_prob=_prob(row.get("showed_prob")),
        composite_score=round(float(row.get("composite_score") or 0), 4),
        each_way_value=bool(row.get("each_way_value", False)),
        low_odds=bool(row.get("low_odds", False)),
        # value-betting layer — None/False when the price-free model is absent.
        # value_win_prob is MARKET-ADJUSTED (F-L recalibrated against the
        # effective price); value_win_prob_independent is the price-free line.
        value_win_prob=_prob(row.get("value_win_prob")),
        value_win_prob_independent=_prob(row.get("value_win_prob_independent")),
        value_edge=_prob(row.get("value_edge")),
        expected_value=_prob(row.get("expected_value")),
        value_bet=bool(row.get("value_bet", False)),
        # Whether the runner has prior in-window form — consumed by
        # models.value.find_value_bets to gate the formless-longshot trap.
        value_supported=bool(row.get("value_supported", True)),
        # data-quality flags for the UI (see _add_data_quality)
        data_completeness=_prob(row.get("data_completeness")),
        confidence=conf,
        first_time_runner=bool(row.get("first_time_runner", False)),
        # de-vig provenance (audit req 5/8): the exact price + source the fair
        # line comes from; falls back to the fused consensus line (dec_clean).
        reference_odds=ref_odds,
        reference_source=ref_source,
        # the horse's OWN trailing win rate — empirical, never a model output
        empirical_win_rate=_prob(row.get("historical_win_rate")),
    ).to_dict()

    # Additive unified multi-line keys (catboost/lgbm/market prob + per-line EV).
    # lgbm_* keys are present only when the LightGBM line scored this runner; the
    # rest are always present (value may be null). See models.predict_unified.
    try:
        from models.predict_unified import unified_runner_keys
        d.update(unified_runner_keys(row))
    except Exception:  # noqa: BLE001 — enrichment is additive, never fatal
        pass
    return d


def _load_calibrator(path: Path) -> Optional[object]:
    """Load a pickled isotonic calibrator, or None if absent/unreadable.

    A calibrator only needs a ``.predict(p) -> p`` method; missing files are a
    normal state (model trained before calibration was added) and never fatal.
    """
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception as exc:  # noqa: BLE001
        logger.warning("predictor: calibrator %s unreadable: %s", path.name, exc)
        return None


def _load_cache() -> dict:
    """Return the existing predictions.json payload, or {} if absent/corrupt."""
    try:
        with open(_CACHE_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _fire_alerts(
    races: list[dict],
    old_top_by_race: dict[str, str],
    old_odds_by_horse: dict[str, float],
) -> None:
    """Dispatch race-soon, new-top-pick, and odds-drop notifications (non-blocking).

    Only races that have NOT yet started are eligible, and only those inside
    ``alert_window_minutes``. Previously just `notify_race_soon` was time-gated,
    so every rebuild re-announced top-pick changes and price moves for races that
    had already been run — the cache holds the whole day, so a rebuild produced a
    burst of alerts about finished racing.

    Every alert is also fingerprinted (see utils.alert_dedupe). The odds-drop
    fingerprint includes the new price, so a horse that shortens *further* still
    notifies while the same drop seen on repeated rebuilds notifies once.
    """
    try:
        notifier = get_notifier()
        race_soon_min: int = notifier.race_soon_minutes
        threshold_pct: float = notifier.odds_drop_threshold_pct
        window_min: float = float(
            get_config().get("notifications", {}).get("events", {})
            .get("alert_window_minutes", 180)
        )
        now_dt = now()
        skipped_started = 0

        for race in races:
            race_key = f"{race['venue']}|{race['race_time']}"

            # Parse race_time string → aware datetime for time-to-go calculation
            minutes_to_go: Optional[float] = None
            try:
                rt = datetime.fromisoformat(race["race_time"])
                if rt.tzinfo is None:
                    rt = rt.replace(tzinfo=timezone.utc)
                minutes_to_go = (rt - now_dt).total_seconds() / 60.0
            except (ValueError, TypeError, KeyError):
                pass

            # An unparseable race_time is not evidence the race is upcoming, so
            # stay silent rather than risk alerting on finished racing.
            if minutes_to_go is None or minutes_to_go <= 0:
                skipped_started += 1
                continue
            if window_min > 0 and minutes_to_go > window_min:
                continue

            selections = race.get("selections", [])
            if selections:
                top = selections[0]
                horse = top["horse_name"]
                score = top.get("composite_score", 0.0)
                odds = top.get("decimal_odds") or 0.0

                # Race starting soon
                if minutes_to_go <= race_soon_min and not alert_dedupe.seen(
                    alert_dedupe.fingerprint("soon", race_key)
                ):
                    notifier.notify_race_soon(
                        race["venue"], race["race_time"],
                        horse, score, odds, int(minutes_to_go),
                    )

                # New top pick (leader changed since last prediction run)
                old_top = old_top_by_race.get(race_key)
                if old_top and old_top != horse and not alert_dedupe.seen(
                    alert_dedupe.fingerprint("top", race_key, horse)
                ):
                    notifier.notify_new_top_pick(
                        race["venue"], race["race_time"], horse, score, odds,
                    )

            # Odds-drop check across all runners in this race
            all_runners = selections + race.get("excluded_low_odds", [])
            for runner in all_runners:
                horse = runner["horse_name"]
                new_dec = runner.get("decimal_odds")
                old_dec = old_odds_by_horse.get((race_key, horse))
                if new_dec and old_dec and old_dec > 0:
                    drop_pct = (old_dec - new_dec) / old_dec * 100.0
                    if drop_pct >= threshold_pct and not alert_dedupe.seen(
                        # The new price is part of the id: shortening again is a
                        # fresh alert, the same move re-seen is not.
                        alert_dedupe.fingerprint(
                            "drop", race_key, horse, f"{new_dec:.2f}"
                        )
                    ):
                        notifier.notify_odds_drop(
                            horse, race["venue"], old_dec, new_dec, drop_pct,
                        )

        if skipped_started:
            logger.info(
                "predictor: %d race(s) already started — no alerts sent for them",
                skipped_started,
            )
    except Exception as exc:
        logger.error("predictor: alert dispatch failed: %s", exc)


_INFERENCE_FEATURES_PATH = _BASE / "data" / "inference_features.parquet"


def _write_inference_features(
    scored: pd.DataFrame,
    feature_cols: list[str],
    value_feature_cols: list[str],
) -> None:
    """Persist the scored runners' feature rows for the UI's per-prediction SHAP.

    The detail pages explain a runner with :mod:`models.explain`, which needs the
    exact feature row the model saw. Writing it here (keyed on ``horse_id``) means
    today's live runners get race-accurate drivers instead of the historical
    fallback. Best-effort and fully guarded: any failure is logged and swallowed
    so it can never interfere with the prediction cache write.
    """
    try:
        if scored is None or scored.empty:
            return
        identity = ["horse_id", "horse_name", "venue", "race_date", "race_time"]
        wanted = list(dict.fromkeys(
            identity + list(feature_cols) + list(value_feature_cols)
        ))
        keep = [c for c in wanted if c in scored.columns]
        if "horse_id" not in keep:
            return
        out = scored[keep].copy()
        _INFERENCE_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _INFERENCE_FEATURES_PATH.with_suffix(".tmp")
        out.to_parquet(tmp, index=False)
        tmp.replace(_INFERENCE_FEATURES_PATH)
        logger.info("predictor: wrote %d inference feature rows → %s",
                    len(out), _INFERENCE_FEATURES_PATH.name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("predictor: inference-feature write skipped: %s", exc)


def _write_cache(payload: dict) -> None:
    """Atomic write to data/predictions.json, then register TTL in centralized cache."""
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    tmp.replace(_CACHE_PATH)
    try:
        from utils.cache import get_cache
        get_cache().set("predictions", payload)
    except Exception as _exc:
        logger.debug("predictor: cache registration skipped: %s", _exc)
    logger.info("predictor: wrote %s", _CACHE_PATH)


def _manifest_path_for_display() -> str:
    """The manifest path, relative to the repo root when possible.

    Falls back to the absolute path (e.g. under a test's tmp_path, which is
    never inside ``_BASE``) rather than raising — this field is disclosure
    only, never used to re-open the file.
    """
    try:
        return str(_MODEL_MANIFEST_PATH.relative_to(_BASE))
    except ValueError:
        return str(_MODEL_MANIFEST_PATH)


def _write_manifest(manifest: dict) -> None:
    """Best-effort atomic write of the served-model manifest (Stage 20 / B7).

    A saved, on-disk manifest is what lets "the provenance on this ticket
    resolves to a saved manifest" be checked after the fact, not just asserted
    in memory at prediction time. Mirrors ``_write_cache``'s atomic-replace
    pattern; a write failure here must never block writing predictions.json.
    """
    try:
        _MODEL_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _MODEL_MANIFEST_PATH.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, default=str)
        tmp.replace(_MODEL_MANIFEST_PATH)
    except OSError as exc:
        logger.warning("predictor: could not write model manifest: %s", exc)


# ── module-level convenience ───────────────────────────────────────────────────

def predict(unified: Optional[pd.DataFrame] = None) -> list[dict]:
    """One-shot convenience wrapper. Creates a Predictor, loads, and predicts."""
    p = Predictor()
    return p.predict(unified=unified)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _main() -> None:
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="Score live runners and print top-3 selections per race"
    )
    ap.add_argument("--model-dir", default=None, help="Override model directory")
    ap.add_argument(
        "--min-odds", type=float, default=None,
        help="Minimum decimal odds for selections (default: 2.50)",
    )
    args = ap.parse_args()

    p = Predictor(model_dir=args.model_dir, min_selection_odds=args.min_odds)
    if not p.load():
        sys.exit(1)

    races = p.predict()
    if not races:
        print("No live runners found.")
        return

    for race in races:
        sep = "─" * 62
        print(f"\n{sep}")
        print(f"  {race['venue']}  |  {race['race_time']}  |  {race['field_size']} runners")
        print(sep)
        for s in race["selections"]:
            odds_str = f"{s['decimal_odds']:.2f}" if s["decimal_odds"] else "  N/A"
            # Headline win prob = normalized (field-coherent) when present, else the
            # raw calibrated marginal — mirrors what the UI/API present.
            won_head = s.get("won_prob_normalized")
            if won_head is None:
                won_head = s.get("won_prob")
            won_str = f"{won_head:.3f}" if won_head is not None else " ----"
            plc_str = f"{s['placed_2_prob']:.3f}" if s["placed_2_prob"] is not None else " ----"
            ew_tag = "  [EW]" if s["each_way_value"] else ""
            ev = s.get("expected_value")
            val_tag = (
                f"  [VALUE +{ev:.0%}]" if s.get("value_bet") and ev is not None else ""
            )
            print(
                f"  #{s['rank']}  {s['horse_name']:<27} {odds_str}  "
                f"win={won_str}  pl2={plc_str}  score={s['composite_score']:.3f}"
                f"{ew_tag}{val_tag}"
            )
        excl = [r["horse_name"] for r in race["excluded_low_odds"]]
        if excl:
            print(f"  [Excluded <{p._min_odds:.2f}]: {', '.join(excl)}")

    value_picks = sum(
        1
        for r in races
        for s in r["selections"] + r.get("excluded_low_odds", [])
        if s.get("value_bet")
    )
    print(f"\n{value_picks} value bet(s) flagged across {len(races)} races.")
    print(f"Predictions cached → {_CACHE_PATH}")


if __name__ == "__main__":
    _main()
