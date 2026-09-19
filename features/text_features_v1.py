"""Versioned, optional text-feature group (step 16 guarded harness).

Joins ``llm.text_archive`` text observations onto race/runner rows using a
point-in-time decision cutoff, then classifies each observation through the
existing tri-state schema (``llm.text_features.TEXT_FEATURES``). This module
is deliberately **not** part of ``models.features.FEATURE_COLS`` /
``PRICE_FREE_FEATURE_COLS`` / ``CANDIDATE_INDEPENDENT_FEATURE_COLS`` — step 16
found the only archived text corpus is a single day / single source with no
timestamped history in any later development window, so forecast validation
stays DEFERRED_DATA (see ``memory/improvement/stages/16.md``) and nothing here
reaches a trained or promoted model. ``models.features.CANDIDATE_TEXT_FEATURE_COLS``
is a separate, clearly-labelled list for exactly this reason.

Versioning
----------
``TEXT_FEATURE_GROUP_VERSION`` bumps whenever this module's column set or join
semantics change — independent of ``llm.text_features.SCHEMA_VERSION``, which
versions the underlying tri-state extraction schema, and of
``llm.hosted_adapter.HOSTED_PROMPT_VERSION``.

Decision cutoff
----------------
There is no deployed "minutes before off" feature-freeze constant anywhere in
this codebase (``docs/improvement/CONTRACTS.md``, "Prediction cutoff"); this
module reuses that document's provisional 10-minute research assumption
(:data:`PREDICTION_CUTOFF_MINUTES_BEFORE_OFF`) purely for research splits. It
does not change any deployed behaviour.

Provenance columns (per row, prefix configurable, default ``text_v1``)
------------------------------------------------------------------------
* ``{prefix}_available`` -- whether ANY archived text existed at-or-before the
  decision cutoff for this runner/race, independent of whether any feature
  fired.
* ``{prefix}_source`` -- ``"none"`` | ``"regex"`` | ``"hosted_<model>"`` |
  ``"<backend>_fallback_regex"`` -- mirrors
  ``llm.text_features.ExtractionResult.backend`` (never mislabels a fallback
  as the backend that failed).
* ``{prefix}_published_at_known`` -- False whenever the archived row's own
  ``published_at`` is unknown (Sporting Life Spotlight never supplies one).
  The join never uses ``published_at`` for the cutoff decision (only
  ``fetched_at`` is real point-in-time information here — see
  ``llm.text_archive.TextArchive.as_of``); this column is provenance only.
* ``{prefix}_{feature}`` -- 1.0 / 0.0 / NaN (tri-state: NaN is unknown, never
  coerced to 0 — a downstream consumer must opt in to collapsing it).
* ``{prefix}_{feature}_evidence_present`` -- whether an evidence phrase was
  recorded (bool only; the phrase itself is not carried into a numeric model
  matrix — call :func:`extract_with_evidence` directly for audit use).

Odds/tips separation
---------------------
None of ``llm.text_features.TEXT_FEATURES`` is derived from a price or
tipster-language mention (verified structurally: no pattern references odds,
prices, "nap", "best bet" or similar). :func:`flag_price_or_tip_language` is a
separate, non-feature diagnostic so a future stage adding subjective/tip
features cannot accidentally fold them into this factual, independent branch
without a conscious new column.

No-lookahead contract
----------------------
The join calls ``TextArchive.as_of(..., as_of=cutoff)``, which filters
strictly on ``fetched_at <= cutoff`` — never ``archived_at``, never a later
edit's row. A later comment or a source edit (a new archive row with a later
``fetched_at``) can only change what a LATER cutoff sees; it structurally
cannot reach back and change an earlier one. Regression-tested in
``tests/features/test_text_features_v1.py``.
"""
from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from llm.text_archive import TextArchive, get_text_archive, runner_key
from llm.text_features import TEXT_FEATURES, FeatureValue, TextFeatureExtractor
from utils.logger import get_logger
from utils.text_norm import minute_key, norm_venue

_log = get_logger(__name__)

# Bump when this module's column set or join semantics change.
TEXT_FEATURE_GROUP_VERSION = 1

# Provisional research-only assumption (CONTRACTS.md "Prediction cutoff") —
# not a deployed feature-freeze rule.
PREDICTION_CUTOFF_MINUTES_BEFORE_OFF = 10

_RACING_TZ = ZoneInfo("Europe/Dublin")
_FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in TEXT_FEATURES)

# Numeric tri-state columns only — safe to feed into a model matrix once (and
# only once) a stage has real evidence to support it. Availability/source/
# evidence-present columns are provenance/diagnostics, not model inputs.
CANDIDATE_TEXT_FEATURE_COLS: list[str] = [f"text_v1_{name}" for name in _FEATURE_NAMES]

_PRICE_TOKENS = (
    "/1", "/2", "/4", "/8", "evens", "evs", " sp ", "odds of", "each-way",
    "each way", "e/w", "priced at", "nap ", "best bet", "next best",
)


def _archive_key(race_uid: Any) -> str:
    """Normalise a training-matrix ``race_uid`` into the archive's own key.

    ``llm.text_archive`` rows are written via ``execution.race_facts.race_facts_key``,
    which lower/normalises the venue (``norm_venue``); the training matrix's
    ``race_uid`` preserves the source's original casing (``"Ripon|..."`` vs
    ``"ripon|..."``). A byte-equality join between the two therefore silently
    matches nothing even for a genuinely shared race — reproduced live this
    stage (0/21 archived races matched the rebuilt matrix before this fix).
    Step 11 hit the identical mismatch joining measured-timing keys and fixed
    it the same way (``reports/improvement/11/02_build_experiment_matrix.py::race_key_of``);
    this mirrors that, not a new pattern.
    """
    text = "" if race_uid is None else str(race_uid)
    if "|" not in text:
        return text
    venue, stamp = text.split("|", 1)
    return f"{norm_venue(venue)}|{minute_key(stamp)}"


def _parse_race_uid_local_dt(race_uid: Any) -> Optional[datetime]:
    """Recover the race's local off-time from ``"<Venue>|<local ISO datetime>"``.

    Returns ``None`` (eligibility cannot be established) for anything that
    does not parse — never guesses a time.
    """
    text = "" if race_uid is None else str(race_uid)
    if "|" not in text:
        return None
    _, stamp = text.split("|", 1)
    try:
        naive = datetime.fromisoformat(stamp.strip())
    except ValueError:
        return None
    return naive.replace(tzinfo=_RACING_TZ)


def decision_cutoff(race_uid: Any, *, minutes_before_off: int = PREDICTION_CUTOFF_MINUTES_BEFORE_OFF
                     ) -> Optional[datetime]:
    """The research decision-cutoff instant for one race, or ``None`` if
    ``race_uid`` cannot be parsed (eligibility cannot be established)."""
    off = _parse_race_uid_local_dt(race_uid)
    if off is None:
        return None
    from datetime import timedelta
    return off - timedelta(minutes=minutes_before_off)


def flag_price_or_tip_language(text: Optional[str]) -> bool:
    """Cheap diagnostic: does this text mention a price/odds or tipster
    phrase? Never used to derive a factual feature — see module docstring."""
    if not text:
        return False
    lowered = f" {str(text).lower()} "
    return any(tok in lowered for tok in _PRICE_TOKENS)


def build_cache_only_hosted_backend(
    *, model: str, reasoning_effort: str = "low", max_output_tokens: int = 1600,
    cache=None,
):
    """A hosted backend that can only ever return a cache hit or "unavailable".

    Constructed with a throwaway, zero-budget, non-persisted ledger — a cache
    miss raises ``BudgetExceeded`` inside ``ledger.reserve`` BEFORE any network
    attempt, and nothing here ever touches the real programme-lifetime ledger
    (``data/cache/llm/hosted_budget_ledger.json``) or spends a cent. This is
    how step 16 satisfies "reuse frozen cached extractions within the same
    cumulative hosted budget" without any risk of new spend.
    """
    from llm.hosted_adapter import OpenRouterBackend
    from llm.hosted_budget import HostedBudgetLedger
    from utils.cache import Cache

    scratch_path = Path(tempfile.gettempdir()) / f"stage16_cache_only_ledger_{uuid.uuid4().hex}.json"
    ledger = HostedBudgetLedger(path=scratch_path, max_cost_usd=0.0, max_requests=0)
    cache = cache if cache is not None else Cache(
        cache_dir=Path(__file__).resolve().parents[1] / "data" / "cache" / "llm",
        default_ttl=2_592_000,
    )
    return OpenRouterBackend(
        api_key="cache-only-no-network", model=model, ledger=ledger,
        price_input_per_token=0.0, price_output_per_token=0.0,
        price_ceiling_input_per_token=1.0, price_ceiling_output_per_token=1.0,
        max_output_tokens=max_output_tokens, reasoning_effort=reasoning_effort,
        cache=cache,
    )


@dataclass(frozen=True)
class TextJoinConfig:
    cutoff_minutes_before_off: int = PREDICTION_CUTOFF_MINUTES_BEFORE_OFF
    prefix: str = "text_v1"


def text_feature_columns(prefix: str = "text_v1") -> list[str]:
    cols = [f"{prefix}_available", f"{prefix}_source", f"{prefix}_published_at_known"]
    for name in _FEATURE_NAMES:
        cols.append(f"{prefix}_{name}")
        cols.append(f"{prefix}_{name}_evidence_present")
    return cols


def _empty_row(prefix: str) -> dict:
    out = {f"{prefix}_available": False, f"{prefix}_source": "none",
           f"{prefix}_published_at_known": False}
    for name in _FEATURE_NAMES:
        out[f"{prefix}_{name}"] = float("nan")
        out[f"{prefix}_{name}_evidence_present"] = False
    return out


def _features_row(prefix: str, source: str, published_at_known: bool,
                   features: dict[str, FeatureValue]) -> dict:
    out = {f"{prefix}_available": True, f"{prefix}_source": source,
           f"{prefix}_published_at_known": published_at_known}
    for name in _FEATURE_NAMES:
        fv = features.get(name, FeatureValue(None))
        out[f"{prefix}_{name}"] = float("nan") if fv.value is None else float(bool(fv.value))
        out[f"{prefix}_{name}_evidence_present"] = fv.evidence is not None
    return out


def _extract_hosted_cache_only(backend, text: str, horse_name: Optional[str]):
    """Returns (source_label, features) or None on cache miss (unavailable)."""
    from llm.hosted_budget import BudgetExceeded
    try:
        result = backend.extract(text, horse_name=horse_name)
    except BudgetExceeded:
        return None
    if not result.schema_valid and not result.is_cache_hit:
        return None
    label = f"hosted_{result.model.replace('/', '_')}" if result.is_cache_hit else "hosted_cache_miss"
    return label, result.features


def attach_text_features(
    df: pd.DataFrame,
    *,
    extractor: Optional[TextFeatureExtractor] = None,
    hosted_backend: Optional[Any] = None,
    archive: Optional[TextArchive] = None,
    config: TextJoinConfig = TextJoinConfig(),
) -> pd.DataFrame:
    """Return a copy of ``df`` with this text-feature group's columns attached.

    ``extractor`` defaults to a regex-only extractor (explicitly enabled here
    — this call site, not the global ``llm.enabled`` config, controls it; the
    default promoted path is untouched). When ``hosted_backend`` is supplied
    it is preferred; a cache miss (see :func:`build_cache_only_hosted_backend`)
    falls back to regex on that row rather than silently emitting unknown,
    matching ``llm.text_features``'s own fallback labelling convention.

    Only rows whose ``race_uid`` has at least one archived observation are
    looked up individually — every other row takes the fast "no text" default
    without touching the archive, so this scales to the full training matrix.
    """
    archive = archive if archive is not None else get_text_archive()
    extractor = extractor if extractor is not None else TextFeatureExtractor(
        enabled=True, backend="regex")
    prefix = config.prefix

    known = archive.known_race_uids()
    out_rows: list[dict] = [None] * len(df)  # type: ignore[list-item]
    race_uids = df["race_uid"].astype(str).to_numpy()
    horse_names = df["horse_name"].to_numpy() if "horse_name" in df.columns else [None] * len(df)
    horse_ids = df["horse_id"].to_numpy() if "horse_id" in df.columns else [None] * len(df)

    for i in range(len(df)):
        ruid = race_uids[i]
        akey = _archive_key(ruid)
        if akey not in known:
            out_rows[i] = _empty_row(prefix)
            continue
        rkey = runner_key(horse_names[i], horse_ids[i])
        if not rkey:
            out_rows[i] = _empty_row(prefix)  # eligibility cannot be established
            continue
        cutoff = decision_cutoff(ruid, minutes_before_off=config.cutoff_minutes_before_off)
        if cutoff is None:
            out_rows[i] = _empty_row(prefix)  # eligibility cannot be established
            continue
        archived = archive.as_of(akey, horse_name=horse_names[i], horse_id=horse_ids[i], as_of=cutoff)
        if archived is None:
            out_rows[i] = _empty_row(prefix)
            continue

        published_known = archived.published_at is not None
        if hosted_backend is not None:
            hit = _extract_hosted_cache_only(hosted_backend, archived.text, archived.horse_name)
            if hit is not None:
                label, features = hit
                out_rows[i] = _features_row(prefix, label, published_known, features)
                continue
            # Cache miss in cache-only mode: fall back to regex, labelled honestly.
            features = extractor.extract_detailed(archived.text).features
            out_rows[i] = _features_row(prefix, "hosted_cache_miss_regex_fallback",
                                        published_known, features)
            continue

        result = extractor.extract_detailed(archived.text)
        out_rows[i] = _features_row(prefix, result.backend, published_known, result.features)

    added = pd.DataFrame(out_rows, index=df.index)
    return pd.concat([df, added], axis=1)


def text_coverage_report(df: pd.DataFrame, *, archive: Optional[TextArchive] = None,
                          config: TextJoinConfig = TextJoinConfig()) -> dict:
    """Honest coverage counts — never a claimed corpus size (mirrors
    ``llm.text_archive.TextArchive.coverage``'s own honesty convention)."""
    archive = archive if archive is not None else get_text_archive()
    known = archive.known_race_uids()
    race_uids = df["race_uid"].astype(str)
    eligible_mask = race_uids.map(_archive_key).isin(known)
    eligible = df.loc[eligible_mask]
    return {
        "total_rows": int(len(df)),
        "total_races": int(race_uids.nunique()),
        "archive_known_race_uids": len(known),
        "rows_with_race_in_archive": int(eligible_mask.sum()),
        "races_with_race_in_archive": int(eligible["race_uid"].nunique()) if len(eligible) else 0,
    }
