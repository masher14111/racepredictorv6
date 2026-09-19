"""Capped OpenRouter DeepSeek *shadow* extraction over daily archived race text (Stage 19).

Pipeline position
------------------
This module is a consumer of two already-shipped, independently tested
components -- it adds no new HTTP client and no new spend ledger:

* ``llm/hosted_adapter.py::OpenRouterBackend`` -- the actual hosted call,
  strict grounding validation, per-record cache, reserve-then-commit cost
  accounting (Stage 15).
* ``llm/hosted_budget.py::HostedBudgetLedger`` -- the durable, file-backed,
  PROGRAMME-lifetime spend/request cap (``data/cache/llm/hosted_budget_ledger.json``,
  shared with the Stage-15 benchmark; never reset, never renewed per day).

What is new here is *shadow* orchestration: selecting which archived
comments (``llm/text_archive.py``) still need extracting, bounding one daily
batch, and durably recording the result **separately from any model
feature** -- this module never writes to ``models.features.FEATURE_COLS`` or
``features/text_features_v1.py``'s candidate columns, and ``llm.enabled``
(the switch that lets ``llm/text_features.py`` actually influence a
forecast) is untouched by anything here. "Shadow" means: extract and record,
never feed into a probability.

Extraction-availability honesty
--------------------------------
``text_archive.fetched_at`` is when our scraper saw the raw comment
(typically pre-race, since it comes off the racecard). The LLM extraction
over that text can happen much later -- a resumed batch, a delayed daily
cycle, a retried failure. ``llm_shadow_extractions.extraction_time`` records
that separately and is the ONLY honest "this feature became available"
instant; any future consumer that wants a strictly pre-race feature must
filter on ``extraction_time``, not on the underlying text's
``fetched_at``/``published_at`` -- otherwise a batch run after a race has
gone off would let its output masquerade as information that existed before
the race. :meth:`ShadowExtractionStore.as_of` enforces exactly this filter,
mirroring ``TextArchive.as_of``'s own point-in-time contract.

Fail-closed, bounded, resumable
--------------------------------
* Disabled config, absent ``OPENROUTER_API_KEY``, unverifiable/over-ceiling
  pricing, or an exhausted programme cap each stop the run with a labelled
  reason and touch neither the ledger nor the network -- never a crash, and
  the caller (``scripts/refresh.py``) treats this step as best-effort, same
  as every other scraper in that pipeline.
* Only archived rows that do not already have a matching extraction (same
  ``archive_row_id``/model/schema/prompt version) are selected, oldest
  ``fetched_at`` first, bounded to ``batch_size_per_run`` -- a rerun after a
  partial batch (crash, exhausted cap) resumes from wherever it left off
  without re-spending on already-recorded rows, and never loads the whole
  archive into memory or submits it to the provider.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from llm.hosted_adapter import HOSTED_PROMPT_VERSION, HostedExtraction, OpenRouterBackend
from llm.hosted_budget import BudgetExceeded, HostedBudgetLedger, PricingUnbounded
from llm.text_archive import ArchivedText, TextArchive, get_text_archive
from llm.text_features import SCHEMA_VERSION
from utils.cache import Cache
from utils.logger import get_logger
from utils.storage import DEFAULT_DB_PATH
from utils.storage.migrations import apply_migrations
from utils.storage.pool import ConnectionPool

logger = get_logger(__name__)

_BASE = Path(__file__).resolve().parents[1]
_CACHE_DIR = _BASE / "data" / "cache" / "llm"
_MODELS_LISTING_CACHE = _CACHE_DIR / "openrouter_models_listing_cache.json"
_HEALTH_PATH = _BASE / "data" / "execution" / "shadow_extraction_health.json"
_MODELS_URL = "https://openrouter.ai/api/v1/models"

_DEFAULTS = {
    "enabled": False,
    "model": "deepseek/deepseek-v4.1-flash",
    "reasoning_effort": "low",
    "max_output_tokens": 1600,
    "max_retries": 2,
    "timeout_s": 30.0,
    "batch_size_per_run": 25,
    "price_ceiling_input_per_million_usd": 0.3,
    "price_ceiling_output_per_million_usd": 1.2,
    "max_cost_usd": 5.0,
    "max_requests": 1200,
}


def _load_config() -> dict:
    cfg = dict(_DEFAULTS)
    try:
        from utils.config_loader import get

        cfg["enabled"] = bool(get("llm.hosted_shadow.enabled", cfg["enabled"]))
        cfg["model"] = str(get("llm.hosted_shadow.model", cfg["model"]))
        cfg["reasoning_effort"] = str(get("llm.hosted_shadow.reasoning_effort", cfg["reasoning_effort"]))
        cfg["max_output_tokens"] = int(get("llm.hosted_shadow.max_output_tokens", cfg["max_output_tokens"]))
        cfg["max_retries"] = int(get("llm.hosted_shadow.max_retries", cfg["max_retries"]))
        cfg["timeout_s"] = float(get("llm.hosted_shadow.timeout_s", cfg["timeout_s"]))
        cfg["batch_size_per_run"] = int(get("llm.hosted_shadow.batch_size_per_run", cfg["batch_size_per_run"]))
        cfg["price_ceiling_input_per_million_usd"] = float(
            get("llm.hosted_shadow.price_ceiling_input_per_million_usd", cfg["price_ceiling_input_per_million_usd"])
        )
        cfg["price_ceiling_output_per_million_usd"] = float(
            get("llm.hosted_shadow.price_ceiling_output_per_million_usd", cfg["price_ceiling_output_per_million_usd"])
        )
        cfg["max_cost_usd"] = float(get("llm.hosted_shadow.max_cost_usd", cfg["max_cost_usd"]))
        cfg["max_requests"] = int(get("llm.hosted_shadow.max_requests", cfg["max_requests"]))
    except Exception as exc:  # noqa: BLE001 - config absent/invalid -> safe defaults
        logger.debug("shadow_extraction: config unavailable (%s) - using defaults", exc)
    return cfg


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


# ── append-only extraction store ────────────────────────────────────────────

_COLUMNS = (
    "archive_row_id", "content_hash", "source", "race_uid", "runner_key",
    "horse_id", "horse_name", "text_published_at", "text_fetched_at",
    "extraction_time", "provider", "model", "backend", "schema_version",
    "prompt_version", "schema_valid", "is_cache_hit", "failure_reason",
    "cost_usd", "features", "raw_response_hash",
)
_INSERT_SQL = (
    "INSERT OR IGNORE INTO llm_shadow_extractions (" + ", ".join(_COLUMNS) + ") "
    "VALUES (" + ", ".join("?" * len(_COLUMNS)) + ")"
)


@dataclass(frozen=True)
class ShadowExtractionRow:
    id: int
    archive_row_id: int
    content_hash: str
    source: str
    race_uid: str
    runner_key: str
    horse_id: Optional[str]
    horse_name: Optional[str]
    text_published_at: Optional[str]
    text_fetched_at: str
    extraction_time: str
    provider: str
    model: str
    backend: str
    schema_version: int
    prompt_version: int
    schema_valid: bool
    is_cache_hit: bool
    failure_reason: Optional[str]
    cost_usd: float
    features: dict
    raw_response_hash: Optional[str]

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> "ShadowExtractionRow":
        try:
            features = json.loads(row["features"]) if row["features"] else {}
        except (TypeError, ValueError):
            features = {}
        return cls(
            id=row["id"], archive_row_id=row["archive_row_id"],
            content_hash=row["content_hash"], source=row["source"],
            race_uid=row["race_uid"], runner_key=row["runner_key"],
            horse_id=row["horse_id"], horse_name=row["horse_name"],
            text_published_at=row["text_published_at"], text_fetched_at=row["text_fetched_at"],
            extraction_time=row["extraction_time"], provider=row["provider"],
            model=row["model"], backend=row["backend"],
            schema_version=row["schema_version"], prompt_version=row["prompt_version"],
            schema_valid=bool(row["schema_valid"]), is_cache_hit=bool(row["is_cache_hit"]),
            failure_reason=row["failure_reason"], cost_usd=row["cost_usd"],
            features=features, raw_response_hash=row["raw_response_hash"],
        )


class ShadowExtractionStore:
    """Reader/writer for the append-only ``llm_shadow_extractions`` table."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        self._pool = ConnectionPool(self.db_path)
        apply_migrations(self._pool)

    def close(self) -> None:
        self._pool.close_all()

    # -- writing ------------------------------------------------------------
    def record(
        self,
        *,
        archive_row: ArchivedText,
        extraction: HostedExtraction,
        extraction_time: Optional[datetime] = None,
    ) -> Optional[ShadowExtractionRow]:
        """Append one extraction outcome, keyed on the source row + model/version.

        Idempotent on ``(archive_row_id, model, schema_version, prompt_version)``:
        a resumed batch that re-selects an already-recorded row (should not
        normally happen, since the caller's own SELECT excludes them, but a
        concurrent/duplicate call must still never overwrite a prior result)
        is a silent no-op, matching ``TextArchive.record``'s own idempotency.
        """
        et = extraction_time or _now()
        params = (
            archive_row.id, archive_row.content_hash, archive_row.source,
            archive_row.race_uid, archive_row.runner_key,
            archive_row.horse_id, archive_row.horse_name,
            archive_row.published_at, archive_row.fetched_at,
            _iso(et), extraction.provider, extraction.model, extraction.backend,
            extraction.schema_version, extraction.prompt_version,
            int(extraction.schema_valid), int(extraction.is_cache_hit),
            extraction.failure_reason, extraction.cost_usd,
            json.dumps({k: v.to_dict() for k, v in extraction.features.items()}),
            extraction.raw_response_hash,
        )
        conn = self._pool.connection()
        with self._pool.write_lock():
            conn.execute(_INSERT_SQL, params)
            conn.commit()
            row = conn.execute(
                "SELECT * FROM llm_shadow_extractions WHERE archive_row_id=? "
                "AND model=? AND schema_version=? AND prompt_version=?",
                (archive_row.id, extraction.model, extraction.schema_version, extraction.prompt_version),
            ).fetchone()
        return ShadowExtractionRow._from_row(row) if row else None

    # -- selecting unprocessed archived rows ---------------------------------
    def select_unprocessed(
        self, *, model: str, schema_version: int, prompt_version: int, limit: int,
    ) -> list[ArchivedText]:
        """Archived comments with no matching extraction yet, oldest-fetched first.

        A single indexed SQL join against ``text_archive`` -- this never loads
        the whole archive into memory and never hands the caller more than
        ``limit`` rows, which is what keeps one daily batch's provider
        exposure bounded to exactly the comments actually being submitted.
        """
        conn = self._pool.connection()
        cur = conn.execute(
            """
            SELECT t.* FROM text_archive t
            LEFT JOIN llm_shadow_extractions s
                ON s.archive_row_id = t.id AND s.model = ? AND s.schema_version = ?
                   AND s.prompt_version = ?
            WHERE s.id IS NULL
            ORDER BY t.fetched_at ASC, t.id ASC
            LIMIT ?
            """,
            (model, schema_version, prompt_version, int(limit)),
        )
        return [ArchivedText._from_row(r) for r in cur.fetchall()]

    def count_unprocessed(self, *, model: str, schema_version: int, prompt_version: int) -> int:
        conn = self._pool.connection()
        row = conn.execute(
            """
            SELECT COUNT(*) AS n FROM text_archive t
            LEFT JOIN llm_shadow_extractions s
                ON s.archive_row_id = t.id AND s.model = ? AND s.schema_version = ?
                   AND s.prompt_version = ?
            WHERE s.id IS NULL
            """,
            (model, schema_version, prompt_version),
        ).fetchone()
        return int(row["n"])

    # -- reading --------------------------------------------------------------
    def as_of(
        self, race_uid: str, runner_key: str, *, as_of: Optional[datetime] = None,
    ) -> Optional[ShadowExtractionRow]:
        """The most recent extraction actually AVAILABLE (``extraction_time``)
        at or before ``as_of`` (default now) -- never the underlying text's
        ``fetched_at``. See module docstring: this is the point-in-time
        contract that stops a late batch from masquerading as pre-race data.
        """
        cutoff = _iso(as_of or _now())
        conn = self._pool.connection()
        row = conn.execute(
            "SELECT * FROM llm_shadow_extractions WHERE race_uid=? AND runner_key=? "
            "AND extraction_time<=? ORDER BY extraction_time DESC, id DESC LIMIT 1",
            (race_uid, runner_key, cutoff),
        ).fetchone()
        return ShadowExtractionRow._from_row(row) if row else None

    def coverage(self) -> dict:
        conn = self._pool.connection()
        row = conn.execute(
            "SELECT COUNT(*) AS n, COUNT(DISTINCT race_uid) AS races, "
            "SUM(schema_valid) AS valid, SUM(is_cache_hit) AS cache_hits, "
            "SUM(cost_usd) AS cost, MIN(extraction_time) AS earliest, "
            "MAX(extraction_time) AS latest FROM llm_shadow_extractions"
        ).fetchone()
        return {
            "rows": row["n"], "races": row["races"], "schema_valid": row["valid"] or 0,
            "cache_hits": row["cache_hits"] or 0, "cost_usd_recorded": row["cost"] or 0.0,
            "earliest_extraction_time": row["earliest"], "latest_extraction_time": row["latest"],
        }


_SINGLETON: Optional[ShadowExtractionStore] = None


def get_shadow_store() -> ShadowExtractionStore:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = ShadowExtractionStore()
    return _SINGLETON


# ── pricing verification (fail closed BEFORE any extraction call) ──────────

class PricingUnverifiable(RuntimeError):
    """Live and cached pricing are both unavailable -- refuse to proceed unbounded."""


def _default_fetch_models_listing(timeout_s: float = 15.0) -> dict:
    import httpx

    resp = httpx.get(_MODELS_URL, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def verify_current_pricing(
    model_id: str,
    *,
    ceiling_input_per_million_usd: float,
    ceiling_output_per_million_usd: float,
    fetch_listing: Optional[Callable[[], dict]] = None,
    cache_path: Path = _MODELS_LISTING_CACHE,
) -> dict:
    """Verify ``model_id``'s live OpenRouter pricing against the configured
    ceiling BEFORE any paid call, mirroring ``scripts/verify_hosted_pricing.py``'s
    logic exactly (Stage 15) but fetching fresh rather than reading a frozen
    snapshot, since pricing is checked on every run, not once.

    Falls back to the last successfully-fetched listing (cached to
    ``cache_path``) if the live fetch fails -- e.g. a transient network
    outage should not silently disable shadow collection every single day --
    but a cold start with no cache and no live fetch fails closed
    (:class:`PricingUnverifiable`), never assumes pricing is within bounds.
    """
    fetch = fetch_listing or _default_fetch_models_listing
    listing: Optional[dict] = None
    source = "live"
    try:
        listing = fetch()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(listing), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("shadow_extraction: live pricing fetch failed (%s); trying cache", exc)
        if cache_path.exists():
            try:
                listing = json.loads(cache_path.read_text(encoding="utf-8"))
                source = "cached_fallback"
            except (OSError, ValueError):
                listing = None
    if listing is None:
        raise PricingUnverifiable(
            f"no live pricing available and no cached snapshot at {cache_path}"
        )

    models = listing.get("data", [])
    match = next((m for m in models if m.get("id") == model_id), None)
    if match is None:
        raise PricingUnverifiable(f"model {model_id!r} not present in the OpenRouter listing ({source})")

    pricing = match.get("pricing") or {}
    try:
        base_in, base_out = float(pricing["prompt"]), float(pricing["completion"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PricingUnverifiable(f"malformed pricing block for {model_id!r}: {exc}") from exc
    overrides = pricing.get("overrides") or []
    all_in = [base_in] + [float(o["prompt"]) for o in overrides if "prompt" in o]
    all_out = [base_out] + [float(o["completion"]) for o in overrides if "completion" in o]
    worst_in, worst_out = max(all_in), max(all_out)

    ceiling_in = ceiling_input_per_million_usd / 1_000_000
    ceiling_out = ceiling_output_per_million_usd / 1_000_000
    within_ceiling = worst_in <= ceiling_in and worst_out <= ceiling_out

    result = {
        "model_id": model_id,
        "provider": "openrouter",
        "pricing_source": source,
        "base_price_input_per_token": base_in,
        "base_price_output_per_token": base_out,
        "worst_case_price_input_per_token": worst_in,
        "worst_case_price_output_per_token": worst_out,
        "ceiling_input_per_token": ceiling_in,
        "ceiling_output_per_token": ceiling_out,
        "within_ceiling": within_ceiling,
        "n_time_of_day_overrides": len(overrides),
        "verified_at": _iso(_now()),
    }
    if not within_ceiling:
        raise PricingUnbounded(f"pricing exceeds configured ceiling, refusing to proceed: {result}")
    return result


# ── orchestration ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ShadowExtractionSummary:
    enabled: bool
    api_key_present: bool
    pricing_verified: bool
    model: str
    reason: Optional[str]
    stop_reason: str
    requested: int = 0
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    cache_hits: int = 0
    unprocessed_remaining: int = 0
    cost_usd_this_run: float = 0.0
    budget_snapshot: dict = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _empty_ledger_snapshot(cfg: dict) -> dict:
    return {
        "max_cost_usd": cfg["max_cost_usd"], "max_requests": cfg["max_requests"],
        "committed_cost_usd": None, "committed_requests": None,
        "remaining_cost_usd": None, "remaining_requests": None,
    }


def run_shadow_extraction(
    *,
    config: Optional[dict] = None,
    api_key: Optional[str] = None,
    archive: Optional[TextArchive] = None,
    store: Optional[ShadowExtractionStore] = None,
    ledger: Optional[HostedBudgetLedger] = None,
    cache: Optional[Cache] = None,
    backend: Optional[OpenRouterBackend] = None,
    fetch_listing: Optional[Callable[[], dict]] = None,
    http_post: Optional[Callable] = None,
    now: Optional[datetime] = None,
    pricing_cache_path: Path = _MODELS_LISTING_CACHE,
) -> ShadowExtractionSummary:
    """Run one bounded batch of hosted shadow extraction over unprocessed
    dated archived comments. Never raises -- every failure mode returns a
    labelled summary instead, since this is a best-effort step inside the
    daily refresh pipeline (like every scraper it runs alongside).

    Injectable ``backend``/``fetch_listing``/``http_post``/``ledger``/``cache``
    are for tests; a live daily run supplies none of them and gets the real
    OpenRouter transport, the real shared ledger file and the real archive.
    """
    started = now or _now()
    cfg = config or _load_config()
    model = cfg["model"]

    if not cfg["enabled"]:
        return ShadowExtractionSummary(
            enabled=False, api_key_present=False, pricing_verified=False, model=model,
            reason="disabled_by_config", stop_reason="disabled_by_config",
            started_at=_iso(started), finished_at=_iso(_now()),
        )

    resolved_key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
    if not resolved_key:
        return ShadowExtractionSummary(
            enabled=True, api_key_present=False, pricing_verified=False, model=model,
            reason="OPENROUTER_API_KEY_absent", stop_reason="api_key_absent",
            budget_snapshot=_empty_ledger_snapshot(cfg),
            started_at=_iso(started), finished_at=_iso(_now()),
        )

    try:
        pricing = verify_current_pricing(
            model,
            ceiling_input_per_million_usd=cfg["price_ceiling_input_per_million_usd"],
            ceiling_output_per_million_usd=cfg["price_ceiling_output_per_million_usd"],
            fetch_listing=fetch_listing,
            cache_path=pricing_cache_path,
        )
    except (PricingUnverifiable, PricingUnbounded) as exc:
        logger.warning("shadow_extraction: pricing verification failed (%s)", exc)
        return ShadowExtractionSummary(
            enabled=True, api_key_present=True, pricing_verified=False, model=model,
            reason=str(exc), stop_reason="pricing_unverifiable",
            budget_snapshot=_empty_ledger_snapshot(cfg),
            started_at=_iso(started), finished_at=_iso(_now()),
        )

    ledger = ledger or HostedBudgetLedger(
        None, max_cost_usd=cfg["max_cost_usd"], max_requests=cfg["max_requests"]
    )
    cache = cache if cache is not None else Cache(cache_dir=_CACHE_DIR, default_ttl=2_592_000)
    archive = archive or get_text_archive()
    store = store or get_shadow_store()

    try:
        real_backend = backend or OpenRouterBackend(
            api_key=resolved_key, model=model, ledger=ledger,
            price_input_per_token=pricing["worst_case_price_input_per_token"],
            price_output_per_token=pricing["worst_case_price_output_per_token"],
            price_ceiling_input_per_token=pricing["ceiling_input_per_token"],
            price_ceiling_output_per_token=pricing["ceiling_output_per_token"],
            max_output_tokens=cfg["max_output_tokens"], reasoning_effort=cfg["reasoning_effort"],
            timeout_s=cfg["timeout_s"], max_retries=cfg["max_retries"], cache=cache,
            http_post=http_post,
        )
    except PricingUnbounded as exc:
        return ShadowExtractionSummary(
            enabled=True, api_key_present=True, pricing_verified=False, model=model,
            reason=str(exc), stop_reason="pricing_exceeds_ceiling",
            budget_snapshot=ledger.snapshot(),
            started_at=_iso(started), finished_at=_iso(_now()),
        )

    schema_version = SCHEMA_VERSION
    prompt_version = HOSTED_PROMPT_VERSION
    batch_size = cfg["batch_size_per_run"]

    candidates = store.select_unprocessed(
        model=model, schema_version=schema_version, prompt_version=prompt_version, limit=batch_size,
    )
    requested = len(candidates)
    processed = succeeded = failed = cache_hits = 0
    cost_this_run = 0.0
    stop_reason = "completed_batch" if requested == batch_size else "no_more_candidates"
    if requested == 0:
        stop_reason = "no_candidates"

    for row in candidates:
        try:
            result = real_backend.extract(row.text, horse_name=row.horse_name)
        except BudgetExceeded as exc:
            logger.warning("shadow_extraction: budget exhausted mid-batch (%s)", exc)
            stop_reason = "budget_exhausted"
            break
        except Exception as exc:  # noqa: BLE001 - one bad row must not kill the batch
            logger.warning("shadow_extraction: extraction failed for archive row %s (%s)", row.id, exc)
            failed += 1
            processed += 1
            continue

        # `now` (when explicitly injected, e.g. by a test) fixes every row in
        # this batch to the same instant; a real run leaves it None and gets a
        # fresh wall-clock reading per row, which is what makes a slow batch's
        # later rows honestly later than its earlier ones.
        extraction_time = now or _now()
        store.record(archive_row=row, extraction=result, extraction_time=extraction_time)

        processed += 1
        cache_hits += int(result.is_cache_hit)
        if not result.is_cache_hit:
            # A cache hit replays the ORIGINAL call's recorded cost_usd
            # (informational, from whichever prior run actually paid for it)
            # with is_cache_hit=True -- counting it again here would report
            # spend that did not happen in this run and never touched the
            # ledger a second time (llm/hosted_adapter.py::extract).
            cost_this_run += result.cost_usd
        if result.schema_valid:
            succeeded += 1
        else:
            failed += 1

    remaining = store.count_unprocessed(
        model=model, schema_version=schema_version, prompt_version=prompt_version,
    )

    return ShadowExtractionSummary(
        enabled=True, api_key_present=True, pricing_verified=True, model=model,
        reason=None, stop_reason=stop_reason,
        requested=requested, processed=processed, succeeded=succeeded, failed=failed,
        cache_hits=cache_hits, unprocessed_remaining=remaining,
        cost_usd_this_run=cost_this_run, budget_snapshot=ledger.snapshot(),
        started_at=_iso(started), finished_at=_iso(_now()),
    )


def write_health_report(summary: ShadowExtractionSummary, path: Path = _HEALTH_PATH) -> str:
    """Persist a non-secret health summary (no API key, no raw text/prompt)
    so an operator/dashboard can see extraction/budget status without reading
    logs -- mirrors ``utils.source_health``'s disk-backed pattern."""
    payload = summary.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(path)


def main() -> int:
    summary = run_shadow_extraction()
    write_health_report(summary)
    print(json.dumps(summary.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
