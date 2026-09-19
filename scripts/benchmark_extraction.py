"""Step 15 — score repaired regex (and, optionally, the hosted backend)
against the frozen prompt/schema and the reviewed label subset.

Usage
-----
    .venv\\Scripts\\python.exe -m scripts.benchmark_extraction --regex-only
    .venv\\Scripts\\python.exe -m scripts.benchmark_extraction --hosted \\
        --price-input-per-token 3e-7 --price-output-per-token 1.2e-6 \\
        --max-records 243

Local backends (Qwen3.5-9B / Hermes-4-14B via Ollama) are intentionally not
invoked here when no Ollama server is reachable — see
``check_local_runtime()`` — the caller is expected to check that first and
record DEFERRED_DATA rather than have this script hang on a dead connection.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Optional

import psutil

from llm.hosted_adapter import HOSTED_PROMPT_VERSION, OpenRouterBackend
from llm.hosted_budget import HostedBudgetLedger
from llm.text_features import PROMPT_VERSION, SCHEMA_VERSION, FeatureValue, RegexBackend, TEXT_FEATURES
from utils.cache import Cache

BASE = Path(__file__).resolve().parents[1]
CORPUS_DIR = BASE / "data" / "audit" / "15"
REPORT_DIR = BASE / "reports" / "improvement" / "15"
_FEATURE_NAMES = tuple(f.name for f in TEXT_FEATURES)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def check_local_runtime(host: str = "http://localhost:11434", timeout: float = 2.0) -> bool:
    """True iff an Ollama server actually answers on ``host``. Never raises."""
    try:
        import httpx

        resp = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


# -- metrics ------------------------------------------------------------------

def confusion_against_labels(
    predictions: dict[str, dict[str, FeatureValue]],
    labels: list[dict],
) -> dict:
    """predictions: content_hash -> {feature: FeatureValue}. labels: reviewed rows."""
    per_field: dict[str, dict] = {
        name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "pred_unknown": 0, "label_unknown": 0, "n": 0}
        for name in _FEATURE_NAMES
    }
    for row in labels:
        h = row["content_hash"]
        pred = predictions.get(h)
        if pred is None:
            continue
        for name in _FEATURE_NAMES:
            gold = row["features"][name]["value"]
            pv = pred[name].value
            stat = per_field[name]
            stat["n"] += 1
            if gold is None:
                stat["label_unknown"] += 1
            if pv is None:
                stat["pred_unknown"] += 1
            # Treat regex's "False, no evidence" as a non-claim: only score a
            # true positive/false positive/false negative on the True class,
            # since that is the only class a phrase-matcher can assert with
            # evidence. This mirrors llm/text_features.py's own documented
            # semantics (RegexBackend never emits unknown by design).
            if gold is True and pv is True:
                stat["tp"] += 1
            elif gold is True and pv is not True:
                stat["fn"] += 1
            elif gold is not True and pv is True:
                stat["fp"] += 1
            else:
                stat["tn"] += 1
    for name, stat in per_field.items():
        tp, fp, fn = stat["tp"], stat["fp"], stat["fn"]
        stat["precision"] = tp / (tp + fp) if (tp + fp) else None
        stat["recall"] = tp / (tp + fn) if (tp + fn) else None
    return per_field


def _rss_mb() -> float:
    """Current process resident set size, MB. Coarse (whole-process, not
    per-call) but portable to Windows, where ``resource.ru_maxrss`` does not
    exist."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


# -- regex scoring --------------------------------------------------------------

def run_regex(records: list[dict]) -> tuple[dict[str, dict], list[float], dict]:
    backend = RegexBackend()
    predictions: dict[str, dict[str, FeatureValue]] = {}
    latencies = []
    tracemalloc.start()
    peak_rss_mb = _rss_mb()
    for r in records:
        start = time.perf_counter()
        feats = backend.classify_detailed(r["text"])
        latencies.append((time.perf_counter() - start) * 1000.0)
        predictions[r["content_hash"]] = feats
        peak_rss_mb = max(peak_rss_mb, _rss_mb())
    _, py_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory = {
        "process_rss_peak_mb": peak_rss_mb,
        "python_alloc_peak_mb": py_peak_bytes / (1024 * 1024),
        "note": "whole-process RSS sampled after each record (coarse); python_alloc_peak_mb is tracemalloc's traced-allocation peak for this loop only.",
    }
    return predictions, latencies, memory


# -- hosted scoring ---------------------------------------------------------------

def run_hosted(
    records: list[dict],
    *,
    api_key: str,
    model: str,
    price_input_per_token: float,
    price_output_per_token: float,
    price_ceiling_input_per_token: float,
    price_ceiling_output_per_token: float,
    max_cost_usd: float,
    max_requests: int,
    max_output_tokens: int,
    max_retries: int,
    reasoning_effort: str = "low",
) -> dict:
    ledger = HostedBudgetLedger(
        None, max_cost_usd=max_cost_usd, max_requests=max_requests
    )
    cache = Cache(cache_dir=BASE / "data" / "cache" / "llm", default_ttl=2_592_000)
    backend = OpenRouterBackend(
        api_key=api_key, model=model, ledger=ledger,
        price_input_per_token=price_input_per_token,
        price_output_per_token=price_output_per_token,
        price_ceiling_input_per_token=price_ceiling_input_per_token,
        price_ceiling_output_per_token=price_ceiling_output_per_token,
        max_output_tokens=max_output_tokens, max_retries=max_retries, cache=cache,
        reasoning_effort=reasoning_effort,
    )
    predictions: dict[str, dict[str, FeatureValue]] = {}
    latencies: list[float] = []
    schema_valid_count = 0
    failure_count = 0
    cache_hits = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_reasoning_tokens = 0
    total_cost = 0.0
    n_scored = 0
    stop_reason = "completed_all_records"
    peak_rss_mb = _rss_mb()
    for r in records:
        try:
            result = backend.extract(r["text"], horse_name=r.get("horse_name"))
        except Exception as exc:  # BudgetExceeded or PricingUnbounded — stop, don't crash the run
            stop_reason = f"stopped_early:{type(exc).__name__}:{exc}"
            break
        predictions[r["content_hash"]] = result.features
        latencies.append(result.latency_ms)
        schema_valid_count += int(result.schema_valid)
        failure_count += int(not result.schema_valid)
        cache_hits += int(result.is_cache_hit)
        total_prompt_tokens += result.prompt_tokens
        total_completion_tokens += result.completion_tokens
        total_reasoning_tokens += result.reasoning_tokens
        total_cost += result.cost_usd
        n_scored += 1
        peak_rss_mb = max(peak_rss_mb, _rss_mb())

    return {
        "predictions": predictions,
        "latencies_ms": latencies,
        "memory": {
            "process_rss_peak_mb": peak_rss_mb,
            "note": "whole local client process RSS sampled after each call — network-bound; the hosted MODEL's own runtime memory is not observable from the client and is not reported.",
        },
        "n_scored": n_scored,
        "n_requested": len(records),
        "schema_valid_count": schema_valid_count,
        "failure_count": failure_count,
        "cache_hits": cache_hits,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_reasoning_tokens": total_reasoning_tokens,
        "total_cost_usd": total_cost,
        "stop_reason": stop_reason,
        "ledger_snapshot": ledger.snapshot(),
        "model": model,
    }


# -- CLI ------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hosted", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    parser.add_argument("--price-input-per-token", type=float, default=None)
    parser.add_argument("--price-output-per-token", type=float, default=None)
    parser.add_argument("--price-ceiling-input-per-token", type=float, default=0.3e-6)
    parser.add_argument("--price-ceiling-output-per-token", type=float, default=1.2e-6)
    parser.add_argument("--max-cost-usd", type=float, default=5.0)
    parser.add_argument("--max-requests", type=int, default=1200)
    parser.add_argument("--max-output-tokens", type=int, default=800)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--reasoning-effort", default="low",
        help="OpenRouter unified reasoning effort ('low'/'medium'/'high', or "
        "'' to omit the field entirely). A reasoning-capable model can spend "
        "its whole max-output-tokens budget on chain-of-thought and return no "
        "JSON at all if this is left uncontrolled.",
    )
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    blind_eval = load_jsonl(CORPUS_DIR / "blind_eval.jsonl")
    prompt_dev = load_jsonl(CORPUS_DIR / "prompt_dev.jsonl")
    labels = load_jsonl(CORPUS_DIR / "labels_reviewed.jsonl")
    all_records = prompt_dev + blind_eval
    if args.max_records:
        all_records = all_records[: args.max_records]

    scorecard: dict = {
        "corpus_size_scored": len(all_records),
        "reviewed_label_count": len(labels),
        "schema_version": SCHEMA_VERSION,
        "regex_prompt_version": PROMPT_VERSION,
        "hosted_prompt_version": HOSTED_PROMPT_VERSION,
    }

    regex_preds, regex_latencies, regex_memory = run_regex(all_records)
    scorecard["regex"] = {
        "backend": "regex",
        "n_scored": len(all_records),
        "schema_valid_rate": 1.0,  # deterministic matcher always returns a complete result
        "latency_ms": {
            "mean": statistics.mean(regex_latencies) if regex_latencies else None,
            "p50": percentile(regex_latencies, 0.50),
            "p95": percentile(regex_latencies, 0.95),
            "max": max(regex_latencies) if regex_latencies else None,
        },
        "memory": regex_memory,
        "cost_usd_per_1000": 0.0,
        "confusion_vs_reviewed_labels": confusion_against_labels(regex_preds, labels),
    }

    if args.hosted:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            scorecard["hosted"] = {"status": "DEFERRED_DATA", "reason": "OPENROUTER_API_KEY absent from process/user environment"}
        elif args.price_input_per_token is None or args.price_output_per_token is None:
            scorecard["hosted"] = {"status": "BLOCKED", "reason": "pricing not supplied/verified — refusing to call hosted API unbounded"}
        else:
            hosted = run_hosted(
                all_records, api_key=api_key, model=args.model,
                price_input_per_token=args.price_input_per_token,
                price_output_per_token=args.price_output_per_token,
                price_ceiling_input_per_token=args.price_ceiling_input_per_token,
                price_ceiling_output_per_token=args.price_ceiling_output_per_token,
                max_cost_usd=args.max_cost_usd, max_requests=args.max_requests,
                max_output_tokens=args.max_output_tokens, max_retries=args.max_retries,
                reasoning_effort=args.reasoning_effort,
            )
            hosted_lat = hosted["latencies_ms"]
            n = max(hosted["n_scored"], 1)
            scorecard["hosted"] = {
                "status": "SCORED" if hosted["n_scored"] else "DEFERRED_DATA",
                "backend": "hosted_openrouter",
                "provider": "openrouter",
                "model": hosted["model"],
                "reasoning_effort": args.reasoning_effort,
                "n_scored": hosted["n_scored"],
                "n_requested": hosted["n_requested"],
                "stop_reason": hosted["stop_reason"],
                "schema_valid_rate": hosted["schema_valid_count"] / n,
                "failure_count": hosted["failure_count"],
                "cache_hits": hosted["cache_hits"],
                "total_prompt_tokens": hosted["total_prompt_tokens"],
                "total_completion_tokens": hosted["total_completion_tokens"],
                "total_reasoning_tokens": hosted["total_reasoning_tokens"],
                "total_cost_usd": hosted["total_cost_usd"],
                "cost_usd_per_1000": hosted["total_cost_usd"] / n * 1000,
                "memory": hosted["memory"],
                "latency_ms": {
                    "mean": statistics.mean(hosted_lat) if hosted_lat else None,
                    "p50": percentile(hosted_lat, 0.50),
                    "p95": percentile(hosted_lat, 0.95),
                    "max": max(hosted_lat) if hosted_lat else None,
                },
                "ledger_snapshot": hosted["ledger_snapshot"],
                "confusion_vs_reviewed_labels": confusion_against_labels(hosted["predictions"], labels),
            }
    else:
        scorecard["hosted"] = {"status": "NOT_REQUESTED"}

    scorecard["local_qwen3.5_9b"] = {"status": "DEFERRED_DATA", "reason": "not evaluated in this invocation"}
    scorecard["local_hermes4_14b"] = {"status": "DEFERRED_DATA", "reason": "not evaluated in this invocation"}

    out_path = REPORT_DIR / ("scorecard_hosted.json" if args.hosted else "scorecard_regex_only.json")
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(scorecard, fh, indent=2)
    print(json.dumps({k: v for k, v in scorecard.items() if k not in ("regex", "hosted")}, indent=2))
    print(f"written -> {out_path}")


if __name__ == "__main__":
    main()
