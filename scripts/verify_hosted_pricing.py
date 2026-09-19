"""Step 15 — verify deepseek/deepseek-v4.1-flash's live OpenRouter pricing
against docs/improvement/runner.json's configured ceilings BEFORE any paid
call (prompts/15 scope 5: "fail closed if cost cannot be bounded").

Reads a previously-fetched public ``/models`` listing (no API key required
for that endpoint) rather than re-fetching on every invocation, so this can
be re-run offline against the same evidence snapshot used for the actual
benchmark run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DEFAULT_RAW = BASE / "reports" / "improvement" / "15" / "openrouter_models_raw.json"
RUNNER_JSON = BASE / "docs" / "improvement" / "runner.json"


def verify(model_id: str, raw_path: Path = DEFAULT_RAW) -> dict:
    runner_cfg = json.loads(RUNNER_JSON.read_text(encoding="utf-8"))["hosted_trial"]
    ceiling_in = runner_cfg["provider_max_input_per_million_usd"] / 1_000_000
    ceiling_out = runner_cfg["provider_max_output_per_million_usd"] / 1_000_000

    models = json.loads(raw_path.read_text(encoding="utf-8"))["data"]
    match = next((m for m in models if m.get("id") == model_id), None)
    if match is None:
        raise SystemExit(f"model {model_id!r} not found in {raw_path}")

    pricing = match["pricing"]
    base_in, base_out = float(pricing["prompt"]), float(pricing["completion"])
    overrides = pricing.get("overrides", [])
    all_in = [base_in] + [float(o["prompt"]) for o in overrides]
    all_out = [base_out] + [float(o["completion"]) for o in overrides]
    worst_in, worst_out = max(all_in), max(all_out)

    result = {
        "model_id": model_id,
        "provider": "openrouter",
        "base_price_input_per_token": base_in,
        "base_price_output_per_token": base_out,
        "worst_case_price_input_per_token": worst_in,
        "worst_case_price_output_per_token": worst_out,
        "ceiling_input_per_token": ceiling_in,
        "ceiling_output_per_token": ceiling_out,
        "within_ceiling": worst_in <= ceiling_in and worst_out <= ceiling_out,
        "n_time_of_day_overrides": len(overrides),
        "source_file": str(raw_path.relative_to(BASE)),
    }
    if not result["within_ceiling"]:
        raise SystemExit(f"PRICING EXCEEDS CEILING — refusing to proceed: {result}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW)
    args = parser.parse_args()
    out = verify(args.model, args.raw_path)
    print(json.dumps(out, indent=2))
    out_path = BASE / "reports" / "improvement" / "15" / "hosted_pricing_verification.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
