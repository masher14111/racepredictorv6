"""Step 15 — build the reproducible text-extraction evaluation corpus.

Pulls every distinct comment currently in ``text_archive`` (append-only,
Stage 14), deduplicates by content hash (the archive can hold more than one
row per hash if the same text was re-fetched), and splits deterministically
(sorted by content hash, not insertion order or any random seed) into:

* ``prompt_dev``  — a small slice used only to sanity-check that a frozen
  prompt/schema produces valid structured output. Never scored.
* ``blind_eval``  — the remainder. A fixed-size prefix of this slice (sorted
  the same deterministic way) is the *reviewed* subset scored for
  precision/recall; the rest is scored only on backend-agreement,
  schema-validity, latency and cost (no gold label required for those).

This script does not invent data: if the archive holds fewer than the
500-1,000-comment/multi-date/multi-source target from prompts/15, the
manifest records the exact shortfall instead of silently proceeding as if
the target were met.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DB_PATH = BASE / "data" / "races.db"
OUT_DIR = BASE / "data" / "audit" / "15"
TARGET_MIN = 500
TARGET_MAX = 1000
PROMPT_DEV_SIZE = 20
REVIEWED_SAMPLE_SIZE = 60


def load_unique_rows() -> list[dict]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT content_hash, source, race_uid, runner_key, horse_name, text, "
        "fetched_at, published_at FROM text_archive ORDER BY content_hash"
    ).fetchall()
    conn.close()
    seen: dict[str, dict] = {}
    for r in rows:
        h = r["content_hash"]
        if h not in seen:
            seen[h] = dict(r)
    return sorted(seen.values(), key=lambda d: d["content_hash"])


def build(out_dir: Path = OUT_DIR) -> dict:
    records = load_unique_rows()
    dates = sorted({r["fetched_at"][:10] for r in records})
    sources = sorted({r["source"] for r in records})

    prompt_dev = records[:PROMPT_DEV_SIZE]
    blind_eval = records[PROMPT_DEV_SIZE:]
    reviewed_ids = {r["content_hash"] for r in blind_eval[:REVIEWED_SAMPLE_SIZE]}

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "prompt_dev.jsonl").open("w", encoding="utf-8") as fh:
        for r in prompt_dev:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / "blind_eval.jsonl").open("w", encoding="utf-8") as fh:
        for r in blind_eval:
            r2 = dict(r)
            r2["in_reviewed_sample"] = r["content_hash"] in reviewed_ids
            fh.write(json.dumps(r2, ensure_ascii=False) + "\n")

    manifest = {
        "source_table": "text_archive",
        "total_rows_seen": len(records),
        "distinct_dates": dates,
        "distinct_sources": sources,
        "target_min": TARGET_MIN,
        "target_max": TARGET_MAX,
        "target_met": len(records) >= TARGET_MIN and len(dates) > 1,
        "gap": (
            f"{len(records)} distinct comments from {len(sources)} source(s) on "
            f"{len(dates)} calendar date(s) ({dates}); target was {TARGET_MIN}-"
            f"{TARGET_MAX} across DATES AND SOURCES (plural). Shortfall is both "
            "volume and diversity — text_archive currently only has Stage 14's "
            "backfilled single-day Spotlight import (D46); no live scrape has run "
            "in this programme yet."
        ),
        "prompt_dev_count": len(prompt_dev),
        "blind_eval_count": len(blind_eval),
        "reviewed_sample_count": len(reviewed_ids),
        "reviewed_sample_note": (
            "The first 60 blind_eval records (deterministic, content-hash order) "
            "are labelled by a single LLM reviewer (Claude Sonnet 5, this stage) "
            "reading the raw text directly against the TEXT_FEATURES "
            "definitions in llm/text_features.py. These are NOT human-reviewed "
            "gold labels — review_status='single_reviewer_llm_unverified' in "
            "labels_reviewed.jsonl, no second reviewer or adjudication step."
        ),
    }
    with (out_dir / "corpus_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    m = build(args.out_dir)
    print(json.dumps(m, indent=2))
