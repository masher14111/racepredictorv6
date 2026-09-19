"""Step 15 — materialize the reviewed label set for the 60-comment sample.

Combines the hand-authored ``LABELS`` in ``scripts/_stage15_labels_source.py``
(index -> {feature: (value, evidence)}) with the deterministic reviewed
sample built by ``scripts/build_extraction_eval_corpus.py``, validates every
non-null evidence string is an actual case-insensitive substring of its
source comment (a labelling bug otherwise), fills every unlabelled feature
in as unknown (None) — "not addressed" is the honest default, not False —
and writes ``data/audit/15/labels_reviewed.jsonl``.
"""
from __future__ import annotations

import json
from pathlib import Path

from llm.text_features import TEXT_FEATURES

from scripts._stage15_labels_source import LABELS

BASE = Path(__file__).resolve().parents[1]
CORPUS_DIR = BASE / "data" / "audit" / "15"
_FEATURE_NAMES = tuple(f.name for f in TEXT_FEATURES)

REVIEW_STATUS = "single_reviewer_llm_unverified"
REVIEWER = "claude-sonnet-5 (this stage, single pass, no second reviewer/adjudication)"


def load_sample() -> list[dict]:
    with (CORPUS_DIR / "blind_eval.jsonl").open(encoding="utf-8") as fh:
        recs = [json.loads(line) for line in fh]
    return [r for r in recs if r["in_reviewed_sample"]]


def build() -> list[dict]:
    sample = load_sample()
    if len(sample) != len(LABELS):
        raise SystemExit(
            f"sample size {len(sample)} != labelled indices {len(LABELS)} — "
            "corpus build changed under the labels; re-review before scoring"
        )

    out: list[dict] = []
    errors: list[str] = []
    for idx, rec in enumerate(sample):
        text = rec["text"]
        text_lower = text.lower()
        given = LABELS.get(idx, {})
        features: dict[str, dict] = {}
        for name in _FEATURE_NAMES:
            if name in given:
                value, evidence = given[name]
            else:
                value, evidence = None, None
            if evidence is not None and evidence.lower() not in text_lower:
                errors.append(
                    f"record {idx} ({rec['content_hash'][:10]}) feature={name!r}: "
                    f"evidence {evidence!r} is not a substring of the text"
                )
            features[name] = {"value": value, "evidence": evidence}
        out.append(
            {
                "content_hash": rec["content_hash"],
                "race_uid": rec["race_uid"],
                "horse_name": rec["horse_name"],
                "text": text,
                "features": features,
                "review_status": REVIEW_STATUS,
                "reviewer": REVIEWER,
            }
        )

    if errors:
        raise SystemExit("evidence validation failed:\n" + "\n".join(errors))

    out_path = CORPUS_DIR / "labels_reviewed.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    true_counts = {name: 0 for name in _FEATURE_NAMES}
    false_counts = {name: 0 for name in _FEATURE_NAMES}
    unknown_counts = {name: 0 for name in _FEATURE_NAMES}
    for row in out:
        for name, fv in row["features"].items():
            if fv["value"] is True:
                true_counts[name] += 1
            elif fv["value"] is False:
                false_counts[name] += 1
            else:
                unknown_counts[name] += 1
    summary = {
        "n_records": len(out),
        "review_status": REVIEW_STATUS,
        "true_counts": true_counts,
        "false_counts": false_counts,
        "unknown_counts": unknown_counts,
    }
    with (CORPUS_DIR / "labels_reviewed_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return out


if __name__ == "__main__":
    rows = build()
    print(f"wrote {len(rows)} reviewed label records -> data/audit/15/labels_reviewed.jsonl")
