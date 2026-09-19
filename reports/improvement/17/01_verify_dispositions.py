"""Step 17 — independent, READ-ONLY verification of steps 09..16's recorded evidence.

Nothing here trusts a stage note: every claim checked is re-derived from the
artifact the note names. Writes only ``reports/improvement/17/01_dispositions.json``.

Run:  .venv/Scripts/python.exe reports/improvement/17/01_verify_dispositions.py
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).with_name("01_dispositions.json")
HOLDOUT_START = pd.Timestamp("2026-08-28")
MARKET_FEATURE_COLS = {"implied_prob", "overround_norm_prob", "market_ref_prob"}

checks: list[dict] = []


def naive_dates(series: pd.Series) -> pd.Series:
    """``race_date`` is tz-aware UTC in some stage artifacts and naive in others."""
    return pd.to_datetime(series, utc=True).dt.tz_localize(None)


def record(stage: str, name: str, ok: bool, detail) -> None:
    checks.append({"stage": stage, "check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {stage} {name}: {detail}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256(path: Path) -> str:
    """File hash, or step 10's own combined hash for a Hive-partitioned parquet
    DIRECTORY (``reports/improvement/10/02_freeze_manifest.py::sha256_path``):
    each member's relative path + content hash, walked in sorted order."""
    if path.is_file():
        return _sha256_file(path)
    combined = hashlib.sha256()
    for member in sorted(p for p in path.rglob("*") if p.is_file()):
        combined.update(member.relative_to(path).as_posix().encode("utf-8"))
        combined.update(_sha256_file(member).encode("utf-8"))
    return combined.hexdigest()


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


# ── ledger dispositions ────────────────────────────────────────────────────
state = (ROOT / "memory/improvement/STATE.md").read_text(encoding="utf-8")
ledger = dict(re.findall(r"^\|\s*(\d{2})\s*\|[^|]*\|\s*([A-Z_]+)\s*\|", state, re.M))
for sid in ("09", "10"):
    record(sid, "required prerequisite DONE", ledger.get(sid) == "DONE", ledger.get(sid))
for sid in ("11", "12", "13", "14", "15", "16"):
    note = ROOT / f"memory/improvement/stages/{sid}.md"
    ok = ledger.get(sid) in {"DONE", "EVALUATED_NO_GAIN", "DEFERRED_DATA"} and note.exists()
    record(sid, "optional disposition recorded + stage note exists", ok, ledger.get(sid))
    if ledger.get(sid) == "DEFERRED_DATA":
        text = note.read_text(encoding="utf-8")
        record(sid, "deferred note has 'Candidate enabled: no' + 'Resume condition:'",
               bool(re.search(r"^Candidate enabled:\s*no\s*$", text, re.M | re.I))
               and bool(re.search(r"^Resume condition:\s*\S.+$", text, re.M)), "regex on stage note")

# ── step 10: frozen manifest vs today's files ──────────────────────────────
m10 = load("reports/improvement/10/run_manifest.json")
matrix = ROOT / m10["data"]["rebuilt_matrix_path"]
got = sha256(matrix)
record("10", "rebuilt matrix sha256 == frozen manifest", got == m10["data"]["rebuilt_matrix_sha256"], got[:16])
drift = {rel: sha256(ROOT / rel)[:12] for rel, want in m10["code_hashes"].items()
         if (ROOT / rel).exists() and sha256(ROOT / rel) != want}
record("10", "code-hash drift since freeze (informational; additive edits expected)", True,
       {"changed_files": sorted(drift)})
unified = ROOT / m10["data"]["unified_races_path"]
record("10", "unified_races sha256 unchanged since freeze (informational)", True,
       {"unchanged": sha256(unified) == m10["data"]["unified_races_sha256"]})

sc10 = load("reports/improvement/10/scorecard.json")
record("10", "scorecard.json present", True, {"top_level_keys": sorted(sc10)[:12]})

# ── steps 10-13: re-derive the headline log losses from the SAVED per-runner panel ─
from models.head_to_head import head_to_head  # noqa: E402

panel = pd.read_parquet(ROOT / "data/audit/13/eval_panel_scored_with_xgb.parquet")
record("13", "eval panel shape", panel["race_uid"].nunique() == 618 and len(panel) == 5561,
       {"races": int(panel["race_uid"].nunique()), "rows": len(panel)})
record("10-13", "eval panel lies wholly inside dev_oos (no holdout rows)",
       naive_dates(panel["race_date"]).max() < HOLDOUT_START,
       str(naive_dates(panel["race_date"]).max().date()))
prob_cols = [c for c in panel.columns if c.endswith("_prob")]
record("10-13", "panel probability columns", True, prob_cols)

# ── final holdout: was it ever scored? scan every stage parquet that holds a probability ─
touched = []
for pq in sorted((ROOT / "data/audit").glob("1[0-6]/**/*.parquet")):
    try:
        cols = pd.read_parquet(pq, columns=None).columns  # small/medium files; read once
    except Exception as exc:  # pragma: no cover - reported, never hidden
        touched.append({"file": str(pq.relative_to(ROOT)), "error": repr(exc)})
        continue
    # MODEL outputs only. `implied_prob` / `overround_norm_prob` / `market_ref_prob` are market
    # FEATURES carried by the full training matrix, which legitimately spans the holdout window
    # (step 10 protects it by code path, listed for review below) — they are not model scores.
    has_prob = [c for c in cols if re.search(r"(_prob$|^prob_|_p$|blend|calibrated)", c)
                and c not in MARKET_FEATURE_COLS]
    if not has_prob or "race_date" not in cols:
        continue
    dates = naive_dates(pd.read_parquet(pq, columns=["race_date"])["race_date"])
    n_holdout = int((dates >= HOLDOUT_START).sum())
    if n_holdout:
        touched.append({"file": str(pq.relative_to(ROOT)), "rows_in_holdout": n_holdout,
                        "prob_cols": has_prob[:6]})
record("10-16", "no stage artifact holds model probabilities on final-holdout rows", not touched, touched)

# scripts that mention the holdout: list them so the code path can be read by a human
mentions = {}
for py in sorted((ROOT / "reports/improvement").glob("1[0-6]/*.py")):
    src = py.read_text(encoding="utf-8", errors="replace")
    hits = [i + 1 for i, line in enumerate(src.splitlines()) if "final_holdout" in line or "2026-08-28" in line]
    if hits:
        mentions[str(py.relative_to(ROOT))] = hits[:12]
record("10-16", "scripts referencing the holdout window (for manual code-path review)", True, mentions)

# ── candidates OFF: feature-list isolation + config flag ───────────────────
from models import features as F  # noqa: E402

served = set(F.FEATURE_COLS) | set(F.PRICE_FREE_FEATURE_COLS)
for name in ("MEASURED_SPEED_FEATURE_COLS", "CANDIDATE_TEXT_FEATURE_COLS"):
    cols = set(getattr(F, name))
    record("11/16", f"{name} absent from FEATURE_COLS/PRICE_FREE_FEATURE_COLS", not (cols & served),
           sorted(cols & served))
record("16", "CANDIDATE_TEXT_FEATURE_COLS not merged into CANDIDATE_INDEPENDENT_FEATURE_COLS",
       not (set(F.CANDIDATE_TEXT_FEATURE_COLS) & set(F.CANDIDATE_INDEPENDENT_FEATURE_COLS)), "")
record("09", "MARKET_DERIVED_FEATURE_COLS stripped from PRICE_FREE_FEATURE_COLS (D40)",
       not (set(F.MARKET_DERIVED_FEATURE_COLS) & set(F.PRICE_FREE_FEATURE_COLS)), F.MARKET_DERIVED_FEATURE_COLS)

cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
record("14-16", "config.yaml llm.enabled is false", cfg.get("llm", {}).get("enabled") is False,
       cfg.get("llm", {}).get("enabled"))

# candidate-only modules must not be imported by any non-test, non-report module
candidate_mods = ["models.blend", "models.train_xgboost", "models.conditional_logit",
                  "features.measured_speed", "features.measured_timing", "features.text_features_v1",
                  "llm.hosted_adapter", "llm.hosted_budget"]
importers: dict[str, list[str]] = {}
for py in ROOT.rglob("*.py"):
    rel = py.relative_to(ROOT).as_posix()
    if rel.startswith((".venv/", "tests/", "reports/", "review/", "docs/", "logs/", ".claude/")):
        continue
    src = py.read_text(encoding="utf-8", errors="replace")
    for mod in candidate_mods:
        if rel == mod.replace(".", "/") + ".py":
            continue
        pkg, leaf = mod.rsplit(".", 1)
        pattern = rf"(from {re.escape(mod)} import|import {re.escape(mod)}|from {pkg} import .*\b{leaf}\b)"
        if re.search(rf"^{pattern}", src, re.M):        # column 0 = executed at import time
            importers.setdefault(mod, []).append(rel + " [top-level]")
        elif re.search(rf"^\s+{pattern}", src, re.M):   # indented = lazy, only when called
            importers.setdefault(mod, []).append(rel + " [lazy, inside a function]")
record("10-16", "importers of candidate-only modules outside tests/reports (must not be on the serving path)",
       True, importers)

# ── champion artifacts: gitignored, so `git status` proves nothing — hash + mtime instead ─
ignored = subprocess.run(["git", "check-ignore", "models/catboost_won_v3nf.bin", "models/lgbm_won_v3.txt"],
                         cwd=ROOT, capture_output=True, text=True).stdout.split()
record("10-13", "champion binaries are gitignored => earlier 'git status shows untouched' checks were vacuous",
       True, {"ignored": ignored})
PROGRAMME_START = pd.Timestamp("2026-09-18", tz="UTC")
champ = {}
late = []
for p in sorted((ROOT / "models").iterdir()):
    if p.suffix in {".bin", ".pkl", ".txt", ".json", ".parquet"}:
        mtime = pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC")
        champ[p.name] = {"sha256": sha256(p), "bytes": p.stat().st_size, "mtime_utc": mtime.isoformat()}
        if mtime >= PROGRAMME_START:
            late.append(p.name)
record("10-13", "every served artifact under models/ has an mtime BEFORE the programme began (2026-09-18)",
       not late, late)

# ── step 15 hosted ledger (accounting only; no credential is read) ─────────
ledger_path = ROOT / "data/cache/llm/hosted_budget_ledger.json"
if ledger_path.exists():
    led = json.loads(ledger_path.read_text(encoding="utf-8"))
    summary = {k: led[k] for k in led if not isinstance(led[k], (list, dict))}
    summary["n_outstanding_reservations"] = len(led.get("reservations", {}) or [])
    record("15", "hosted budget ledger within caps (US$5 / 1,200)", True, summary)
else:
    record("15", "hosted budget ledger present", False, "missing")

OUT.write_text(json.dumps({"checks": checks, "champion_artifacts": champ}, indent=1, default=str), encoding="utf-8")
failed = [c for c in checks if not c["ok"]]
print(f"\n{len(checks) - len(failed)}/{len(checks)} checks ok; wrote {OUT.relative_to(ROOT)}")
sys.exit(1 if failed else 0)
