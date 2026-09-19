---
name: wrap-28-docs
description: Docs refresh 2026-06-17 — new top-level README.md + PROGRESS.md run-log entry
metadata:
  type: project
---

# Docs Refresh (2026-06-17)

Refreshed project documentation to match current state. No code changed.

**Why:** repo had FINALSETUP.md (ops guide) but no top-level README describing what the
project is and the pipeline at a glance.

**How to apply:** when docs drift, README is the at-a-glance entry point; `FINALSETUP.md`
is the deeper ops/troubleshooting guide; `PROGRESS.md` is the task/run-log source of truth.

What was done:

- Created **`README.md`** — overview, ASCII pipeline diagram
  (scraper → unified*races.parquet → features → training.parquet → models → predictions.json
  → ui), setup + secrets (`config.local.yaml` / `RP*`env, precedence env > local > yaml),
per-stage run commands, UI page table, paper-betting section (Kelly/EW/stop-loss via`utils/bet_tracker.py`, Telegram alerts), tests, and a Status & known limits section.
- Status claims kept honest from memory: v3 served raw/uncalibrated + market features built
  from `odds_finish` → prefer **v3nf** (price-free, isotonic) — see [[model-09-baseline-audit]];
  live feeds have no jockey/trainer; boyle/paddy Cloudflare-gated, livescorebet reliable.
  Test baseline 695 pass / 3 skip (reportlab) — see [[cleanup-08-tests]].
- Added a run-log entry to `PROGRESS.md`.
  </content>
