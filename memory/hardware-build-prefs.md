---
name: hardware-build-prefs
description: Local hardware + preference to use GPU (mainly), then RAM/CPU for faster builds/training/backfills
metadata:
  type: feedback
---

User's local machine — prefer GPU first, then RAM/CPU, for any heavy build, training, hyperparameter search, or backfill. Maximise hardware use for speed.

**Hardware:**

- GPU: NVIDIA RTX 5070 Ti, 16 GB VRAM (primary compute — use it).
- RAM: 32 GB @ 3200 MHz.
- CPU: Intel i5-10500, 3.10 GHz (6 cores / 12 threads).

**Why:** User explicitly wants faster builds and to lean on the GPU as the main workhorse, with RAM/CPU as secondary parallelism.

**How to apply:**

- CatBoost training already set for GPU in `config.yaml` (`model.task_type: GPU`, `devices: "0"`) — keep it on GPU; it auto-falls back to CPU only if no GPU. Don't switch to CPU.
- `model.thread_count: -1` = all CPU cores for the CPU-side work; keep it.
- For scrapers/backfills, use the parallel worker knobs in `config.yaml` (`backbone_workers`, `enrichment_workers`, per-domain `max_concurrent`) rather than single-threaded runs.
- Watch the 16 GB VRAM ceiling and 32 GB RAM ceiling on big feature matrices / Optuna trials — chunk if needed, but default to the GPU path.
- Pipeline rebuild commands are in `data/README.md` / [[cleanup-05-data-artifacts]]; run them on GPU where applicable.
