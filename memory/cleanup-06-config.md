---
name: cleanup-06-config
description: Config consistency audit — bet_tracker section added, orphaned extractor key removed, 3 UI files routed through config_loader (2026-06-16)
metadata:
  type: project
---

Config consistency audit on branch `chore/config-audit` (off master).

**Config surface:** all reads go through `utils/config_loader.py` (`get()` / `get_config()`),
which loads `config.yaml`, deep-merges git-ignored `config.local.yaml` (secrets), then applies
`RP_`-prefixed env overrides. See [[review-02-utils-bugs]] for the loader's history.

**Keys ADDED to config.yaml:**

- `bet_tracker:` section (was in `_KNOWN_TOP_LEVEL_KEYS` but had no section — defaults always won).
  Fields: `initial_bankroll` 1000.0, `flat_stake` 10.0, `kelly_fraction` 0.25, `stop_loss_pct` 0.20.
  Read by `utils/bet_tracker.BetTracker` via the 3 UI bet/perf pages.
- `model.random_seed: 42` — was hardcoded as `random_state=42` in `models/tuner.py` CV shuffle.

**Keys REMOVED (orphaned):**

- `betsp_historical.extractor: stub` — never read; `scraper/betsp_historical.py` hardcodes
  `StubExtractor()`.

**Stray `yaml.safe_load` → routed through config_loader:**

- `ui/bet_placer.py`, `ui/performance.py`, `ui/performance_dashboard.py` now call
  `config_loader.get("bet_tracker", {})`; removed `import yaml` + unused `_CFG_PATH`.
- `ui/settings.py` KEPT raw yaml read+write — it's the config _editor_ (atomic save-back), a
  legitimate exception.

**UI tooltips:** expanded `help=` text on the Stake Strategy selectbox and Flat stake input in
`ui/bet_placer.py` to explain Flat vs Fractional/Full Kelly and what flat_stake means.

**Left as code (domain constants, not promoted):** Optuna search ranges in `tuner.py`,
`_COMPOSITE_W` blend weights + `_AE_TOLERANCE`/bucket thresholds in `models/predictor.py` /
`models/evaluate.py`.

**Tests:** `pytest tests` → 687 passed, 3 skipped. (Root `pytest` errors on `tools/slice_test.py`,
a live-network script pytest tries to collect — pre-existing, unrelated.)
