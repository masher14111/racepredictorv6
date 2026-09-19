# Quick start

The short version. `README.md` explains the design, `FINALSETUP.md` is the full
operations manual — this page is just the buttons to press.

> **Status: paper only.** The model is currently rated **NO-GO** — it does not
> beat the closing market, so the system deliberately produces **no betting
> recommendations**. Seeing "0 candidates" is the system working correctly, not
> a bug. Nothing here can stake real money.

---

## 0. No typing — double-click launchers

The numbered `.bat` files in this folder run everything below for you. They use
the project's own `.venv` directly, so there is nothing to activate, and they
work from wherever the folder lives.

| file                          | what it does                                                      |
| ----------------------------- | ----------------------------------------------------------------- |
| `0 - First-Time Setup.bat`    | once per PC: scraper browser (Playwright Chromium), DB, config check — and rebuilds `.venv` if it is missing |
| `1 - Full Daily Run.bat`      | **the everyday button** — §2's three commands in order, ending with the dashboard |
| `2 - Open Dashboard Only.bat` | just the dashboard at `http://localhost:8501` (no scraping); local-only |
| `3 - Run Tests.bat`           | §5's health check, with a plain PASSED / FAILED verdict           |

Add `--dry-run` to any of them to print the commands without running them.

---

## 1. First-time build

Do this once. Takes a few minutes; the repo already ships trained models and
historical data, so you are not starting from scratch.

```powershell
# from the project root (the folder containing this file) — currently
# C:\Users\mshr\Documents\Race Predictor v4\Race Predictor v4

# a) activate the virtual environment
.\.venv\Scripts\Activate.ps1

# b) install dependencies (first time, or after a git pull that changes them)
pip install -r requirements.txt
pip install -r requirements-dev.txt      # only if you want to run the tests
playwright install                       # one-time: browser binaries for the scrapers

# c) set up the database
python -m utils.storage migrate
python -m utils.storage version          # expect: current=5 target=5

# d) check the config loads
python -c "from utils.config_loader import get_config; print('config OK', bool(get_config()))"
```

Secrets (bookmaker logins, proxy details) go in `config.local.yaml`, which is
gitignored. See §3 of `FINALSETUP.md` if you need to create it.

**That's it — you're built.** Skip to §2.

<details>
<summary>Only if you have no trained models at all (rare)</summary>

```powershell
python pipeline_full.py --skip-scrape
```

Rebuilds features and trains both model lines. Roughly 30–60 minutes. Add
`--resume` if it dies partway and you want to pick up where it stopped.

</details>

---

## 2. Every day — just start it

Three commands. Run them in order.

```powershell
.\.venv\Scripts\Activate.ps1

# 1) get today's racing: scrape -> normalize -> predict (also takes today's
#    first live price snapshot)
python -m scripts.refresh

# 2) gate -> issue paper tickets -> settle yesterday's -> reports
#    (--no-scrape: step 1 already captured today's first snapshot)
python -m scripts.daily_paper_loop --no-scrape

# 3) open the dashboard
streamlit run ui/app.py
```

`daily_paper_loop` is idempotent — re-running it later the same day never
double-issues a ticket or double-counts a settlement; a day it never ran on is
recorded as a **gap**, not silently treated as a clean no-bet day. Drop
`--no-scrape` on a later same-day run if you want it to grab another intraday
price poll on top. It is paper-only and does not change the model gate
(currently NO-GO). See `HANDOFF.md`'s Stage 6 section for the scheduled
(unattended) form of this command.

The dashboard opens in your browser. The pages you'll actually use:

| page                   | what it tells you                                       |
| ---------------------- | ------------------------------------------------------- |
| **Today**              | today's cards and model probabilities                   |
| **Model Honesty**      | whether the model is beating the market (currently: no) |
| **Forward Validation** | paper-betting evidence and the release gate             |

If the scrapers are being blocked or you just want to re-score what you already
have, skip the scrape:

```powershell
python -m scripts.refresh --no-scrape
python -m scripts.daily_paper_loop --no-scrape
```

---

## 3. Checking the paper-betting view

This runs the honest evaluation and writes today's dated reports.

```powershell
python -m scripts.paper_betting
```

It prints a verdict line at the end and writes three files into `reports/`:

- `forward_validation_<date>.md` — the full picture, kept in three separate
  lanes: historical backtest, paper bets, and real money (permanently empty).
- `todays_candidates_<date>.md` — today's card. While the model is NO-GO this
  will list **zero candidates** and explain, runner by runner, why each one was
  passed over.
- `candidate_decisions_<date>.csv` — the same thing as a spreadsheet.

Want just today's card without re-running the backtest (much faster):

```powershell
python -m scripts.paper_betting --candidates-only
```

---

## 4. When to rebuild, and how much

Most days you rebuild nothing — `scripts.refresh` is enough.

| situation                                  | command                                 | roughly                  |
| ------------------------------------------ | --------------------------------------- | ------------------------ |
| Normal day                                 | `python -m scripts.refresh`             | 2–5 min                  |
| Scrapers blocked, re-score only            | `python -m scripts.refresh --no-scrape` | under 1 min              |
| Results look stale / features changed      | `python pipeline_full.py --resume`      | 10–40 min                |
| Full retrain from scratch                  | `python pipeline_full.py --skip-scrape` | 30–60 min                |
| Full retrain _with_ a fresh history scrape | `python pipeline_full.py`               | hours, and often blocked |

`--resume` skips any stage whose output is already up to date, so it is almost
always the one you want. `--force` ignores that and redoes everything.

> **If you retrain, the honesty numbers are no longer valid.** A new model has
> not been audited, so re-run `python -m scripts.paper_betting` afterwards and
> read the verdict before trusting anything on screen.

---

## 5. Checking it still works

```powershell
python -m pytest -q
```

Expect **1993 passed**. It takes a few minutes. Do not add `--timeout` — the
plugin that flag needs isn't installed and it will just error.

Quick data sanity check at any time:

```powershell
python scripts/data_health.py
```

---

## 6. If something breaks

| symptom                                    | what to do                                                           |
| ------------------------------------------ | -------------------------------------------------------------------- |
| `config OK` fails                          | `config.local.yaml` is missing or malformed — see `FINALSETUP.md` §3 |
| Scrapers return nothing                    | Usually Cloudflare. Use `--no-scrape` and try again later            |
| Dashboard shows stale races                | Re-run `python -m scripts.refresh`                                   |
| `python -m utils.storage version` mismatch | Run `python -m utils.storage migrate`                                |
| Zero candidates                            | **Not a fault.** The model is NO-GO; see §3                          |
| Tests fail after a `git pull`              | `pip install -r requirements.txt` first                              |
| `.venv` broken (folder moved / Python reinstalled) | Virtualenvs are not relocatable. Delete `.venv`, then `python -m venv .venv` and `.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt` (exact known-good versions) |

---

## 7. What isn't built yet

As of Stage 6, live point-in-time price capture, the daily paper loop
(capture → gate → issue → settle → reports), and the eight-week
forward-validation window (freeze/reset on any model or config change, gap
accounting for days the loop never ran) all exist and are wired in — see
`HANDOFF.md`'s Stage 6 section for what actually runs and how to check on it.

Still missing:

- the eight-week forward window itself has not accumulated enough real days
  yet to reach a verdict — `FORWARD GATE NOT MET` is expected and correct for
  some time; do **not** read the window's mere existence as progress toward a
  release;
- the unattended schedule is documented (the exact `schtasks` command is in
  `HANDOFF.md`) but not installed — it only runs when you run it yourself, or
  register the task.

The honest position stands regardless: **the model does not beat the market,
so the system does not recommend bets.**
