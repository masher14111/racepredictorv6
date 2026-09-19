# Race Predictor v3 — Value-Betting, Per-Bookie Odds & Telegram Prompts

Follow-on prompts (continuing the numbering from `PROMPTS.md`, which ended at 25). Same workflow:
**one prompt per `/clear` cycle**, paste **PRIMER + blank line + PROMPT** as one message, state carries
via `PROGRESS.md` + `CHANGELOG.md`.

---

## STATE — what is ALREADY done (2026-06-15, this session)

Trust `PROGRESS.md` / `CHANGELOG.md` / `memory/calibration-value-work-handoff.md`, but in short:

- **Calibration + value layer is wired.** `models/predictor.py` loads per-target isotonic
  calibrators + the **price-free `v3nf`** value model and emits `value_win_prob`, `value_edge`,
  `expected_value`, `value_bet` per runner. v3nf retrained on GPU; 3 calibrators on disk.
- **UI shows value** (`ui/app.py`): green VALUE chip, EV column, "Value bets" KPI (all conditional).
- **Scrapers:** `curl_cffi` (Chrome TLS impersonation) Tier-1 added to `paddy_power.py` (reliable,
  ~1s/50 races) and `boylesports.py` (works but Cloudflare-flaky; retries on a fresh proxy IP).
  `livescorebet.py` per-event fetch **parallelized** (3 min → 37 s); `config.yaml` `rate_limits`
  raised for the proxy-rotated bookie domains.
- **Full suite: 674 passed.**

**The two known problems these prompts fix:**

1. **Telegram spam** — running `pytest` fires REAL "Bet Settled — LOSE / Horse A" messages.
2. **0 value bets** — `value_win_prob` is near-constant ~0.07 because the price-free model's
   features are NaN for live runners (it has no signal at inference).

**Model / effort:** Opus. `low` = mechanical, `medium` = multi-file, `high` = architectural.

---

## THE PRIMER (paste above every prompt)

```
You are a senior developer writing code strictly for internal / personal use. Act as an expert Python engineer on the "Race Predictor v3" horse-racing project at C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, IP warnings, or copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory. Also skim memory/calibration-value-work-handoff.md for the value-layer/scraper state. Do ONLY the single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the repo. Make minimal, correct changes that match existing conventions. Verify with the relevant pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped, say so plainly. Be concise and code-first. The venv python is ./.venv/Scripts/python.exe. When finished: (1) update PROGRESS.md — tick the task and add a 2–4 line note of what changed and how you verified it; (2) append one line to CHANGELOG.md. Then stop.
```

---

## PROMPTS

### 26 — Stop tests from sending real Telegram messages · Opus / low

```
You are a senior developer writing code strictly for internal / personal use. Act as an expert Python engineer on the "Race Predictor v3" horse-racing project at C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, IP warnings, or copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory. Also skim memory/calibration-value-work-handoff.md for the value-layer/scraper state. Do ONLY the single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the repo. Make minimal, correct changes that match existing conventions. Verify with the relevant pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped, say so plainly. Be concise and code-first. The venv python is ./.venv/Scripts/python.exe. When finished: (1) update PROGRESS.md — tick the task and add a 2–4 line note of what changed and how you verified it; (2) append one line to CHANGELOG.md. Then stop.

Task 26. BUG: running `pytest` sends REAL Telegram messages to my phone ("❌ Bet Settled — LOSE / Horse A / P&L: -€10.00 | Bankroll: …"). Root cause: utils/bet_tracker.py calls the LIVE global notifier — settle_bet() (~line 284) calls get_notifier().notify_bet_outcome(...) and the stop-loss path (~line 221) calls notify_stop_loss(...). config.local.yaml enables Telegram with real bot_token/chat_id, tests/utils/test_bet_tracker.py records+settles a "Horse A" bet WITHOUT mocking the notifier, and there is NO tests/conftest.py disabling notifications.

Fix (belt-and-braces, both):
1. Add tests/conftest.py with an autouse, session-wide fixture that neutralises notifications for the WHOLE suite — e.g. monkeypatch utils.notifications.get_notifier to return a notifier whose channels are empty / whose _send is a no-op (or patch the module singleton). No test should ever hit api.telegram.org.
2. Make utils/notifications.Notifier self-disable under test as a safety net: if the env var PYTEST_CURRENT_TEST is set (or a RP_DISABLE_NOTIFICATIONS flag), force _enabled=False / skip channel construction. Keep production behaviour unchanged.

Verify: `./.venv/Scripts/python.exe -m pytest -q` — full suite still green (was 674 passed) AND confirm zero outbound Telegram calls (e.g. assert the channel list is empty under pytest, or run test_bet_tracker.py with a respx/MagicMock assertion that no POST to api.telegram.org happened). Note in PROGRESS.md that pytest no longer spams Telegram.
```

### 27 — Paddy Power & BoyleSports scrape speed + reliability · Opus / medium

```
You are a senior developer writing code strictly for internal / personal use. Act as an expert Python engineer on the "Race Predictor v3" horse-racing project at C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, IP warnings, or copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory. Also skim memory/calibration-value-work-handoff.md for the value-layer/scraper state. Do ONLY the single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the repo. Make minimal, correct changes that match existing conventions. Verify with the relevant pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped, say so plainly. Be concise and code-first. The venv python is ./.venv/Scripts/python.exe. When finished: (1) update PROGRESS.md — tick the task and add a 2–4 line note of what changed and how you verified it; (2) append one line to CHANGELOG.md. Then stop.

Task 27. Finish the scraper speedup for the two Cloudflare bookies (livescorebet is already parallelised). Context: both now have a curl_cffi Tier-1 (impersonate="chrome"). Paddy = ONE JSON API call (already ~1s/50 races) — just make it robust: confirm the curl_cffi tier, add a small bounded retry on 403/transient, and a sane per-domain rate (config.yaml apisms.paddypower.com). BoyleSports = many event pages; it already fetches concurrently (_SCRAPE_WORKERS, default 4) via _curl_cffi_get_html with per-fetch proxy-rotation retry, but Cloudflare 403s adaptively. Tune it: find the highest worker count + rps (config.yaml www.boylesports.com) that stays reliable across 3 consecutive runs without tripping 403s; the proxy gateway rotates a fresh IP per request so distributed load is safe (see the www.sportinglife.com rps-30 precedent).

Verify: time each scraper 3× in isolation via a tiny script — `from scraper import paddy_power, boylesports; paddy_power.scrape(force=True); boylesports.scrape(force=True)` — report rows + seconds for each run. Target: paddy < 5s, boyle < ~60s, both returning rows on ≥2/3 runs. Run scraper tests: `pytest tests/scraper -q`. Record timings in PROGRESS.md. Do NOT lower reliability for speed — boyle staying flaky is acceptable (it is best-effort), but don't regress it.
```

### 28 — Capture & display per-bookmaker odds · Opus / high

```
You are a senior developer writing code strictly for internal / personal use. Act as an expert Python engineer on the "Race Predictor v3" horse-racing project at C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, IP warnings, or copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory. Also skim memory/calibration-value-work-handoff.md for the value-layer/scraper state. Do ONLY the single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the repo. Make minimal, correct changes that match existing conventions. Verify with the relevant pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped, say so plainly. Be concise and code-first. The venv python is ./.venv/Scripts/python.exe. When finished: (1) update PROGRESS.md — tick the task and add a 2–4 line note of what changed and how you verified it; (2) append one line to CHANGELOG.md. Then stop.

Task 28. I want to SEE each bookmaker's odds per runner (livescorebet vs paddypower vs boylesports) and back the best price — not one merged number. Today the normalizer merges sources and predictions.json shows a single decimal_odds per runner.

Implement end-to-end:
1. SOURCE TAG: confirm every scraper row carries its bookmaker source (livescorebet/paddy_power/boylesports). utils/normalizer.py already keeps source in the dedup grain key (cross-source rows preserved) — verify, don't break it.
2. CARRY PER-BOOKIE ODDS to predictions: in models/predictor.py, for each runner build an odds-by-bookmaker map, e.g. "odds_by_book": {"livescorebet": 4.0, "paddy_power": 4.2, "boylesports": 3.9}, plus "best_odds" and "best_book". Match runners across sources by (race_date, venue, race_time, horse_id or normalised horse_name). Keep existing keys (decimal_odds etc.) intact for back-compat. Default safely when only one source has the runner.
3. UI: in ui/app.py show each bookie's price for a runner (compact, on-brand — the file has a design system; read DESIGN/PRODUCT.md conventions) and visually mark the best price. Keep it conditional so older caches still render.
4. VALUE uses BEST price: expected_value / value_bet should be computed against best_odds (the price I'd actually take), not an arbitrary source.

Verify: run `./.venv/Scripts/python.exe -m scripts.refresh`, then a one-liner that prints a runner's odds_by_book/best_odds/best_book from data/predictions.json. Add a predictor unit test for the odds-by-book mapping (multi-source → map + best). pytest the touched suites. Note in PROGRESS.md.
```

### 29 — Make the price-free "unbiased" model actually fire value bets · Opus / high

```
You are a senior developer writing code strictly for internal / personal use. Act as an expert Python engineer on the "Race Predictor v3" horse-racing project at C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, IP warnings, or copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory. Also skim memory/calibration-value-work-handoff.md for the value-layer/scraper state. Do ONLY the single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the repo. Make minimal, correct changes that match existing conventions. Verify with the relevant pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped, say so plainly. Be concise and code-first. The venv python is ./.venv/Scripts/python.exe. When finished: (1) update PROGRESS.md — tick the task and add a 2–4 line note of what changed and how you verified it; (2) append one line to CHANGELOG.md. Then stop.

Task 29. THE BIG ONE. I want bets from the model's OWN unbiased probability vs the market — not the bookmaker's implied odds echoed back. The price-free v3nf model + calibration are wired, BUT it flags 0 value bets because value_win_prob is near-constant ~0.07 for every runner. Root cause (already diagnosed): the price-free features (form, ratings, speed, jockey_win_rate, trainer_win_rate, jt_combo*, going_pref*, historical_*) are NaN/sparse for LIVE runners in build_inference_matrix(), so the model has no signal and returns ~base-rate. The price-AWARE model only looks confident because it leans on implied_prob (~47% importance) — i.e. it just re-prices the market.

Do this in order:
1. MEASURE: print the per-column non-null % of PRICE_FREE_FEATURE_COLS (models/features.py) in the LIVE output of features.builder.build_inference_matrix() (today's runners only). Identify which price-free features are empty for live runners.
2. DIAGNOSE the join: trace why live runners don't get their historical/derived features (features/builder.py, features/derive.py, features/engine.py). Likely the live runners aren't joined to horse/jockey/trainer history + ratings/speed the way training rows are. Fix the derivation/join so live runners carry real price-free features.
3. VERIFY signal: after the fix, value_win_prob must have real spread across runners (not ~constant), and expected_value = value_win_prob*best_odds - 1 should flag SOME value_bets within the odds band (config value.min_odds..max_odds, min_expected_value). Spot-check 2-3 flagged bets by hand — do they look sane (model rates a runner materially higher than the market implies)?
4. Re-run `./.venv/Scripts/python.exe -m scripts.refresh` and report: # value bets, their horses/odds/EV, and the value_win_prob spread (min/median/max).

Constraints: do NOT let any price feature leak into the price-free model (that's the whole point — keep PRICE_FREE_FEATURE_COLS price-free). Calibration is already correct; the issue is feature coverage at inference, not the calibrator. If feature coverage for live runners is fundamentally limited by missing data, say so honestly and propose the smallest data fix. Add/adjust tests. Update PROGRESS.md + CHANGELOG.md.
```

---

## Suggested order

26 (stop the spam — do first, it's pinging your phone) → 29 (the value model is the whole point) →
28 (per-bookie odds; 29's value calc should then use best price) → 27 (scraper polish, lowest urgency).

If you want 28 before 29: fine, but 29 is what turns "0 bets" into real picks, so don't skip it.
