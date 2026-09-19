# Race Predictor v3 — Setup Guide

A machine-learning horse racing predictor with a live Streamlit dashboard, multi-bookie odds scraping, Telegram alerts, and automated PDF reports.

---

## 1. Prerequisites

| Requirement | Version                                      |
| ----------- | -------------------------------------------- |
| Python      | 3.14                                         |
| OS          | Windows 11 (tested), macOS/Linux should work |
| Internet    | Required for scraping and Telegram           |

You will also need **Playwright's Chromium browser** installed (used as a fallback scraper when sites serve Cloudflare challenges). The install command is in the next section.

---

## 2. Installation

From the project root, run these three commands in order:

```
pip install -r requirements.txt
playwright install chromium
```

That's it. No database setup is needed — the SQLite database at `data/races.db` is created automatically on first run.

---

## 3. Configuration

All settings live in `config.yaml` at the project root. Most defaults are fine. The sections below cover only the keys you are likely to need to change.

### 3.1 Proxy pool (`proxy_pool`)

The predictor routes all scraping requests through a residential proxy to avoid IP bans.

```yaml
proxy_pool:
  enabled: true
  proxies:
    - http://YOUR_LOGIN:YOUR_PASSWORD@gw.dataimpulse.com:823
```

**How to update:** Sign in to your DataImpulse dashboard, copy the credentials for an Ireland-targeted residential proxy, and replace the existing line in the `proxies` list. The format must be `http://LOGIN:PASSWORD@HOST:PORT`.

If you don't have a DataImpulse account you can set `enabled: false` to scrape without a proxy, but expect more Cloudflare blocks on BoyleSports and Timeform.

### 3.2 Timeform session cookie (`timeform.session_cookie`)

```yaml
timeform:
  session_cookie: "" # paste your Timeform subscriber cookie here
```

**What it unlocks:** Timeform Form Rating (TFR), pace figures, and detailed historical form. Without it, the scraper still works but returns only publicly available race card data — TFR columns will be blank.

**How to get it:**

1. Log in to timeform.com with your subscriber account in Chrome/Firefox.
2. Open DevTools → Application → Cookies → `www.timeform.com`.
3. Copy the value of the `.ASPXAUTH` or `TFSession` cookie (whichever is present).
4. Paste it as the `session_cookie` value in `config.yaml`.

The cookie typically expires after 30 days. If predictions lose TFR data, refresh the cookie.

### 3.3 Telegram notifications (`notifications.channels.telegram`)

The bot token and chat ID are already filled in for you:

```yaml
notifications:
  enabled: true
  channels:
    telegram:
      enabled: true
      bot_token: "8648732935:AAErLnphLrBGxcpCO7SAwcpbXxbKAF71Q5g"
      chat_id: "6618994572"
```

**To verify the bot is working:** Open Telegram and send any message to your bot, then trigger a test alert from the Settings page in the UI (see section 6). If no message arrives within 10 seconds, double-check that `notifications.enabled` and `notifications.channels.telegram.enabled` are both `true`.

**Events that trigger alerts:**

- `race_soon` — a race you have a prediction for is starting within 10 minutes (configurable via `race_soon_minutes`)
- `odds_drop` — live odds shorten by 10 % or more (configurable via `odds_drop_threshold_pct`)

### 3.4 Scheduled PDF reports (`reporter`)

```yaml
reporter:
  enabled: false # change to true to start background reports
  interval_hours: 24 # how often a report is generated
  output_dir: reports # folder relative to project root
```

Leave `enabled: false` until you have run the app at least once and the model is trained. Once predictions are flowing, set `enabled: true` and restart the app — a PDF summary will appear in `reports/` every 24 hours.

### 3.5 Other keys worth knowing

| Key                      | Default | What it does                                       |
| ------------------------ | ------- | -------------------------------------------------- |
| `each_way_threshold`     | `8.0`   | Minimum decimal odds to suggest each-way bet       |
| `low_odds_threshold`     | `2.0`   | Odds below this are flagged as odds-on             |
| `gbp_eur_rate`           | `1.18`  | Static GBP→EUR conversion rate displayed in the UI |
| `model_weights.lightgbm` | `0.6`   | Blend weight for the LightGBM model vs baseline    |
| `scrape_interval`        | `3600`  | Seconds between full scrape cycles                 |

---

## 4. Running the app

From the project root:

```
streamlit run ui/app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open it in your browser. The sidebar lets you navigate between pages.

---

## 5. Running the test suite

```
python -m pytest tests/ -v
```

All 604 tests should pass (3 known skips, 0 failures). The test suite does not require a live internet connection.

---

## 6. Feature walkthrough

### Live Races

Shows today's race cards with live odds pulled from LivescoreBet and BoyleSports. Each runner card displays current win odds, the model's predicted probability, and an each-way flag if odds are above the threshold. Odds update automatically when the scraper cycle completes.

### Predictions

A filterable table of all model predictions for upcoming races. Columns include predicted win probability, expected value, confidence band, jockey, trainer, and going preference. You can sort by any column or filter down to a specific venue or race type. Click any row to open the full runner breakdown.

### Performance Dashboard

Historical accuracy charts showing model ROI, precision/recall curves, and a confusion matrix for win/place/show predictions. The drift panel shows the latest PSI score for each feature — a score above 0.2 means the model has seen significant distribution shift and may need retraining. Use the "Retrain" button on this page to kick off a new training run.

### Bet Placer

An interface to log bets manually and track your bankroll. Enter the bookie, stake, odds, and race, and the system records the bet and marks it settled once results are scraped. Running P&L and ROI are shown at the top. This page does not place bets automatically — it is a record-keeping tool.

### Race Compare

Side-by-side comparison of two or more runners across the same race or different races. Useful for handicap and each-way decisions. Shows going ratings, speed figures, jockey/trainer strike rates, and recent form. Drag runner cards to reorder the comparison columns.

### Settings

Live editor for key `config.yaml` values without leaving the browser. Changes are written back to disk immediately. Also contains a **Test Telegram** button that fires a test notification to verify your bot credentials, and a **Force Scrape** button to trigger an immediate scrape cycle outside the normal interval.

---

## 7. Scheduled PDF reports

When `reporter.enabled: true` is set in `config.yaml`, a background scheduler fires every `interval_hours` hours (default 24) and writes a PDF summary to `reports/`. The PDF includes:

- Top 5 predictions by expected value for the next race day
- Yesterday's results vs predictions (win rate, ROI)
- Model drift summary (current PSI scores)
- Any Telegram alerts fired in the last 24 hours

The scheduler starts when the app starts and stops when you close it. Reports are never deleted automatically — clean up `reports/` manually if disk space is a concern.

---

## 8. Telegram alerts

### Verifying the bot works

1. Open the app and go to **Settings**.
2. Click **Test Telegram** — this sends a test message immediately.
3. If the message arrives in your Telegram chat, the bot is configured correctly.
4. If it doesn't arrive, check:
   - `notifications.enabled: true` (top-level flag)
   - `notifications.channels.telegram.enabled: true`
   - The `bot_token` and `chat_id` values match your BotFather bot and your Telegram user/chat ID

### What triggers real alerts

| Event     | Condition                                                             |
| --------- | --------------------------------------------------------------------- |
| Race soon | Race start ≤ 10 minutes away and you have an active prediction for it |
| Odds drop | Any tracked runner's live odds shorten by ≥ 10 % since last check     |

Both thresholds are adjustable in `config.yaml` under `notifications.events`.

---

## 9. Known limitations

- **Paddy Power** — The Paddy Power scraper targets `apisms.paddypower.com`. The endpoint structure was inferred from public DevTools traces and has not been confirmed against a live session. Expect occasional 404s or empty odds until the endpoint is verified.

- **Timeform TFR** — Full Timeform Form Ratings and pace data require a paid Timeform subscriber account. Without a valid `session_cookie`, these columns will be blank in the Predictions table and the Performance Dashboard. The model will still predict using the remaining features.

- **BoyleSports each-way terms** — BoyleSports does not expose each-way fraction/place terms in their HTML. The predictor cross-sources this from whichever other bookie (LivescoreBet, Paddy Power) has already been scraped for the same race. If neither has been scraped yet, the EW terms cell will show "unknown".

---

## 10. Troubleshooting

### Cloudflare 403 on BoyleSports or Timeform

The proxy is either blacklisted or Cloudflare has tightened detection. The scraper automatically retries with Playwright's headless Chromium (which passes Cloudflare's JS challenge). If Playwright is not installed, run `playwright install chromium`. If the 403 persists, rotate your DataImpulse proxy credentials.

### Empty predictions / Predictions table is blank

The model has not been trained yet. Go to **Performance Dashboard** and click **Retrain**. Training requires at least some historical data in `data/historical/` — run a manual scrape first via **Settings → Force Scrape**, wait for it to complete, then retrain.

### Telegram alerts not sending

Check in order:

1. `notifications.enabled` is `true` in `config.yaml`
2. `notifications.channels.telegram.enabled` is `true`
3. `bot_token` and `chat_id` are correct (use **Settings → Test Telegram** to verify)
4. Your bot has been started in Telegram — send `/start` to your bot in the Telegram app if you haven't already

### `ImportError` on startup

Most likely a missing dependency. Run `pip install -r requirements.txt` again. If the error names a specific package (e.g. `catboost`), install it directly with `pip install catboost`.

### `sqlite3.OperationalError: no such table`

The database hasn't been initialised. Delete `data/races.db` if it exists and restart the app — the schema is created on first launch.

### Jockey / trainer names showing blank in Predictions

This was a known bug (fixed in v3). If you see blank names, confirm you are running the latest code — the fix is in `models/predictor.py` and reads from `jockey_name` / `trainer_name` columns in the feature frame.
