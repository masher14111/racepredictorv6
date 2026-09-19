"""Task 08 validation: bounded real LivescoreBet scrape -> live_odds.parquet ->
normalizer -> inference -> predictor, asserting odds_decimal/implied_prob survive."""
import logging
logging.disable(logging.CRITICAL)
import os
from datetime import datetime

import pandas as pd

from utils.timezone import TZ
from scraper.livescorebet import (
    LivescoreBetClient, _extract_event_ids, _parse_event, _rows_to_df, _PARQUET_PATH,
)
from utils import normalizer as N

MAX_EVENTS = 40

c = LivescoreBetClient()
home = c.get_racing_home()
ids = _extract_event_ids(home)
rows = []
for eid in ids:
    try:
        rows.extend(_parse_event(c.get_event(eid)))
    except Exception as exc:
        print("  event fail", eid, exc)
    if len([r for r in rows]) and len(set(r["race_id"] for r in rows)) >= MAX_EVENTS:
        break

df = _rows_to_df(rows, datetime.now(tz=TZ))
print(f"[scrape] {len(df)} runner rows across {df['race_id'].nunique()} races")
print(f"[scrape] odds_decimal non-null: {df['odds_decimal'].notna().sum()}/{len(df)}")

# Persist a clean real snapshot, replacing the synthetic rows.
df.to_parquet(_PARQUET_PATH, engine="pyarrow", index=False)
print(f"[persist] wrote real data to {os.path.relpath(_PARQUET_PATH)}")

# --- normalizer maps odds_decimal into the canonical/unified schema ---
live_only = N.normalize(sources={"live_odds": df}, write=False)
print(f"[normalize] {len(live_only)} unified rows, "
      f"odds_decimal non-null: {live_only['odds_decimal'].notna().sum()}")

# --- end-to-end through the real predictor (build_inference -> derive -> score) ---
live_only["position"] = pd.NA  # upcoming: no result
from models.predictor import Predictor
pred = Predictor()
races = pred.predict(unified=live_only)
print(f"[predict] {len(races)} races scored")
n_dec = n_ip = n_sel = 0
shown = 0
for r in races:
    for s in r.get("selections", []):
        n_sel += 1
        if s.get("decimal_odds") is not None:
            n_dec += 1
        if s.get("implied_prob") not in (None, 0):
            n_ip += 1
        if shown < 6 and s.get("decimal_odds"):
            print(f"   {r['venue']:<22} {s['horse_name']:<20} "
                  f"decimal_odds={s['decimal_odds']} implied_prob={s['implied_prob']}")
            shown += 1
print(f"[predict] selections={n_sel} with decimal_odds={n_dec} with implied_prob={n_ip}")
assert n_dec > 0, "FAIL: no decimal_odds in predictor output"
assert n_ip > 0, "FAIL: implied_prob not derived"
print("PASS: live odds flow scrape -> normalize -> predict -> decimal_odds + implied_prob")
