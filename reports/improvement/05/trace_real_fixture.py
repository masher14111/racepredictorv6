"""Stage 05 evidence: trace ONE real, previously-captured Timeform racecard
(tests/scraper/timeform/fixtures/racecard_single.html -- a genuine saved page,
York 2026-06-13) through provider-id parsing -> normalize -> fuse, and show a
live-odds-only row for the SAME runner losing the jockey/trainer/draw/weight
priority fight to the declared Timeform row. Read-only: writes nothing to any
shared data path.
"""
import pandas as pd

from scraper.timeform import parser
from utils.normalizer import _from_timeform
from features.fuse import fuse_sources

FIX = "tests/scraper/timeform/fixtures/racecard_single.html"

html = open(FIX, encoding="utf-8").read()
rows = parser.parse_event(html, venue="York", going="good", race_time="2026-06-13T13:50")
r = rows[0]
print("1) PARSED (real fixture, row 0):")
print(f"   horse={r.horse_name!r} horse_id={r.horse_id} timeform_race_id={r.timeform_race_id}")
print(f"   jockey={r.jockey_name!r} jockey_id={r.jockey_id}")
print(f"   trainer={r.trainer_name!r} trainer_id={r.trainer_id}")
print(f"   draw={r.draw} weight_lbs={r.weight_lbs} age={r.age} OR={r.official_rating} "
      f"equip={r.equipment} colour={r.colour} sex={r.sex} status={r.runner_status}")

native = pd.DataFrame([{**r.__dict__, "race_date": "2026-06-13", "position": None,
                        "market_type": "WIN", "region": "uk", "source": "timeform",
                        "fetched_at": "2026-06-13T09:00:00+00:00"}])
tf = _from_timeform(native)
print("\n2) NORMALIZED (canonical row):")
row = tf.iloc[0]
print(f"   horse_id={row['horse_id']} src_horse_id={row['src_horse_id']} "
      f"src_race_id={row['src_race_id']}")
print(f"   jockey_id={row['jockey_id']} src_jockey_id={row['src_jockey_id']} "
      f"jockey_name={row['jockey_name']}")
print(f"   draw={row['draw']} weight_lbs={row['weight_lbs']} official_rating={row['official_rating']}")

# A same-runner live ODDS row for the SAME horse/race, as livescorebet emits it:
# horse + price only, no jockey/trainer/draw/weight -- and fetched an hour later.
live_only = row.to_dict()
live_only.update({
    "source": "livescorebet",
    # whole-second precision, matching the timeform row's format (mixed sub-second
    # precision across rows makes pandas' to_datetime format-inference emit NaT --
    # a pre-existing fuse.py quirk, not a Stage 05 defect; sidestepped here)
    "fetched_at": pd.Timestamp.now(tz="UTC").floor("s").isoformat(),
    "jockey_name": pd.NA, "jockey_id": pd.NA, "trainer_name": pd.NA, "trainer_id": pd.NA,
    "draw": pd.NA, "weight_lbs": pd.NA, "official_rating": pd.NA,
    "odds_decimal": 9.0, "validation_status": "VALID",
})
combined = pd.DataFrame([row.to_dict(), live_only])
fused = fuse_sources(combined)
print(f"\n3) FUSED ({len(fused)} row(s) -- one runner, one WIN market):")
f = fused.iloc[0]
print(f"   source priority winner per-column: jockey_id={f['jockey_id']} "
      f"(declared, from timeform, NOT invented) draw={f['draw']} weight_lbs={f['weight_lbs']}")
print(f"   odds_decimal={f['odds_decimal']} (from the live odds row -- market-specific, "
      f"never taken from timeform)")
assert f["jockey_id"] == row["jockey_id"], "declared jockey_id must win over the odds-only row"
assert f["draw"] == row["draw"]
assert pd.notna(f["odds_decimal"])
print("\nOK: declared card wins non-market fields; live odds row still supplies price.")
