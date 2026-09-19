"""Demo: per-prediction SHAP explanation on a sample race."""
import pandas as pd

from models.explain import get_explainer

df = pd.read_parquet("data/features/training_full.parquet")
# pick a real race from the test tail with a sane field size
df = df.sort_values("race_date")
counts = df.groupby("race_uid")["horse_id"].transform("size")
race = df[(counts.between(8, 14)) & (df["race_date"] >= "2026-06-01")].iloc[:1]
uid = race["race_uid"].iloc[0]
field = df[df["race_uid"] == uid].copy()

print(f"Race {uid}  |  {field['venue'].iloc[0]}  |  {field['race_date'].iloc[0]}  "
      f"|  {len(field)} runners\n")

ex = get_explainer(target="won", version_tag="v3nf")
expl = ex.explain_frame(field, top_k=4)

# rank by predicted prob
order = sorted(range(len(field)), key=lambda i: expl[i]["predicted_prob"], reverse=True)
for rank, i in enumerate(order[:3], 1):
    row = field.iloc[i]; e = expl[i]
    won = row.get("won")
    print(f"#{rank}  {str(row.get('horse_name','?')):<24} "
          f"win_prob(raw)={e['predicted_prob']:.3f}  (actually won={won})")
    print(f"      + drivers:")
    for d in e["top_positive"]:
        v = "n/a" if d["value"] is None else f"{d['value']:.3g}"
        print(f"        {d['label']:<32} {v:>8}   margin {d['contribution']:+.3f}")
    print(f"      - drivers:")
    for d in e["top_negative"]:
        v = "n/a" if d["value"] is None else f"{d['value']:.3g}"
        print(f"        {d['label']:<32} {v:>8}   margin {d['contribution']:+.3f}")
    print()
