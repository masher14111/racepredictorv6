"""Compute v4 head-to-head-vs-market metrics on the 2026-05-22..06-12 window,
mirroring racing_ingestion's phase4 methodology (within-race normalization,
proportional de-vig, EV>0 flat bets, 2% commission, CLV vs closing line)."""
import numpy as np, pandas as pd

START, END = "2026-05-22", "2026-06-12"
COMM = 0.02

scored = pd.read_parquet("data/backtests/20260617_112807/scored.parquet")
scored["race_date"] = pd.to_datetime(scored["race_date"]).dt.tz_localize(None)
feat = pd.read_parquet("data/features.parquet",
                       columns=["race_date", "horse_id", "race_uid"])
feat["race_date"] = pd.to_datetime(feat["race_date"]).dt.tz_localize(None)
feat["d"] = feat["race_date"].dt.normalize()
scored["d"] = scored["race_date"].dt.normalize()

df = scored.merge(feat[["d", "horse_id", "race_uid"]].drop_duplicates(),
                  on=["d", "horse_id"], how="left")
m = (df["d"] >= START) & (df["d"] <= END)
df = df[m].copy()
print(f"rows={len(df)} race_uid nulls={df['race_uid'].isna().sum()} "
      f"races={df['race_uid'].nunique()} winners={int(df['won'].sum())}")
df = df.dropna(subset=["race_uid"])

# keep races with exactly one winner and >=2 runners
g = df.groupby("race_uid")
ok = g["won"].transform("sum").eq(1) & g["horse_id"].transform("size").ge(2)
df = df[ok].copy()

def norm(col):
    s = df.groupby("race_uid")[col].transform("sum")
    return df[col] / s

df["model_n"] = norm("prob")          # within-race model prob
df["mkt_n"] = norm("implied")         # proportional de-vig of market

races = df["race_uid"].nunique()
def race_logloss(pcol):
    w = df[df["won"] == 1]
    p = w[pcol].clip(1e-15, 1 - 1e-15)
    return float((-np.log(p)).mean())

def brier(pcol):
    return float(((df[pcol] - df["won"]) ** 2).mean())

def ece(pcol, bins=10):
    q = pd.cut(df[pcol], np.linspace(0, 1, bins + 1), include_lowest=True)
    t = df.groupby(q, observed=True).apply(
        lambda x: pd.Series({"n": len(x), "p": x[pcol].mean(), "a": x["won"].mean()}))
    return float((t["n"] / t["n"].sum() * (t["p"] - t["a"]).abs()).sum())

print(f"\n=== v4 head-to-head ({races} races) ===")
print(f"log-loss : model {race_logloss('model_n'):.4f} | market {race_logloss('mkt_n'):.4f}"
      f"  -> {'MODEL' if race_logloss('model_n')<race_logloss('mkt_n') else 'MARKET'}")
print(f"Brier    : model {brier('model_n'):.4f} | market {brier('mkt_n'):.4f}")
print(f"ECE      : model {ece('model_n'):.4f} | market {ece('mkt_n'):.4f}")

# EV>0 flat-bet betting at bet_price (ppwap), commission on winnings, CLV vs close
df["ev2"] = df["model_n"] * df["bet_price"] - 1
b = df[df["ev2"] > 0].copy()
stake = 10.0
gross = np.where(b["won"] == 1, (b["bet_price"] - 1) * stake, -stake)
comm = np.where(b["won"] == 1, (b["bet_price"] - 1) * stake * COMM, 0.0)
b["pnl"] = gross - comm
roi = b["pnl"].sum() / (len(b) * stake)
clv = np.log(b["bet_price"] / b["close_price"])
print(f"\n=== v4 EV>0 betting (settle at bet_price, 2% comm) ===")
print(f"bets={len(b)} strike={b['won'].mean():.3f} ROI={roi*100:+.2f}% "
      f"P&L={b['pnl'].sum():+.2f} meanCLV={clv.mean():+.4f} CLVbeat={(clv>0).mean():.3f}")

# per odds band
b["band"] = pd.cut(b["bet_price"], [0, 4, 8, 15, 30, 1e9],
                   labels=["<4", "4-8", "8-15", "15-30", "30+"])
for nm, x in b.groupby("band", observed=True):
    gr = np.where(x["won"] == 1, (x["bet_price"] - 1) * stake, -stake)
    cm = np.where(x["won"] == 1, (x["bet_price"] - 1) * stake * COMM, 0.0)
    pnl = (gr - cm).sum()
    print(f"  {nm:>6}: bets={len(x):4d} strike={x['won'].mean():.3f} "
          f"ROI={pnl/(len(x)*stake)*100:+.2f}% P&L={pnl:+.1f}")
