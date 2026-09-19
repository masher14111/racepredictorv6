"""Step 12: select the power-blend exponents on the blend_dev period ONLY.

Reads data/audit/12/walkforward_oos.parquet, restricts to the frozen blend_dev
window, and exhaustively scores the frozen (alpha, beta) grid by race-level log
loss for two model lines:

  combined_indep  -- price-free conditional logit x reference market line
  combined_market -- market-assisted conditional logit x reference market line

The calib and eval periods are never read here. Writes the selected blends to
data/audit/12/blend/ and the full selection surface (including the
independent-only and market-only boundary controls) to
reports/improvement/12/blend_selection.json.

Run: .venv/Scripts/python.exe reports/improvement/12/03_fit_blend.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd

from models.blend import PowerBlend, race_level_log_loss, select_power_blend

MANIFEST = "reports/improvement/12/blend_manifest.json"
PANEL = "data/audit/12/walkforward_oos.parquet"
BLEND_DIR = "data/audit/12/blend"
OUT = "reports/improvement/12/blend_selection.json"

LINES = {"combined_indep": "condlogit_indep_prob",
         "combined_market": "condlogit_market_prob"}


def main() -> None:
    manifest = json.load(open(MANIFEST, "r", encoding="utf-8"))
    alpha_grid = tuple(manifest["blend_grid"]["alpha"])
    beta_grid = tuple(manifest["blend_grid"]["beta"])

    panel = pd.read_parquet(PANEL)
    dev = panel[panel["period"] == "blend_dev"].reset_index(drop=True)
    # Complete reference books only — identical rule to models.head_to_head.
    complete = dev["market_ref_prob"].notna().groupby(dev["race_uid"]).transform("all")
    dev = dev[complete.to_numpy()].reset_index(drop=True)

    y = dev["won"].to_numpy(float)
    rid = dev["race_uid"].to_numpy()
    ref = dev["market_ref_prob"].to_numpy(float)

    out = {
        "manifest": MANIFEST,
        "panel": PANEL,
        "period": "blend_dev",
        "window": manifest["windows"]["blend_dev"],
        "n_rows": int(len(dev)),
        "n_races": int(dev["race_uid"].nunique()),
        "market_only_race_log_loss": race_level_log_loss(y, ref, rid),
        "lines": {},
    }
    print(f"blend_dev: {out['n_rows']} rows / {out['n_races']} races; "
          f"market-only race log loss {out['market_only_race_log_loss']:.5f}")

    os.makedirs(BLEND_DIR, exist_ok=True)
    for name, col in LINES.items():
        p_model = dev[col].to_numpy(float)
        blend, table = select_power_blend(
            p_model, ref, rid, y, alpha_grid=alpha_grid, beta_grid=beta_grid)
        blend.metadata.update({
            "line": name,
            "model_prob_col": col,
            "reference_prob_col": "market_ref_prob",
            "selected_on": "blend_dev (walk-forward OOS, 2026-02-01..2026-06-12)",
            "n_races": out["n_races"],
        })
        path = os.path.join(BLEND_DIR, f"{name}.json")
        blend.save(path)

        def at(a, b):
            m = (table.alpha == a) & (table.beta == b)
            return float(table.loc[m, "race_log_loss"].iloc[0]) if m.any() else float("nan")

        out["lines"][name] = {
            "artifact": path,
            "alpha": blend.alpha,
            "beta": blend.beta,
            "blend_dev_race_log_loss": blend.metadata["dev_race_log_loss"],
            "boundary_independent_only_alpha1_beta0": at(1.0, 0.0),
            "boundary_market_only_alpha0_beta1": at(0.0, 1.0),
            "grid": table.to_dict(orient="records"),
        }
        print(f"{name}: alpha={blend.alpha} beta={blend.beta} "
              f"LL={blend.metadata['dev_race_log_loss']:.5f} | "
              f"indep-only {at(1.0, 0.0):.5f} | market-only {at(0.0, 1.0):.5f}")

        # Diagnostic arm: the unconstrained search can (and does) land on the
        # alpha=0 edge, i.e. "use the market and discard the model". That is a
        # real answer, but it collapses the combined candidate onto the market
        # line and so tells the eval panel nothing about how much the model
        # could contribute at best. This second arm is the same grid restricted
        # to alpha >= 0.1, still selected on blend_dev alone.
        forced = table[(table.alpha >= 0.1) & table["race_log_loss"].notna()]
        fbest = forced.loc[forced["race_log_loss"].idxmin()]
        fblend = PowerBlend(fbest["alpha"], fbest["beta"], metadata={
            "line": f"{name}_forced", "model_prob_col": col,
            "reference_prob_col": "market_ref_prob",
            "selected_by": "race_level_log_loss, constrained to alpha >= 0.1",
            "dev_race_log_loss": float(fbest["race_log_loss"]),
            "selected_on": "blend_dev (walk-forward OOS, 2026-02-01..2026-06-12)",
            "n_races": out["n_races"],
        })
        fpath = os.path.join(BLEND_DIR, f"{name}_forced.json")
        fblend.save(fpath)
        out["lines"][f"{name}_forced"] = {
            "artifact": fpath,
            "alpha": fblend.alpha,
            "beta": fblend.beta,
            "blend_dev_race_log_loss": fblend.metadata["dev_race_log_loss"],
            "constraint": "alpha >= 0.1",
            "note": "diagnostic arm; shares combined_* grid, no separate search",
        }
        print(f"{name}_forced (alpha>=0.1): alpha={fblend.alpha} beta={fblend.beta} "
              f"LL={fblend.metadata['dev_race_log_loss']:.5f}")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {OUT}")
    print("BLEND_SELECTION_12_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
