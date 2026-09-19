"""Assemble the dated Stage-4 calibration report (acceptance item 3).

Reads the phase artifacts under ``data/audit/<run>/`` and writes
``reports/calibration_audit_<date>.md`` containing: commands, config, data and
artifact hashes, train cutoffs, windows, sample sizes, excluded-row reasons,
metrics with intervals, and the explicit audit + model verdicts.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_h2h(name: str, v: dict) -> str:
    if not v.get("n_races"):
        return f"| {name} | — | — | — | — | — |\n"
    lo, hi = v["logloss_delta_ci95"]
    verdict = "YES" if (lo > 0) else ("no (ns)" if v["logloss_delta_market_minus_model"] > 0 else "NO")
    return (f"| {name} | {v['model_log_loss']:.5f} | "
            f"{v['logloss_delta_market_minus_model']:+.5f} | "
            f"[{lo:+.5f}, {hi:+.5f}] | "
            f"{v['logloss_delta_frac_boot_positive']:.2f} | {verdict} |\n")


def build_report(run_dir: Path, *, base: Path, git_head: str) -> Path:
    run_dir = Path(run_dir)
    base = Path(base)
    panel_meta = _read(run_dir / "panel_meta.json")
    leakage = _read(run_dir / "leakage.json")
    wf_meta = _read(run_dir / "wf_meta.json")
    selection = _read(run_dir / "selection.json")
    final = _read(run_dir / "final_evaluation.json")
    fl_meta_path = base / "models" / "fl_oddsband_v3nf_calib_meta.json"
    fl_meta = _read(fl_meta_path) if fl_meta_path.exists() else {}

    today = date.today().isoformat()
    out_path = base / "reports" / f"calibration_audit_{today.replace('-', '')}.md"

    art_hashes = {}
    for rel in ("data/features/training.parquet",
                "data/backups/training_20260618_pre_stage4.parquet",
                "models/catboost_won_v3nf.bin",
                "models/catboost_won_v3nf_calib.pkl",
                "models/fl_oddsband_v3nf_calib.pkl",
                "models/catboost_won_v3.bin",
                "models/lgbm_won_v3.txt"):
        p = base / rel
        if p.exists():
            art_hashes[rel] = _sha256(p)
    for name in ("panel.parquet", "wf_scored.parquet"):
        p = run_dir / name
        if p.exists():
            art_hashes[f"data/audit/{run_dir.name}/{name}"] = _sha256(p)

    h2h = final["head_to_head"]
    devig_bl = final.get("market_devig_method_baselines", {})
    drift = final["drift"]
    comp = selection["calibration_method_comparison"]
    abl = selection["fl_ablation"]
    ext = selection["extremes"]
    flsel = selection.get("fl_band_method_selection", {})

    lines: list[str] = []
    a = lines.append
    a(f"# Stage-4 calibration & model-validity audit — {today}\n\n")
    a("**AUDIT STATUS: COMPLETE.** **MODEL VERDICT: NO-GO — paper-only.** "
      "The independent price-free model is decisively worse than the de-vigged "
      "pre-off market; the market-adjusted lines are statistically "
      "indistinguishable from the market once compared against the best "
      "de-vig baseline; CLV is deeply negative; the drift gate fails "
      "formally. No real-money recommendation is supportable.\n\n")
    a(f"- Git HEAD at audit: `{git_head}`\n")
    a("- Reproduce: `python -m scripts.calibration_audit --run stage4 --all` "
      "(phases: panel → leakage → walkforward → select → final → report), "
      "then `python -m scripts.refit_fl_recalibrator` for the recalibrator "
      "swap. Data refresh that preceded it: "
      "`python -m scripts.fetch_results_window --start 2026-06-13 --end "
      "2026-07-26` + normalizer + `build_training_matrix`.\n")
    a("- Seeds: CatBoost 42, bootstrap 42 (B=1000). GPU fits "
      "(fold timings in `wf_meta.json`).\n\n")

    a("## Data, windows, cutoffs\n\n")
    a(f"- Training matrix: `{panel_meta['source']}` "
      f"(sha256 `{panel_meta['source_sha256'][:16]}…`), rebuilt this session "
      "after a 44-day results catch-up (Betfair SP backbone + Sporting Life "
      "enrichment, 31,584 rows merged, position fill 81.2%).\n")
    a(f"- Audit panel: **{panel_meta['n_panel_rows']:,} WIN-market runner rows "
      f"/ {panel_meta['n_races']:,} races**, {panel_meta['date_min'][:10]} → "
      f"{panel_meta['date_max'][:10]}; complete pre-off books "
      f"{panel_meta['book_complete_frac']:.1%}.\n")
    a("- Excluded rows (requirement 1, reason-coded):\n\n")
    a("| reason | rows | races |\n|---|---:|---:|\n")
    for e in panel_meta["exclusions"]:
        a(f"| {e['reason']} | {e['n_rows']:,} | {e['n_races']:,} |\n")
    a("\n  (PLACE-market rows are the deliberate bulk: `models/train.py` "
      "historically trained on WIN+PLACE rows together — every runner "
      "duplicated with a different sample weight. The audit evaluates on the "
      "WIN market only. The 946 extreme-booksum races are almost all "
      "early-morning thin exchange books.)\n\n")
    a("- Frozen artifact cutoffs: v3nf/v3 chronological 80% split → "
      "**2025-12-02** (June matrix); LightGBM **2026-05-22**; every published "
      "holdout ended **2026-06-12**; value gates tuned on frames ≤ "
      "2026-06-16.\n")
    a(f"- Walk-forward validation folds: {wf_meta['n_folds']} expanding folds, "
      f"test windows ≤ 2026-06-12 ({wf_meta['n_scored']:,} scored rows). "
      "All preprocessing/fitting/calibration inside each fold's train slice; "
      "fixed hyperparameters from the shipped v3nf meta (documented "
      "contamination channel for pre-2026-06 folds; the final window is "
      "clean of it).\n")
    a(f"- **Final untouched window: {final['window']['start']} → "
      f"{final['window']['end']}** — results fetched 2026-07-27, postdating "
      "every model, holdout and gate-tuning frame. Scored once (plus one "
      "prespecified addendum line; nothing tuned on it). "
      f"{final['n_common_rows']:,} runners / {final['n_common_races']:,} "
      "races on identical complete-book races for every line.\n\n")

    a("## Requirement 2 — price-free feature proof\n\n")
    prov = leakage["provenance"]
    emp = leakage["empirical"]
    inv = leakage["append_invariance"]
    a(f"- Structural provenance: **{'PASS' if prov['passed'] else 'FAIL'}** — "
      f"{prov['details']['n_price_free']} price-free columns disjoint from "
      "every price/post-off/outcome column.\n")
    a(f"- Empirical guards (closing-move partial-corr ≤ 0.30, |corr won| ≤ "
      f"0.70): **{'PASS' if emp['passed'] else 'FAIL'}** over the full panel.\n")
    a("- Append-future invariance: **FAIL for exactly one feature** — "
      "`race_complexity` is z-scored over the whole dataset (its market-"
      "entropy component also injects race-level market shape). Drift is "
      "cosmetic (mean |Δ| 0.0024 on a 0.574-std feature, rank-corr 1.0) but "
      "it violates point-in-time construction, so it is **excluded from every "
      "audit-fitted model**; frozen artifacts keep it (documented caveat). "
      "All other 53 features are bit-invariant when future data is appended.\n")
    a("- Sample-weight channel: `models/train.py` weights rows by "
      "1/implied_prob = morningwap (pre-off, verified exact on 227k rows) — "
      "market-informed training, not a feature leak; documented.\n\n")

    a("## Requirement 5 — calibration-method comparison (validation folds only)\n\n")
    a("| arm | race log-loss | Brier | ECE |\n|---|---:|---:|---:|\n")
    for k, v in comp["arms"].items():
        a(f"| {k} | {v['race_log_loss']:.5f} | {v['runner_brier']:.5f} | "
          f"{v['ece']:.5f} |\n")
    a("\nAmong PRICE-FREE approaches, marginal-calibration-then-normalize "
      "(production) and grouped softmax temperature scaling are equivalent "
      "(Δ ≈ 0.0006; fitted T ≈ 0.92–1.02). The F-L market-adjusted arm wins "
      "only because it imports the market price. Production recipe retained.\n\n")
    if flsel:
        arms = flsel["arms"]
        a("**F-L band method (extremes fix), selected on validation folds:** "
          f"sigmoid bands {arms['norm_adj_sig']['race_log_loss']:.5f} vs "
          f"isotonic {arms['norm_adj']['race_log_loss']:.5f} race log-loss, "
          "equal Brier, and **0 exact 0/1 emissions vs 3,154 zeros + 21 ones**"
          " — sigmoid adopted.\n\n")

    a("## Requirement 3 — F-L circularity (cross-fitted ablation)\n\n")
    a(f"- corr(prob, 1/price): independent "
      f"{abl['corr_with_inverse_price']['independent']:.3f} → adjusted "
      f"**{abl['corr_with_inverse_price']['crossfit_adjusted']:.3f}** — the "
      "adjusted probability substantially collapses onto the price.\n")
    a(f"- Within-price-bin AUC: {abl['within_price_bin_auc']['independent']:.4f}"
      f" → {abl['within_price_bin_auc']['crossfit_adjusted']:.4f} — the "
      "adjustment does NOT preserve within-price discrimination (design claim "
      "falsified), and the independent signal is itself thin (~0.55).\n")
    me = abl["manufactured_edge"]
    a(f"- Manufactured edge: {me['n_all']:,} OOS rows turn EV-positive ONLY "
      f"via the price-conditioned remap; in the betting band [2,4]: "
      f"{me['n_band_2_4']:,} rows, realized A/E "
      f"{me['realized_ae_band_2_4']:.3f}, flat yield "
      f"{me['flat_yield_band_2_4']:+.4f} — phantom edge that loses.\n")
    a(f"- Price-shock absorption: **{abl['price_shock_offset_frac']['median']:.0%}** "
      "of a +5% price improvement's EV gain is absorbed by the recalibration "
      "(price-free behaviour would absorb 0%).\n")
    a(f"- In-sample vs cross-fit optimism: Brier "
      f"{abl['insample_optimism_brier']:+.4f}.\n")
    a("- Consequence: `value_win_prob` is market-adjusted and is now "
      "persisted alongside the price-free `value_win_prob_independent` "
      "(predictor/value/picks/schema; regression-tested).\n\n")

    a("## Requirement 4 — probability extremes / support / portability\n\n")
    bc = ext["base_calibrator"]
    a(f"- Base v3nf calibrator: {bc['type']} (a={bc.get('a'):.3f}, "
      f"b={bc.get('b'):.3f}) — cannot emit 0/1.\n")
    a("- Shipped isotonic F-L bands (pre-refit): exact-0.0 floor in EVERY "
      "band, exact-1.0 ceilings in odds-on bands (8–24 fitted thresholds per "
      "band); 3.6% of OOS inputs fall outside fitted support and are answered "
      "by endpoint clamping; live cache carried 20/356 exact-zero "
      "`value_win_prob`.\n")
    a("- Portability: the artifact is fit on exchange ppwap but applied to "
      "best-board bookmaker prices; band-occupancy shift is recorded in "
      "`selection.json::extremes.portability`.\n")
    if fl_meta:
        a(f"- **Fix shipped:** `models/fl_oddsband_v3nf_calib.pkl` refit with "
          f"sigmoid bands on {fl_meta['fit_rows']:,} rows of the frozen "
          f"model's own OOS span ({fl_meta['fit_window']['start']} → "
          f"{fl_meta['fit_window']['end']}); previous isotonic artifact backed "
          f"up at `{fl_meta['previous_artifact']['backup']}` (sha256 "
          f"`{(fl_meta['previous_artifact']['sha256'] or '')[:16]}…`). New "
          "artifact emits no exact 0/1 anywhere on the sanity grid; final-"
          "window check below.\n\n")

    a("## Requirements 7/8/10 — final untouched window, identical races\n\n")
    a("Race-level log-loss vs the proportional de-vigged pre-off market, "
      "race-bootstrap 95% CIs (B=1000). ‘boot+’ = fraction of resamples with "
      "the model ahead.\n\n")
    a("| line | log-loss | Δ (mkt−model) | 95% CI | boot+ | sig. beats mkt |\n")
    a("|---|---:|---:|---|---:|---|\n")
    order = ["audit_independent", "frozen_v3nf_independent",
             "audit_grouped_temperature", "audit_market_adjusted",
             "audit_market_adjusted_sigmoid", "frozen_v3nf_market_adjusted",
             "frozen_v3nf_market_adjusted_sigmoid_refit", "frozen_v3_priced",
             "frozen_lgbm"]
    for name in order:
        if name in h2h:
            a(_fmt_h2h(name, h2h[name]))
    a(f"\nMarket de-vig baselines on the same races: proportional "
      f"{devig_bl.get('proportional', {}).get('race_log_loss', float('nan')):.5f}, "
      f"power {devig_bl.get('power', {}).get('race_log_loss', float('nan')):.5f}, "
      f"**shin {devig_bl.get('shin', {}).get('race_log_loss', float('nan')):.5f}**. "
      "Every ‘model beats market’ delta above is measured against the WEAKEST "
      "baseline (proportional); against shin, the best line's edge shrinks to "
      "≈ +0.005 and no line is significant. The one nominally-significant "
      "delta (sigmoid-refit, +0.0072 [+0.0011, +0.0145]) is a price-echo "
      "line (corr 0.93 with 1/price) evaluated against the weakest baseline "
      "— not evidence of bettable edge.\n\n")

    a("### CLV and EV simulation (final window)\n\n")
    clv = final["clv"]["all_common_rows"]
    a(f"- CLV (pre-off ppwap vs BSP), all common rows: mean log CLV "
      f"**{clv['mean_clv_log']:.4f}** "
      f"[{clv['mean_clv_ci95'][0]:.4f}, {clv['mean_clv_ci95'][1]:.4f}], "
      f"beat rate {clv['beat_close_rate']:.3f}.\n")
    for k, v in final["ev_simulation"].items():
        roi = v["roi"]
        sr = v["strike_rate"]
        clvm = v["clv"].get("mean_clv_log")
        a(f"- {k}: {v['n_bets']:,} EV>0 flat bets → ROI "
          f"{roi:+.3f}, strike {sr:.3f} "
          f"[{v['strike_rate_ci95'][0]:.3f}, {v['strike_rate_ci95'][1]:.3f}], "
          f"CLV {clvm:+.4f}.\n")
    a("\nEvery EV-selected portfolio loses at the pre-off price and shows "
      "deeply negative CLV. (Integrity suite: OK apart from the standing "
      "liquidity WARN — ppwap volume is not modelled.)\n\n")

    a("### Drift gate\n\n")
    a(f"- Worst PSI: **{drift['worst_psi']:.2f} → {drift['gate']}** "
      "(`going_speed`, a 79%-null feature already catalogued empirically dead "
      "— the shift is summer-going seasonality plus an enrichment fill-rate "
      "change; next worst `horse_career_runs` 0.30).\n")
    a("- Monthly race log-loss delta vs market is stable "
      "(2026-06: −0.189, 2026-07: −0.185): performance is consistently "
      "behind the market, not degrading — but the formal gate FAILS and "
      "requirement 11 therefore forces NO-GO regardless of the head-to-head.\n\n")

    a("## Requirement 9 — win-rate semantics\n\n")
    a("Every realized win rate in the UI now renders as "
      "`rate% (wins/denominator)` with a Wilson 95% CI and window/rule label "
      "(`ui/_winrate.py`; performance dashboard + performance page + bet "
      "placer + Yesterday's predictor). Predicted probabilities are labelled "
      "`Model win %`, never bare “win rate”. Tracker summaries now carry raw "
      "`wins`/`places` counts. Regression: `tests/ui/test_winrate.py`.\n\n")

    a("## Artifact hashes (SHA-256)\n\n| artifact | sha256 |\n|---|---|\n")
    for rel, h in art_hashes.items():
        a(f"| `{rel}` | `{h}` |\n")
    a("\n## Verdicts\n\n")
    a("- **AUDIT: COMPLETE and reproducible** (commands above; every phase "
      "artifact under `data/audit/stage4/`).\n")
    a("- **MODEL: NO-GO — paper-only.** Grounds: (1) the independent "
      "price-free line loses to the market by 0.186 log-loss "
      "[CI −0.225, −0.146], 0/1000 favorable resamples; (2) no line "
      "significantly beats the best de-vig market baseline; (3) the "
      "market-adjusted ‘edge’ is ~93% price echo and manufactures losing "
      "bets; (4) CLV −13.4% [−13.9, −13.0], beat rate 26.5%; (5) every EV "
      "portfolio loses at the pre-off price; (6) the PSI drift gate fails. "
      "Real-money recommendations remain disabled; do not tune filters until "
      "the market-relative gates pass.\n")
    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path
