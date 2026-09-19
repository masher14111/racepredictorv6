"""Stage-4 leak-free calibration and model-validity audit (live-repair-04).

Modules
-------
panel       — validated evaluation panel: WIN-only, deduped, reason-coded
              exclusions, complete-book flags (the audit's requirement-1 contract
              for the historical betsp-derived frame).
leakage     — requirement-2 proofs for the price-free v3nf feature set.
walkforward — expanding-window walk-forward with model fitting, base
              calibration, F-L recalibration and calibration-method arms all
              fit strictly inside each fold's train slice.
extremes    — probability-extreme / support / portability diagnostics for the
              persisted calibrator artifacts.
metrics_panel — head-to-head vs the de-vigged market with race-level bootstrap
              confidence intervals and subgroup tables.

Everything here is read-only over the project's data artifacts: the audit
writes only under ``data/audit/`` and ``reports/``.
"""
