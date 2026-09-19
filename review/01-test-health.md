# Race Predictor v3 — Test Health Report

_Generated: 2026-06-13_

---

## Summary

| Metric      |   Count |
| ----------- | ------: |
| Collected   |     607 |
| **Passed**  | **604** |
| **Skipped** |   **3** |
| Failed      |       0 |
| Error       |       0 |

Run time: 34.68 s on Python 3.14.4 / pytest 9.0.3 / Windows 11.

**Overall health: GREEN.** No failures, no errors. All 604 executable tests pass.

---

## Failures

_None._

---

## Skipped / xfail

All three skips are in `tests/utils/test_reporter.py` and are guarded by `pytest.importorskip()`:

| Test                                      | Reason                                                                          |
| ----------------------------------------- | ------------------------------------------------------------------------------- |
| `test_export_pdf_produces_file`           | `reportlab` not installed → `pytest.importorskip("reportlab")` skips at runtime |
| `test_export_pdf_with_data_produces_file` | Same — `reportlab` missing                                                      |
| `test_generate_all_produces_four_outputs` | Same — `reportlab` missing                                                      |

No `xfail` markers found anywhere in the suite.

---

## Missing deps

`reportlab` is listed in `requirements.txt` but **not installed** in the active Python environment.
All other packages in `requirements.txt` are present (matplotlib, catboost, optuna, streamlit, etc. all import cleanly — confirmed by passing test suite which exercises them all).

**Install command:**

```powershell
pip install reportlab
```

---

## Next

Install `reportlab` (`pip install reportlab`) to un-skip the 3 PDF-export tests and confirm the full reporter pipeline, then proceed to unlocking finishing-position data (betsp/timeform results) so the training matrix gains rows.
