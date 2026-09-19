"""Validated evaluation panel for the Stage-4 audit (requirement 1).

The live scrapers pass ``utils.market_validation``'s race-level contract, but
the historical training frame comes from a different source entirely — the
Betfair SP daily files — which the live contract never sees. This module is the
equivalent contract for that frame, with the same fail-closed, reason-coded
philosophy:

* **WIN market only.** ``models.train`` historically trained over WIN + PLACE
  rows together (every runner duplicated with the place-market price); the audit
  panel keeps exactly one WIN row per runner.
* **Deduplicated** on (race_uid, horse_id).
* **Labelled** — a row must carry a 0/1 ``won`` outcome.
* **Priced** — a row must carry a usable pre-off execution price
  (ppwap -> morningwap, decimal > 1).
* **Plausible field** — 2..40 runners (mirrors market_validation's bounds).
* **Plausible book** — the race's pre-off implied-probability sum must fall in
  [0.80, 1.80] (the live contract's extreme_booksum bounds). Betfair SP books
  hover near 1.0 by construction, so violations mark data corruption, not vig.
* **Complete book flag** — every runner priced. Only complete books can be
  de-vigged, so ``head_to_head`` drops partial races; the flag makes that
  exclusion measurable instead of silent.

Nothing in the Betfair SP feed can smuggle a specials market into the WIN file
(market identity comes from the *file requested*, not from parsing a mixed
page), so the live contract's specials/duplicate-selection threat model does not
apply here; the checks above are the meaningful subset. Exclusions are returned
as a reason-coded frame so the calibration report can state exactly what was
dropped and why (requirement: "excluded-row reasons").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

# Mirrors utils.market_validation's race-level bounds.
FIELD_MIN, FIELD_MAX = 2, 40
BOOKSUM_MIN, BOOKSUM_MAX = 0.80, 1.80


@dataclass
class PanelResult:
    """The validated panel plus a reason-coded account of every exclusion."""

    panel: pd.DataFrame
    exclusions: pd.DataFrame               # columns: reason, n_rows, n_races
    n_input_rows: int = 0
    notes: list = field(default_factory=list)


def _exclusion(reason: str, rows: int, races: int) -> dict:
    return {"reason": reason, "n_rows": int(rows), "n_races": int(races)}


def build_audit_panel(df: pd.DataFrame, *, price_col: str = "ppwap",
                      fallback_price_col: str = "morningwap",
                      close_col: str = "odds_finish",
                      max_price: float = 1000.0) -> PanelResult:
    """Validate a training-matrix-shaped frame into the audit evaluation panel.

    Input must carry: race_uid, race_date, horse_id, won, market_type, and the
    price columns. Output panel columns: everything in ``df`` plus ``bet_price``
    (the pre-off execution price), ``close_price`` (BSP), ``book_complete``
    (bool), ``booksum`` (race implied sum), ``field_size_panel`` (validated
    per-race runner count).
    """
    excl: list[dict] = []
    n_input = len(df)
    out = df.copy()

    # 1. WIN market only (drop the PLACE duplication models.train trains over).
    if "market_type" in out.columns:
        mt = out["market_type"].astype(str).str.upper()
        drop = mt != "WIN"
        if drop.any():
            races = out.loc[drop, "race_uid"].nunique() if "race_uid" in out else 0
            excl.append(_exclusion("non_win_market_row", drop.sum(), races))
            out = out[~drop]

    # 2. Labelled outcome.
    won = pd.to_numeric(out.get("won"), errors="coerce")
    drop = ~won.isin([0, 1])
    if drop.any():
        excl.append(_exclusion("missing_outcome", drop.sum(),
                               out.loc[drop, "race_uid"].nunique()))
        out = out[~drop]

    # 3. Deduplicate (race_uid, horse_id).
    dup = out.duplicated(subset=["race_uid", "horse_id"], keep="first")
    if dup.any():
        excl.append(_exclusion("duplicate_runner_row", dup.sum(),
                               out.loc[dup, "race_uid"].nunique()))
        out = out[~dup]

    # 4. Usable pre-off price.
    price = pd.to_numeric(out.get(price_col), errors="coerce")
    fb = pd.to_numeric(out.get(fallback_price_col), errors="coerce")
    price = price.where((price > 1.0) & (price <= max_price))
    fb = fb.where((fb > 1.0) & (fb <= max_price))
    bet_price = price.fillna(fb)
    drop = bet_price.isna()
    if drop.any():
        excl.append(_exclusion("no_valid_preoff_price", drop.sum(),
                               out.loc[drop, "race_uid"].nunique()))
        out = out[~drop]
        bet_price = bet_price[~drop]
    out = out.assign(bet_price=bet_price.to_numpy(dtype=float))

    close = pd.to_numeric(out.get(close_col), errors="coerce")
    out["close_price"] = close.where((close > 1.0) & (close <= max_price)).to_numpy()

    # 5. Race-level checks over the retained rows.
    grp = out.groupby("race_uid", sort=False)
    fs = grp["horse_id"].transform("size")
    out["field_size_panel"] = fs.to_numpy()
    imp = 1.0 / out["bet_price"]
    out["booksum"] = imp.groupby(out["race_uid"]).transform("sum").to_numpy()

    bad_field = (fs < FIELD_MIN) | (fs > FIELD_MAX)
    if bad_field.any():
        excl.append(_exclusion(f"implausible_field_size(<{FIELD_MIN}|>{FIELD_MAX})",
                               bad_field.sum(), out.loc[bad_field, "race_uid"].nunique()))
        out = out[~bad_field]

    bad_book = (out["booksum"] < BOOKSUM_MIN) | (out["booksum"] > BOOKSUM_MAX)
    if bad_book.any():
        excl.append(_exclusion(f"extreme_booksum(<{BOOKSUM_MIN}|>{BOOKSUM_MAX})",
                               bad_book.sum(), out.loc[bad_book, "race_uid"].nunique()))
        out = out[~bad_book]

    # 6. Complete-book flag: after validation every retained runner is priced, so
    # a race is complete iff its validated field equals its original declared
    # field. Recompute size over retained rows and compare to the max field_size
    # the feature engine recorded (which counted every declared runner).
    retained_fs = out.groupby("race_uid", sort=False)["horse_id"].transform("size")
    declared = pd.to_numeric(out.get("field_size"), errors="coerce")
    out["book_complete"] = np.where(
        declared.notna(), retained_fs.to_numpy() >= declared.to_numpy(),
        True)  # no declared size recorded -> retained field is the field

    out = out.sort_values("race_date").reset_index(drop=True)
    exclusions = pd.DataFrame(excl or [],
                              columns=["reason", "n_rows", "n_races"])
    kept_races = out["race_uid"].nunique() if len(out) else 0
    logger.info("audit.panel: kept %d rows / %d races of %d input rows; "
                "%d exclusion reasons", len(out), kept_races, n_input,
                len(exclusions))
    return PanelResult(panel=out, exclusions=exclusions, n_input_rows=n_input)
