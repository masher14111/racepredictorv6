"""Display helpers for odds: decimal -> traditional fraction, and book labels.

The scrapers and the model work in decimal odds throughout — that is the right
internal representation and none of it changes here. This module exists purely
so the UI can show a price the way a bookmaker's board shows it (5/4, 11/8, 2/1)
instead of 2.25, 2.375, 3.00.

Two things worth knowing about fractional odds:

  * They are a **ratio, not money.** 5/4 means "four units staked returns five
    in profit". It has no currency, so it is never converted between GBP and
    EUR — only the stake and the returns are. Prefixing a price with a currency
    symbol is a category error.
  * Bookmakers do not show arbitrary reduced fractions. They quote from a fixed
    traditional ladder (1/5, 2/9, 1/4 ... 11/8, 6/4, 13/8 ...), which is why a
    board reads 6/4 rather than the arithmetically-identical 3/2, and 11/8
    rather than 1.375/1. ``to_fraction`` snaps to that ladder, so a scraped
    decimal always renders as a price a punter would recognise.
"""

from __future__ import annotations

from bisect import bisect_left
from fractions import Fraction
from typing import Iterable, Optional

# The traditional UK/IRE board ladder, shortest price first. Decimal = n/d + 1.
_LADDER: tuple[tuple[int, int], ...] = (
    (1, 10), (1, 9), (1, 8), (1, 7), (1, 6), (1, 5), (2, 9), (1, 4), (2, 7),
    (3, 10), (1, 3), (4, 11), (2, 5), (4, 9), (1, 2), (8, 15), (4, 7), (8, 13),
    (4, 6), (8, 11), (4, 5), (5, 6), (10, 11), (1, 1), (11, 10), (6, 5), (5, 4),
    (11, 8), (6, 4), (13, 8), (7, 4), (15, 8), (2, 1), (85, 40), (9, 4), (5, 2),
    (11, 4), (3, 1), (100, 30), (7, 2), (4, 1), (9, 2), (5, 1), (11, 2), (6, 1),
    (13, 2), (7, 1), (15, 2), (8, 1), (17, 2), (9, 1), (19, 2), (10, 1), (11, 1),
    (12, 1), (14, 1), (16, 1), (18, 1), (20, 1), (22, 1), (25, 1), (28, 1),
    (33, 1), (40, 1), (50, 1), (66, 1), (80, 1), (100, 1), (125, 1), (150, 1),
    (200, 1), (250, 1), (300, 1), (400, 1), (500, 1), (1000, 1),
)

# Decimal equivalents, ascending — kept module-level so lookup is a bisect.
_LADDER_DEC: tuple[float, ...] = tuple(n / d + 1.0 for n, d in _LADDER)

# Short labels for the card. Anything unmapped falls back to a tidied slug.
_BOOK_LABELS = {
    "paddy_power": "Paddy",
    "paddypower": "Paddy",
    "boylesports": "Boyle",
    "livescorebet": "LSB",
    "betfair": "Betfair",
    "betfair_sp": "BSP",
}


def to_fraction(decimal_odds: Optional[float]) -> str:
    """Render decimal odds as the traditional fraction a board would show.

    Snaps to the bookmaker ladder, so 2.25 -> "5/4", 2.50 -> "6/4" (not 3/2) and
    2.0 -> "Evens". Returns "—" for a missing or non-positive price.
    """
    if decimal_odds is None:
        return "—"
    try:
        dec = float(decimal_odds)
    except (TypeError, ValueError):
        return "—"
    if dec != dec or dec <= 1.0:  # NaN, or no profit leg
        return "—"

    if dec >= _LADDER_DEC[-1]:
        # Past the top of the board: reduce honestly rather than clamp.
        frac = Fraction(dec - 1.0).limit_denominator(4)
        return f"{frac.numerator}/{frac.denominator}"

    i = bisect_left(_LADDER_DEC, dec)
    if i == 0:
        n, d = _LADDER[0]
    else:
        lo, hi = _LADDER_DEC[i - 1], _LADDER_DEC[i]
        n, d = _LADDER[i - 1] if (dec - lo) <= (hi - dec) else _LADDER[i]

    if (n, d) == (1, 1):
        return "Evens"
    return f"{n}/{d}"


def book_label(source: Optional[str]) -> str:
    """Short display name for a bookmaker source key."""
    key = (source or "").strip().lower()
    if key in _BOOK_LABELS:
        return _BOOK_LABELS[key]
    return key.replace("_", " ").title() if key else "—"


def format_book_prices(
    odds_by_book: Optional[dict],
    order: Optional[Iterable[str]] = None,
) -> list[dict]:
    """Normalise an ``{source: decimal}`` map into sorted display rows.

    Returns dicts of ``{book, label, decimal, fraction, is_best}`` ordered by
    price (best first), so the card can render "Paddy 1/2 · Boyle 6/4" and mark
    the best available price.
    """
    if not odds_by_book:
        return []

    rows: list[dict] = []
    for book, dec in odds_by_book.items():
        try:
            dec_f = float(dec)
        except (TypeError, ValueError):
            continue
        if dec_f != dec_f or dec_f <= 1.0:
            continue
        rows.append({
            "book": book,
            "label": book_label(book),
            "decimal": dec_f,
            "fraction": to_fraction(dec_f),
            "is_best": False,
        })
    if not rows:
        return []

    if order:
        rank = {b: i for i, b in enumerate(order)}
        rows.sort(key=lambda r: (rank.get(r["book"], len(rank)), -r["decimal"]))
    else:
        rows.sort(key=lambda r: -r["decimal"])

    best = max(r["decimal"] for r in rows)
    for r in rows:
        r["is_best"] = r["decimal"] == best
    return rows
