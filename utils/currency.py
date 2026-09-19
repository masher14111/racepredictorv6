from utils.config_loader import get_config

GBP_EUR_RATE: float = float(get_config().get("gbp_eur_rate", 1.18))

# Northern Irish courses. Politically UK, but both run under Horse Racing
# Ireland and Irish books price them in EUR — so they are deliberately NOT in
# the GBP set below and need no conversion. Listed explicitly so the decision is
# visible rather than an accident of omission.
_NI_COURSES = {"down royal", "downpatrick"}

# Known UK (GBP) racecourses. Anything not listed defaults to EUR (Irish/other).
_UK_COURSES = {
    "ascot", "aintree", "ayr", "bangor", "bath", "beverley", "brighton",
    "carlisle", "cartmel", "catterick", "chelmsford", "cheltenham", "chepstow",
    "chester", "doncaster", "epsom", "exeter", "fakenham", "ffos las",
    "fontwell", "goodwood", "hamilton", "haydock", "hereford", "hexham",
    "huntingdon", "kelso", "kempton", "leicester", "lingfield", "ludlow",
    "market rasen", "musselburgh", "newbury", "newcastle", "newmarket",
    "newton abbot", "nottingham", "perth", "plumpton", "pontefract", "redcar",
    "ripon", "salisbury", "sandown", "sedgefield", "southwell", "stratford",
    "taunton", "thirsk", "uttoxeter", "warwick", "wetherby", "wincanton",
    "windsor", "wolverhampton", "worcester", "yarmouth", "york",
}


def currency_for_venue(venue: str, country_code: str = None) -> str:
    """Return 'GBP' for UK races, else 'EUR'. country_code (ISO-2) overrides venue."""
    name = (venue or "").strip().lower()
    # A NI course is EUR even when the feed tags it GB — see _NI_COURSES.
    if name in _NI_COURSES:
        return "EUR"
    if country_code:
        return "GBP" if country_code.upper() in ("GB", "UK") else "EUR"
    return "GBP" if name in _UK_COURSES else "EUR"


def to_eur(amount: float, currency: str) -> float:
    """Convert a monetary amount to EUR. EUR/unknown pass through; GBP uses rate."""
    if currency == "GBP":
        return amount * GBP_EUR_RATE
    return amount


def from_eur(amount: float, currency: str) -> float:
    """Convert a EUR amount into the venue's currency — the inverse of to_eur.

    Used to show what a UK book would actually debit and pay for a stake that is
    budgeted in euro. Note that ODDS are never converted: they are a ratio, so a
    EUR stake at a GBP book returns the same multiple and the rate cancels. Only
    the amounts a bookmaker displays need this.
    """
    if currency == "GBP" and GBP_EUR_RATE:
        return amount / GBP_EUR_RATE
    return amount
