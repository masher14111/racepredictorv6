"""Shared text normalizers for cross-source joining (venue/horse/minute)."""
import re
from datetime import datetime

import pandas as pd

_COUNTRY_RE = re.compile(r"\((?:ire|gb|fr|usa|ger|aus|nz)\)", re.IGNORECASE)


def norm_horse(name: str) -> str:
    text = _COUNTRY_RE.sub("", str(name or "")).lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_venue(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def minute_key(value) -> str:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%dT%H:%M")
    text = str(value or "")
    m = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})", text)
    return m.group(1) if m else text
