"""Timeform URL builders and racecard-link extraction."""
import re
from datetime import date
from typing import Optional

from lxml import html as lxml_html

from utils.text_norm import norm_venue

_BASE = "https://www.timeform.com"
_RACECARDS = f"{_BASE}/horse-racing/racecards"

# A real per-race link: /horse-racing/racecards/{course}/{date}/{HHMM}/{courseId}/{raceNo}/{slug}
# meeting-summary links have fewer segments and must be skipped.
_EVENT_RE = re.compile(
    r"^/horse-racing/racecards/[^/]+/\d{4}-\d{2}-\d{2}/\d{3,4}/\d+/\d+/")


def index_url(day: Optional[date]) -> str:
    """Racecards index URL; today's cards when day is None."""
    if day is None:
        return _RACECARDS
    return f"{_RACECARDS}/{day.isoformat()}"


def _base_going(text: str) -> Optional[str]:
    """'GoingSoft (Good to Soft in Places)' -> 'soft'. Returns a hyphenated slug."""
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    t = re.sub(r"^Going", "", t)          # strip the "Going" label prefix
    t = t.split("(")[0].strip()           # drop the "(... in places)" detail
    if not t:
        return None
    return re.sub(r"\s+to\s+", "-to-", t.lower()).replace(" ", "-")


def meeting_goings(index_html: str) -> dict:
    """Map normalized venue slug -> base going string, read from the index meetings."""
    tree = lxml_html.fromstring(index_html)
    out = {}
    for meeting in tree.xpath(
            '//*[contains(concat(" ", normalize-space(@class), " "),'
            ' " w-racecard-grid-meeting ")]'):
        course = meeting.xpath('.//*[contains(@class,"w-racecard-grid-course")]')
        going = meeting.xpath('.//*[contains(@class,"w-racecard-grid-info-going")]')
        if not course:
            continue
        venue = norm_venue(course[0].text_content())
        base = _base_going(going[0].text_content()) if going else None
        if venue and base:
            out[venue] = base
    return out


def event_links(index_html: str) -> list:
    """Absolute URLs of each individual racecard on an index page (deduped, ordered)."""
    tree = lxml_html.fromstring(index_html)
    seen, out = set(), []
    for href in tree.xpath('//a[@href]/@href'):
        if _EVENT_RE.match(href) and href not in seen:
            seen.add(href)
            out.append(_BASE + href)
    return out
