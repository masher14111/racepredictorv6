"""At The Races results source (illustrative selectors — confirm via DevTools)."""
import re
from datetime import date
from typing import Optional

from lxml import html as lxml_html

from scraper.betsp.results.base import RawResult, ResultRow, ResultsSource
from utils.logger import get_logger

logger = get_logger(__name__)

_RESULTS_INDEX = "https://www.attheraces.com/results/{ymd}"
_ID_RE = re.compile(r"/(?:jockey|trainer)/([a-z0-9-]+)", re.IGNORECASE)


def _id(href: Optional[str]) -> Optional[str]:
    if not href:
        return None
    m = _ID_RE.search(href)
    return m.group(1) if m else None


def _text(nodes) -> str:
    return (nodes[0].text_content() or "").strip() if nodes else ""


class AtTheRaces(ResultsSource):
    name = "at_the_races"

    def fetch_raw(self, day: date, get_html) -> list:
        url = _RESULTS_INDEX.format(ymd=day.strftime("%Y-%m-%d"))
        try:
            html = get_html(url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AtTheRaces fetch failed %s: %s", url, exc)
            return []
        return [RawResult(source=self.name, race_date=day, url=url, html=html)]

    def parse(self, raw: RawResult) -> list:
        tree = lxml_html.fromstring(raw.html)
        rows = []
        for sec in tree.xpath('//*[contains(@class,"result")]'):
            going = sec.get("data-going") or None
            venue = sec.get("data-course") or raw.venue or ""
            race_dt = sec.get("data-time") or raw.race_date.isoformat()
            for run in sec.xpath('.//*[contains(@class,"runner")]'):
                pos = run.get("data-pos")
                horse = _text(run.xpath('.//*[contains(@class,"horse-name")]'))
                if not horse:
                    continue
                jockey_a = run.xpath('.//a[contains(@class,"jockey-link")]')
                trainer_a = run.xpath('.//a[contains(@class,"trainer-link")]')
                rows.append(ResultRow(
                    race_date=race_dt, venue=venue, horse_name=horse, source=self.name,
                    position=int(pos) if pos and pos.isdigit() else None, going=going,
                    jockey_id=_id(jockey_a[0].get("href")) if jockey_a else None,
                    jockey_name=_text(jockey_a) or None,
                    trainer_id=_id(trainer_a[0].get("href")) if trainer_a else None,
                    trainer_name=_text(trainer_a) or None,
                ))
        return rows
