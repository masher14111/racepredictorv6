"""Racing Post results source (illustrative selectors — confirm via DevTools)."""
import re
from datetime import date
from typing import Optional

from lxml import html as lxml_html

from scraper.betsp.results.base import RawResult, ResultRow, ResultsSource
from utils.logger import get_logger

logger = get_logger(__name__)

_RESULTS_INDEX = "https://www.racingpost.com/results/{ymd}"
_ID_RE = re.compile(r"/profile/(?:jockey|trainer|horse)/([a-z0-9-]+)", re.IGNORECASE)


def _id(href: Optional[str]) -> Optional[str]:
    if not href:
        return None
    m = _ID_RE.search(href)
    return m.group(1) if m else None


class RacingPost(ResultsSource):
    name = "racing_post"

    def fetch_raw(self, day: date, get_html) -> list:
        url = _RESULTS_INDEX.format(ymd=day.strftime("%Y-%m-%d"))
        try:
            html = get_html(url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RacingPost fetch failed %s: %s", url, exc)
            return []
        return [RawResult(source=self.name, race_date=day, url=url, html=html)]

    def parse(self, raw: RawResult) -> list:
        tree = lxml_html.fromstring(raw.html)
        rows = []
        for sec in tree.xpath('//*[contains(@class,"rp-result")]'):
            going = sec.get("data-going") or None
            venue = sec.get("data-course") or raw.venue or ""
            race_dt = sec.get("data-time") or raw.race_date.isoformat()
            for row in sec.xpath('.//*[contains(@class,"rp-horse-row")]'):
                pos = row.get("data-position")
                horse_a = row.xpath('.//a[contains(@class,"rp-horse")]')
                jockey_a = row.xpath('.//a[contains(@class,"rp-jockey")]')
                trainer_a = row.xpath('.//a[contains(@class,"rp-trainer")]')
                if not horse_a:
                    continue
                rows.append(ResultRow(
                    race_date=race_dt, venue=venue,
                    horse_name=(horse_a[0].text_content() or "").strip(),
                    source=self.name,
                    position=int(pos) if pos and pos.isdigit() else None,
                    going=going,
                    jockey_id=_id(jockey_a[0].get("href")) if jockey_a else None,
                    jockey_name=((jockey_a[0].text_content() or "").strip() or None)
                    if jockey_a else None,
                    trainer_id=_id(trainer_a[0].get("href")) if trainer_a else None,
                    trainer_name=((trainer_a[0].text_content() or "").strip() or None)
                    if trainer_a else None,
                ))
        return rows
