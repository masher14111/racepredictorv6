# betSP Historical Data Fetcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scraper/betsp_historical.py` (+ a `scraper/betsp/` package) that fetches Betfair SP daily price files, enriches them with jockey/trainer/position/going from results sites, and writes a year-partitioned parquet at `data/historical/betsp.parquet`.

**Architecture:** Betfair SP CSVs form the spine (race_date, horse, odds_finish/BSP, win_lose, distance, venue, market signals). Three results sources (Sporting Life, Racing Post, At The Races) are scraped via the existing httpx→Playwright→Selenium tiers, raw HTML stored under `data/historical/raw/`, parsed by a pluggable `StubExtractor` (LLM impl deferred), then left-joined onto the spine by (date, course, horse). A `writer` outputs year-partitioned parquet with a read-merge-write per partition.

**Tech Stack:** Python 3.14, pandas, pyarrow, httpx, lxml, playwright, selenium, pytz, pytest, respx, pytest-mock. Reuses `utils/timezone`, `utils/logger`, `scraper/_selenium_fallback`.

**Note on git:** The project has no git repo yet. Commit steps below are written as optional checkpoints; run them only after `git init` (or skip until the repo is initialized).

---

## File Structure

```
scraper/betsp_historical.py        — PUBLIC orchestrator: fetch(years, force) -> pd.DataFrame
scraper/betsp/__init__.py
scraper/betsp/betfair_sp.py        — daily SP CSV download + parse → backbone rows
scraper/betsp/raw_store.py         — persist raw HTML under data/historical/raw/{source}/{year}/{date}/
scraper/betsp/results/__init__.py
scraper/betsp/results/base.py      — ResultsSource ABC, RawResult & ResultRow dataclasses
scraper/betsp/results/sporting_life.py
scraper/betsp/results/racing_post.py
scraper/betsp/results/at_the_races.py
scraper/betsp/extractor.py         — PerfExtractor ABC + StubExtractor
scraper/betsp/joiner.py            — normalize + left-join results onto backbone
scraper/betsp/writer.py            — year-partitioned parquet read-merge-write
tests/scraper/betsp/...            — one test module per unit
```

---

## Task 1: Config + package scaffold

**Files:**

- Modify: `config.yaml`
- Create: `scraper/betsp/__init__.py`, `scraper/betsp/results/__init__.py`
- Create: `tests/scraper/betsp/__init__.py`

- [ ] **Step 1: Add config block** to `config.yaml`:

```yaml
betsp_historical:
  rolling_years: 3
  regions: [uk, ire]
  markets: [win, place]
  results_sources: [sporting_life, racing_post, at_the_races]
  source_priority: [racing_post, sporting_life, at_the_races]
  extractor: stub
  max_days_per_run: 0
  parquet_path: data/historical/betsp.parquet
  raw_path: data/historical/raw
```

- [ ] **Step 2: Create empty package **init** files** (each a single docstring line).

- [ ] **Step 3: Commit (optional)** `git add config.yaml scraper/betsp tests/scraper/betsp && git commit -m "chore: scaffold betsp_historical package + config"`

---

## Task 2: Betfair SP fetcher — column mapping & price/distance/date parsing

**Files:**

- Create: `scraper/betsp/betfair_sp.py`
- Test: `tests/scraper/betsp/test_betfair_sp.py`

Real CSV header (confirmed via recon):
`event_id,menu_hint,event_name,event_dt,selection_id,selection_name,win_lose,bsp,ppwap,morningwap,ppmax,ppmin,ipmax,ipmin,morningtradedvol,pptradedvol,iptradedvol`

- [ ] **Step 1: Write failing tests** in `tests/scraper/betsp/test_betfair_sp.py`:

```python
import pandas as pd
from scraper.betsp import betfair_sp


SAMPLE_CSV = (
    "event_id,menu_hint,event_name,event_dt,selection_id,selection_name,win_lose,bsp,"
    "ppwap,morningwap,ppmax,ppmin,ipmax,ipmin,morningtradedvol,pptradedvol,iptradedvol\n"
    "258715928,Nottingham 31st May,1m2f Hcap,31-05-2026 16:55,96505154,Sharp Romance (IRE),1,2.51,"
    "2.49,2.99,3.0,2.46,2.74,1.01,3337.16,77465.12,38963.54\n"
    "258715934,Nottingham 31st May,5f Nov Stks,31-05-2026 17:25,69684787,Penelope Valentine,0,1001,"
    "8.80,11.3,14.5,8.4,1000.0,3.2,886.3,12135.31,5770.85\n"
)


def test_parse_distance_variants():
    assert betfair_sp._parse_distance("1m2f Hcap") == "1m2f"
    assert betfair_sp._parse_distance("5f Nov Stks") == "5f"
    assert betfair_sp._parse_distance("2m Chase") == "2m"
    assert betfair_sp._parse_distance("Maiden") is None


def test_parse_csv_maps_columns_and_nulls_bsp_1001():
    df = betfair_sp.parse_csv(SAMPLE_CSV, region="UK", market="WIN")
    assert list(df["horse_name"]) == ["Sharp Romance (IRE)", "Penelope Valentine"]
    assert df.loc[0, "horse_id"] == 96505154
    assert df.loc[0, "odds_finish"] == 2.51
    # bsp 1001 -> null (no SP backers)
    assert pd.isna(df.loc[1, "odds_finish"])
    assert list(df["win_lose"]) == [1, 0]
    assert df.loc[0, "distance"] == "1m2f"
    assert df.loc[0, "venue"] == "Nottingham"
    assert df.loc[0, "market_type"] == "WIN"
    assert df.loc[0, "region"] == "UK"


def test_parse_csv_race_date_is_dublin_aware():
    df = betfair_sp.parse_csv(SAMPLE_CSV, region="UK", market="WIN")
    ts = df.loc[0, "race_date"]
    assert ts.year == 2026 and ts.hour == 16 and ts.minute == 55
    assert ts.tzinfo is not None


def test_file_url():
    url = betfair_sp.file_url("uk", "win", 2026, 5, 31)
    assert url == (
        "https://promo.betfair.com/betfairsp/prices/dwbfpricesukwin31052026.csv"
    )
```

- [ ] **Step 2: Run to verify failure**
      Run: `pytest tests/scraper/betsp/test_betfair_sp.py -v`
      Expected: FAIL (module/functions not defined)

- [ ] **Step 3: Implement** `scraper/betsp/betfair_sp.py`:

```python
"""Betfair SP daily price file fetcher and parser (the betSP spine)."""
import io
import re
import time as _time
from datetime import datetime
from typing import Optional

import httpx
import pandas as pd

from utils.logger import get_logger
from utils.timezone import TZ

logger = get_logger(__name__)

_BASE = "https://promo.betfair.com/betfairsp/prices"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
    )
}
# 1m, 1m2f, 5f, 2m4f etc. — first distance token in the event name.
_DISTANCE_RE = re.compile(r"\b(\d+m(?:\d+f)?|\d+f)\b", re.IGNORECASE)

# Backbone columns produced by this module (subset of the final schema).
BACKBONE_COLUMNS = [
    "race_date", "venue", "horse_id", "horse_name", "odds_finish",
    "win_lose", "distance", "market_type", "region", "morningwap", "ppwap",
]


def file_url(region: str, market: str, year: int, month: int, day: int) -> str:
    """Build a daily SP file URL. region in {uk,ire}, market in {win,place}."""
    ddmmyyyy = f"{day:02d}{month:02d}{year:04d}"
    return f"{_BASE}/dwbfprices{region.lower()}{market.lower()}{ddmmyyyy}.csv"


def _parse_distance(event_name: str) -> Optional[str]:
    m = _DISTANCE_RE.search(event_name or "")
    return m.group(1).lower() if m else None


def _parse_dt(raw: str) -> Optional[datetime]:
    """'31-05-2026 16:55' -> Dublin-aware datetime."""
    try:
        naive = datetime.strptime(str(raw).strip(), "%d-%m-%Y %H:%M")
    except (ValueError, TypeError):
        return None
    return TZ.localize(naive)  # pytz: localize naive, never tzinfo=TZ


def _venue_from_menu_hint(menu_hint: str) -> str:
    # "Nottingham 31st May" -> "Nottingham"; strip trailing date words.
    text = str(menu_hint or "").strip()
    text = re.sub(r"\s+\d+(st|nd|rd|th)\s+\w+.*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def parse_csv(text: str, region: str, market: str) -> pd.DataFrame:
    """Parse one SP CSV body into backbone rows."""
    raw = pd.read_csv(io.StringIO(text))
    raw.columns = [c.strip().lower() for c in raw.columns]
    if raw.empty:
        return pd.DataFrame(columns=BACKBONE_COLUMNS)
    out = pd.DataFrame()
    out["race_date"] = raw["event_dt"].map(_parse_dt)
    out["venue"] = raw["menu_hint"].map(_venue_from_menu_hint)
    out["horse_id"] = pd.to_numeric(raw["selection_id"], errors="coerce").astype("Int64")
    out["horse_name"] = raw["selection_name"].astype(str).str.strip()
    bsp = pd.to_numeric(raw["bsp"], errors="coerce")
    out["odds_finish"] = bsp.where(bsp < 1001.0)  # 1001 = no SP backers -> null
    out["win_lose"] = pd.to_numeric(raw["win_lose"], errors="coerce").astype("Int8")
    out["distance"] = raw["event_name"].map(_parse_distance)
    out["market_type"] = market.upper()
    out["region"] = region.upper()
    out["morningwap"] = pd.to_numeric(raw.get("morningwap"), errors="coerce")
    out["ppwap"] = pd.to_numeric(raw.get("ppwap"), errors="coerce")
    return out[BACKBONE_COLUMNS]


def fetch_csv(url: str, max_retries: int = 3) -> Optional[str]:
    """GET a daily SP file. 404 -> None (no racing/not published)."""
    delay = 1.0
    for attempt in range(max_retries):
        try:
            with httpx.Client(headers=_HEADERS, timeout=20) as client:
                resp = client.get(url, follow_redirects=True)
            if resp.status_code == 404:
                logger.debug("SP file 404 (skipping): %s", url)
                return None
            resp.raise_for_status()
            return resp.text
        except httpx.HTTPError as exc:
            logger.warning("SP fetch error %s (attempt %d): %s", url, attempt + 1, exc)
            _time.sleep(delay)
            delay *= 2
    logger.warning("SP fetch giving up: %s", url)
    return None
```

- [ ] **Step 4: Run to verify pass**
      Run: `pytest tests/scraper/betsp/test_betfair_sp.py -v`
      Expected: PASS (4 tests)

- [ ] **Step 5: Commit (optional)** `git add scraper/betsp/betfair_sp.py tests/scraper/betsp/test_betfair_sp.py && git commit -m "feat(betsp): Betfair SP CSV fetch + parse"`

---

## Task 3: Results contracts — RawResult, ResultRow, ResultsSource ABC

**Files:**

- Create: `scraper/betsp/results/base.py`
- Test: `tests/scraper/betsp/test_results_base.py`

- [ ] **Step 1: Write failing test**:

```python
from scraper.betsp.results.base import RawResult, ResultRow, ResultsSource


def test_resultrow_defaults_nullable_fields():
    row = ResultRow(race_date="2026-05-31T16:55", venue="Nottingham",
                    horse_name="Sharp Romance", source="sporting_life")
    assert row.jockey_id is None and row.trainer_id is None
    assert row.position is None and row.going is None


def test_resultssource_is_abstract():
    import pytest
    with pytest.raises(TypeError):
        ResultsSource()  # abstract, cannot instantiate
```

- [ ] **Step 2: Run to verify failure**
      Run: `pytest tests/scraper/betsp/test_results_base.py -v`
      Expected: FAIL (import error)

- [ ] **Step 3: Implement** `scraper/betsp/results/base.py`:

```python
"""Contracts shared by all results sources."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class RawResult:
    """A captured raw document for one race (or results page)."""
    source: str
    race_date: date
    url: str
    html: str
    venue: Optional[str] = None


@dataclass
class ResultRow:
    """Normalized enrichment record for one horse-run."""
    race_date: str            # ISO timestamp or date string for join keying
    venue: str
    horse_name: str
    source: str
    jockey_id: Optional[str] = None
    trainer_id: Optional[str] = None
    position: Optional[int] = None
    going: Optional[str] = None


class ResultsSource(ABC):
    """A results website that yields RawResult docs and parses them to ResultRows."""

    name: str = "base"

    @abstractmethod
    def fetch_raw(self, day: date, get_html) -> list:
        """Return a list[RawResult] for the given day using get_html(url)->str."""

    @abstractmethod
    def parse(self, raw: RawResult) -> list:
        """Parse one RawResult into list[ResultRow]."""
```

- [ ] **Step 4: Run to verify pass**
      Run: `pytest tests/scraper/betsp/test_results_base.py -v`
      Expected: PASS

- [ ] **Step 5: Commit (optional)** `git add scraper/betsp/results/base.py tests/scraper/betsp/test_results_base.py && git commit -m "feat(betsp): results source contracts"`

---

## Task 4: Sporting Life results source (primary)

**Files:**

- Create: `scraper/betsp/results/sporting_life.py`
- Test: `tests/scraper/betsp/test_results_sporting_life.py`

> NOTE: Sporting Life's live DOM must be confirmed via DevTools before trusting the
> selectors below. The test uses a representative fixture; if the live structure
> differs, update `_RESULT_ROW_XPATH` and the per-cell extraction, keeping the
> public `parse()` contract identical. Mark the fixture clearly as illustrative.

- [ ] **Step 1: Write failing test** with an illustrative fixture:

```python
from datetime import date
from scraper.betsp.results.base import RawResult
from scraper.betsp.results.sporting_life import SportingLife

FIXTURE = """
<html><body>
<section class="race" data-going="Good to Soft" data-course="Nottingham"
         data-time="2026-05-31T16:55:00">
  <table class="results-table">
    <tr class="result-row">
      <td class="position">1</td>
      <td class="horse"><a href="/horse/123">Sharp Romance</a></td>
      <td class="jockey"><a href="/jockey/oscar-456">O Murphy</a></td>
      <td class="trainer"><a href="/trainer/charlie-789">C Appleby</a></td>
    </tr>
    <tr class="result-row">
      <td class="position">2</td>
      <td class="horse"><a href="/horse/124">Penelope Valentine</a></td>
      <td class="jockey"><a href="/jockey/tom-457">T Marquand</a></td>
      <td class="trainer"><a href="/trainer/sara-790">S Crane</a></td>
    </tr>
  </table>
</section>
</body></html>
"""


def test_parse_extracts_rows():
    src = SportingLife()
    raw = RawResult(source="sporting_life", race_date=date(2026, 5, 31),
                    url="http://x", html=FIXTURE, venue="Nottingham")
    rows = src.parse(raw)
    assert len(rows) == 2
    r0 = rows[0]
    assert r0.horse_name == "Sharp Romance"
    assert r0.position == 1
    assert r0.jockey_id == "oscar-456"
    assert r0.trainer_id == "charlie-789"
    assert r0.going == "Good to Soft"
    assert r0.venue == "Nottingham"
    assert r0.source == "sporting_life"
```

- [ ] **Step 2: Run to verify failure**
      Run: `pytest tests/scraper/betsp/test_results_sporting_life.py -v`
      Expected: FAIL (module not defined)

- [ ] **Step 3: Implement** `scraper/betsp/results/sporting_life.py`:

```python
"""Sporting Life results source (illustrative selectors — confirm via DevTools)."""
import re
from datetime import date
from typing import Optional

from lxml import html as lxml_html

from scraper.betsp.results.base import RawResult, ResultRow, ResultsSource
from utils.logger import get_logger

logger = get_logger(__name__)

_RESULTS_INDEX = "https://www.sportinglife.com/racing/results/{ymd}"
_ID_RE = re.compile(r"/(?:jockey|trainer|horse)/([a-z0-9-]+)", re.IGNORECASE)


def _id_from_href(href: Optional[str]) -> Optional[str]:
    if not href:
        return None
    m = _ID_RE.search(href)
    return m.group(1) if m else None


def _text(node) -> str:
    return (node.text_content() or "").strip() if node is not None else ""


class SportingLife(ResultsSource):
    name = "sporting_life"

    def fetch_raw(self, day: date, get_html) -> list:
        url = _RESULTS_INDEX.format(ymd=day.strftime("%Y-%m-%d"))
        try:
            html = get_html(url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SportingLife fetch failed %s: %s", url, exc)
            return []
        return [RawResult(source=self.name, race_date=day, url=url, html=html)]

    def parse(self, raw: RawResult) -> list:
        tree = lxml_html.fromstring(raw.html)
        rows = []
        for sec in tree.xpath('//section[contains(@class,"race")]'):
            going = sec.get("data-going") or None
            venue = sec.get("data-course") or raw.venue or ""
            race_dt = sec.get("data-time") or (raw.race_date.isoformat())
            for tr in sec.xpath('.//tr[contains(@class,"result-row")]'):
                pos_txt = _text(tr.xpath('.//td[contains(@class,"position")]')[0]) \
                    if tr.xpath('.//td[contains(@class,"position")]') else ""
                try:
                    position = int(re.sub(r"\D", "", pos_txt)) if pos_txt else None
                except ValueError:
                    position = None
                horse_a = tr.xpath('.//td[contains(@class,"horse")]//a')
                jockey_a = tr.xpath('.//td[contains(@class,"jockey")]//a')
                trainer_a = tr.xpath('.//td[contains(@class,"trainer")]//a')
                horse_name = _text(horse_a[0]) if horse_a else ""
                if not horse_name:
                    continue
                rows.append(ResultRow(
                    race_date=race_dt, venue=venue, horse_name=horse_name,
                    source=self.name, position=position, going=going,
                    jockey_id=_id_from_href(jockey_a[0].get("href")) if jockey_a else None,
                    trainer_id=_id_from_href(trainer_a[0].get("href")) if trainer_a else None,
                ))
        return rows
```

- [ ] **Step 4: Run to verify pass**
      Run: `pytest tests/scraper/betsp/test_results_sporting_life.py -v`
      Expected: PASS

- [ ] **Step 5: Commit (optional)** `git add scraper/betsp/results/sporting_life.py tests/scraper/betsp/test_results_sporting_life.py && git commit -m "feat(betsp): Sporting Life results source"`

---

## Task 5: Racing Post & At The Races sources (same contract)

**Files:**

- Create: `scraper/betsp/results/racing_post.py`, `scraper/betsp/results/at_the_races.py`
- Test: `tests/scraper/betsp/test_results_other_sources.py`

> NOTE: Both selectors are illustrative — confirm via DevTools. Racing Post is
> Cloudflare-protected; it will rely on the Playwright/Selenium tiers at runtime.
> The fixture proves the parse contract; live selectors get tuned later.

- [ ] **Step 1: Write failing test**:

```python
from datetime import date
from scraper.betsp.results.base import RawResult
from scraper.betsp.results.racing_post import RacingPost
from scraper.betsp.results.at_the_races import AtTheRaces

RP_FIXTURE = """
<div class="rp-result" data-going="Good" data-course="Nottingham" data-time="2026-05-31T16:55">
  <div class="rp-horse-row" data-position="1">
    <a class="rp-horse" href="/profile/horse/123/x">Sharp Romance</a>
    <a class="rp-jockey" href="/profile/jockey/oscar-456/x">O Murphy</a>
    <a class="rp-trainer" href="/profile/trainer/charlie-789/x">C Appleby</a>
  </div>
</div>
"""

ATR_FIXTURE = """
<div class="result" data-going="Soft" data-course="Nottingham" data-time="2026-05-31T16:55">
  <div class="runner" data-pos="1">
    <span class="horse-name">Sharp Romance</span>
    <a class="jockey-link" href="/jockey/oscar-456">O Murphy</a>
    <a class="trainer-link" href="/trainer/charlie-789">C Appleby</a>
  </div>
</div>
"""


def test_racing_post_parse():
    rows = RacingPost().parse(RawResult("racing_post", date(2026, 5, 31), "u", RP_FIXTURE))
    assert rows[0].horse_name == "Sharp Romance"
    assert rows[0].position == 1
    assert rows[0].jockey_id == "oscar-456"
    assert rows[0].trainer_id == "charlie-789"
    assert rows[0].going == "Good"


def test_at_the_races_parse():
    rows = AtTheRaces().parse(RawResult("at_the_races", date(2026, 5, 31), "u", ATR_FIXTURE))
    assert rows[0].horse_name == "Sharp Romance"
    assert rows[0].position == 1
    assert rows[0].jockey_id == "oscar-456"
    assert rows[0].going == "Soft"
```

- [ ] **Step 2: Run to verify failure**
      Run: `pytest tests/scraper/betsp/test_results_other_sources.py -v`
      Expected: FAIL

- [ ] **Step 3: Implement** `scraper/betsp/results/racing_post.py`:

```python
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
                    trainer_id=_id(trainer_a[0].get("href")) if trainer_a else None,
                ))
        return rows
```

- [ ] **Step 4: Implement** `scraper/betsp/results/at_the_races.py`:

```python
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
                    trainer_id=_id(trainer_a[0].get("href")) if trainer_a else None,
                ))
        return rows
```

- [ ] **Step 5: Run to verify pass**
      Run: `pytest tests/scraper/betsp/test_results_other_sources.py -v`
      Expected: PASS

- [ ] **Step 6: Commit (optional)** `git add scraper/betsp/results/racing_post.py scraper/betsp/results/at_the_races.py tests/scraper/betsp/test_results_other_sources.py && git commit -m "feat(betsp): Racing Post + At The Races results sources"`

---

## Task 6: Raw store

**Files:**

- Create: `scraper/betsp/raw_store.py`
- Test: `tests/scraper/betsp/test_raw_store.py`

- [ ] **Step 1: Write failing test**:

```python
from datetime import date
from scraper.betsp.results.base import RawResult
from scraper.betsp import raw_store


def test_write_and_path(tmp_path):
    raw = RawResult(source="sporting_life", race_date=date(2026, 5, 31),
                    url="http://x/results", html="<html>hi</html>")
    p = raw_store.write(raw, root=str(tmp_path))
    assert p.endswith(".html")
    assert "sporting_life" in p and "2026" in p and "2026-05-31" in p
    with open(p, encoding="utf-8") as f:
        assert f.read() == "<html>hi</html>"
```

- [ ] **Step 2: Run to verify failure**
      Run: `pytest tests/scraper/betsp/test_raw_store.py -v`
      Expected: FAIL

- [ ] **Step 3: Implement** `scraper/betsp/raw_store.py`:

```python
"""Persist captured raw results documents for later (LLM) re-extraction."""
import hashlib
import os

from scraper.betsp.results.base import RawResult
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "historical", "raw")
)


def write(raw: RawResult, root: str = _DEFAULT_ROOT) -> str:
    """Store raw.html under {root}/{source}/{year}/{date}/{hash}.html. Returns path."""
    year = f"{raw.race_date.year:04d}"
    day = raw.race_date.isoformat()
    digest = hashlib.sha1(raw.url.encode("utf-8")).hexdigest()[:12]
    folder = os.path.join(root, raw.source, year, day)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{digest}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(raw.html or "")
    logger.debug("raw_store wrote %s", path)
    return path
```

- [ ] **Step 4: Run to verify pass**
      Run: `pytest tests/scraper/betsp/test_raw_store.py -v`
      Expected: PASS

- [ ] **Step 5: Commit (optional)** `git add scraper/betsp/raw_store.py tests/scraper/betsp/test_raw_store.py && git commit -m "feat(betsp): raw results store"`

---

## Task 7: Stub extractor

**Files:**

- Create: `scraper/betsp/extractor.py`
- Test: `tests/scraper/betsp/test_extractor.py`

- [ ] **Step 1: Write failing test**:

```python
from datetime import date
from scraper.betsp.results.base import RawResult, ResultRow, ResultsSource
from scraper.betsp.extractor import PerfExtractor, StubExtractor


class _FakeSource(ResultsSource):
    name = "fake"
    def fetch_raw(self, day, get_html):
        return []
    def parse(self, raw):
        return [ResultRow(race_date="2026-05-31T16:55", venue="Nottingham",
                          horse_name="Sharp Romance", source="fake", position=1)]


def test_stub_extractor_delegates_to_source_parse():
    raw = RawResult("fake", date(2026, 5, 31), "u", "<html></html>")
    rows = StubExtractor().extract(raw, _FakeSource())
    assert len(rows) == 1 and rows[0].position == 1


def test_stub_drops_rows_without_horse_name():
    class _Bad(_FakeSource):
        def parse(self, raw):
            return [ResultRow(race_date="x", venue="y", horse_name="", source="fake")]
    raw = RawResult("fake", date(2026, 5, 31), "u", "<html></html>")
    assert StubExtractor().extract(raw, _Bad()) == []


def test_perfextractor_is_abstract():
    import pytest
    with pytest.raises(TypeError):
        PerfExtractor()
```

- [ ] **Step 2: Run to verify failure**
      Run: `pytest tests/scraper/betsp/test_extractor.py -v`
      Expected: FAIL

- [ ] **Step 3: Implement** `scraper/betsp/extractor.py`:

```python
"""Pluggable enrichment extractor. StubExtractor now; LLMExtractor later."""
from abc import ABC, abstractmethod

from scraper.betsp.results.base import RawResult, ResultsSource
from utils.logger import get_logger

logger = get_logger(__name__)


class PerfExtractor(ABC):
    @abstractmethod
    def extract(self, raw: RawResult, source: ResultsSource) -> list:
        """Return list[ResultRow] from a raw doc."""


class StubExtractor(PerfExtractor):
    """Deterministic extractor: delegates to the source's own parser + validates."""

    def extract(self, raw: RawResult, source: ResultsSource) -> list:
        try:
            rows = source.parse(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("StubExtractor parse failed (%s): %s", source.name, exc)
            return []
        valid = [r for r in rows if (r.horse_name or "").strip()]
        if len(valid) != len(rows):
            logger.debug("StubExtractor dropped %d invalid rows", len(rows) - len(valid))
        return valid
```

- [ ] **Step 4: Run to verify pass**
      Run: `pytest tests/scraper/betsp/test_extractor.py -v`
      Expected: PASS

- [ ] **Step 5: Commit (optional)** `git add scraper/betsp/extractor.py tests/scraper/betsp/test_extractor.py && git commit -m "feat(betsp): stub extractor + interface"`

---

## Task 8: Joiner

**Files:**

- Create: `scraper/betsp/joiner.py`
- Test: `tests/scraper/betsp/test_joiner.py`

- [ ] **Step 1: Write failing test**:

```python
import pandas as pd
from scraper.betsp.results.base import ResultRow
from scraper.betsp import joiner


def _backbone():
    return pd.DataFrame([{
        "race_date": pd.Timestamp("2026-05-31T16:55", tz="Europe/Dublin"),
        "venue": "Nottingham", "horse_id": 96505154,
        "horse_name": "Sharp Romance (IRE)", "odds_finish": 2.51, "win_lose": 1,
        "distance": "1m2f", "market_type": "WIN", "region": "UK",
        "morningwap": 2.99, "ppwap": 2.49,
    }])


def test_normalize_horse_strips_country_and_punct():
    assert joiner._norm_horse("Sharp Romance (IRE)") == "sharp romance"
    assert joiner._norm_horse("O'Brien's Pride") == "obriens pride"


def test_join_fills_enrichment_on_match():
    rows = [ResultRow(race_date="2026-05-31T16:55", venue="Nottingham",
                      horse_name="Sharp Romance", source="sporting_life",
                      jockey_id="oscar-456", trainer_id="charlie-789",
                      position=1, going="Good to Soft")]
    out = joiner.join(_backbone(), rows, priority=["sporting_life"])
    assert out.loc[0, "jockey_id"] == "oscar-456"
    assert out.loc[0, "position"] == 1
    assert out.loc[0, "going"] == "Good to Soft"
    assert out.loc[0, "result_source"] == "sporting_life"


def test_unmatched_backbone_keeps_nulls():
    out = joiner.join(_backbone(), [], priority=["sporting_life"])
    assert pd.isna(out.loc[0, "jockey_id"])
    assert pd.isna(out.loc[0, "result_source"])
    # backbone row is retained
    assert len(out) == 1


def test_source_priority_wins_conflict():
    rows = [
        ResultRow(race_date="2026-05-31T16:55", venue="Nottingham",
                  horse_name="Sharp Romance", source="at_the_races", position=3),
        ResultRow(race_date="2026-05-31T16:55", venue="Nottingham",
                  horse_name="Sharp Romance", source="racing_post", position=1),
    ]
    out = joiner.join(_backbone(), rows,
                      priority=["racing_post", "sporting_life", "at_the_races"])
    assert out.loc[0, "position"] == 1
    assert out.loc[0, "result_source"] == "racing_post"
```

- [ ] **Step 2: Run to verify failure**
      Run: `pytest tests/scraper/betsp/test_joiner.py -v`
      Expected: FAIL

- [ ] **Step 3: Implement** `scraper/betsp/joiner.py`:

```python
"""Left-join results enrichment onto the Betfair SP backbone."""
import re
from datetime import datetime

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

ENRICH_COLS = ["jockey_id", "trainer_id", "position", "going", "result_source"]
_COUNTRY_RE = re.compile(r"\((?:ire|gb|fr|usa|ger|aus|nz)\)", re.IGNORECASE)


def _norm_horse(name: str) -> str:
    text = _COUNTRY_RE.sub("", str(name or "")).lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _norm_venue(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def _minute_key(value) -> str:
    """Normalize a timestamp/ISO string to 'YYYY-MM-DDTHH:MM'."""
    if isinstance(value, pd.Timestamp) or isinstance(value, datetime):
        return value.strftime("%Y-%m-%dT%H:%M")
    text = str(value or "")
    # Accept 'YYYY-MM-DDTHH:MM[:SS][...]' -> trim to minute.
    m = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})", text)
    return m.group(1) if m else text


def _key(venue, horse, when) -> tuple:
    return (_norm_venue(venue), _norm_horse(horse), _minute_key(when))


def join(backbone: pd.DataFrame, result_rows: list, priority: list) -> pd.DataFrame:
    """Return backbone with ENRICH_COLS filled from result_rows (left join)."""
    out = backbone.copy()
    for col in ENRICH_COLS:
        out[col] = pd.NA

    rank = {name: i for i, name in enumerate(priority)}
    best: dict = {}
    for r in result_rows:
        k = _key(r.venue, r.horse_name, r.race_date)
        cur = best.get(k)
        r_rank = rank.get(r.source, len(priority))
        if cur is None or r_rank < cur[0]:
            best[k] = (r_rank, r)
        elif r_rank != cur[0]:
            continue
        else:
            # same source-rank, different value on position -> data quality note
            if cur[1].position is not None and r.position is not None \
                    and cur[1].position != r.position:
                logger.warning("data_quality: position conflict for %s", k)

    for idx, row in out.iterrows():
        k = _key(row["venue"], row["horse_name"], row["race_date"])
        hit = best.get(k)
        if not hit:
            continue
        r = hit[1]
        out.at[idx, "jockey_id"] = r.jockey_id
        out.at[idx, "trainer_id"] = r.trainer_id
        out.at[idx, "position"] = r.position
        out.at[idx, "going"] = r.going
        out.at[idx, "result_source"] = r.source
    return out
```

- [ ] **Step 4: Run to verify pass**
      Run: `pytest tests/scraper/betsp/test_joiner.py -v`
      Expected: PASS

- [ ] **Step 5: Commit (optional)** `git add scraper/betsp/joiner.py tests/scraper/betsp/test_joiner.py && git commit -m "feat(betsp): backbone-results joiner"`

---

## Task 9: Writer (year-partitioned parquet)

**Files:**

- Create: `scraper/betsp/writer.py`
- Test: `tests/scraper/betsp/test_writer.py`

- [ ] **Step 1: Write failing test**:

```python
import pandas as pd
from scraper.betsp import writer


def _df(horse_id, year=2026):
    return pd.DataFrame([{
        "race_date": pd.Timestamp(f"{year}-05-31T16:55", tz="Europe/Dublin"),
        "venue": "Nottingham", "horse_id": horse_id, "horse_name": "X",
        "jockey_id": None, "trainer_id": None, "odds_finish": 2.5, "position": None,
        "win_lose": 1, "going": None, "distance": "1m2f", "market_type": "WIN",
        "region": "UK", "morningwap": 2.9, "ppwap": 2.4, "result_source": None,
        "fetched_at": pd.Timestamp("2026-06-12", tz="UTC"),
    }])


def test_write_creates_year_partition(tmp_path):
    path = str(tmp_path / "betsp.parquet")
    writer.write(_df(1), path=path)
    back = pd.read_parquet(path)
    assert "year" in back.columns and set(back["year"]) == {2026}
    assert len(back) == 1


def test_write_dedupes_on_key(tmp_path):
    path = str(tmp_path / "betsp.parquet")
    writer.write(_df(1), path=path)
    writer.write(_df(1), path=path)  # same key -> still one row
    back = pd.read_parquet(path)
    assert len(back) == 1


def test_write_merges_distinct_rows(tmp_path):
    path = str(tmp_path / "betsp.parquet")
    writer.write(_df(1), path=path)
    writer.write(_df(2), path=path)
    back = pd.read_parquet(path)
    assert len(back) == 2
```

- [ ] **Step 2: Run to verify failure**
      Run: `pytest tests/scraper/betsp/test_writer.py -v`
      Expected: FAIL

- [ ] **Step 3: Implement** `scraper/betsp/writer.py`:

```python
"""Year-partitioned parquet writer with read-merge-write dedupe."""
import os

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "historical", "betsp.parquet")
)

FINAL_COLUMNS = [
    "race_date", "venue", "horse_id", "horse_name", "jockey_id", "trainer_id",
    "odds_finish", "position", "win_lose", "going", "distance", "market_type",
    "region", "morningwap", "ppwap", "result_source", "fetched_at", "year",
]
_DEDUPE_KEY = ["race_date", "venue", "horse_id", "market_type"]


def _add_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = pd.to_datetime(df["race_date"], utc=True).dt.year.astype(int)
    return df


def write(df: pd.DataFrame, path: str = _DEFAULT_PATH) -> None:
    """Merge df into the partitioned dataset at `path`, dedupe, write per year."""
    if df.empty:
        logger.debug("writer: empty df, nothing to write")
        return
    df = _add_year(df)
    os.makedirs(path, exist_ok=True)
    if os.path.exists(path) and os.listdir(path):
        try:
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("writer: could not read existing dataset (%s)", exc)
    df = df.drop_duplicates(subset=_DEDUPE_KEY, keep="last")
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[FINAL_COLUMNS]
    df.to_parquet(path, engine="pyarrow", index=False, partition_cols=["year"])
    logger.debug("writer: wrote %d rows across %d years",
                 len(df), df["year"].nunique())
```

- [ ] **Step 4: Run to verify pass**
      Run: `pytest tests/scraper/betsp/test_writer.py -v`
      Expected: PASS

- [ ] **Step 5: Commit (optional)** `git add scraper/betsp/writer.py tests/scraper/betsp/test_writer.py && git commit -m "feat(betsp): year-partitioned parquet writer"`

---

## Task 10: Orchestrator + tier fetcher

**Files:**

- Create: `scraper/betsp_historical.py`
- Test: `tests/scraper/betsp/test_betsp_historical.py`

- [ ] **Step 1: Write failing test** (all upstreams mocked):

```python
from datetime import date
import pandas as pd
import scraper.betsp_historical as bh
from scraper.betsp.betfair_sp import BACKBONE_COLUMNS


SAMPLE_CSV = (
    "event_id,menu_hint,event_name,event_dt,selection_id,selection_name,win_lose,bsp,"
    "ppwap,morningwap,ppmax,ppmin,ipmax,ipmin,morningtradedvol,pptradedvol,iptradedvol\n"
    "258715928,Nottingham 31st May,1m2f Hcap,31-05-2026 16:55,96505154,Sharp Romance (IRE),1,2.51,"
    "2.49,2.99,3.0,2.46,2.74,1.01,3337.16,77465.12,38963.54\n"
)

SL_FIXTURE = """
<section class="race" data-going="Good to Soft" data-course="Nottingham"
         data-time="2026-05-31T16:55:00">
  <table class="results-table">
    <tr class="result-row">
      <td class="position">1</td>
      <td class="horse"><a href="/horse/123">Sharp Romance</a></td>
      <td class="jockey"><a href="/jockey/oscar-456">O Murphy</a></td>
      <td class="trainer"><a href="/trainer/charlie-789">C Appleby</a></td>
    </tr>
  </table>
</section>
"""


def test_fetch_end_to_end(tmp_path, monkeypatch):
    # Betfair: only the uk/win file for one day returns data; others 404 (None).
    def fake_fetch_csv(url, max_retries=3):
        return SAMPLE_CSV if "ukwin31052026" in url else None
    monkeypatch.setattr(bh.betfair_sp, "fetch_csv", fake_fetch_csv)

    # Results: every source returns the Sporting-Life-shaped fixture via get_html.
    monkeypatch.setattr(bh, "_results_get_html", lambda url: SL_FIXTURE)

    # One-day window.
    monkeypatch.setattr(bh, "_iter_days",
                        lambda years: [date(2026, 5, 31)])

    out_path = str(tmp_path / "betsp.parquet")
    df = bh.fetch(years=[2026], force=True,
                  parquet_path=out_path, raw_root=str(tmp_path / "raw"))

    assert not df.empty
    assert df.loc[0, "horse_name"] == "Sharp Romance (IRE)"
    assert df.loc[0, "odds_finish"] == 2.51
    # enrichment joined in
    assert df.loc[0, "position"] == 1
    assert df.loc[0, "jockey_id"] == "oscar-456"
    # parquet persisted + year-partitioned
    back = pd.read_parquet(out_path)
    assert set(back["year"]) == {2026}
```

- [ ] **Step 2: Run to verify failure**
      Run: `pytest tests/scraper/betsp/test_betsp_historical.py -v`
      Expected: FAIL

- [ ] **Step 3: Implement** `scraper/betsp_historical.py`:

```python
"""betSP historical fetcher — PUBLIC orchestrator.

Betfair SP daily price files form the spine; results sites enrich
jockey/trainer/position/going; output is a year-partitioned parquet.
"""
import os
import time as _time
from datetime import date, timedelta
from typing import Callable, Optional

import pandas as pd
import yaml

from scraper.betsp import betfair_sp, joiner, raw_store, writer
from scraper.betsp.extractor import StubExtractor
from scraper.betsp.results.at_the_races import AtTheRaces
from scraper.betsp.results.racing_post import RacingPost
from scraper.betsp.results.sporting_life import SportingLife
from utils.logger import get_logger
from utils.timezone import now as _now

logger = get_logger(__name__)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
_SOURCES = {
    "sporting_life": SportingLife,
    "racing_post": RacingPost,
    "at_the_races": AtTheRaces,
}


def _load_cfg() -> dict:
    with open(_CONFIG_PATH) as f:
        return (yaml.safe_load(f) or {}).get("betsp_historical", {})


def _iter_days(years: list) -> list:
    days = []
    for y in years:
        d = date(y, 1, 1)
        end = date(y, 12, 31)
        while d <= end:
            days.append(d)
            d += timedelta(days=1)
    return days


def _resolve_years(cfg: dict, years: Optional[list]) -> list:
    if years:
        return years
    span = int(cfg.get("rolling_years", 3))
    this_year = _now().year
    return list(range(this_year - span + 1, this_year + 1))


# Default live HTML fetcher for results pages (httpx; tiers added at runtime).
def _results_get_html(url: str) -> str:
    import httpx
    headers = {"User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36")}
    with httpx.Client(headers=headers, timeout=20) as client:
        resp = client.get(url, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def _fetch_backbone(cfg: dict, days: list) -> pd.DataFrame:
    regions = cfg.get("regions", ["uk", "ire"])
    markets = cfg.get("markets", ["win", "place"])
    frames = []
    for d in days:
        for region in regions:
            for market in markets:
                url = betfair_sp.file_url(region, market, d.year, d.month, d.day)
                text = betfair_sp.fetch_csv(url)
                if not text:
                    continue
                try:
                    frames.append(betfair_sp.parse_csv(text, region, market))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("backbone parse failed %s: %s", url, exc)
    if not frames:
        return pd.DataFrame(columns=betfair_sp.BACKBONE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _fetch_enrichment(cfg: dict, days: list, raw_root: str,
                      get_html: Callable[[str], str]) -> list:
    source_names = cfg.get("results_sources", list(_SOURCES))
    extractor = StubExtractor()
    rows = []
    for name in source_names:
        src_cls = _SOURCES.get(name)
        if not src_cls:
            logger.warning("unknown results source: %s", name)
            continue
        src = src_cls()
        for d in days:
            for raw in src.fetch_raw(d, get_html):
                try:
                    raw_store.write(raw, root=raw_root)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("raw_store failed: %s", exc)
                rows.extend(extractor.extract(raw, src))
    return rows


def fetch(years: Optional[list] = None, force: bool = False,
          parquet_path: Optional[str] = None,
          raw_root: Optional[str] = None) -> pd.DataFrame:
    """Build/refresh the betSP historical dataset; returns the joined DataFrame."""
    cfg = _load_cfg()
    parquet_path = parquet_path or cfg.get("parquet_path",
                                           "data/historical/betsp.parquet")
    raw_root = raw_root or cfg.get("raw_path", "data/historical/raw")
    priority = cfg.get("source_priority",
                       ["racing_post", "sporting_life", "at_the_races"])

    years = _resolve_years(cfg, years)
    days = _iter_days(years)
    max_days = int(cfg.get("max_days_per_run", 0) or 0)
    if max_days:
        days = days[:max_days]

    logger.info("betsp_historical: %d day(s) across years %s", len(days), years)
    backbone = _fetch_backbone(cfg, days)
    if backbone.empty:
        logger.warning("betsp_historical: no Betfair SP data fetched")
        return backbone

    result_rows = _fetch_enrichment(cfg, days, raw_root, _results_get_html)
    joined = joiner.join(backbone, result_rows, priority=priority)
    joined["fetched_at"] = pd.Timestamp(_now()).tz_convert("UTC")
    writer.write(joined, path=parquet_path)
    return joined
```

- [ ] **Step 4: Run to verify pass**
      Run: `pytest tests/scraper/betsp/test_betsp_historical.py -v`
      Expected: PASS

- [ ] **Step 5: Run full betsp suite**
      Run: `pytest tests/scraper/betsp/ -v`
      Expected: all PASS

- [ ] **Step 6: Commit (optional)** `git add scraper/betsp_historical.py tests/scraper/betsp/test_betsp_historical.py && git commit -m "feat(betsp): orchestrator end-to-end"`

---

## Task 11: Regression + docs

- [ ] **Step 1: Run the whole test suite** to confirm nothing else broke
      Run: `pytest -q`
      Expected: existing suite + new betsp tests all PASS

- [ ] **Step 2: Update memory** `memory/project-overview.md` — add a `scraper/betsp_historical.py` section summarizing the spine+enrichment design, the confirmed SP CSV columns, the year-partitioned parquet path, and the "results selectors are illustrative, confirm via DevTools" caveat. Add an index line to `memory/MEMORY.md` if a new memory file is created.

- [ ] **Step 3: Commit (optional)** `git add memory && git commit -m "docs: record betsp_historical in project memory"`

---

## Self-Review

**Spec coverage:** every spec section maps to a task — recon/columns→Task 2; schema→Tasks 2/8/9; Betfair fetcher→Task 2; results sources→Tasks 3-5; raw store→Task 6; extractor seam→Task 7; join logic→Task 8; writer/partition/dedupe→Task 9; config→Task 1; orchestrator/error-handling→Task 10; testing→every task. LLMExtractor is explicitly out of scope (interface seam built in Task 7).

**Known caveat carried from spec:** results-site selectors (Tasks 4-5) are illustrative and must be confirmed against live DOM via DevTools — flagged inline in those tasks and to be recorded in memory (Task 11). This mirrors how the BoyleSports scraper was first built against assumed structure then corrected.
