# Timeform Historical Fetcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scraper/timeform_historical.py` (+ `scraper/timeform/` package) that scrapes Timeform racecards (ratings, pace, class, going, form) through the proxy/anti-bot ladder and writes a standardized year-partitioned `data/historical/timeform.parquet` that fuses with `betsp.parquet`.

**Architecture:** Mirrors `betsp_historical` — separated units (client, pages, parser, extractor, features, writer) behind a public `fetch()` orchestrator. Reuses `scraper/betsp/raw_store.py` and `scraper/_selenium_fallback.py`. Selectors confirmed from live DOM saved in `_probe_out/`.

**Tech Stack:** Python 3.14, httpx, lxml (`from lxml import html as lxml_html`), pandas, pyarrow, pytz (`utils/timezone.TZ`), pytest, respx.

**Spec:** `docs/superpowers/specs/2026-06-13-timeform-historical-design.md`

---

## File Structure

| File                             | Responsibility                                                              |
| -------------------------------- | --------------------------------------------------------------------------- |
| `scraper/timeform/__init__.py`   | package marker                                                              |
| `scraper/timeform/pages.py`      | URL builders + pagination (pure functions)                                  |
| `scraper/timeform/parser.py`     | `RunRow` dataclass + `parse_index` + `parse_event`                          |
| `scraper/timeform/extractor.py`  | `PerfExtractor` ABC + `StubExtractor`                                       |
| `scraper/timeform/features.py`   | `going_speed`, `class_change`, leak-safe trailing rates                     |
| `scraper/timeform/writer.py`     | year-partitioned parquet, read-merge-write dedupe                           |
| `scraper/timeform/client.py`     | anti-bot fetch ladder + ProxyRotator + session auth                         |
| `scraper/timeform_historical.py` | public `fetch()` orchestrator                                               |
| `utils/text_norm.py`             | shared `norm_venue`/`norm_horse`/`minute_key` (extracted from betsp joiner) |
| `tests/scraper/timeform/`        | one test module per unit + fixtures                                         |

---

## Task 0: Scaffolding, config, fixtures, shared normalizers

**Files:**

- Create: `scraper/timeform/__init__.py` (empty)
- Create: `tests/scraper/timeform/__init__.py` (empty)
- Create: `tests/scraper/timeform/fixtures/racecard_single.html` (copy from `_probe_out/`)
- Create: `tests/scraper/timeform/fixtures/racecards_index.html` (copy from `_probe_out/httpx_racecards_index.html`)
- Create: `utils/text_norm.py`
- Test: `tests/utils/test_text_norm.py`
- Modify: `config.yaml` (add `timeform:` block)

- [ ] **Step 1: Init git if absent (enables the commit steps)**

Run:

```bash
cd "C:\Users\mshr\Desktop\Race Predictor v3"
git rev-parse --is-inside-work-tree 2>$null || git init
```

If git is not desired, skip every "Commit" step in this plan.

- [ ] **Step 2: Copy real DOM fixtures for parser tests**

```powershell
New-Item -ItemType Directory -Force tests\scraper\timeform\fixtures
Copy-Item _probe_out\racecard_single.html tests\scraper\timeform\fixtures\racecard_single.html
Copy-Item _probe_out\httpx_racecards_index.html tests\scraper\timeform\fixtures\racecards_index.html
```

- [ ] **Step 3: Add the config block** to `config.yaml` (append at end):

```yaml
timeform:
  racecards_url: https://www.timeform.com/horse-racing/racecards
  rolling_years: 3
  regions: [uk, ire]
  request_delay: 1.5 # seconds between page fetches
  lookback_months: 12
  lookback_runs: 20
  place_positions: 3 # top-N finish counts as a place
  max_events_per_run: 0 # 0 = unlimited
  parquet_path: data/historical/timeform.parquet
  raw_path: data/historical/raw
  session_cookie: "" # paste a Timeform subscriber session cookie to unlock TFR/pace
  going_speed_map:
    firm: 5
    good-to-firm: 4
    good: 3
    good-to-soft: 2
    soft: 1
    heavy: 0
```

- [ ] **Step 4: Write failing test for shared normalizers**

`tests/utils/test_text_norm.py`:

```python
from utils.text_norm import norm_venue, norm_horse, minute_key


def test_norm_horse_strips_country_and_punct():
    assert norm_horse("Prince Of The Seas (IRE)") == "prince of the seas"


def test_norm_venue_lowercases_and_strips():
    assert norm_venue("Sandown-Park") == "sandownpark"


def test_minute_key_truncates_iso_to_minute():
    assert minute_key("2026-06-13T13:50:21+01:00") == "2026-06-13T13:50"
```

- [ ] **Step 5: Run test to verify it fails**

Run: `pytest tests/utils/test_text_norm.py -v`
Expected: FAIL with `ModuleNotFoundError: utils.text_norm`

- [ ] **Step 6: Implement `utils/text_norm.py`** (logic lifted verbatim from `scraper/betsp/joiner.py` so betsp and timeform align):

```python
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
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/utils/test_text_norm.py -v`
Expected: PASS (3 passed)

- [ ] **Step 8: Point betsp joiner at the shared helper (DRY, no behavior change)**

In `scraper/betsp/joiner.py`, replace the local `_norm_horse`/`_norm_venue`/`_minute_key`
function bodies with imports:

```python
from utils.text_norm import norm_horse as _norm_horse
from utils.text_norm import norm_venue as _norm_venue
from utils.text_norm import minute_key as _minute_key
```

Delete the now-duplicated local definitions (lines defining them).

- [ ] **Step 9: Verify betsp tests still pass**

Run: `pytest tests/scraper/betsp/ -q`
Expected: PASS (same count as before)

- [ ] **Step 10: Commit**

```bash
git add scraper/timeform/__init__.py tests/scraper/timeform/ utils/text_norm.py tests/utils/test_text_norm.py config.yaml scraper/betsp/joiner.py
git commit -m "feat(timeform): scaffold package, config, shared normalizers, fixtures"
```

---

## Task 1: `pages.py` — URL builders + pagination

**Files:**

- Create: `scraper/timeform/pages.py`
- Test: `tests/scraper/timeform/test_pages.py`

- [ ] **Step 1: Write failing tests**

`tests/scraper/timeform/test_pages.py`:

```python
from datetime import date

from scraper.timeform import pages


def test_index_url_for_date():
    assert pages.index_url(date(2026, 6, 13)) == \
        "https://www.timeform.com/horse-racing/racecards/2026-06-13"


def test_index_url_today_has_no_date_suffix():
    assert pages.index_url(None) == \
        "https://www.timeform.com/horse-racing/racecards"


def test_event_links_extracts_and_absolutizes():
    html = (
        '<a href="/horse-racing/racecards/york/2026-06-13/1350/62/1/x">A</a>'
        '<a href="/horse-racing/racecards/meeting-summary/chester/2026-06-12/12">skip</a>'
        '<a href="/horse-racing/racecards/bath/2026-06-13/1320/4/1/y">B</a>'
    )
    links = pages.event_links(html)
    assert links == [
        "https://www.timeform.com/horse-racing/racecards/york/2026-06-13/1350/62/1/x",
        "https://www.timeform.com/horse-racing/racecards/bath/2026-06-13/1320/4/1/y",
    ]
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/scraper/timeform/test_pages.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement `scraper/timeform/pages.py`**

```python
"""Timeform URL builders and racecard-link extraction."""
import re
from datetime import date
from typing import Optional

from lxml import html as lxml_html

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


def event_links(index_html: str) -> list:
    """Absolute URLs of each individual racecard on an index page (deduped, ordered)."""
    tree = lxml_html.fromstring(index_html)
    seen, out = set(), []
    for href in tree.xpath('//a[@href]/@href'):
        if _EVENT_RE.match(href) and href not in seen:
            seen.add(href)
            out.append(_BASE + href)
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/scraper/timeform/test_pages.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Verify against the real index fixture**

Run:

```bash
python -c "from scraper.timeform import pages; print(len(pages.event_links(open('tests/scraper/timeform/fixtures/racecards_index.html',encoding='utf-8').read())))"
```

Expected: a positive integer (real racecard count, e.g. 30+). If 0, the `_EVENT_RE` needs adjusting to the fixture.

- [ ] **Step 6: Commit**

```bash
git add scraper/timeform/pages.py tests/scraper/timeform/test_pages.py
git commit -m "feat(timeform): URL builders and racecard-link extraction"
```

---

## Task 2: `parser.py` — RunRow + parse_event

**Files:**

- Create: `scraper/timeform/parser.py`
- Test: `tests/scraper/timeform/test_parser.py`

Confirmed DOM: race header `rp-header` text contains the race name with a trailing
`(N)` class band, `Distance : 1m 3f 188y`, and time. Runners are `rp-horse-row`
containers; per runner — `rp-entry-number`, `rp-td-horse-name` (name may carry a
trailing `(draw)` and a country code), `rp-td-horse-jockey`, `rp-td-horse-form`
(figures like `50318-586`), `rp-td-horse-tfr` (empty unless subscribed).

- [ ] **Step 1: Write failing tests (against the real fixture)**

`tests/scraper/timeform/test_parser.py`:

```python
import os

from scraper.timeform import parser

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "racecard_single.html")


def _html():
    with open(FIX, encoding="utf-8") as f:
        return f.read()


def test_parse_event_returns_rows():
    rows = parser.parse_event(_html(), venue="York", going="good")
    assert len(rows) >= 8


def test_parse_event_extracts_core_fields():
    rows = parser.parse_event(_html(), venue="York", going="good")
    r = rows[0]
    assert r.horse_name and r.horse_name.isupper() is False or r.horse_name
    assert r.jockey_name
    assert r.recent_form  # e.g. "50318-586"
    assert r.race_class == 3            # from "(3)" in the header
    assert r.distance == "1m 3f 188y"
    assert r.going == "good"
    assert r.venue == "York"
    # TFR is paywalled in the free fixture -> None
    assert r.timeform_rating is None


def test_parse_class_band_from_header():
    assert parser._class_from_header("QUEEN MOTHER'S CUP HANDICAP (3) Distance : 1m") == 3
    assert parser._class_from_header("Some Maiden Stakes Distance : 5f") is None
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/scraper/timeform/test_parser.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement `scraper/timeform/parser.py`**

```python
"""Parse a Timeform racecard page into per-runner RunRow records."""
import re
from dataclasses import dataclass, field
from typing import Optional

from lxml import html as lxml_html


@dataclass
class RunRow:
    venue: str
    race_time: Optional[str] = None
    horse_name: str = ""
    horse_id: Optional[str] = None
    jockey_name: Optional[str] = None
    jockey_id: Optional[str] = None
    trainer_name: Optional[str] = None
    trainer_id: Optional[str] = None
    timeform_rating: Optional[float] = None   # paywalled -> None in free mode
    pace_rating: Optional[float] = None        # paywalled -> None in free mode
    race_class: Optional[int] = None
    distance: Optional[str] = None
    going: Optional[str] = None
    recent_form: Optional[str] = None          # free finishing-figures string
    region: Optional[str] = None


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _class_from_header(header_text: str) -> Optional[int]:
    """Timeform prints the class band as a parenthesised digit in the race name."""
    # Take the segment before "Distance" to avoid matching rating bands like (0-90).
    head = header_text.split("Distance")[0]
    m = re.search(r"\((\d)\)", head)
    return int(m.group(1)) if m else None


def _distance_from_header(header_text: str) -> Optional[str]:
    m = re.search(r"Distance\s*:\s*([\dmfyo ]+?)\s+(?:Prize|Rated|Age|Race|Surface|$)",
                  header_text)
    return _clean(m.group(1)) if m else None


def _float_or_none(text: str) -> Optional[float]:
    t = _clean(text)
    return float(t) if re.fullmatch(r"\d+(\.\d+)?", t) else None


def parse_event(html: str, venue: str, going: Optional[str] = None,
                region: Optional[str] = None, race_time: Optional[str] = None) -> list:
    tree = lxml_html.fromstring(html)
    header_els = tree.xpath('//*[contains(@class,"rp-header")]')
    header = _clean(header_els[0].text_content()) if header_els else ""
    race_class = _class_from_header(header)
    distance = _distance_from_header(header)

    rows = []
    seen_numbers = set()
    for row in tree.xpath('//*[contains(@class,"rp-horse-row")]'):
        num_el = row.xpath('.//*[contains(@class,"rp-entry-number")]')
        number = _clean(num_el[0].text_content()) if num_el else ""
        if number and number in seen_numbers:
            continue   # rp-horse-row spans 2 sub-rows; keep first per runner
        if number:
            seen_numbers.add(number)

        name_el = row.xpath('.//*[contains(@class,"rp-td-horse-name")]')
        if not name_el:
            continue
        # name cell carries trailing "(draw) D" noise -> strip parenthetical+suffix
        raw_name = _clean(name_el[0].text_content())
        horse_name = re.sub(r"\s*\(\d+\).*$", "", raw_name).strip()

        jockey_el = row.xpath('.//*[contains(@class,"rp-td-horse-jockey")]')
        form_el = row.xpath('.//*[contains(@class,"rp-td-horse-form")]')
        tfr_el = row.xpath('.//*[contains(@class,"rp-td-horse-tfr")]')

        rows.append(RunRow(
            venue=venue,
            race_time=race_time,
            horse_name=horse_name,
            jockey_name=_clean(jockey_el[0].text_content()) if jockey_el else None,
            recent_form=_clean(form_el[0].text_content()) or None if form_el else None,
            timeform_rating=_float_or_none(tfr_el[0].text_content()) if tfr_el else None,
            race_class=race_class,
            distance=distance,
            going=going,
            region=region,
        ))
    return rows
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/scraper/timeform/test_parser.py -v`
Expected: PASS (3 passed). If a field assertion fails, inspect the fixture with
`python -c "from lxml import html as H; t=H.fromstring(open('tests/scraper/timeform/fixtures/racecard_single.html',encoding='utf-8').read()); print(t.xpath('//*[contains(@class,\"rp-td-horse-name\")]')[0].text_content())"`
and adjust the selector/regex.

- [ ] **Step 5: Commit**

```bash
git add scraper/timeform/parser.py tests/scraper/timeform/test_parser.py
git commit -m "feat(timeform): racecard parser (RunRow, class/distance/form/jockey)"
```

---

## Task 3: `extractor.py` — PerfExtractor ABC + StubExtractor

**Files:**

- Create: `scraper/timeform/extractor.py`
- Test: `tests/scraper/timeform/test_extractor.py`

- [ ] **Step 1: Write failing test**

`tests/scraper/timeform/test_extractor.py`:

```python
import os

from scraper.timeform.extractor import StubExtractor

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "racecard_single.html")


def test_stub_extractor_delegates_to_parser():
    html = open(FIX, encoding="utf-8").read()
    rows = StubExtractor().extract(html, venue="York", going="good")
    assert rows and rows[0].horse_name
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/scraper/timeform/test_extractor.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement `scraper/timeform/extractor.py`**

```python
"""Perf extractor interface. StubExtractor delegates to the HTML parser;
an LLMExtractor reading raw_store blobs is the deferred next step."""
from abc import ABC, abstractmethod

from scraper.timeform import parser


class PerfExtractor(ABC):
    @abstractmethod
    def extract(self, html: str, **ctx) -> list:
        ...


class StubExtractor(PerfExtractor):
    def extract(self, html: str, **ctx) -> list:
        return parser.parse_event(html, **ctx)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/scraper/timeform/test_extractor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scraper/timeform/extractor.py tests/scraper/timeform/test_extractor.py
git commit -m "feat(timeform): PerfExtractor ABC + StubExtractor"
```

---

## Task 4: `features.py` — going_speed, class_change, leak-safe trailing rates

**Files:**

- Create: `scraper/timeform/features.py`
- Test: `tests/scraper/timeform/test_features.py`

- [ ] **Step 1: Write failing tests (leak-safety is the key one)**

`tests/scraper/timeform/test_features.py`:

```python
import pandas as pd

from scraper.timeform import features


def test_going_speed_maps_via_config():
    df = pd.DataFrame({"going": ["Good", "good-to-soft", "Heavy", "unknown"]})
    out = features.add_going_speed(df, {"good": 3, "good-to-soft": 2, "heavy": 0})
    assert list(out["going_speed"]) == [3, 2, 0, None]


def test_class_change_vs_previous_run():
    df = pd.DataFrame({
        "horse_name": ["A", "A", "A"],
        "race_date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"], utc=True),
        "race_class": [4, 3, 3],
    })
    out = features.add_class_change(df).sort_values("race_date")
    # first run has no prior -> NA; then 4->3 = -1 (up in class); 3->3 = 0
    assert pd.isna(out.iloc[0]["class_change"])
    assert out.iloc[1]["class_change"] == -1
    assert out.iloc[2]["class_change"] == 0


def test_trailing_win_rate_excludes_same_day_row():
    # A won on 2026-03-01. Its OWN win must not count toward its 2026-03-01 rate.
    df = pd.DataFrame({
        "horse_name": ["A", "A", "A"],
        "race_date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"], utc=True),
        "position": [1, 2, 1],
    })
    out = features.add_trailing_rates(
        df, entity="horse_name", win_col="historical_win_rate",
        place_col="historical_place_rate", lookback_months=12,
        lookback_runs=20, place_positions=3).sort_values("race_date")
    # On 2026-03-01: prior runs are pos 1 and pos 2 -> 1 win of 2 = 0.5, not 2/3.
    assert out.iloc[2]["historical_win_rate"] == 0.5
    assert out.iloc[2]["runs_in_window"] == 2


def test_trailing_rate_cold_start_is_null():
    df = pd.DataFrame({
        "horse_name": ["A"],
        "race_date": pd.to_datetime(["2026-03-01"], utc=True),
        "position": [1],
    })
    out = features.add_trailing_rates(
        df, entity="horse_name", win_col="historical_win_rate",
        place_col="historical_place_rate", lookback_months=12,
        lookback_runs=20, place_positions=3)
    assert pd.isna(out.iloc[0]["historical_win_rate"])
    assert out.iloc[0]["runs_in_window"] == 0
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/scraper/timeform/test_features.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement `scraper/timeform/features.py`**

```python
"""Derived Timeform features: going_speed, class_change, leak-safe trailing rates."""
import pandas as pd


def add_going_speed(df: pd.DataFrame, going_map: dict) -> pd.DataFrame:
    out = df.copy()
    norm = {str(k).lower(): v for k, v in (going_map or {}).items()}
    out["going_speed"] = out["going"].map(
        lambda g: norm.get(str(g).strip().lower()) if pd.notna(g) else None)
    return out


def add_class_change(df: pd.DataFrame) -> pd.DataFrame:
    """class_change = this race_class minus the horse's previous race_class.
    Negative = stepping UP in class (lower band number is higher class)."""
    out = df.copy().sort_values(["horse_name", "race_date"])
    prev = out.groupby("horse_name")["race_class"].shift(1)
    out["class_change"] = out["race_class"] - prev
    return out


def add_trailing_rates(df: pd.DataFrame, entity: str, win_col: str, place_col: str,
                       lookback_months: int, lookback_runs: int,
                       place_positions: int) -> pd.DataFrame:
    """Point-in-time win/place rate from runs STRICTLY before each row's race_date,
    bounded by lookback_months AND lookback_runs (tighter wins). Leak-free."""
    out = df.copy().reset_index(drop=True)
    out[win_col] = pd.NA
    out[place_col] = pd.NA
    out["runs_in_window"] = 0
    window = pd.DateOffset(months=lookback_months)

    for _, grp in out.groupby(entity):
        g = grp.sort_values("race_date")
        idx = list(g.index)
        for pos, i in enumerate(idx):
            cutoff = out.at[i, "race_date"]
            prior = g.loc[idx[:pos]]
            prior = prior[prior["race_date"] >= (cutoff - window)]
            if lookback_runs:
                prior = prior.tail(lookback_runs)
            n = len(prior)
            out.at[i, "runs_in_window"] = n
            if n:
                wins = (prior["position"] == 1).sum()
                places = (prior["position"] <= place_positions).sum()
                out.at[i, win_col] = wins / n
                out.at[i, place_col] = places / n
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/scraper/timeform/test_features.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add scraper/timeform/features.py tests/scraper/timeform/test_features.py
git commit -m "feat(timeform): going_speed, class_change, leak-safe trailing rates"
```

---

## Task 5: `writer.py` — year-partitioned parquet with dedupe

**Files:**

- Create: `scraper/timeform/writer.py`
- Test: `tests/scraper/timeform/test_writer.py`

- [ ] **Step 1: Write failing tests**

`tests/scraper/timeform/test_writer.py`:

```python
import pandas as pd

from scraper.timeform import writer


def _row(**kw):
    base = dict(race_date=pd.Timestamp("2026-06-13", tz="UTC"), venue="York",
                race_time="2026-06-13T13:50", horse_name="A", source="timeform")
    base.update(kw)
    return base


def test_write_partitions_by_year_and_dedupes(tmp_path):
    path = str(tmp_path / "timeform.parquet")
    writer.write(pd.DataFrame([_row(timeform_rating=100)]), path=path)
    # same dedupe key, newer value -> last wins, not duplicated
    writer.write(pd.DataFrame([_row(timeform_rating=120)]), path=path)
    df = pd.read_parquet(path)
    assert len(df) == 1
    assert df.iloc[0]["timeform_rating"] == 120
    assert df.iloc[0]["year"] == 2026


def test_write_empty_is_noop(tmp_path):
    path = str(tmp_path / "tf.parquet")
    writer.write(pd.DataFrame(), path=path)  # must not raise
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/scraper/timeform/test_writer.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement `scraper/timeform/writer.py`** (pattern from `scraper/betsp/writer.py`, incl. the `shutil.rmtree`-before-partition-write gotcha):

```python
"""Year-partitioned parquet writer for the Timeform dataset (read-merge-write)."""
import os
import shutil

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

FINAL_COLUMNS = [
    "race_date", "venue", "race_time", "horse_name", "horse_id",
    "jockey_name", "jockey_id", "trainer_name", "trainer_id", "position",
    "timeform_rating", "pace_rating", "race_class", "going", "going_speed",
    "distance", "class_change", "recent_form", "historical_win_rate",
    "historical_place_rate", "jockey_win_rate", "runs_in_window",
    "region", "source", "fetched_at", "year",
]
_DEDUPE_KEY = ["race_date", "venue", "race_time", "horse_name"]


def _add_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = pd.to_datetime(df["race_date"], utc=True).dt.year.astype(int)
    return df


def write(df: pd.DataFrame, path: str) -> None:
    if df is None or df.empty:
        logger.debug("timeform writer: empty df, nothing to write")
        return
    df = _add_year(df)
    if os.path.exists(path) and os.listdir(path):
        try:
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("timeform writer: could not read existing (%s)", exc)
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    df = df.drop_duplicates(subset=_DEDUPE_KEY, keep="last")
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[FINAL_COLUMNS]
    df.to_parquet(path, engine="pyarrow", index=False, partition_cols=["year"])
    logger.debug("timeform writer: wrote %d rows", len(df))
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/scraper/timeform/test_writer.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scraper/timeform/writer.py tests/scraper/timeform/test_writer.py
git commit -m "feat(timeform): year-partitioned parquet writer with dedupe"
```

---

## Task 6: `client.py` — anti-bot fetch ladder + proxy + session

**Files:**

- Create: `scraper/timeform/client.py`
- Test: `tests/scraper/timeform/test_client.py`

Reuses `scraper.paddy_power.ProxyRotator` for proxy rotation and
`scraper._selenium_fallback.SeleniumFallback` for the Selenium tier.

- [ ] **Step 1: Write failing tests (respx-mocked transports)**

`tests/scraper/timeform/test_client.py`:

```python
import httpx
import pytest
import respx

from scraper.timeform.client import TimeformClient, TimeformError


def _cfg(**kw):
    base = {"session_cookie": "", "request_delay": 0}
    base.update(kw)
    return base


@respx.mock
def test_get_html_returns_body_on_200():
    respx.get("https://x.test/card").mock(
        return_value=httpx.Response(200, html="<html>ok rp-horse-row</html>"))
    c = TimeformClient(_cfg(), browser_fetch=None, selenium_fetch=None)
    assert "rp-horse-row" in c.get_html("https://x.test/card")


@respx.mock
def test_waf_challenge_raises_when_no_browser_tier():
    respx.get("https://x.test/r").mock(
        return_value=httpx.Response(200, html="<title>Azure WAF</title>"))
    c = TimeformClient(_cfg(), browser_fetch=None, selenium_fetch=None)
    with pytest.raises(TimeformError):
        c.get_html("https://x.test/r")


@respx.mock
def test_403_falls_through_to_browser_tier():
    respx.get("https://x.test/r").mock(return_value=httpx.Response(403))
    c = TimeformClient(_cfg(), browser_fetch=lambda u: "<html>rp-horse-row via browser</html>",
                       selenium_fetch=None)
    assert "via browser" in c.get_html("https://x.test/r")


@respx.mock
def test_session_cookie_is_sent():
    captured = {}

    def _capture(request):
        captured["cookie"] = request.headers.get("cookie")
        return httpx.Response(200, html="<html>rp-horse-row</html>")

    respx.get("https://x.test/c").mock(side_effect=_capture)
    c = TimeformClient(_cfg(session_cookie="sess=abc"), browser_fetch=None, selenium_fetch=None)
    c.get_html("https://x.test/c")
    assert captured["cookie"] == "sess=abc"
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/scraper/timeform/test_client.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement `scraper/timeform/client.py`**

```python
"""Timeform fetch ladder: httpx(+proxy) -> browser -> selenium -> raise.
WAF/paywall detection routes to the next tier; results section stays out of scope."""
import re
import time
from typing import Callable, Optional

import httpx

from scraper.paddy_power import ProxyRotator
from utils.logger import get_logger

logger = get_logger(__name__)


class TimeformError(RuntimeError):
    pass


_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36")
_WAF_RE = re.compile(r"Azure WAF|/\.azwaf/|cf-browser-verification", re.I)


def _looks_blocked(html: str) -> bool:
    return bool(_WAF_RE.search(html or "")) or "<html" not in (html or "").lower()


class TimeformClient:
    def __init__(self, cfg: dict, proxy_rotator: Optional[ProxyRotator] = None,
                 browser_fetch: Optional[Callable[[str], str]] = None,
                 selenium_fetch: Optional[Callable[[str], str]] = None,
                 max_retries: int = 3):
        self._cfg = cfg or {}
        self._rotator = proxy_rotator or ProxyRotator()
        self._browser_fetch = browser_fetch
        self._selenium_fetch = selenium_fetch
        self._max_retries = max_retries
        self._delay = float(self._cfg.get("request_delay", 1.5) or 0)

    def _headers(self) -> dict:
        h = {"User-Agent": _UA,
             "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
             "Accept-Language": "en-GB,en;q=0.9"}
        cookie = self._cfg.get("session_cookie")
        if cookie:
            h["Cookie"] = cookie
        return h

    def _httpx_get(self, url: str) -> Optional[str]:
        delay = 1
        for attempt in range(self._max_retries):
            proxy = self._rotator.next()
            kwargs = {"headers": self._headers(), "timeout": 25, "follow_redirects": True}
            if proxy:
                kwargs["proxy"] = proxy
            try:
                with httpx.Client(**kwargs) as client:
                    resp = client.get(url)
                if resp.status_code == 200 and not _looks_blocked(resp.text):
                    return resp.text
                logger.info("timeform httpx tier blocked (%s) on %s", resp.status_code, url)
                return None
            except httpx.HTTPError as exc:
                logger.warning("timeform httpx error %s (attempt %d)", exc, attempt + 1)
                time.sleep(delay)
                delay *= 2
        return None

    def get_html(self, url: str) -> str:
        if self._delay:
            time.sleep(self._delay)
        body = self._httpx_get(url)
        if body:
            return body
        for tier in (self._browser_fetch, self._selenium_fetch):
            if tier is None:
                continue
            try:
                body = tier(url)
            except Exception as exc:  # noqa: BLE001
                logger.warning("timeform fallback tier failed: %s", exc)
                continue
            if body and not _looks_blocked(body):
                return body
        raise TimeformError(f"all tiers failed/blocked for {url}")
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/scraper/timeform/test_client.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add scraper/timeform/client.py tests/scraper/timeform/test_client.py
git commit -m "feat(timeform): anti-bot fetch ladder with proxy and session cookie"
```

---

## Task 7: `timeform_historical.py` — public orchestrator

**Files:**

- Create: `scraper/timeform_historical.py`
- Test: `tests/scraper/test_timeform_historical.py`

- [ ] **Step 1: Write failing test (all network injected)**

`tests/scraper/test_timeform_historical.py`:

```python
import os

import pandas as pd

from scraper import timeform_historical as th

FIX = os.path.join(os.path.dirname(__file__), "timeform", "fixtures")


def _index_html():
    return open(os.path.join(FIX, "racecards_index.html"), encoding="utf-8").read()


def _card_html():
    return open(os.path.join(FIX, "racecard_single.html"), encoding="utf-8").read()


def test_fetch_builds_dataframe_and_writes(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_get_html(url):
        calls["n"] += 1
        return _index_html() if "racecards" in url and url.count("/") < 6 else _card_html()

    path = str(tmp_path / "timeform.parquet")
    df = th.fetch(days=[__import__("datetime").date(2026, 6, 13)],
                  get_html=fake_get_html, parquet_path=path, max_events=1)
    assert not df.empty
    assert {"timeform_rating", "race_class", "going_speed",
            "historical_win_rate"}.issubset(df.columns)
    assert os.path.exists(path)
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/scraper/test_timeform_historical.py -v`
Expected: FAIL `ModuleNotFoundError` / `AttributeError`

- [ ] **Step 3: Implement `scraper/timeform_historical.py`**

```python
"""Timeform historical fetcher — PUBLIC orchestrator.

Scrapes racecards (ratings/pace/class/going/form) and writes a year-partitioned
parquet that fuses with betsp.parquet. Positions/results are sourced separately;
trailing win/place rates compute when a `position` column is present, else stay null.
"""
import os
from datetime import date, timedelta
from typing import Callable, Optional

import pandas as pd
import yaml

from scraper.timeform import features, pages, writer
from scraper.timeform.client import TimeformClient
from scraper.timeform.extractor import StubExtractor
from utils.logger import get_logger
from utils.timezone import now as _now

logger = get_logger(__name__)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def _load_cfg() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("timeform", {})


def _resolve_days(cfg: dict, days: Optional[list]) -> list:
    if days:
        return days
    span = int(cfg.get("rolling_years", 3))
    today = _now().date()
    out, d = [], date(today.year - span + 1, 1, 1)
    while d <= today:
        out.append(d)
        d += timedelta(days=1)
    return out


def fetch(days: Optional[list] = None, get_html: Optional[Callable[[str], str]] = None,
          parquet_path: Optional[str] = None, max_events: Optional[int] = None) -> pd.DataFrame:
    cfg = _load_cfg()
    parquet_path = parquet_path or cfg.get("parquet_path", "data/historical/timeform.parquet")
    max_events = cfg.get("max_events_per_run", 0) if max_events is None else max_events
    if get_html is None:
        get_html = TimeformClient(cfg).get_html

    days = _resolve_days(cfg, days)
    extractor = StubExtractor()
    rows = []
    event_count = 0
    for d in days:
        try:
            index_html = get_html(pages.index_url(d))
        except Exception as exc:  # noqa: BLE001
            logger.warning("timeform index fetch failed for %s: %s", d, exc)
            continue
        for url in pages.event_links(index_html):
            if max_events and event_count >= max_events:
                break
            event_count += 1
            venue = url.split("/racecards/")[1].split("/")[0]
            try:
                html = get_html(url)
                rows.extend(extractor.extract(html, venue=venue))
            except Exception as exc:  # noqa: BLE001
                logger.warning("timeform event failed %s: %s", url, exc)

    if not rows:
        logger.warning("timeform: no rows scraped")
        return pd.DataFrame()

    df = pd.DataFrame([r.__dict__ for r in rows])
    df["race_date"] = pd.to_datetime(
        [d for d in days[:1] * len(df)], utc=True) if "race_date" not in df else df["race_date"]
    # race_date from the first day in scope when not parsed per-row (racecards are dated by URL)
    df["source"] = "timeform"
    df["fetched_at"] = pd.Timestamp(_now()).tz_convert("UTC")

    df = features.add_going_speed(df, cfg.get("going_speed_map", {}))
    df = features.add_class_change(df)
    if "position" in df.columns and df["position"].notna().any():
        df = features.add_trailing_rates(
            df, entity="horse_name", win_col="historical_win_rate",
            place_col="historical_place_rate",
            lookback_months=int(cfg.get("lookback_months", 12)),
            lookback_runs=int(cfg.get("lookback_runs", 20)),
            place_positions=int(cfg.get("place_positions", 3)))
    else:
        for col in ("historical_win_rate", "historical_place_rate", "jockey_win_rate"):
            df[col] = pd.NA
        df["runs_in_window"] = 0

    writer.write(df, path=parquet_path)
    return df
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/scraper/test_timeform_historical.py -v`
Expected: PASS. If `race_date` handling errors on the fixture, set `df["race_date"]`
explicitly from the loop day before building features (the URL carries the date).

- [ ] **Step 5: Run the whole suite**

Run: `pytest -q`
Expected: all green (128 prior + new timeform tests).

- [ ] **Step 6: Commit**

```bash
git add scraper/timeform_historical.py tests/scraper/test_timeform_historical.py
git commit -m "feat(timeform): public fetch() orchestrator wiring all units"
```

---

## Task 8: Live smoke test + cleanup

**Files:**

- Delete: `probe_timeform.py`, `_probe_out/`

- [ ] **Step 1: Live smoke test through the proxy (one day, a few events)**

Run:

```bash
python -c "from scraper import timeform_historical as th; import datetime; df=th.fetch(days=[datetime.date(2026,6,13)], max_events=3); print(df[['venue','horse_name','race_class','distance','recent_form','going_speed']].head(20)); print('rows', len(df))"
```

Expected: real York/Bath/etc. runners with `race_class`, `distance`, `recent_form`
populated; `timeform_rating` null (paywalled). Confirms selectors hold on live HTML.

- [ ] **Step 2: If selectors drifted**, update `scraper/timeform/parser.py` xpath/regex
      against the live output and re-run Task 2 tests.

- [ ] **Step 3: Remove the throwaway probe**

```powershell
Remove-Item probe_timeform.py
Remove-Item -Recurse -Force _probe_out
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore(timeform): remove throwaway probe after selector confirmation"
```

---

## Self-Review Notes

- **Spec coverage:** ratings (Task 2, session-gated), pace (Task 2 field, session-gated),
  race_class (Task 2), going_speed (Task 4), class_change (Task 4),
  historical_win_rate/place/jockey (Task 4, deferred-position aware), pagination
  (Task 1), anti-bot (Task 6), parquet output (Task 5), orchestrator (Task 7),
  join keys via shared normalizers (Task 0). All covered.
- **Deferred by design:** `position`, `pace_rating`, `timeform_rating` values depend on
  the separate results scraper / subscriber session — columns exist and the math runs
  when data is present (Task 7 conditional).
- **`jockey_win_rate`** column is emitted null in this build; wiring it through
  `add_trailing_rates(entity="jockey_name", ...)` is a one-call follow-up once jockey
  positions exist — intentionally not forced now (YAGNI without positions).

```

```
