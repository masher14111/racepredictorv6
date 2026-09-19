"""Parse a Timeform racecard page into per-runner RunRow records."""
import re
from dataclasses import dataclass
from typing import Optional

from lxml import html as lxml_html


@dataclass
class RunRow:
    venue: str
    race_time: Optional[str] = None
    horse_name: str = ""
    horse_id: Optional[str] = None             # native Timeform id, e.g. "000000614450"
    jockey_name: Optional[str] = None
    jockey_id: Optional[str] = None            # native Timeform id
    trainer_name: Optional[str] = None
    trainer_id: Optional[str] = None           # native Timeform id
    timeform_rating: Optional[float] = None   # paywalled -> None in free mode
    pace_rating: Optional[float] = None        # paywalled -> None in free mode
    race_class: Optional[int] = None
    distance: Optional[str] = None
    going: Optional[str] = None
    recent_form: Optional[str] = None          # free finishing-figures string
    region: Optional[str] = None
    # Declared-card fields free of the Timeform Race Passes paywall.
    draw: Optional[int] = None
    weight_lbs: Optional[int] = None
    age: Optional[int] = None
    official_rating: Optional[int] = None       # "(OR)" column
    equipment: Optional[str] = None             # raw code, e.g. "t" (tongue strap)
    colour: Optional[str] = None                # e.g. "b" (bay)
    sex: Optional[str] = None                   # e.g. "g" (gelding)
    runner_status: str = "RUNNER"                # "RUNNER" | "NON_RUNNER"
    timeform_race_id: Optional[str] = None      # native per-event id, e.g. "2026-06-13-62-1"


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


def _strip_name_noise(raw_name: str) -> str:
    """Drop trailing draw/headgear noise: '(20H)', '(39H) D BF' -> removed.
    A country code like '(IRE)' (no leading digit) is preserved."""
    return re.sub(r"\s*\(\d+[A-Za-z]*\).*$", "", _clean(raw_name)).strip()


def _float_or_none(text: str) -> Optional[float]:
    t = _clean(text)
    return float(t) if re.fullmatch(r"\d+(\.\d+)?", t) else None


def _int_or_none(text: str) -> Optional[int]:
    t = re.sub(r"[^\d]", "", _clean(text))
    return int(t) if t else None


_HORSE_ID_RE = re.compile(r"/horse-racing/horse/form/[^/]+/(\d+)/")

# Free-tier non-runner markers observed across Timeform/racing-site racecards
# ("NON RUNNER", "(WD)" for withdrawn, "(NR)"). No live withdrawal example was
# captured in the saved fixture (it predates any late scratching), so this is
# a best-effort text/class heuristic, not verified against a real live
# withdrawal page -- see memory/improvement/stages/05.md.
_NON_RUNNER_RE = re.compile(r"non[\s-]?runner|\bwithdrawn\b|\(NR\)|\(WD\)", re.I)


def _horse_id_from_href(name_el) -> Optional[str]:
    hrefs = name_el.xpath('.//a[contains(@class,"rp-horse")]/@href') or name_el.xpath('.//a/@href')
    for href in hrefs:
        m = _HORSE_ID_RE.search(href)
        if m:
            return m.group(1)
    return None


def _colour_sex(row) -> tuple:
    """('b', 'g') from the free pedigree cell's 'Colour & Gender' span, e.g. 'b g'."""
    els = row.xpath('.//*[@title="Colour & Gender"]')
    if not els:
        return None, None
    parts = _clean(els[0].text_content()).split()
    if not parts:
        return None, None
    if len(parts) == 1:
        return None, parts[0]
    return " ".join(parts[:-1]), parts[-1]


def _is_non_runner(row, horse_name: str) -> bool:
    row_class = " ".join(row.xpath('./@class') + row.xpath('.//*/@class'))
    if _NON_RUNNER_RE.search(row_class):
        return True
    return bool(_NON_RUNNER_RE.search(horse_name))


def parse_event(html: str, venue: str, going: Optional[str] = None,
                region: Optional[str] = None, race_time: Optional[str] = None) -> list:
    tree = lxml_html.fromstring(html)
    header_els = tree.xpath('//*[contains(@class,"rp-header")]')
    header = _clean(header_els[0].text_content()) if header_els else ""
    race_class = _class_from_header(header)
    distance = _distance_from_header(header)

    rows = []
    # Exact class token "rp-horse-row" — excludes nested rp-horse-row-1/-2 sub-rows.
    runner_rows = tree.xpath(
        '//*[contains(concat(" ", normalize-space(@class), " "), " rp-horse-row ")]')
    for row in runner_rows:
        name_el = row.xpath('.//*[contains(@class,"rp-td-horse-name")]')
        if not name_el:
            continue
        # name cell carries trailing "(draw) D" noise -> strip parenthetical+suffix
        raw_name = name_el[0].text_content()
        horse_name = _strip_name_noise(raw_name)

        jockey_el = row.xpath('.//*[contains(@class,"rp-td-horse-jockey")]')
        trainer_el = row.xpath(
            './/*[contains(concat(" ", normalize-space(@class), " "), " rp-td-horse-trainer ")]')
        form_el = row.xpath('.//*[contains(@class,"rp-td-horse-form")]')
        tfr_el = row.xpath('.//*[contains(@class,"rp-td-horse-tfr")]')
        age_el = row.xpath('.//*[contains(@class,"rp-td-horse-age")]')
        or_el = row.xpath('.//*[contains(@class,"rp-td-horse-or")]')
        equip_el = row.xpath('.//*[contains(@class,"rp-td-horse-equipment")]')
        colour, sex = _colour_sex(row)

        # tbody-level data attributes carry the native provider ids and the
        # free draw/weight facts directly (no href-parsing needed for those).
        draw = _int_or_none(row.get("data-drawnumber"))
        weight_lbs = _int_or_none(row.get("data-weightsort"))
        jockey_id = _clean(row.get("data-jockey") or "") or None
        trainer_id = _clean(row.get("data-trainer") or "") or None
        timeform_race_id = _clean(row.get("data-raceid") or "") or None

        rows.append(RunRow(
            venue=venue,
            race_time=race_time,
            horse_name=horse_name,
            horse_id=_horse_id_from_href(name_el[0]),
            jockey_name=_clean(jockey_el[0].text_content()) if jockey_el else None,
            jockey_id=jockey_id,
            trainer_name=_clean(trainer_el[0].text_content()) if trainer_el else None,
            trainer_id=trainer_id,
            recent_form=(_clean(form_el[0].text_content()) or None) if form_el else None,
            timeform_rating=_float_or_none(tfr_el[0].text_content()) if tfr_el else None,
            race_class=race_class,
            distance=distance,
            going=going,
            region=region,
            draw=draw,
            weight_lbs=weight_lbs,
            age=_int_or_none(age_el[0].text_content()) if age_el else None,
            official_rating=_int_or_none(or_el[0].text_content()) if or_el else None,
            equipment=(_clean(equip_el[0].text_content()).strip("()") or None)
                if equip_el else None,
            colour=colour,
            sex=sex,
            runner_status="NON_RUNNER" if _is_non_runner(row, raw_name) else "RUNNER",
            timeform_race_id=timeform_race_id,
        ))
    return rows
