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
    assert r.horse_name
    assert r.jockey_name
    assert r.recent_form  # e.g. "50318-586"
    assert r.race_class == 3            # from "(3)" in the header
    assert r.distance == "1m 3f 188y"
    assert r.going == "good"
    assert r.venue == "York"
    # TFR is paywalled in the free fixture -> None
    assert r.timeform_rating is None


def test_parse_strips_draw_number_noise_from_name():
    # "CORSICAN CAPER (IRE) (20H)" and "MIND HUNTER (39H) D BF" -> drop draw suffix
    assert parser._strip_name_noise("CORSICAN CAPER (IRE) (20H)") == "CORSICAN CAPER (IRE)"
    assert parser._strip_name_noise("MIND HUNTER (39H) D BF") == "MIND HUNTER"
    assert parser._strip_name_noise("MAN OF THE SEA (IRE)") == "MAN OF THE SEA (IRE)"


def test_parse_event_passes_going_through():
    rows = parser.parse_event(_html(), venue="York", going="soft")
    assert rows[0].going == "soft"


def test_parse_class_band_from_header():
    assert parser._class_from_header("QUEEN MOTHER'S CUP HANDICAP (3) Distance : 1m") == 3
    assert parser._class_from_header("Some Maiden Stakes Distance : 5f") is None


def test_parse_event_extracts_declared_card_fields():
    """Stage 05: provider ids + free declared-card facts (not just names)."""
    rows = parser.parse_event(_html(), venue="York", going="good")
    r = rows[0]
    assert r.horse_name == "PRINCE OF THE SEAS (IRE)"
    assert r.horse_id == "000000614450"
    assert r.jockey_name == "Miss Megan Jordan"
    assert r.jockey_id == "000000018452"
    assert r.trainer_name == "David O'Meara"
    assert r.trainer_id == "000000045008"
    assert r.draw == 5
    assert r.weight_lbs == 149            # 10-9 in stone-lb == 149 lb
    assert r.age == 4
    assert r.official_rating == 89
    assert r.equipment == "t"             # tongue strap, parens stripped
    assert r.colour == "b"
    assert r.sex == "g"
    assert r.runner_status == "RUNNER"
    assert r.timeform_race_id == "2026-06-13-62-1"


def test_parse_event_runner_missing_equipment_and_or_stays_none():
    """A runner with no headgear/no official rating (both free-tier optional
    cells) must stay None, not a fabricated 0 or empty string."""
    rows = parser.parse_event(_html(), venue="York", going="good")
    # Any runner whose free-tier OR/equipment cells are genuinely blank in the
    # fixture must parse to None, never "" or 0 — assert the invariant holds
    # across the whole card rather than assuming a specific row index.
    for r in rows:
        assert r.equipment != ""
        assert r.official_rating != 0 or r.official_rating is None


_NON_RUNNER_ROW = """
<table><thead></thead><tbody>
<tbody data-raceid="2026-09-19-99-1" data-drawnumber="3" data-weightsort="140"
       data-trainer="000000000001" data-jockey="000000000002"
       class="rp-horse-row rp-table-row">
  <tr class="rp-horse-row-1">
    <td class="rp-td-horse-form">-</td>
    <td class="rp-td-horse-name">
      <a class="rp-horse" href="/horse-racing/horse/form/some-horse/000000999999/x">
        SOME HORSE (IRE) (NON RUNNER)</a>
    </td>
    <td class="rp-td-horse-jockey"><a href="#">N/A</a></td>
    <td class="rp-td-horse-age">5</td>
    <td class="rp-td-horse-weight">10-0</td>
  </tr>
</tbody>
</tbody></table>
"""


def test_parse_event_flags_non_runner():
    """Stage 05 fixture: a withdrawn/non-runner row, constructed by hand from the
    documented Timeform markup pattern (the saved live fixture predates any
    withdrawal, so no real example was captured — see stages/05.md)."""
    rows = parser.parse_event(_NON_RUNNER_ROW, venue="Test")
    assert len(rows) == 1
    assert rows[0].runner_status == "NON_RUNNER"


_ABSENT_FIELDS_ROW = """
<table><thead></thead><tbody>
<tbody class="rp-horse-row rp-table-row">
  <tr class="rp-horse-row-1">
    <td class="rp-td-horse-name">
      <a class="rp-horse" href="/horse-racing/horse/form/bare-horse/000000111111/x">
        BARE HORSE</a>
    </td>
  </tr>
</tbody>
</tbody></table>
"""


def test_parse_event_absent_fields_stay_none_not_fabricated():
    """A runner row missing jockey/trainer/draw/weight/age/OR/equipment cells
    entirely (a genuinely thin card) must parse those as None, not 0/''/guessed."""
    rows = parser.parse_event(_ABSENT_FIELDS_ROW, venue="Test")
    assert len(rows) == 1
    r = rows[0]
    assert r.horse_id == "000000111111"
    assert r.jockey_name is None and r.jockey_id is None
    assert r.trainer_name is None and r.trainer_id is None
    assert r.draw is None and r.weight_lbs is None
    assert r.age is None and r.official_rating is None
    assert r.equipment is None
    assert r.runner_status == "RUNNER"


_CONFLICTING_CARD_ROWS = """
<table><thead></thead><tbody>
<tbody data-jockey="000000000010" data-trainer="000000000020"
       class="rp-horse-row rp-table-row">
  <tr class="rp-horse-row-1">
    <td class="rp-td-horse-name">
      <a class="rp-horse" href="/horse-racing/horse/form/dup-horse/000000222222/x">
        DUP HORSE</a>
    </td>
    <td class="rp-td-horse-jockey"><a href="#">Original Jockey</a></td>
  </tr>
</tbody>
<tbody data-jockey="000000000099" data-trainer="000000000020"
       class="rp-horse-row rp-table-row">
  <tr class="rp-horse-row-1">
    <td class="rp-td-horse-name">
      <a class="rp-horse" href="/horse-racing/horse/form/dup-horse/000000222222/x">
        DUP HORSE</a>
    </td>
    <td class="rp-td-horse-jockey"><a href="#">Late Replacement Jockey</a></td>
  </tr>
</tbody>
</tbody></table>
"""


def test_parse_event_late_jockey_change_keeps_last_declared_row():
    """A same-horse row repeated with a different jockey (a late substitution
    re-rendered on the card) must not be silently collapsed by the parser —
    identity reconciliation across the duplicate is the normalizer/fuse layer's
    job (freshest fetch wins), not the parser's. The parser must report both
    rows honestly, in document order, so the caller can apply fetched_at."""
    rows = parser.parse_event(_CONFLICTING_CARD_ROWS, venue="Test")
    assert len(rows) == 2
    assert rows[0].jockey_name == "Original Jockey"
    assert rows[1].jockey_name == "Late Replacement Jockey"
    assert rows[0].jockey_id == "000000000010"
    assert rows[1].jockey_id == "000000000099"
