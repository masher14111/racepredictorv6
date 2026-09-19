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
