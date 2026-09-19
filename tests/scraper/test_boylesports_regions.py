"""BoyleSports region filtering.

With uk_ire_only, every foreign event fetched is a full proxy request against a
Cloudflare-fronted site whose rows are binned at normalize — so the index filter
is what stops the scraper paying for Churchill Downs to throw it away.

The allowlist must be fail-safe: if BoyleSports renames a slug it has to fall
back rather than silently return an empty card, which is indistinguishable from
a total scrape failure.
"""

import pytest

import scraper.boylesports as bs


def _ev(region: str, course: str = "x") -> dict:
    return {"race_id": f"{region}-{course}", "event_url": f"/{region}/{course}",
            "venue": course.title(), "race_time": "", "region": region,
            "resulted": False}


EVENTS = [
    _ev("uk-ire-featured", "dundalk"),
    _ev("uk-ire-featured", "wolverhampton"),
    _ev("usa", "churchill-downs"),
    _ev("usa", "gulfstream"),
    _ev("virtuals", "sprintvalley"),
]


@pytest.fixture
def regions(monkeypatch):
    def _set(include=(), exclude=("virtuals",)):
        monkeypatch.setattr(bs, "_INCLUDED_REGIONS", {r.lower() for r in include})
        monkeypatch.setattr(bs, "_EXCLUDED_REGIONS", {r.lower() for r in exclude})
    return _set


def test_allowlist_keeps_only_uk_ire(regions):
    regions(include=["uk-ire-featured"])
    kept = bs._filter_regions(EVENTS)
    assert [e["venue"] for e in kept] == ["Dundalk", "Wolverhampton"]


def test_allowlist_drops_the_us_cards_that_wasted_requests(regions):
    regions(include=["uk-ire-featured"])
    kept = {e["region"] for e in bs._filter_regions(EVENTS)}
    assert "usa" not in kept


def test_allowlist_is_case_insensitive(regions):
    regions(include=["UK-IRE-FEATURED"])
    assert len(bs._filter_regions([_ev("uk-ire-featured", "ayr")])) == 1


def test_stale_allowlist_falls_back_instead_of_emptying_the_card(regions, caplog):
    """A renamed slug must not look like a total scrape failure."""
    regions(include=["uk-ire-oldname"], exclude=["virtuals"])
    kept = bs._filter_regions(EVENTS)
    assert len(kept) == 4                      # everything except virtuals
    assert {e["region"] for e in kept} == {"uk-ire-featured", "usa"}
    assert "falling back" in caplog.text
    assert "uk-ire-featured" in caplog.text    # logs the slugs actually seen


def test_blacklist_used_when_no_allowlist_configured(regions):
    regions(include=[], exclude=["virtuals", "usa"])
    kept = {e["region"] for e in bs._filter_regions(EVENTS)}
    assert kept == {"uk-ire-featured"}


def test_empty_index_passes_through(regions):
    regions(include=["uk-ire-featured"])
    assert bs._filter_regions([]) == []


def test_allowlist_matching_everything_keeps_everything(regions):
    regions(include=["uk-ire-featured", "usa", "virtuals"])
    assert len(bs._filter_regions(EVENTS)) == 5
