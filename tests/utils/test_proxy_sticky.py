"""Sticky-session proxy URLs.

A malformed token here fails silently — DataImpulse just ignores it and keeps
rotating, so the scraper would look like it had a sticky session while every
request still came from a new IP. These tests pin the exact wire format:
``login__cr.ie;sessid.<id>;sessttl.<n>`` — `__` introduces the first token, `;`
chains the rest.
"""

import pytest

from utils.proxy_manager import ProxyManager

BASE = "http://user123:secret@gw.dataimpulse.com:823"
WITH_CC = "http://user123__cr.ie:secret@gw.dataimpulse.com:823"


def _user(url: str) -> str:
    return url.split("//", 1)[1].split(":", 1)[0]


# ---------------------------------------------------------------------------
# Token format
# ---------------------------------------------------------------------------
def test_sessid_chains_onto_an_existing_country_token():
    out = ProxyManager._apply_session(WITH_CC, "run1")
    assert _user(out) == "user123__cr.ie;sessid.run1"


def test_sessid_introduces_itself_with_dunder_when_first():
    out = ProxyManager._apply_session(BASE, "run1")
    assert _user(out) == "user123__sessid.run1"


def test_ttl_is_appended_with_a_semicolon():
    out = ProxyManager._apply_session(WITH_CC, "run1", ttl_minutes=10)
    assert _user(out) == "user123__cr.ie;sessid.run1;sessttl.10"


def test_password_host_and_port_survive_untouched():
    out = ProxyManager._apply_session(WITH_CC, "run1", 10)
    assert out.endswith(":secret@gw.dataimpulse.com:823")
    assert out.startswith("http://")


# ---------------------------------------------------------------------------
# Idempotence — a re-applied session must not stack tokens
# ---------------------------------------------------------------------------
def test_reapplying_replaces_rather_than_stacks():
    once = ProxyManager._apply_session(WITH_CC, "run1", 10)
    twice = ProxyManager._apply_session(once, "run2", 20)
    assert _user(twice) == "user123__cr.ie;sessid.run2;sessttl.20"
    assert twice.count("sessid") == 1
    assert twice.count("sessttl") == 1


def test_country_is_still_applied_after_a_session():
    """_apply_country strips only its own token, so the two compose either way."""
    sticky = ProxyManager._apply_session(BASE, "run1")
    out = ProxyManager._apply_country(sticky, "ie")
    assert "sessid.run1" in out and "__cr.ie" in out


# ---------------------------------------------------------------------------
# Hardening
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [("a b", "ab"), ("run/1", "run1"), ("x;y", "xy"), ("p:q", "pq"), ("R2D2", "R2D2")],
)
def test_session_ids_are_sanitised(raw, expected):
    """A stray ';' or ':' in the id would corrupt the username or the URL."""
    assert f"sessid.{expected}" in ProxyManager._apply_session(WITH_CC, raw)


def test_empty_session_id_falls_back_to_a_placeholder():
    assert "sessid.rp" in ProxyManager._apply_session(WITH_CC, "!!!")


def test_url_without_credentials_is_left_alone():
    plain = "http://gw.dataimpulse.com:823"
    assert ProxyManager._apply_session(plain, "run1") == plain


def test_same_id_gives_a_stable_url():
    """DataImpulse returns the same IP for the same sessid — so the URL must be
    byte-identical across calls within a run."""
    a = ProxyManager._apply_session(WITH_CC, "run1", 10)
    b = ProxyManager._apply_session(WITH_CC, "run1", 10)
    assert a == b


def test_different_ids_give_different_urls():
    a = ProxyManager._apply_session(WITH_CC, "run1")
    b = ProxyManager._apply_session(WITH_CC, "run2")
    assert a != b
