"""Chromium discovery for the Botasaurus tier.

Botasaurus raises FileNotFoundError when no real Google Chrome is installed. This
machine has none, so the tier has to fall back to the Chromium Playwright already
ships — otherwise the one path that actually clears Cloudflare never starts.
"""

import os

import pytest

import scraper.boylesports as bs


@pytest.fixture
def fake_fs(monkeypatch, tmp_path):
    """Control both os.path.exists and the Playwright install root."""
    present: set[str] = set()

    monkeypatch.setattr(bs.os.path, "exists", lambda p: str(p) in present)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    def _add(*paths):
        present.update(str(p) for p in paths)

    def _playwright(version="chromium-1223", sub="chrome-win64"):
        exe = tmp_path / "ms-playwright" / version / sub / "chrome.exe"
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_text("")
        _add(exe)
        return str(exe)

    return type("FS", (), {"add": staticmethod(_add),
                           "playwright": staticmethod(_playwright)})


def test_falls_back_to_playwright_chromium(fake_fs, monkeypatch):
    monkeypatch.setattr(bs, "_BS_CFG", {})
    exe = fake_fs.playwright()
    assert bs._resolve_chrome_path() == exe


def test_real_chrome_wins_and_defers_to_autodetect(fake_fs, monkeypatch):
    """With real Chrome present, return None so Botasaurus picks it up itself —
    real Chrome fingerprints better than bare Chromium."""
    monkeypatch.setattr(bs, "_BS_CFG", {})
    fake_fs.add(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    fake_fs.playwright()
    assert bs._resolve_chrome_path() is None


def test_configured_path_overrides_everything(fake_fs, monkeypatch, tmp_path):
    custom = tmp_path / "my-chrome.exe"
    custom.write_text("")
    fake_fs.add(custom)
    fake_fs.playwright()
    monkeypatch.setattr(bs, "_BS_CFG", {"botasaurus_chrome_path": str(custom)})
    assert bs._resolve_chrome_path() == str(custom)


def test_configured_but_missing_path_is_ignored(fake_fs, monkeypatch):
    monkeypatch.setattr(bs, "_BS_CFG", {"botasaurus_chrome_path": r"C:\nope\chrome.exe"})
    exe = fake_fs.playwright()
    assert bs._resolve_chrome_path() == exe


def test_newest_playwright_build_is_preferred(fake_fs, monkeypatch):
    monkeypatch.setattr(bs, "_BS_CFG", {})
    fake_fs.playwright("chromium-1100")
    newest = fake_fs.playwright("chromium-1223")
    assert bs._resolve_chrome_path() == newest


def test_legacy_chrome_win_layout_is_found(fake_fs, monkeypatch):
    monkeypatch.setattr(bs, "_BS_CFG", {})
    exe = fake_fs.playwright(sub="chrome-win")
    assert bs._resolve_chrome_path() == exe


def test_nothing_found_returns_none(fake_fs, monkeypatch):
    """None is safe: Botasaurus then auto-detects and raises its own clear error,
    which the tier catches and skips."""
    monkeypatch.setattr(bs, "_BS_CFG", {})
    assert bs._resolve_chrome_path() is None


def test_missing_localappdata_does_not_raise(monkeypatch):
    monkeypatch.setattr(bs, "_BS_CFG", {})
    monkeypatch.setattr(bs.os.path, "exists", lambda p: False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert bs._resolve_chrome_path() is None
