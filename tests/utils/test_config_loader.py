import copy
import os
import time

import pytest
import yaml

from utils.config_loader import _validate, ConfigError

VALID_CFG = {
    "database_path": "data/races.db",
    "scrape_interval": 3600,
    "each_way_threshold": 8.0,
    "model": {
        "test_size": 0.2,
        "cv_folds": 5,
        "optuna_trials": 50,
        "iterations": 1000,
    },
    "notifications": {"enabled": False},
    "proxy_pool": {"enabled": False, "proxies": []},
}


# ---------------------------------------------------------------------------
# Task 1: _validate()
# ---------------------------------------------------------------------------


def test_validate_valid_config():
    assert _validate(VALID_CFG) == []


def test_validate_missing_database_path():
    cfg = {**VALID_CFG, "database_path": ""}
    errors = _validate(cfg)
    assert any("database_path" in e for e in errors)


def test_validate_scrape_interval_zero():
    cfg = {**VALID_CFG, "scrape_interval": 0}
    errors = _validate(cfg)
    assert any("scrape_interval" in e for e in errors)


def test_validate_test_size_out_of_range():
    cfg = copy.deepcopy(VALID_CFG)
    cfg["model"]["test_size"] = 2.0
    errors = _validate(cfg)
    assert any("test_size" in e for e in errors)


def test_validate_cv_folds_too_low():
    cfg = copy.deepcopy(VALID_CFG)
    cfg["model"]["cv_folds"] = 1
    errors = _validate(cfg)
    assert any("cv_folds" in e for e in errors)


def test_validate_optuna_trials_zero():
    cfg = copy.deepcopy(VALID_CFG)
    cfg["model"]["optuna_trials"] = 0
    errors = _validate(cfg)
    assert any("optuna_trials" in e for e in errors)


def test_validate_iterations_zero():
    cfg = copy.deepcopy(VALID_CFG)
    cfg["model"]["iterations"] = 0
    errors = _validate(cfg)
    assert any("iterations" in e for e in errors)


def test_validate_notifications_enabled_missing_token():
    cfg = copy.deepcopy(VALID_CFG)
    cfg["notifications"] = {
        "enabled": True,
        "channels": {"telegram": {"bot_token": "", "chat_id": "123"}},
    }
    errors = _validate(cfg)
    assert any("bot_token" in e for e in errors)


def test_validate_notifications_enabled_missing_chat_id():
    cfg = copy.deepcopy(VALID_CFG)
    cfg["notifications"] = {
        "enabled": True,
        "channels": {"telegram": {"bot_token": "tok", "chat_id": ""}},
    }
    errors = _validate(cfg)
    assert any("chat_id" in e for e in errors)


def test_validate_notifications_disabled_no_token_ok():
    cfg = copy.deepcopy(VALID_CFG)
    cfg["notifications"] = {"enabled": False}
    assert _validate(cfg) == []


# ---------------------------------------------------------------------------
# Task 2: _apply_env_overrides()
# ---------------------------------------------------------------------------

from utils.config_loader import _apply_env_overrides


def test_env_override_top_level_int(monkeypatch):
    monkeypatch.setenv("RP_SCRAPE_INTERVAL", "999")
    cfg = copy.deepcopy(VALID_CFG)
    _apply_env_overrides(cfg)
    assert cfg["scrape_interval"] == 999


def test_env_override_nested_float(monkeypatch):
    monkeypatch.setenv("RP_MODEL__TEST_SIZE", "0.3")
    cfg = copy.deepcopy(VALID_CFG)
    _apply_env_overrides(cfg)
    assert cfg["model"]["test_size"] == pytest.approx(0.3)


def test_env_override_bool_false(monkeypatch):
    monkeypatch.setenv("RP_NOTIFICATIONS__ENABLED", "false")
    cfg = copy.deepcopy(VALID_CFG)
    cfg["notifications"] = {"enabled": True}
    _apply_env_overrides(cfg)
    assert cfg["notifications"]["enabled"] is False


def test_env_override_bool_true(monkeypatch):
    monkeypatch.setenv("RP_NOTIFICATIONS__ENABLED", "1")
    cfg = copy.deepcopy(VALID_CFG)
    cfg["notifications"] = {"enabled": False}
    _apply_env_overrides(cfg)
    assert cfg["notifications"]["enabled"] is True


def test_env_override_list(monkeypatch):
    monkeypatch.setenv("RP_MODEL__TARGETS", "won,placed_2")
    cfg = copy.deepcopy(VALID_CFG)
    cfg["model"]["targets"] = ["won", "placed_2", "showed"]
    _apply_env_overrides(cfg)
    assert cfg["model"]["targets"] == ["won", "placed_2"]


def test_env_override_unknown_path_ignored(monkeypatch, caplog):
    import logging

    monkeypatch.setenv("RP_NONEXISTENT__KEY", "value")
    cfg = copy.deepcopy(VALID_CFG)
    with caplog.at_level(logging.WARNING, logger="config"):
        _apply_env_overrides(cfg)
    assert any("NONEXISTENT__KEY" in r.message for r in caplog.records)


def test_env_override_bad_cast_ignored(monkeypatch, caplog):
    import logging

    monkeypatch.setenv("RP_SCRAPE_INTERVAL", "not_a_number")
    cfg = copy.deepcopy(VALID_CFG)
    with caplog.at_level(logging.WARNING, logger="config"):
        _apply_env_overrides(cfg)
    assert cfg["scrape_interval"] == 3600  # unchanged


# ---------------------------------------------------------------------------
# Task 3 & 4: ConfigLoader singleton
# ---------------------------------------------------------------------------

import utils.config_loader as cl_module
from utils.config_loader import get_config, get, register_callback, start_watching


@pytest.fixture(autouse=True)
def reset_singleton():
    cl_module._loader = None
    yield
    cl_module._loader = None


def _write_config(tmp_path, data: dict):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")
    return p


def test_get_config_returns_dict(tmp_path, monkeypatch):
    p = _write_config(tmp_path, VALID_CFG)
    monkeypatch.setattr(cl_module, "_CONFIG_PATH", str(p))
    cfg = get_config()
    assert isinstance(cfg, dict)
    assert cfg["scrape_interval"] == 3600


def test_get_config_raises_on_invalid_startup(tmp_path, monkeypatch):
    bad = copy.deepcopy(VALID_CFG)
    bad["scrape_interval"] = -1
    p = _write_config(tmp_path, bad)
    monkeypatch.setattr(cl_module, "_CONFIG_PATH", str(p))
    with pytest.raises(ConfigError):
        get_config()


def test_get_dotted_key(tmp_path, monkeypatch):
    p = _write_config(tmp_path, VALID_CFG)
    monkeypatch.setattr(cl_module, "_CONFIG_PATH", str(p))
    assert get("model.test_size") == pytest.approx(0.2)


def test_get_missing_key_returns_default(tmp_path, monkeypatch):
    p = _write_config(tmp_path, VALID_CFG)
    monkeypatch.setattr(cl_module, "_CONFIG_PATH", str(p))
    assert get("model.nonexistent", default="fallback") == "fallback"


def test_mtime_unchanged_no_reparse(tmp_path, monkeypatch):
    p = _write_config(tmp_path, VALID_CFG)
    monkeypatch.setattr(cl_module, "_CONFIG_PATH", str(p))
    cfg1 = get_config()
    cfg2 = get_config()
    assert cfg1 is cfg2


def test_mtime_changed_valid_reloads(tmp_path, monkeypatch):
    p = _write_config(tmp_path, VALID_CFG)
    monkeypatch.setattr(cl_module, "_CONFIG_PATH", str(p))
    get_config()

    updated = copy.deepcopy(VALID_CFG)
    updated["scrape_interval"] = 7200
    p.write_text(yaml.dump(updated), encoding="utf-8")
    os.utime(str(p), (time.time() + 1, time.time() + 1))

    cfg2 = get_config()
    assert cfg2["scrape_interval"] == 7200


def test_mtime_changed_invalid_keeps_previous(tmp_path, monkeypatch, caplog):
    import logging

    p = _write_config(tmp_path, VALID_CFG)
    monkeypatch.setattr(cl_module, "_CONFIG_PATH", str(p))
    get_config()

    bad = copy.deepcopy(VALID_CFG)
    bad["scrape_interval"] = -1
    p.write_text(yaml.dump(bad), encoding="utf-8")
    os.utime(str(p), (time.time() + 2, time.time() + 2))

    with caplog.at_level(logging.ERROR, logger="config"):
        cfg2 = get_config()

    assert cfg2["scrape_interval"] == 3600
    assert any("scrape_interval" in r.message for r in caplog.records)


def test_register_callback_called_on_reload(tmp_path, monkeypatch):
    p = _write_config(tmp_path, VALID_CFG)
    monkeypatch.setattr(cl_module, "_CONFIG_PATH", str(p))

    calls = []
    register_callback(lambda cfg: calls.append(cfg["scrape_interval"]))

    updated = copy.deepcopy(VALID_CFG)
    updated["scrape_interval"] = 9999
    p.write_text(yaml.dump(updated), encoding="utf-8")
    os.utime(str(p), (time.time() + 1, time.time() + 1))

    get_config()
    assert calls == [9999]


def test_register_callback_not_duplicated(tmp_path, monkeypatch):
    p = _write_config(tmp_path, VALID_CFG)
    monkeypatch.setattr(cl_module, "_CONFIG_PATH", str(p))
    fn = lambda cfg: None
    register_callback(fn)
    register_callback(fn)
    assert cl_module._get_loader()._callbacks.count(fn) == 1


def test_start_watching_idempotent(tmp_path, monkeypatch):
    p = _write_config(tmp_path, VALID_CFG)
    monkeypatch.setattr(cl_module, "_CONFIG_PATH", str(p))
    start_watching()
    start_watching()
    assert cl_module._get_loader()._watch_started is True


def test_start_watching_without_watchdog(tmp_path, monkeypatch, caplog):
    import builtins
    import logging

    p = _write_config(tmp_path, VALID_CFG)
    monkeypatch.setattr(cl_module, "_CONFIG_PATH", str(p))

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name.startswith("watchdog"):
            raise ImportError("watchdog not installed")
        return real_import(name, *args, **kwargs)

    with caplog.at_level(logging.WARNING, logger="config"):
        monkeypatch.setattr(builtins, "__import__", mock_import)
        start_watching()

    assert any("watchdog" in r.message for r in caplog.records)
    assert cl_module._get_loader()._watch_started is False
