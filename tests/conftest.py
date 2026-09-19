"""Session-wide test safety net: no test may ever hit api.telegram.org.

Two belts:
  1. Set RP_DISABLE_NOTIFICATIONS so any Notifier built during the suite
     self-disables (see utils.notifications.Notifier.__init__).
  2. Autouse fixture replaces the module singleton with a notifier that has
     no channels, so get_notifier().notify_*() are guaranteed no-ops.
"""

import os

import pytest

# Belt 1: ensure self-disable even before any fixture runs (collection-time).
os.environ.setdefault("RP_DISABLE_NOTIFICATIONS", "1")


@pytest.fixture(autouse=True, scope="session")
def _disable_notifications():
    import utils.notifications as notifications

    # Force a fresh, channel-less singleton (RP_DISABLE_NOTIFICATIONS is set).
    notifications._notifier = None
    notifier = notifications.get_notifier()
    assert notifier._active is False, "notifier must have no active channels under pytest"
    notifications._notifier = notifier
    yield


@pytest.fixture(autouse=True)
def _isolate_source_health(tmp_path, monkeypatch):
    """No test may ever write the real data/source_health.json.

    Scraper/refresh tests that exercise record_success/record_failure
    transitively (not just tests/utils/test_source_health.py directly) used to
    pollute the real file with test-run telemetry. Every test gets its own
    throwaway path.
    """
    import utils.source_health as source_health

    monkeypatch.setattr(source_health, "_PATH", tmp_path / "source_health.json")
    yield


@pytest.fixture(autouse=True)
def _isolate_data_artifacts(tmp_path, monkeypatch):
    """No test may ever write the real data/ artefacts the app serves from.

    Predictor/builder tests that run ``predict()`` / ``build_training_matrix()``
    on their default paths used to overwrite the real ``data/features.parquet``
    and ``data/inference_features.parquet`` with fixture rows, and register a
    fixture payload under the shared cache's ``predictions`` key — which
    ``utils.reporter`` prefers over ``data/predictions.json`` until the TTL
    lapses. Every test gets throwaway paths and a throwaway cache singleton.
    """
    import features.builder as builder
    import models.predictor as predictor
    import utils.alert_dedupe as alert_dedupe
    import utils.cache as cache

    monkeypatch.setattr(builder, "_DEFAULT_FEATURES", str(tmp_path / "features.parquet"))
    monkeypatch.setattr(
        predictor, "_INFERENCE_FEATURES_PATH", tmp_path / "inference_features.parquet"
    )
    monkeypatch.setattr(cache, "_DEFAULT_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cache, "_instance", None)
    # Alert de-dup state: a test that exercises _fire_alerts would otherwise
    # write the real data/cache/sent_alerts.json and suppress the user's own
    # next notification.
    monkeypatch.setattr(alert_dedupe, "_PATH", str(tmp_path / "sent_alerts.json"))
    yield
