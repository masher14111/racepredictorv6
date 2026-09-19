"""Tests for llm/text_features.py — stubbed backend, no live model call."""
import pytest

from llm.text_features import (
    TEXT_FEATURES,
    ExtractionResult,
    FeatureValue,
    OllamaBackend,
    RegexBackend,
    TextFeatureExtractor,
)
from utils.cache import Cache

_FEATURE_NAMES = {f.name for f in TEXT_FEATURES}


class _StubBackend:
    """Records calls; returns a canned detailed result (or None to force fallback)."""

    name = "stub"
    model = "stub-model"

    def __init__(self, result=None):
        self.result = result
        self.calls = 0

    def classify_detailed(self, text):
        self.calls += 1
        return self.result


def _extractor(tmp_path, backend, *, enabled=True):
    return TextFeatureExtractor(
        enabled=enabled,
        backend_impl=backend,
        cache=Cache(cache_dir=tmp_path / "llm", default_ttl=60),
    )


def _full(overrides: dict) -> dict:
    """A complete tri-state result with every other feature unknown."""
    out = {name: FeatureValue(None) for name in _FEATURE_NAMES}
    out.update(overrides)
    return out


# ── no-op / disabled behaviour ───────────────────────────────────────────────

def test_disabled_returns_empty(tmp_path):
    stub = _StubBackend(_full({"trip_trouble": FeatureValue(True, "hampered")}))
    ext = _extractor(tmp_path, stub, enabled=False)
    assert ext.extract("hampered and short of room") == {}
    assert stub.calls == 0  # backend never invoked when disabled


@pytest.mark.parametrize("text", [None, "", "   ", "\n\t"])
def test_blank_text_returns_empty(tmp_path, text):
    stub = _StubBackend(_full({"trip_trouble": FeatureValue(True, "hampered")}))
    ext = _extractor(tmp_path, stub)
    assert ext.extract(text) == {}
    assert stub.calls == 0


# ── schema / caching ─────────────────────────────────────────────────────────

def test_result_has_exactly_schema_keys(tmp_path):
    stub = _StubBackend(_full({"trip_trouble": FeatureValue(True, "hampered")}))
    ext = _extractor(tmp_path, stub)
    out = ext.extract_detailed("some comment")
    assert set(out.features) == _FEATURE_NAMES
    assert out.features["trip_trouble"].value is True
    assert out.features["trip_trouble"].evidence == "hampered"
    assert out.features["ground_excuse"].value is None  # unmentioned -> unknown, not False


def test_second_call_is_cached(tmp_path):
    stub = _StubBackend(_full({"course_winner": FeatureValue(True, "course winner")}))
    ext = _extractor(tmp_path, stub)
    first = ext.extract_detailed("won here last year")
    second = ext.extract_detailed("won here last year")
    assert first == second
    assert stub.calls == 1  # cache hit on the second call


def test_as_bools_collapses_unknown_to_false(tmp_path):
    stub = _StubBackend(_full({"course_winner": FeatureValue(True, "course winner")}))
    ext = _extractor(tmp_path, stub)
    out = ext.extract("won here last year")
    assert out["course_winner"] is True
    assert out["ground_excuse"] is False  # legacy view: unknown -> False
    assert all(isinstance(v, bool) for v in out.values())


# ── fallback path + correctly labelled fallback cache records ───────────────

def test_none_result_falls_back_to_regex_and_is_labelled(tmp_path):
    stub = _StubBackend(None)  # simulates Ollama down
    stub.name = "ollama"
    ext = _extractor(tmp_path, stub)
    out = ext.extract_detailed("badly hampered and short of room two out")
    assert stub.calls == 1
    assert out.features["trip_trouble"].value is True  # regex fallback caught it
    assert out.backend == "ollama_fallback_regex"
    assert out.is_fallback is True
    assert out.failure_reason == "ollama_backend_failed"

    # The cache record itself must carry the same honest label, not the
    # configured backend's name — a later working Ollama call must not read
    # back a regex result pretending to be an Ollama one.
    second = ext.extract_detailed("badly hampered and short of room two out")
    assert stub.calls == 1  # served from the (correctly labelled) fallback cache
    assert second.backend == "ollama_fallback_regex"
    assert second.is_fallback is True


def test_regex_configured_backend_is_never_labelled_fallback(tmp_path):
    rb = RegexBackend()
    ext = _extractor(tmp_path, rb)
    out = ext.extract_detailed("hampered and short of room")
    assert out.backend == "regex"
    assert out.is_fallback is False
    assert out.failure_reason is None


def test_schema_failure_malformed_backend_output_falls_back(tmp_path):
    # Every feature entry is an unusable type -> whole response rejected.
    class _BadBackend:
        name = "ollama"
        model = "bad-model"

        def classify_detailed(self, text):
            return None  # malformed upstream JSON already converted to failure

    ext = _extractor(tmp_path, _BadBackend())
    out = ext.extract_detailed("hampered")
    assert out.backend == "ollama_fallback_regex"
    assert out.failure_reason == "ollama_backend_failed"
    assert out.features["trip_trouble"].value is True


# ── regex backend correctness: true / false / unknown / negation ───────────

def test_regex_backend_true_strings():
    rb = RegexBackend()
    out = rb.classify_detailed("Needs better ground; stamina in doubt over this trip.")
    assert out["ground_excuse"].value is True
    assert out["ground_excuse"].evidence
    assert out["distance_excuse"].value is True
    assert out["trip_trouble"].value is False  # not mentioned -> regex reports False
    assert set(out) == _FEATURE_NAMES


def test_regex_backend_false_on_neutral_text():
    rb = RegexBackend()
    out = rb.classify_detailed("Won well last time and looks fairly treated here.")
    assert all(fv.value is False for fv in out.values())


def test_regex_backend_ground_excuse_negation_not_misread():
    """The bug this stage repairs: 'suited by the soft ground' is NOT an excuse."""
    rb = RegexBackend()
    out = rb.classify_detailed("A confirmed mudlark, well suited by the soft ground today.")
    assert out["ground_excuse"].value is False
    assert "suited by the soft ground" in out["ground_excuse"].evidence.lower()


def test_regex_backend_ground_excuse_true_on_genuine_negation():
    rb = RegexBackend()
    out = rb.classify_detailed("Struggled last time, clearly unsuited by the soft ground.")
    assert out["ground_excuse"].value is True
    assert "unsuited by the soft ground" in out["ground_excuse"].evidence.lower()

    out2 = rb.classify_detailed("Not suited by the soft ground on this evidence.")
    assert out2["ground_excuse"].value is True


def test_regex_backend_contradictory_comment_negation_wins():
    """A comment that both names the ground and explicitly denies the excuse."""
    rb = RegexBackend()
    text = "Ground was testing but this sort is well suited by the soft ground."
    out = rb.classify_detailed(text)
    assert out["ground_excuse"].value is False


def test_regex_backend_unmentioned_feature_has_no_evidence():
    rb = RegexBackend()
    out = rb.classify_detailed("Course winner last time out.")
    assert out["course_winner"].value is True
    assert out["headgear_first_time"].value is False
    assert out["headgear_first_time"].evidence is None


# ── Ollama backend: grounding + unknown handling ─────────────────────────────

def test_ollama_backend_returns_none_without_server(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("blocked in test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert OllamaBackend().classify_detailed("hampered") is None


def test_ollama_backend_rejects_ungrounded_true_claim(monkeypatch):
    backend = OllamaBackend()
    payload = {name: {"value": False, "evidence": None} for name in _FEATURE_NAMES}
    payload["trip_trouble"] = {"value": True, "evidence": "this phrase is not in the text"}
    monkeypatch.setattr(backend, "_call", lambda text: payload)
    out = backend.classify_detailed("Won easily with no trouble at all.")
    # Claimed true but evidence isn't a substring of the real text -> unknown, not True.
    assert out["trip_trouble"].value is None


def test_ollama_backend_accepts_grounded_true_claim(monkeypatch):
    backend = OllamaBackend()
    payload = {name: {"value": False, "evidence": None} for name in _FEATURE_NAMES}
    payload["trip_trouble"] = {"value": True, "evidence": "badly hampered"}
    monkeypatch.setattr(backend, "_call", lambda text: payload)
    out = backend.classify_detailed("Travelled well but was badly hampered two out.")
    assert out["trip_trouble"].value is True
    assert out["trip_trouble"].evidence == "badly hampered"


def test_ollama_backend_honors_explicit_unknown(monkeypatch):
    backend = OllamaBackend()
    payload = {name: {"value": "unknown", "evidence": None} for name in _FEATURE_NAMES}
    monkeypatch.setattr(backend, "_call", lambda text: payload)
    out = backend.classify_detailed("A short, uninformative comment.")
    assert all(fv.value is None for fv in out.values())


def test_ollama_backend_falls_back_when_every_entry_malformed(monkeypatch):
    backend = OllamaBackend()
    monkeypatch.setattr(backend, "_call", lambda text: {"unexpected_key": 123})
    assert backend.classify_detailed("hampered") is None


# ── ExtractionResult round-trip (cache serialisation) ────────────────────────

def test_extraction_result_roundtrips_through_dict():
    result = ExtractionResult(
        features={"trip_trouble": FeatureValue(True, "hampered"), "ground_excuse": FeatureValue(None)},
        backend="ollama_fallback_regex", model="", schema_version=2, prompt_version=1,
        is_fallback=True, failure_reason="ollama_backend_failed",
    )
    rebuilt = ExtractionResult.from_dict(result.to_dict())
    assert rebuilt == result
