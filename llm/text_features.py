"""Free-text → structured, evidence-grounded features for race commentary.

Pipeline position
-----------------
This module is **strictly off the probability hot path**. It maps free text
(spotlight / in-running comments / going notes) to a fixed set of tri-state
"excuse / angle" flags. It produces no win probability and no EV, and nothing
here is read by ``models.predictor`` or listed in
``models.features.FEATURE_COLS``. It exists so that, once evaluated
(step 15/16), these flags can be joined in deliberately.

Tri-state schema
-----------------
Every feature resolves to a :class:`FeatureValue` with ``value`` in
``{True, False, None}`` plus an ``evidence`` phrase:

* ``True``  — the text explicitly supports the claim; ``evidence`` is the
  exact phrase (a real substring of the input) that grounds it.
* ``False`` — the text explicitly contradicts the claim, or (regex backend)
  no supporting phrase was found. ``evidence`` may be the contradicting
  phrase or ``None``.
* ``None``  (unknown) — the text simply does not address the claim, or a
  backend's claim could not be grounded in the supplied text. **Unknown is
  never coerced to False** — "not mentioned" and "explicitly absent" are kept
  distinct so a downstream consumer cannot silently read "no evidence found"
  as "positively ruled out".

An LLM backend result whose claimed ``evidence`` is not an actual substring of
the input text (case-insensitive) is rejected: that feature is forced to
unknown with a recorded ``failure_reason`` rather than trusted ungrounded.

Backends
--------
* ``OllamaBackend`` — calls a local Ollama server (``/api/generate`` with
  ``format=json``). Any failure (server down, bad JSON, ungrounded/malformed
  schema) returns ``None`` so the caller falls back to regex. No network call
  in CI.
* ``RegexBackend`` — deterministic keyword/phrase matching. Always returns a
  full result (never unknown for regex); used as the fallback and as the
  default when ``llm.backend`` is not ``"ollama"``.

Results are disk-cached, keyed on text hash + the ACTUAL backend/model that
produced them (never the configured one — a fallback is labelled as a
fallback, not silently attributed to the backend that failed) + prompt/schema
version, via ``utils.cache.Cache``.

Configuration (all optional; absent ``llm`` section => disabled / no-op)::

    llm:
      enabled: false
      backend: regex            # "ollama" | "regex"
      cache_ttl: 2592000        # seconds (30 days)
      ollama:
        host: http://localhost:11434
        model: llama3.1
        timeout: 30
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

from utils.cache import Cache
from utils.logger import get_logger

_log = get_logger(__name__)

_BASE = Path(__file__).resolve().parents[1]
_CACHE_DIR = _BASE / "data" / "cache" / "llm"

# Bump when the feature set or their meaning changes — invalidates the cache.
SCHEMA_VERSION = 2
# Bump when the LLM prompt wording changes (the deterministic regex behaviour
# is versioned by SCHEMA_VERSION alone since it has no prompt).
PROMPT_VERSION = 1


# ── feature schema ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TextFeature:
    """One tri-state feature derivable from race text."""

    name: str
    description: str  # used in the LLM prompt
    patterns: tuple[str, ...] = field(default=())  # regex fallback: True evidence
    negative_patterns: tuple[str, ...] = field(default=())  # explicit False evidence


TEXT_FEATURES: tuple[TextFeature, ...] = (
    TextFeature(
        "ground_excuse",
        "the going/ground did not suit the horse, or it wants different ground "
        "(NOT true merely because ground is mentioned — only when the text says "
        "the ground is or was a problem for this horse)",
        (
            r"\bunsuited by the (?:soft|heavy|firm|good) ground\b",
            r"\bnot suited by the (?:soft|heavy|firm|good) ground\b",
            r"\bdoesn'?t (?:appreciate|handle|act on) (?:the )?(?:soft|heavy|firm|good) ground\b",
            r"\bdid not (?:appreciate|handle|act on) (?:the )?(?:soft|heavy|firm|good) ground\b",
            r"\bnever (?:really )?(?:took to|handled|acted on) the ground\b",
            r"\bneed(?:s|ed)? (?:better|faster|softer|easier) ground\b",
            r"\bprefer(?:s|red)? (?:faster|softer|firmer) ground\b",
            r"\bground (?:against|too (?:soft|firm|quick|slow))\b",
            r"\bnot at (?:his|her|its) best on the (?:soft|heavy|firm) ground\b",
        ),
        # These explicitly say the ground SUITS the horse — the exact phrase the
        # unrepaired pattern used to misread as an excuse via an optional "un".
        # The negative lookbehind is what the original bug lacked: without it,
        # "not suited by the ground" would (wrongly) match here too, since the
        # literal substring "suited by the ... ground" is present in both a
        # positive statement and its negation.
        (
            r"(?<!not )(?<!never )\bsuited by the (?:soft|heavy|firm|good) ground\b",
            r"\bwell suited by (?:the|this) ground\b",
            r"\bappreciat(?:es|ed) the (?:soft|heavy|firm|good) ground\b",
            r"\bat (?:his|her|its) best on the (?:soft|heavy|firm|good) ground\b",
        ),
    ),
    TextFeature(
        "distance_excuse",
        "the race distance/trip was too short or too long, or stamina was in doubt",
        (
            r"\bstamina (?:in doubt|may be stretched|to prove)\b",
            r"\bneed(?:s|ed)? (?:further|a (?:longer|shorter) trip)\b",
            r"\btrip (?:too (?:short|sharp|long)|may (?:be|prove) (?:inadequate|too far))\b",
            r"\bmay (?:want|prefer) (?:shorter|further|a stiffer test)\b",
            r"\bdidn'?t (?:get|stay) the trip\b",
            r"\bdid not (?:get|stay) the trip\b",
        ),
        (
            r"\bstays? the trip (?:well|comfortably|with ease)\b",
            r"\bno stamina (?:doubts|concerns)\b",
            r"\bstamina is not in doubt\b",
        ),
    ),
    TextFeature(
        "trip_trouble",
        "the horse met in-running trouble: hampered, no clear run, short of room, "
        "stumbled, badly drawn, or lost its action",
        (
            r"\bhampered\b",
            r"\bno (?:clear )?run\b",
            r"\bshort of room\b",
            r"\b(?:badly|poorly) drawn\b",
            r"\bstumbled\b",
            r"\blost (?:his|her|its) action\b",
            r"\bsqueezed (?:up|out)\b",
            r"\bchecked\b",
        ),
        (
            r"\bno (?:trouble|excuses) in running\b",
            r"\btrouble[- ]free (?:run|trip)\b",
            r"\bmet no trouble\b",
        ),
    ),
    TextFeature(
        "headgear_first_time",
        "the horse wears headgear (blinkers, cheekpieces, hood, visor) for the first time",
        (
            r"\b(?:blinkers|cheekpieces|hood|visor|tongue tie)\b[^.]*\bfirst time\b",
            r"\bfirst[- ]time\b[^.]*\b(?:blinkers|cheekpieces|hood|visor)\b",
            r"\bwears? (?:a )?(?:hood|visor|blinkers|cheekpieces) for the first time\b",
        ),
        (),
    ),
    TextFeature(
        "returning_from_layoff",
        "the horse is returning from an absence, layoff or long break",
        (
            r"\boff (?:the track )?since\b",
            r"\breturn(?:s|ing)? (?:from|after) (?:a|an) (?:absence|break|layoff|lay-off)\b",
            r"\b(?:long )?absence\b",
            r"\bfresh(?:ened)? up\b",
            r"\bafter (?:a )?(?:break|wind (?:op|surgery))\b",
        ),
        (),
    ),
    TextFeature(
        "course_winner",
        "the horse has previously won at this course (course or course-and-distance winner)",
        (
            r"\bcourse(?:[- ]and[- ]distance)? winner\b",
            r"\bC ?& ?D winner\b",
            r"\bwon (?:here|at this (?:course|track))\b",
            r"\bcourse specialist\b",
        ),
        (
            r"\bcourse form (?:is )?(?:poor|unproven)\b",
            r"\byet to win (?:here|at this course)\b",
        ),
    ),
)

_FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in TEXT_FEATURES)


# ── value objects ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeatureValue:
    """One feature's tri-state result plus the phrase that grounds it.

    ``evidence`` is ``None`` for unknown, and for a regex ``False`` where no
    negative phrase specifically matched (there is simply nothing to quote).
    """

    value: Optional[bool]
    evidence: Optional[str] = None

    def to_dict(self) -> dict:
        return {"value": self.value, "evidence": self.evidence}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureValue":
        return cls(value=d.get("value"), evidence=d.get("evidence"))


_UNKNOWN: dict[str, FeatureValue] = {name: FeatureValue(None) for name in _FEATURE_NAMES}


@dataclass(frozen=True)
class ExtractionResult:
    """Full, auditable outcome of one ``extract_detailed`` call.

    ``backend`` and ``model`` always describe what ACTUALLY produced
    ``features`` — never the configured backend if it failed and a fallback
    ran. ``is_fallback`` and ``failure_reason`` make that failure visible
    rather than silently absorbed.
    """

    features: dict[str, FeatureValue]
    backend: str
    model: str
    schema_version: int
    prompt_version: int
    is_fallback: bool = False
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "features": {k: v.to_dict() for k, v in self.features.items()},
            "backend": self.backend,
            "model": self.model,
            "schema_version": self.schema_version,
            "prompt_version": self.prompt_version,
            "is_fallback": self.is_fallback,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractionResult":
        return cls(
            features={k: FeatureValue.from_dict(v) for k, v in d.get("features", {}).items()},
            backend=d.get("backend", "unknown"),
            model=d.get("model", ""),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            prompt_version=d.get("prompt_version", PROMPT_VERSION),
            is_fallback=bool(d.get("is_fallback", False)),
            failure_reason=d.get("failure_reason"),
        )

    def as_bools(self) -> dict[str, bool]:
        """Legacy view: unknown collapses to False. Prefer ``features`` directly."""
        return {name: bool(fv.value) for name, fv in self.features.items()}


def _disabled_result() -> ExtractionResult:
    return ExtractionResult(
        features=dict(_UNKNOWN), backend="disabled", model="",
        schema_version=SCHEMA_VERSION, prompt_version=PROMPT_VERSION,
        is_fallback=False, failure_reason="llm_disabled",
    )


def _empty_text_result() -> ExtractionResult:
    return ExtractionResult(
        features=dict(_UNKNOWN), backend="none", model="",
        schema_version=SCHEMA_VERSION, prompt_version=PROMPT_VERSION,
        is_fallback=False, failure_reason="empty_text",
    )


# ── backends ────────────────────────────────────────────────────────────────

class Backend(Protocol):
    name: str

    def classify_detailed(self, text: str) -> Optional[dict[str, FeatureValue]]:
        """Return {feature: FeatureValue} for all features, or None on failure."""
        ...


class RegexBackend:
    """Deterministic phrase matching. Always returns a complete result.

    Never returns unknown: absence of any matching phrase (positive or
    negative) means the regex backend has no opinion and reports False with
    no evidence — that is what "no keyword found" honestly means for a
    fixed-pattern matcher. Genuine unknown-vs-false is the LLM backend's job.
    """

    name = "regex"

    def __init__(self) -> None:
        self._positive = {
            f.name: [re.compile(p, re.IGNORECASE) for p in f.patterns]
            for f in TEXT_FEATURES
        }
        self._negative = {
            f.name: [re.compile(p, re.IGNORECASE) for p in f.negative_patterns]
            for f in TEXT_FEATURES
        }

    @staticmethod
    def _first_match(regexes: list[re.Pattern], text: str) -> Optional[str]:
        for rx in regexes:
            m = rx.search(text)
            if m:
                return m.group(0)
        return None

    def classify_detailed(self, text: str) -> dict[str, FeatureValue]:
        out: dict[str, FeatureValue] = {}
        for name in _FEATURE_NAMES:
            # A negative (explicitly-contradicting) phrase is checked first: a
            # comment cannot simultaneously carry the excuse and its explicit
            # denial, and the denial is the more specific claim about ground
            # like "suited by the soft ground" (see TEXT_FEATURES docstring).
            neg = self._first_match(self._negative[name], text)
            if neg is not None:
                out[name] = FeatureValue(False, neg)
                continue
            pos = self._first_match(self._positive[name], text)
            out[name] = FeatureValue(bool(pos), pos)
        return out

    def classify(self, text: str) -> dict[str, bool]:
        """Legacy bool-only view, kept for existing callers/tests."""
        return {name: bool(fv.value) for name, fv in self.classify_detailed(text).items()}


class OllamaBackend:
    """Local Ollama server. Returns None on any failure → caller falls back."""

    name = "ollama"

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.1",
        timeout: float = 30.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _prompt(self, text: str) -> str:
        lines = "\n".join(f"- {f.name}: {f.description}" for f in TEXT_FEATURES)
        return (
            "You analyse UK/Irish horse-racing comments. Given the comment, decide "
            "for each feature below whether the text supports it (true), explicitly "
            "contradicts it (false), or does not address it at all (unknown). Only "
            "use evidence from the supplied text; never infer a fact the text does "
            "not state. Respond with a single JSON object whose keys are exactly the "
            "feature names. Each value must itself be an object "
            '{"value": true|false|"unknown", "evidence": "<exact short phrase copied '
            'verbatim from the comment, or empty string for unknown/false with no '
            'quotable phrase>"}. Do not add commentary outside the JSON object.\n\n'
            f"Features:\n{lines}\n\n"
            f"Comment:\n\"\"\"{text}\"\"\"\n"
        )

    def _call(self, text: str) -> Optional[dict]:
        try:
            import httpx
        except ImportError:  # pragma: no cover - httpx is a project dep
            _log.warning("llm: httpx unavailable — cannot reach Ollama")
            return None
        try:
            resp = httpx.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": self._prompt(text),
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            parsed = json.loads(raw)
        except Exception as exc:
            _log.warning("llm: Ollama call failed (%s) — falling back to regex", exc)
            return None
        if not isinstance(parsed, dict):
            _log.warning("llm: Ollama returned non-object JSON — falling back")
            return None
        return parsed

    def classify_detailed(self, text: str) -> Optional[dict[str, FeatureValue]]:
        """Strict schema parse: every claim must be grounded in ``text`` or is unknown.

        Returns ``None`` (triggering the caller's regex fallback) only when the
        whole response is unusable (network/parse failure, or every single
        feature was malformed). A response that parses but has some ungrounded
        claims is still returned — those specific features are forced to
        unknown with a per-feature ``failure_reason`` folded into ``evidence``
        being ``None``, rather than discarding an otherwise-good response.
        """
        parsed = self._call(text)
        if parsed is None:
            return None
        text_lower = text.lower()
        out: dict[str, FeatureValue] = {}
        malformed = 0
        for name in _FEATURE_NAMES:
            entry = parsed.get(name)
            if isinstance(entry, dict):
                raw_value = entry.get("value")
                evidence = entry.get("evidence") or None
            elif isinstance(entry, bool):
                # Tolerate a plain boolean (older/simpler model output) but it
                # carries no evidence, so it cannot be grounded as True.
                raw_value, evidence = entry, None
            else:
                malformed += 1
                out[name] = FeatureValue(None)
                continue

            if raw_value is None or (isinstance(raw_value, str) and raw_value.lower() == "unknown"):
                out[name] = FeatureValue(None)
                continue
            if not isinstance(raw_value, bool):
                malformed += 1
                out[name] = FeatureValue(None)
                continue
            if raw_value is True:
                if evidence and str(evidence).strip().lower() in text_lower:
                    out[name] = FeatureValue(True, str(evidence).strip())
                else:
                    # Claimed true but not grounded in the supplied text — never
                    # trust an ungrounded positive claim; degrade to unknown.
                    out[name] = FeatureValue(None)
            else:
                out[name] = FeatureValue(False, str(evidence).strip() if evidence else None)

        if malformed == len(_FEATURE_NAMES):
            _log.warning("llm: Ollama response had no usable feature entries — falling back")
            return None
        return out


# ── extractor ───────────────────────────────────────────────────────────────

class TextFeatureExtractor:
    """Text → tri-state features, with disk cache and graceful no-op.

    Parameters
    ----------
    enabled, backend, ollama_*, cache_ttl
        Override config; when ``None`` they are read from the ``llm`` config
        section (which defaults to disabled).
    backend_impl
        Inject a concrete backend (used by tests to stub out the model). When
        given, it is used directly and config backend selection is bypassed.
    cache
        Inject a ``Cache``; defaults to a dedicated ``data/cache/llm`` store.
    """

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        backend: Optional[str] = None,
        ollama_host: Optional[str] = None,
        ollama_model: Optional[str] = None,
        ollama_timeout: Optional[float] = None,
        cache_ttl: Optional[int] = None,
        backend_impl: Optional[Backend] = None,
        cache: Optional[Cache] = None,
    ) -> None:
        cfg = self._load_config()
        self.enabled = cfg["enabled"] if enabled is None else bool(enabled)
        self._cache_ttl = cfg["cache_ttl"] if cache_ttl is None else int(cache_ttl)
        self._cache = cache if cache is not None else Cache(
            cache_dir=_CACHE_DIR, default_ttl=self._cache_ttl
        )

        self._regex = RegexBackend()
        self._configured_name: str
        if backend_impl is not None:
            self._backend: Backend = backend_impl
            self._configured_name = getattr(backend_impl, "name", "custom")
        else:
            chosen = (backend or cfg["backend"]).lower()
            if chosen == "ollama":
                self._backend = OllamaBackend(
                    host=ollama_host or cfg["ollama_host"],
                    model=ollama_model or cfg["ollama_model"],
                    timeout=ollama_timeout if ollama_timeout is not None else cfg["ollama_timeout"],
                )
                self._configured_name = "ollama"
            else:
                self._backend = self._regex
                self._configured_name = "regex"

    @staticmethod
    def _load_config() -> dict:
        defaults = {
            "enabled": False,
            "backend": "regex",
            "cache_ttl": 2_592_000,
            "ollama_host": "http://localhost:11434",
            "ollama_model": "llama3.1",
            "ollama_timeout": 30.0,
        }
        try:
            from utils.config_loader import get
            defaults["enabled"] = bool(get("llm.enabled", defaults["enabled"]))
            defaults["backend"] = str(get("llm.backend", defaults["backend"]))
            defaults["cache_ttl"] = int(get("llm.cache_ttl", defaults["cache_ttl"]))
            defaults["ollama_host"] = str(get("llm.ollama.host", defaults["ollama_host"]))
            defaults["ollama_model"] = str(get("llm.ollama.model", defaults["ollama_model"]))
            defaults["ollama_timeout"] = float(get("llm.ollama.timeout", defaults["ollama_timeout"]))
        except Exception as exc:  # config absent/invalid → stay with safe defaults
            _log.debug("llm: config unavailable (%s) — using defaults", exc)
        return defaults

    @staticmethod
    def _cache_key(text: str, backend_name: str, model: str) -> str:
        digest = hashlib.sha1(
            f"{SCHEMA_VERSION}|{PROMPT_VERSION}|{backend_name}|{model}|{text}".encode("utf-8")
        ).hexdigest()
        return f"text_{backend_name}_{digest}"

    def extract_detailed(self, text: Optional[str]) -> ExtractionResult:
        """Return the full, auditable extraction outcome. Never raises."""
        if not self.enabled:
            return _disabled_result()
        if not text or not str(text).strip():
            return _empty_text_result()
        text = str(text).strip()

        configured_name = self._configured_name
        configured_model = getattr(self._backend, "model", "")
        is_fallback_candidate = configured_name != "regex"
        fallback_name = f"{configured_name}_fallback_regex" if is_fallback_candidate else "regex"

        # Check both possible outcomes' cache slots BEFORE calling the backend at
        # all: a persistently-down Ollama server must not be re-invoked on every
        # single call just because its own (failed) key never gets populated —
        # the fallback result is cached under its own honestly-labelled key.
        optimistic_key = self._cache_key(text, configured_name, configured_model)
        cached = self._cache.get(optimistic_key)
        if isinstance(cached, dict) and cached.get("backend") == configured_name:
            return ExtractionResult.from_dict(cached)

        fallback_key = self._cache_key(text, fallback_name, "")
        cached_fb = self._cache.get(fallback_key)
        if isinstance(cached_fb, dict) and cached_fb.get("backend") == fallback_name:
            return ExtractionResult.from_dict(cached_fb)

        detailed = self._backend.classify_detailed(text) if hasattr(self._backend, "classify_detailed") else None
        if detailed is not None:
            result = ExtractionResult(
                features=detailed, backend=configured_name, model=configured_model,
                schema_version=SCHEMA_VERSION, prompt_version=PROMPT_VERSION,
                is_fallback=False, failure_reason=None,
            )
            self._cache.set(optimistic_key, result.to_dict(), ttl=self._cache_ttl)
            return result

        # Backend failed (or is not configured with a real name) → deterministic
        # fallback, labelled as exactly that — never attributed to the backend
        # that failed.
        regex_features = self._regex.classify_detailed(text)
        result = ExtractionResult(
            features=regex_features, backend=fallback_name, model="",
            schema_version=SCHEMA_VERSION, prompt_version=PROMPT_VERSION,
            is_fallback=is_fallback_candidate,
            failure_reason=f"{configured_name}_backend_failed" if is_fallback_candidate else None,
        )
        self._cache.set(fallback_key, result.to_dict(), ttl=self._cache_ttl)
        return result

    def extract(self, text: Optional[str]) -> dict[str, bool]:
        """Legacy convenience: {feature: bool}, unknown collapses to False.

        Kept for backward compatibility with callers that predate the tri-state
        schema. New code should call :meth:`extract_detailed` and look at
        ``FeatureValue.value`` (``None`` = unknown) directly. Returns ``{}`` for
        the same disabled/no-text no-op cases the old implementation did.
        """
        if not self.enabled or not text or not str(text).strip():
            return {}
        return self.extract_detailed(text).as_bools()


# ── module-level convenience ────────────────────────────────────────────────

_default: Optional[TextFeatureExtractor] = None


def _get_default() -> TextFeatureExtractor:
    global _default
    if _default is None:
        _default = TextFeatureExtractor()
    return _default


def extract_text_features(text: Optional[str]) -> dict[str, bool]:
    """Convenience wrapper over a shared, config-driven extractor.

    No-ops (returns ``{}``) unless ``llm.enabled`` is true and ``text`` is given.
    """
    return _get_default().extract(text)


def extract_text_features_detailed(text: Optional[str]) -> ExtractionResult:
    """Convenience wrapper returning the full tri-state, evidence-grounded result."""
    return _get_default().extract_detailed(text)
