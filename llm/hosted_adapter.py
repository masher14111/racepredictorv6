"""Provider-neutral hosted LLM extraction adapter (step 15 benchmark only).

Not wired into ``llm.text_features.TextFeatureExtractor``'s config-driven
backend selection (``llm.backend`` stays ``"regex"``/``"ollama"`` — D46,
Stage 14). This module exists so the step 15 benchmark harness can score a
hosted candidate under the SAME tri-state, evidence-grounded schema
(``TEXT_FEATURES`` / ``FeatureValue``) as the local backends, with its own
budget/cache/telemetry layer since bounded hosted spend has no equivalent in
the free local backends.

Design points required by prompts/15-local-llm-benchmark.md:
* Strict output validation — a claimed ``True`` whose ``evidence`` is not an
  actual substring of the supplied text is forced to unknown, exactly like
  ``OllamaBackend`` (llm/text_features.py). A response that fails to parse
  as the expected JSON schema is a recorded ``schema_valid=False`` failure,
  not a silent fallback mislabelled as a real result.
* Bounded timeouts/retries — ``max_retries`` bounds attempts; every attempt
  (including failed ones) is accounted for in the budget reservation BEFORE
  any network call, so a string of failures cannot overspend the cap.
* Per-record caching keyed on (text hash, model, provider, prompt version,
  schema version) — a cache hit never re-spends budget, and is never
  reported as a fresh billed call.
* No browsing/tools — the request is a single non-tool chat completion; nothing
  here lets the model fetch external context, so any claim must come from the
  supplied comment text alone.
* Identity-substitution probe (``build_prompt`` + ``substitute_identity``) —
  swapping the horse's name for a neutral placeholder before extraction and
  checking the grounded output is unchanged is this module's cheap proxy for
  "is the model reciting a real remembered result, or actually reading the
  text" (full outcome-memorization testing is out of scope; this only checks
  that identity swapping cannot change a text-grounded answer).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Optional

from llm.hosted_budget import BudgetExceeded, HostedBudgetLedger, PricingUnbounded
from llm.text_features import TEXT_FEATURES, FeatureValue, SCHEMA_VERSION
from utils.cache import Cache
from utils.logger import get_logger

_log = get_logger(__name__)

_FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in TEXT_FEATURES)

# Bump on wording change — frozen for the duration of step 15's scoring run
# (reports/improvement/15/frozen_protocol.json records the value in force
# when scoring started).
HOSTED_PROMPT_VERSION = 1


def build_prompt(text: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt). Frozen wording — do not edit
    without bumping HOSTED_PROMPT_VERSION; a scoring run started under one
    version must not be silently compared against a later-worded one."""
    lines = "\n".join(f"- {f.name}: {f.description}" for f in TEXT_FEATURES)
    system = (
        "You extract structured facts from UK/Irish horse-racing preview "
        "comments. You have no identity beyond this task; never mention what "
        "model or company you are. You cannot browse or use tools — use only "
        "the supplied comment text. Never use outside knowledge of real horses, "
        "trainers or race results; if the comment doesn't state something, the "
        "correct answer is unknown, never a guess from memory."
    )
    user = (
        "For each feature below, decide whether the comment supports it "
        "(true), explicitly contradicts it (false), or does not address it "
        "at all (unknown). Respond with ONLY a single JSON object whose keys "
        "are exactly the feature names; each value is "
        '{"value": true|false|"unknown", "evidence": "<exact short phrase '
        'copied verbatim from the comment, or empty string for '
        'unknown/false with no quotable phrase>"}. No commentary, no markdown '
        "fences, no extra keys.\n\n"
        f"Features:\n{lines}\n\n"
        f"Comment:\n\"\"\"{text}\"\"\"\n"
    )
    return system, user


def substitute_identity(text: str, horse_name: Optional[str], placeholder: str = "This Runner") -> str:
    """Replace the horse's own name with a neutral placeholder.

    Used only for the identity-substitution robustness probe: a text-grounded
    extractor's tri-state answers/evidence should be invariant to the
    runner's real name (only the phrase-bearing sentence structure matters),
    whereas a model reciting a memorised real-world result about that named
    horse could answer differently once the name is gone.
    """
    if not horse_name:
        return text
    out = text
    for variant in {horse_name, horse_name.strip()}:
        if variant:
            out = out.replace(variant, placeholder)
    return out


@dataclass(frozen=True)
class HostedExtraction:
    features: dict[str, FeatureValue]
    backend: str
    provider: str
    model: str
    schema_version: int
    prompt_version: int
    schema_valid: bool
    is_cache_hit: bool
    failure_reason: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost_usd: float
    latency_ms: float
    attempts: int
    raw_response_hash: Optional[str]

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["features"] = {k: v.to_dict() for k, v in self.features.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "HostedExtraction":
        d = dict(d)
        d["features"] = {k: FeatureValue.from_dict(v) for k, v in d["features"].items()}
        return cls(**d)


def _unknown_features() -> dict[str, FeatureValue]:
    return {name: FeatureValue(None) for name in _FEATURE_NAMES}


class OpenRouterBackend:
    """Hosted extraction via OpenRouter's chat-completions API.

    Pricing (``price_input_per_token`` / ``price_output_per_token``) and the
    ceiling to enforce it against must be supplied by the CALLER, verified
    against the live OpenRouter ``/models`` listing before construction
    (``scripts/verify_hosted_pricing.py``) — this class never fetches pricing
    itself, so unit tests never need network access to prove the ceiling
    check fires.
    """

    name = "hosted_openrouter"
    provider = "openrouter"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        ledger: HostedBudgetLedger,
        price_input_per_token: float,
        price_output_per_token: float,
        price_ceiling_input_per_token: float,
        price_ceiling_output_per_token: float,
        max_output_tokens: int = 800,
        reasoning_effort: str = "low",
        timeout_s: float = 30.0,
        max_retries: int = 2,
        cache: Optional[Cache] = None,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        http_post=None,
    ) -> None:
        if price_input_per_token > price_ceiling_input_per_token:
            raise PricingUnbounded(
                f"input price {price_input_per_token} exceeds ceiling {price_ceiling_input_per_token}"
            )
        if price_output_per_token > price_ceiling_output_per_token:
            raise PricingUnbounded(
                f"output price {price_output_per_token} exceeds ceiling {price_ceiling_output_per_token}"
            )
        self.api_key = api_key
        self.model = model
        self.ledger = ledger
        self.price_input_per_token = price_input_per_token
        self.price_output_per_token = price_output_per_token
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._cache = cache
        self.base_url = base_url
        # Injectable transport for tests — never hits the network unless the
        # caller passes a real one (or leaves the httpx default, live only).
        self._http_post = http_post

    # -- cost estimation ----------------------------------------------------
    def _estimate_prompt_tokens(self, system: str, user: str) -> int:
        # Conservative: ~3.5 chars/token (English prose skews slightly higher
        # than the common 4 chars/token rule) rounded up, so the reservation
        # over-estimates rather than under-estimates true usage.
        chars = len(system) + len(user)
        return max(1, -(-chars // 3))

    def _worst_case_cost(self, system: str, user: str) -> float:
        prompt_tokens = self._estimate_prompt_tokens(system, user)
        per_attempt = (
            prompt_tokens * self.price_input_per_token
            + self.max_output_tokens * self.price_output_per_token
        )
        return per_attempt * (1 + self.max_retries)

    # -- cache ----------------------------------------------------------------
    def _cache_key(self, text: str) -> str:
        # reasoning_effort AND max_output_tokens are both part of the key: a
        # reasoning-capable model's answer (and whether it fits in the
        # completion budget at all — see reports/improvement/15 evidence,
        # unset/too-low a budget consumed the whole thing on chain-of-thought
        # and left no room for the JSON answer) depends on both, so a call
        # made with a larger budget must never silently read back a stale
        # exhaustion failure cached under a smaller one.
        digest = hashlib.sha256(
            f"{SCHEMA_VERSION}|{HOSTED_PROMPT_VERSION}|{self.provider}|{self.model}|"
            f"{self.reasoning_effort}|{self.max_output_tokens}|{text}".encode("utf-8")
        ).hexdigest()
        return f"hosted_{self.provider}_{self.model.replace('/', '_')}_{digest}"

    # -- transport --------------------------------------------------------------
    def _post(self, system: str, user: str) -> dict:
        if self._http_post is not None:
            return self._http_post(self.base_url, self._headers(), self._payload(system, user), self.timeout_s)
        import httpx  # local import: not required unless a live call happens

        resp = httpx.post(
            self.base_url,
            headers=self._headers(),
            json=self._payload(system, user),
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        return resp.json()

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _payload(self, system: str, user: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
            # Ask OpenRouter to report its own actual billed cost in
            # usage.cost — see the actual_cost comment in extract().
            "usage": {"include": True},
        }
        if self.reasoning_effort:
            # OpenRouter's unified reasoning control (supported_parameters
            # includes "reasoning"/"reasoning_effort" for this model, verified
            # against the live /models listing — scripts/verify_hosted_pricing.py).
            # Without this, a reasoning-capable model can spend the ENTIRE
            # max_tokens budget on chain-of-thought and return no answer
            # content at all (measured: effort left unset -> 176/243 schema
            # failures, every one a reasoning_tokens==max_output_tokens
            # exhaustion, not a genuine extraction miss).
            payload["reasoning"] = {"effort": self.reasoning_effort}
        return payload

    # -- validation -------------------------------------------------------------
    @staticmethod
    def _parse_and_ground(parsed: dict, text: str) -> tuple[dict[str, FeatureValue], bool]:
        """Mirrors OllamaBackend's grounding rule exactly (llm/text_features.py):
        a claimed True whose evidence is not an actual substring is forced to
        unknown. Returns (features, schema_valid)."""
        if not isinstance(parsed, dict):
            return _unknown_features(), False
        text_lower = text.lower()
        out: dict[str, FeatureValue] = {}
        malformed = 0
        for name in _FEATURE_NAMES:
            entry = parsed.get(name)
            if not isinstance(entry, dict):
                malformed += 1
                out[name] = FeatureValue(None)
                continue
            raw_value = entry.get("value")
            evidence = entry.get("evidence") or None
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
                    out[name] = FeatureValue(None)
            else:
                out[name] = FeatureValue(False, str(evidence).strip() if evidence else None)
        schema_valid = malformed == 0
        return out, schema_valid

    # -- public API -------------------------------------------------------------
    def extract(self, text: str, *, horse_name: Optional[str] = None) -> HostedExtraction:
        text = str(text).strip()
        cache_key = self._cache_key(text)
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if isinstance(cached, dict):
                result = HostedExtraction.from_dict(cached)
                return HostedExtraction(
                    **{**result.__dict__, "is_cache_hit": True}
                )

        system, user = build_prompt(text)
        worst_case = self._worst_case_cost(system, user)
        token = self.ledger.reserve(
            worst_case, requests=1, note=f"model={self.model} hash={cache_key[-12:]}"
        )

        start = time.monotonic()
        attempts = 0
        last_exc: Optional[Exception] = None
        raw: Optional[dict] = None
        for attempt in range(1, self.max_retries + 2):
            attempts = attempt
            try:
                raw = self._post(system, user)
                last_exc = None
                break
            except Exception as exc:  # network/HTTP/timeout — bounded retry
                last_exc = exc
                _log.warning("hosted_adapter: attempt %d failed (%s)", attempt, exc)
        latency_ms = (time.monotonic() - start) * 1000.0

        if raw is None:
            self.ledger.release(token, note=f"all {attempts} attempts failed: {last_exc}")
            result = HostedExtraction(
                features=_unknown_features(), backend=self.name, provider=self.provider,
                model=self.model, schema_version=SCHEMA_VERSION, prompt_version=HOSTED_PROMPT_VERSION,
                schema_valid=False, is_cache_hit=False,
                failure_reason=f"transport_failed:{last_exc}", prompt_tokens=0,
                completion_tokens=0, reasoning_tokens=0, cost_usd=0.0,
                latency_ms=latency_ms, attempts=attempts, raw_response_hash=None,
            )
            return result

        usage = raw.get("usage", {}) or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        reasoning_tokens = int(
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        )
        # Prefer OpenRouter's own reported cost (requested via "usage":
        # {"include": true} in _payload) over a static price*tokens estimate
        # — this model's real price has time-of-day-dependent tiers (weekday
        # peak/off-peak), so a caller-supplied static price is only correct
        # at the instant it was measured; the provider's own figure is never
        # stale. Falls back to the static estimate only if the provider omits
        # "cost" (older/non-conforming responses), which is why the caller-
        # supplied price must still be an accurate ceiling-checked value.
        reported_cost = usage.get("cost")
        if reported_cost is not None:
            actual_cost = float(reported_cost)
        else:
            actual_cost = (
                prompt_tokens * self.price_input_per_token
                + completion_tokens * self.price_output_per_token
            )
        actual_model = raw.get("model", self.model)
        if actual_model != self.model:
            _log.warning(
                "hosted_adapter: requested model %r, provider returned %r — recording actual",
                self.model, actual_model,
            )

        try:
            self.ledger.commit(token, actual_cost_usd=actual_cost, requests=1, note=actual_model)
        except BudgetExceeded:
            # Provider billed more than the worst-case reservation covered —
            # fail closed rather than silently accept an unbounded overspend.
            self.ledger.release(token, note="commit_exceeded_reservation")
            raise

        content = ""
        try:
            content = raw["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except Exception as exc:
            features, schema_valid = _unknown_features(), False
            failure_reason = f"json_parse_failed:{exc}"
        else:
            features, schema_valid = self._parse_and_ground(parsed, text)
            failure_reason = None if schema_valid else "some_fields_malformed"

        raw_hash = hashlib.sha256(content.encode("utf-8")).hexdigest() if content else None
        result = HostedExtraction(
            features=features, backend=self.name, provider=self.provider,
            model=actual_model, schema_version=SCHEMA_VERSION, prompt_version=HOSTED_PROMPT_VERSION,
            schema_valid=schema_valid, is_cache_hit=False, failure_reason=failure_reason,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens, cost_usd=actual_cost, latency_ms=latency_ms,
            attempts=attempts, raw_response_hash=raw_hash,
        )
        if self._cache is not None:
            self._cache.set(cache_key, result.to_dict())
        return result
