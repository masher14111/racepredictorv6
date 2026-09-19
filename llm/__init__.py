"""Optional local-LLM text→feature extraction (OFF the probability hot path).

This package turns free-text race commentary / spotlight / going notes into a
small set of structured, tri-state, evidence-grounded features (e.g.
``ground_excuse``, ``distance_excuse``, ``trip_trouble``). It NEVER produces
win probabilities or EV, and its outputs are NOT in
``models.features.FEATURE_COLS`` — wiring them in is a deliberate future step
(step 15/16) that requires a real text feed on disk first.

It no-ops gracefully when ``llm.enabled`` is false (the default) or when no text
is supplied, and degrades from the Ollama backend to a deterministic regex
fallback when no local model is reachable. ``llm.text_archive`` is the
append-only store for the raw incoming text itself, kept separate from feature
extraction.
"""
from llm.text_features import (
    TEXT_FEATURES,
    ExtractionResult,
    FeatureValue,
    TextFeature,
    TextFeatureExtractor,
    extract_text_features,
    extract_text_features_detailed,
)

__all__ = [
    "TEXT_FEATURES",
    "ExtractionResult",
    "FeatureValue",
    "TextFeature",
    "TextFeatureExtractor",
    "extract_text_features",
    "extract_text_features_detailed",
]
