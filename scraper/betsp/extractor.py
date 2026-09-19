"""Pluggable enrichment extractor. StubExtractor now; LLMExtractor later."""
from abc import ABC, abstractmethod

from scraper.betsp.results.base import RawResult, ResultsSource
from utils.logger import get_logger

logger = get_logger(__name__)


class PerfExtractor(ABC):
    @abstractmethod
    def extract(self, raw: RawResult, source: ResultsSource) -> list:
        """Return list[ResultRow] from a raw doc."""


class StubExtractor(PerfExtractor):
    """Deterministic extractor: delegates to the source's own parser + validates."""

    def extract(self, raw: RawResult, source: ResultsSource) -> list:
        try:
            rows = source.parse(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("StubExtractor parse failed (%s): %s", source.name, exc)
            return []
        valid = [r for r in rows if (r.horse_name or "").strip()]
        if len(valid) != len(rows):
            logger.debug("StubExtractor dropped %d invalid rows", len(rows) - len(valid))
        return valid
