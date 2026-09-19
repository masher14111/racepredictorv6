"""Perf extractor interface. StubExtractor delegates to the HTML parser;
an LLMExtractor reading raw_store blobs is the deferred next step."""
from abc import ABC, abstractmethod

from scraper.timeform import parser


class PerfExtractor(ABC):
    @abstractmethod
    def extract(self, html: str, **ctx) -> list:
        ...


class StubExtractor(PerfExtractor):
    def extract(self, html: str, **ctx) -> list:
        return parser.parse_event(html, **ctx)
