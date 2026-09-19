"""Contracts shared by all results sources."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class RawResult:
    """A captured raw document for one race (or results page)."""
    source: str
    race_date: date
    url: str
    html: str
    venue: Optional[str] = None


@dataclass
class ResultRow:
    """Normalized enrichment record for one horse-run."""
    race_date: str            # ISO timestamp or date string for join keying
    venue: str
    horse_name: str
    source: str
    jockey_id: Optional[str] = None
    jockey_name: Optional[str] = None
    trainer_id: Optional[str] = None
    trainer_name: Optional[str] = None
    position: Optional[int] = None
    going: Optional[str] = None


class ResultsSource(ABC):
    """A results website that yields RawResult docs and parses them to ResultRows."""

    name: str = "base"

    @abstractmethod
    def fetch_raw(self, day: date, get_html) -> list:
        """Return a list[RawResult] for the given day using get_html(url)->str."""

    @abstractmethod
    def parse(self, raw: RawResult) -> list:
        """Parse one RawResult into list[ResultRow]."""
