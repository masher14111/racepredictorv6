"""Derived Timeform features.

The leak-safe primitives now live in ``features/derive.py`` (their canonical home,
shared with the /features pipeline). Re-exported here so the Timeform writer and its
existing tests keep importing them from ``scraper.timeform.features`` unchanged.
"""
from features.derive import add_class_change, add_going_speed, add_trailing_rates

__all__ = ["add_going_speed", "add_class_change", "add_trailing_rates"]
