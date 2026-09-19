"""Step 11 probe: what timing/margin fields does the raw archive actually carry?

Read-only. Samples the Sporting Life ``__NEXT_DATA__`` archive to establish,
BEFORE any feature is designed:
  * presence rate of ``race_summary.winning_time`` and ``distance``
  * the vocabulary of per-ride ``finish_distance`` tokens
  * whether ``finish_distance`` is CUMULATIVE (behind winner) or INCREMENTAL
    (behind the horse in front) -- decided by monotonicity, not by assumption
  * whether any sectional field exists at all

Usage:  python reports/improvement/11/00_probe_archive.py [n_days]
"""
from __future__ import annotations

import gzip
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

RAW_ROOT = os.path.join("data", "historical", "raw", "sporting_life")
_NEXT = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)
_SECTIONAL_KEY = re.compile(r"sectional|split_time|furlong_time|pace_figure", re.I)


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
            html = fh.read()
    except OSError:
        return None
    m = _NEXT.search(html)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except ValueError:
        return None


def all_keys(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(k)
            all_keys(v, out)
    elif isinstance(obj, list):
        for v in obj:
            all_keys(v, out)


def main(n_days=40):
    days = []
    for year in sorted(os.listdir(RAW_ROOT)):
        ydir = os.path.join(RAW_ROOT, year)
        if os.path.isdir(ydir):
            days += [os.path.join(ydir, d) for d in sorted(os.listdir(ydir))]
    random.Random(11).shuffle(days)
    days = days[:n_days]

    n_docs = n_races = 0
    have_time = have_dist = 0
    tokens = Counter()
    sectional_keys = set()
    cumulative = incremental = ambiguous = 0
    surfaces, classes = Counter(), Counter()

    for day in days:
        for name in sorted(os.listdir(day)):
            path = os.path.join(day, name)
            doc = load(path)
            n_docs += 1
            if not doc:
                continue
            race = (((doc.get("props") or {}).get("pageProps") or {}).get("race")) or {}
            summary = race.get("race_summary") or {}
            if not summary:
                continue
            n_races += 1
            if str(summary.get("winning_time") or "").strip():
                have_time += 1
            if str(summary.get("distance") or "").strip():
                have_dist += 1
            surfaces[str((summary.get("course_surface") or {}).get("surface") or "")] += 1
            classes[str(summary.get("race_class") or "")] += 1

            keys = set()
            all_keys(race, keys)
            sectional_keys |= {k for k in keys if _SECTIONAL_KEY.search(k)}

            # margin convention: collect (position, finish_distance) in order
            seq = []
            for ride in race.get("rides") or []:
                if not isinstance(ride, dict):
                    continue
                pos = ride.get("finish_position")
                fd = ride.get("finish_distance")
                if fd is not None:
                    tokens[str(fd)] += 1
                if isinstance(pos, int) and pos > 0 and fd is not None:
                    seq.append((pos, str(fd)))
            nums = []
            for pos, fd in sorted(seq):
                v = _num(fd)
                if v is None:
                    nums = []
                    break
                nums.append(v)
            if len(nums) >= 3:
                if all(b >= a for a, b in zip(nums, nums[1:])):
                    cumulative += 1  # consistent with cumulative (also with equal steps)
                else:
                    incremental += 1  # a decrease is impossible under cumulative
            elif nums:
                ambiguous += 1

    print(f"days sampled          : {len(days)}")
    print(f"documents read        : {n_docs}")
    print(f"race documents        : {n_races}")
    print(f"winning_time present  : {have_time} ({100*have_time/max(n_races,1):.2f}%)")
    print(f"distance present      : {have_dist} ({100*have_dist/max(n_races,1):.2f}%)")
    print(f"sectional-like keys   : {sorted(sectional_keys) or 'NONE'}")
    print()
    print(f"margin monotone non-decreasing (cumulative-compatible): {cumulative}")
    print(f"margin decreases somewhere (INCREMENTAL, cumulative impossible): {incremental}")
    print(f"too short to judge   : {ambiguous}")
    print()
    print("surfaces:", dict(surfaces.most_common()))
    print("race_class:", dict(classes.most_common(12)))
    print()
    print(f"distinct finish_distance tokens: {len(tokens)}")
    for tok, n in tokens.most_common(45):
        print(f"  {tok!r:14} {n}")


def _num(tok):
    """Numeric value of a margin token, or None if non-numeric (nk/hd/dist...)."""
    t = str(tok).strip()
    t = t.replace("¼", " 1/4").replace("½", " 1/2").replace("¾", " 3/4")
    m = re.match(r"^(\d+)?\s*(?:(\d+)/(\d+))?$", t)
    if not m or not (m.group(1) or m.group(2)):
        return None
    whole = float(m.group(1) or 0)
    if m.group(2):
        whole += float(m.group(2)) / float(m.group(3))
    return whole


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
