"""Step 15 — hand-authored reviewed labels for the 60-comment blind-eval sample.

Produced by a single LLM reviewer (Claude Sonnet 5, this stage) reading each
comment directly against the TEXT_FEATURES definitions in
llm/text_features.py. NOT human-reviewed gold labels: review_status is
recorded explicitly as "single_reviewer_llm_unverified" in the emitted
labels_reviewed.jsonl (see scripts/label_stage15_sample.py), no second
reviewer or adjudication pass. Every non-null evidence string is verified
(scripts/label_stage15_sample.py) to be an exact case-insensitive substring
of its source comment before the file is written — a mismatch is a labelling
bug, not evidence formatting.

Policy notes applied consistently across all 60 records (documented here so
the numbers are reproducible, not vibes):
* A feature is labelled True whenever the text contains an explicit phrase
  supporting the claim ANYWHERE in the comment (own past-performance lines
  included) — this mirrors how RegexBackend/OllamaBackend actually work:
  context-free phrase support, not entity-linked to "today's" race, since
  the schema itself makes no such claim.
* False requires an EXPLICIT contradiction (e.g. "stays 1.75m" contradicts a
  distance excuse; "runner-up ... at this course" contradicts "course
  winner" — established result at that course was not a win).
* None (unknown) is used whenever the text does not address the claim, or
  only alludes to it too indirectly to ground a specific quoted phrase
  (e.g. "decent record when applying first-time headgear" is a STABLE
  statistic, not a claim about this horse's own gear today).
"""

# index -> {feature: (value, evidence_or_None)}
LABELS: dict[int, dict[str, tuple]] = {
    0: {"distance_excuse": (None, None)},
    1: {"returning_from_layoff": (True, "returning from a break")},
    2: {},
    3: {"trip_trouble": (True, "denied a clear run")},
    4: {
        "headgear_first_time": (False, "first-time hood (discarded here)"),
        "returning_from_layoff": (True, "almost a year off"),
    },
    5: {},
    6: {
        "returning_from_layoff": (True, "on return"),
    },
    7: {},
    8: {"course_winner": (True, "winner of 17-runner handicap here")},
    9: {},
    10: {"returning_from_layoff": (True, "14 months off")},
    11: {},
    12: {
        "headgear_first_time": (True, "Tongue tied first time here"),
        "course_winner": (False, "remote fourth on recent C&D chase debut"),
    },
    13: {"course_winner": (True, "Multiple C&D winner")},
    14: {"distance_excuse": (False, "step up to 6f looks sure to suit her")},
    15: {"headgear_first_time": (True, "First-time blinkers now reached for")},
    16: {"returning_from_layoff": (True, "5 months off")},
    17: {
        "course_winner": (
            False,
            "filled that position in 3-runner conditions stakes at this course",
        )
    },
    18: {},
    19: {},
    20: {
        "course_winner": (
            False,
            "6 lengths second to Beorma (who has won again since) over C&D",
        )
    },
    21: {"returning_from_layoff": (True, "on reappearance")},
    22: {},
    23: {"headgear_first_time": (False, "blinkers added (retained)")},
    24: {"course_winner": (True, "winner of a C&D novice")},
    25: {"returning_from_layoff": (True, "Made a winner return")},
    26: {"course_winner": (True, "Successful twice over C&D in 2024")},
    27: {},
    28: {"returning_from_layoff": (True, "on return")},
    29: {
        "distance_excuse": (
            True, "down markedly in trip here and surely up against it"
        )
    },
    30: {},
    31: {"course_winner": (True, "taking 9-runner handicap at this C&D")},
    32: {},
    33: {},
    34: {},
    35: {"distance_excuse": (False, "stays 1¾m")},
    36: {"returning_from_layoff": (True, "Given a break")},
    37: {},
    38: {},
    39: {},
    40: {"returning_from_layoff": (True, "Made a winning return")},
    41: {"returning_from_layoff": (True, "reappearance run")},
    42: {
        "returning_from_layoff": (True, "on return"),
        "course_winner": (
            False, "narrowly beaten in 7f listed race here in October"
        ),
    },
    43: {"headgear_first_time": (True, "first-time hood")},
    44: {},
    45: {
        "distance_excuse": (False, "stays 2m"),
        "returning_from_layoff": (True, "Back from 8 months off"),
    },
    46: {"returning_from_layoff": (True, "6 months off")},
    47: {},
    48: {"returning_from_layoff": (True, "on return")},
    49: {"distance_excuse": (False, "bred to stay at least 1½m")},
    50: {
        "course_winner": (True, "scoring over this C&D in August"),
        "returning_from_layoff": (True, "on return"),
    },
    51: {
        "course_winner": (
            False, "10¼ lengths sixth of 10 to Glory Hyde over C&D"
        )
    },
    52: {},
    53: {},
    54: {"returning_from_layoff": (True, "after a break")},
    55: {"headgear_first_time": (True, "first-time cheekpieces")},
    56: {
        "returning_from_layoff": (True, "successful return"),
        "trip_trouble": (True, "overcoming interference"),
    },
    57: {"headgear_first_time": (True, "first-time cheekpieces")},
    58: {},
    59: {"headgear_first_time": (True, "First-time hood goes on")},
}
