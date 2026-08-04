"""Roster parser — functional, no classes.

Parses roster.txt (comma-separated slot tokens) into a plain dict
describing starters, flex spots, bench, and IR counts.
"""

from pathlib import Path
from scoring_engine import read_source, POSITIONS


# Tokens that represent a specific position starter
_POSITION_TOKENS = set(POSITIONS)

# Flex tokens and which positions they allow
_FLEX_DEFINITIONS = {
    "WRT": {"RB", "WR", "TE"},
    "WRTQ": {"QB", "RB", "WR", "TE"},
}


def parse_roster(source: str | Path) -> dict:
    """Parse roster spec into dict of slot counts.

    Accepts either a comma-separated string directly, or a file path.
    Tokens: QB, RB, WR, TE, K, DEF → position starters
            WRT → Flex (RB/WR/TE)
            WRTQ → Superflex (QB/RB/WR/TE)
            BN → Bench (any position)
            IR → Injured Reserve (not drafted)

    Returns dict with keys:
        starters: dict[str, int]  — position → count
        flex: list[dict]          — each has 'type' (str) and 'eligible' (list[str])
        bench: int
        ir: int
        total_roster: int         — starters + flex + bench (excludes IR)
    """
    text = read_source(source)
    tokens = [t.strip().upper() for t in text.split(",") if t.strip()]

    starters = {}
    flex_slots = []
    bench = 0
    ir = 0

    for token in tokens:
        if token in _FLEX_DEFINITIONS:
            flex_slots.append({
                "type": token,
                "eligible": sorted(_FLEX_DEFINITIONS[token]),
            })
        elif token == "BN":
            bench += 1
        elif token == "IR":
            ir += 1
        elif token in _POSITION_TOKENS:
            starters[token] = starters.get(token, 0) + 1
        # Unknown tokens silently ignored (defensive)

    total_roster = sum(starters.values()) + len(flex_slots) + bench

    return {
        "starters": dict(sorted(starters.items())),
        "flex": flex_slots,
        "bench": bench,
        "ir": ir,
        "total_roster": total_roster,
    }


