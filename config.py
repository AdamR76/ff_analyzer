"""Configuration loader — functional, no classes.

Reads league settings from env vars with sensible defaults.
All functions return plain dicts.
"""

import os
from pathlib import Path


def load_config(draft_position: int | None = None) -> dict:
    """Return config dict with all league and pipeline settings.

    Reads env vars for overrides, falls back to defaults.
    Callers receive a plain dict — look up keys directly.

    Dict keys:
        num_teams: int          (default 12, env: FF_NUM_TEAMS)
        draft_rounds: int       (default 20, env: FF_DRAFT_ROUNDS)
        draft_position: int|None  (arg override, env: FF_DRAFT_POSITION)
        weight_current: float   (default 0.50, env: FF_WEIGHT_CURRENT)
        weight_prev: float      (default 0.30, env: FF_WEIGHT_PREV)
        weight_oldest: float    (default 0.20, env: FF_WEIGHT_OLDEST)
        scoring_file: Path      (default scoring.txt, env: FF_SCORING_FILE)
        roster_file: Path       (default roster.txt, env: FF_ROSTER_FILE)
        data_dir: Path          (default data/, env: FF_DATA_DIR)
        output_dir: Path        (default output/, env: FF_OUTPUT_DIR)
    """
    return {
        "num_teams": int(os.getenv("FF_NUM_TEAMS", "12")),
        "draft_rounds": int(os.getenv("FF_DRAFT_ROUNDS", "20")),
        "draft_position": (
            draft_position
            if draft_position is not None
            else _parse_optional_int(os.getenv("FF_DRAFT_POSITION", "5"))
        ),
        "weight_current": float(os.getenv("FF_WEIGHT_CURRENT", "0.50")),
        "weight_prev": float(os.getenv("FF_WEIGHT_PREV", "0.30")),
        "weight_oldest": float(os.getenv("FF_WEIGHT_OLDEST", "0.20")),
        "scoring_file": Path(os.getenv("FF_SCORING_FILE", "scoring.txt")),
        "roster_file": Path(os.getenv("FF_ROSTER_FILE", "roster.txt")),
        "data_dir": Path(os.getenv("FF_DATA_DIR", "data")),
        "output_dir": Path(os.getenv("FF_OUTPUT_DIR", "output")),
        # Projection feature flags
        "age_curve_enabled": os.getenv("FF_AGE_CURVE", "true").lower() == "true",
        "wr_breakout_enabled": os.getenv("FF_WR_BREAKOUT", "true").lower() == "true",
        "te_elite_enabled": os.getenv("FF_TE_ELITE", "true").lower() == "true",
        "qb_rushing_baseline_enabled": os.getenv("FF_QB_RUSHING_BASELINE", "true").lower() == "true",
        "injury_model_enabled": os.getenv("FF_INJURY_MODEL", "true").lower() == "true",
        "trend_adjustment_enabled": os.getenv("FF_TREND_ADJUST", "true").lower() == "true",
        "shrinkage_enabled": os.getenv("FF_SHRINKAGE", "true").lower() == "true",
    }


def parse_args(argv: list[str] | None = None) -> dict:
    """Parse CLI args, return overrides dict.

    Supported: --pick N, --scoring FILE, --roster FILE, --rounds N
    Keys match load_config dict keys: pick, scoring_file, roster_file, rounds
    """
    import sys

    args = argv if argv is not None else sys.argv[1:]
    overrides = {}
    i = 0
    while i < len(args):
        if args[i] == "--pick" and i + 1 < len(args):
            overrides["pick"] = int(args[i + 1])
            i += 2
        elif args[i] == "--scoring" and i + 1 < len(args):
            overrides["scoring_file"] = Path(args[i + 1])
            i += 2
        elif args[i] == "--roster" and i + 1 < len(args):
            overrides["roster_file"] = Path(args[i + 1])
            i += 2
        elif args[i] == "--rounds" and i + 1 < len(args):
            overrides["rounds"] = int(args[i + 1])
            i += 2
        elif args[i].startswith("-"):
            import warnings
            warnings.warn(
                f"Unrecognized argument: '{args[i]}'. "
                f"Supported: --pick N, --scoring FILE, --roster FILE"
            )
            # Skip the unknown flag and its next arg if it looks like a value
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                i += 2
            else:
                i += 1
        else:
            i += 1
    return overrides


def _parse_optional_int(value: str | None) -> int | None:
    """Parse env var to int, returning None for empty/unset."""
    if value is None or value.strip() == "":
        return None
    return int(value)
