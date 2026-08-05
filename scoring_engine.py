"""Dynamic scoring engine — functional, no classes.

Parses scoring.txt at runtime. All rules are plain dicts.
No stat-to-points mapping is ever hardcoded.
"""

import re
from pathlib import Path


# ── COLUMN_NAME BRIDGE ──────────────────────────────────────────────
# Maps scoring.txt stat_key to the actual nflreadpy column name.
# Keys NOT in this dict use their own name as-is (identity passthrough).
# The reverse map REVERSE_COLUMN_MAP is computed from this dict and
# used by game-rule evaluation to translate nflreadpy row dicts back
# to scoring-engine keys.

COLUMN_MAP: dict[str, str] = {
    # Passing
    "pass_yd":   "passing_yards",
    "pass_td":   "passing_tds",
    "pass_2pt":  "passing_2pt_conversions",
    "pass_int":  "passing_interceptions",
    "pass_cmp":  "completions",
    "qb_sack":   "sacks_suffered",
    "bonus_qb_1st": "passing_first_downs",
    # pick_six — no direct nflreadpy column (needs pbp)
    # pass_td_40 / pass_td_50 — nflreadpy does not break out TD distance (needs pbp)

    # Rushing
    "rush_yd":   "rushing_yards",
    "rush_td":   "rushing_tds",
    "rush_2pt":  "rushing_2pt_conversions",
    "rush_att":  "carries",
    # rush_td_40 / rush_td_50 — nflreadpy does not break out TD distance (needs pbp)

    # Receiving
    "rec":       "receptions",
    "rec_yd":    "receiving_yards",
    "rec_td":    "receiving_tds",
    "rec_2pt":   "receiving_2pt_conversions",
    "rec_40":    "receiving_40",
    # rec_30_39 — no 30-39 bucket in nflreadpy
    # rec_td_40 / rec_td_50 — TD distance not tracked separately (needs pbp)

    # Kicking
    "pat_made":      "pat_made",
    "fg_miss_0_19":  "fg_missed_0_19",
    "fg_miss_20_29": "fg_missed_20_29",
    "fg_miss_30_39": "fg_missed_30_39",
    "fg_miss_40_49": "fg_missed_40_49",
    "fg_miss_50_59": "fg_missed_50_59",
    "pat_miss":      "pat_missed",
    "fg_yd":         "fg_made_distance",

    # Team Defense (individual player-level; also available in team_stats)
    "def_td":      "def_tds",
    "def_sack":    "def_sacks",
    "def_int":     "def_interceptions",
    "def_fum_rec": "fumble_recovery_opp",
    "def_tfl":     "def_tackles_for_loss",
    "def_safety":  "def_safeties",
    "def_ff":      "def_fumbles_forced",
    # def_blk_kick, def_2pt_ret — not in nflreadpy season stats (needs pbp)
    # def_3nout, def_4down_stop — need play-by-play, not in season stats
    # def_pa_* / def_yd_* — need opponent stats via schedule+pbp join

    # Special Teams
    "st_td":  "special_teams_tds",
    "stp_td": "special_teams_tds",
    # st_ff, st_fum_rec, stp_ff, stp_fum_rec — not in nflreadpy

    # Miscellaneous
    "fum_lost":   "fumbles_lost_total",
    "fum_rec_td": "fumble_recovery_tds",

    # IDP (Individual Defensive Players) — new for LB/DB roster spots
    "idp_td":         "def_tds",
    "idp_sack":       "def_sacks",
    "idp_qb_hit":     "def_qb_hits",
    "idp_tfl":        "def_tackles_for_loss",
    "idp_int":        "def_interceptions",
    "idp_int_yd":     "def_interception_yards",
    "idp_fum_rec":    "fumble_recovery_opp",
    "idp_fum_yd":     "fumble_recovery_yards_opp",
    "idp_ff":         "def_fumbles_forced",
    "idp_safety":     "def_safeties",
    "idp_ast_tackle": "def_tackle_assists",
    "idp_solo_tackle":"def_tackles_solo",
    "idp_pass_def":   "def_pass_defended",
    # idp_tackle — derived in score.py from solo + assist, not mapped here
    # idp_blk_kick — not in nflreadpy season stats (needs pbp)
    # idp_tackle_10 — game rule, checks derived idp_tackle >= 10
    # idp_int_td_50 / idp_fum_td_50 — TD distance not tracked (needs pbp)
}

# Reverse map: nflreadpy column → scoring.txt key.
# Built once at import so game-rule row translation is O(1) per column.
REVERSE_COLUMN_MAP: dict[str, str] = {v: k for k, v in COLUMN_MAP.items()}

# Canonical position list — single source of truth for all modules.
POSITIONS: list[str] = ["QB", "RB", "WR", "TE", "K", "DEF", "LB", "DB"]

# Positions that are individual defensive players (not team defense).
# Used for per-position rule filtering (idp_* rules apply only to these).
IDP_POSITIONS: set[str] = {
    "LB", "DB", "OLB", "ILB", "MLB", "DE", "DT", "NT", "DL",
    "CB", "S", "FS", "SS", "SAF",
}


def normalize_position(pos: str) -> str:
    """Map nflreadpy position codes to canonical POSITIONS list.

    Defensive backs (CB, S, FS, etc.) → DB.
    Front-seven defenders (DE, DT, OLB, etc.) → LB.
    Fullback → RB. Offensive line → '' (skip).
    Unknown positions pass through unchanged.
    """
    pos = pos.upper().strip()
    if pos in ("CB", "S", "FS", "SS", "SAF", "DB"):
        return "DB"
    if pos in ("LB", "OLB", "ILB", "MLB", "NT", "DT", "DE", "DL"):
        return "LB"  # DL/DE/DT/NT all map to LB for IDP scoring
    if pos in ("FB",):
        return "RB"
    if pos in ("G", "C", "T", "OT", "OG", "OL", "LS", "P"):
        return ""  # OL/P don't score fantasy points, skip
    return pos


# ── source reading (shared utility) ──────────────────────────────────


def read_source(source: str | Path) -> str:
    """Read text from a file path, or return inline string directly.

    If source is a Path it is treated as a file. For strings: try as a
    file path first; if that fails, return the string as inline rules text.
    """
    if isinstance(source, Path):
        return source.read_text()

    s = str(source)
    try:
        p = Path(s)
        if p.exists():
            return p.read_text()
    except OSError:
        pass

    return s


def parse_scoring_rules(source: str | Path) -> list[dict]:
    """Parse pipe-delimited scoring rules into list of plain dicts.

    Each rule dict: {"stat_key": str, "points": float, "unit": str}
    Units: per_yd, per_td, per_rec, per_cmp, per_att, per_1st,
           per_int, per_sack, per_fum, per_ff, per_tfl, per_safety,
           per_block, per_punt, per_3nout, per_stop, per_pat,
           per_miss, per_2pt, game

    Skips # comments and blank lines.
    """
    text = read_source(source)
    rules = []

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "|" not in stripped:
            continue

        parts = [p.strip() for p in stripped.split("|")]
        if len(parts) < 3:
            continue

        rules.append({
            "stat_key": parts[0],
            "points": float(parts[1]),
            "unit": parts[2],
        })

    return rules


def score_row(row: dict, rules: list[dict]) -> dict:
    """Apply all scoring rules to a single player-game row.

    row: dict with stat column names as keys
    rules: list of rule dicts from parse_scoring_rules()

    Returns: {"fantasy_points": float, "breakdown": dict[str, float]}
      breakdown maps stat_key -> points scored from that rule.
    """
    total = 0.0
    breakdown = {}

    for rule in rules:
        key = rule["stat_key"]
        unit = rule["unit"]
        multiplier = rule["points"]

        stat_value = row.get(key, 0) or 0

        if unit == "game":
            pts = _eval_game_rule(rule, row)
        elif unit.startswith("per_"):
            pts = stat_value * multiplier
        else:
            pts = 0.0

        total += pts
        breakdown[key] = pts

    return {"fantasy_points": total, "breakdown": breakdown}


def group_rules_by_unit(rules: list[dict]) -> dict[str, list[dict]]:
    """Group parsed rules by unit type for efficient batch application.

    Returns dict like: {"per_yd": [...], "per_td": [...], "game": [...]}
    """
    groups = {}
    for rule in rules:
        unit = rule["unit"]
        if unit not in groups:
            groups[unit] = []
        groups[unit].append(rule)
    return groups


def _eval_game_rule(rule: dict, row: dict) -> float:
    """Evaluate a 'game' unit rule against a row.

    Handles these key patterns (checked in order):

    1. Direct column match — key exists as-is in row data
       e.g. def_pa_0 -> row["def_pa_0"]

    2. Bonus prefix — bonus_* rules use a known category->stat mapping
       e.g. bonus_rush_100 -> rush_yd >= 100
       e.g. bonus_25cmp   -> pass_cmp >= 25
       e.g. bonus_yd_100  -> rush_yd + rec_yd >= 100

    3. Less-than pattern — stat_lt_N
       e.g. def_yd_lt_100 -> def_yd < 100

    4. Plus pattern — stat_N_plus
       e.g. def_pa_35_plus -> def_pa >= 35

    5. Range pattern — stat_lo_hi
       e.g. def_pa_7_13  -> 7 <= def_pa <= 13
       e.g. def_yd_100_199 -> 100 <= def_yd <= 199

    6. Floor pattern — stat_N (N > 0 means >=, N == 0 means ==)
       e.g. def_pa_0 (threshold=0) -> def_pa == 0
    """
    key = rule["stat_key"]
    multiplier = rule["points"]

    # 1. Direct match: key exists as a column in the row
    if key in row:
        val = row[key]
        if val is not None:
            return float(val) * multiplier

    # 2. Bonus rules — need category-to-stat mapping
    if key.startswith("bonus_"):
        return _eval_bonus(key, row, multiplier)

    # 3. Less-than pattern: stat_lt_N (e.g. def_yd_lt_100)
    m = re.match(r'^(.+)_lt_(\d+)$', key)
    if m:
        stat_prefix = m.group(1)
        threshold = int(m.group(2))
        for sn in _find_matching_stats(stat_prefix, row):
            val = row.get(sn, 0) or 0
            if 0 <= val < threshold:
                return multiplier
        return 0.0

    # 4. Plus pattern: stat_N_plus (e.g. def_pa_35_plus)
    m = re.match(r'^(.+)_(\d+)_plus$', key)
    if m:
        stat_prefix = m.group(1)
        threshold = int(m.group(2))
        for sn in _find_matching_stats(stat_prefix, row):
            val = row.get(sn, 0) or 0
            if val >= threshold:
                return multiplier
        return 0.0

    # 5. Range pattern: stat_lo_hi (e.g. def_pa_7_13, def_yd_100_199)
    m = re.match(r'^(.+)_(\d+)_(\d+)$', key)
    if m:
        stat_prefix = m.group(1)
        lo = int(m.group(2))
        hi = int(m.group(3))
        for sn in _find_matching_stats(stat_prefix, row):
            val = row.get(sn, 0) or 0
            if lo <= val <= hi:
                return multiplier
        return 0.0

    # 6. Floor pattern: stat_N (e.g. rec_40 when unit=game)
    #    threshold==0 uses equality (def_pa_0 -> def_pa == 0)
    #    threshold>0 uses >= (bonus_rush_100 -> rush_yd >= 100)
    m = re.match(r'^(.+)_(\d+)$', key)
    if m:
        stat_prefix = m.group(1)
        threshold = int(m.group(2))
        for sn in _find_matching_stats(stat_prefix, row):
            val = row.get(sn, 0) or 0
            if threshold == 0:
                if val == 0:
                    return multiplier
            else:
                if val >= threshold:
                    return multiplier
        return 0.0

    return 0.0


# Mapping from bonus category suffix to actual stat column name.
# Used by _eval_bonus to resolve bonus_* game rules.
# Convention: bonus_{category}_{threshold} -> stat_column >= threshold
#             bonus_{threshold}{category}  -> stat_column >= threshold
# Category 'yd' is special: sums rush_yd + rec_yd.
# To add a new category (e.g. 'sack' -> 'def_sack'), add it here.
_BONUS_CATEGORY_STAT = {
    "rush": "rush_yd",
    "rec": "rec_yd",
    "pass": "pass_yd",
    "cmp": "pass_cmp",
    "car": "rush_att",
}


def _eval_bonus(key: str, row: dict, multiplier: float) -> float:
    """Evaluate a bonus_* game rule.

    Two naming conventions:
      bonus_{category}_{threshold}  e.g. bonus_rush_100, bonus_yd_100
      bonus_{threshold}{category}   e.g. bonus_25cmp, bonus_20car

    The category 'yd' is special: it sums rush_yd + rec_yd.
    """
    rest = key[len("bonus_"):]

    # Pattern: {category}_{threshold} (e.g. rush_100, rec_200, yd_100)
    m = re.match(r'^([a-z]+)_(\d+)$', rest)
    if m:
        category = m.group(1)
        threshold = int(m.group(2))
        return _check_bonus_category(category, threshold, row, multiplier)

    # Pattern: {threshold}{category} (e.g. 25cmp, 20car)
    m = re.match(r'^(\d+)([a-z]+)$', rest)
    if m:
        threshold = int(m.group(1))
        category = m.group(2)
        return _check_bonus_category(category, threshold, row, multiplier)

    raise ValueError(
        f"Unrecognized bonus key format: '{key}'. "
        f"Expected bonus_{{category}}_{{threshold}} (e.g. bonus_rush_100) "
        f"or bonus_{{threshold}}{{category}} (e.g. bonus_25cmp)."
    )


def _check_bonus_category(category: str, threshold: int, row: dict, multiplier: float) -> float:
    """Check a bonus category against the row and return points if threshold met.

    Raises ValueError for unknown categories so new bonus rules fail
    loudly instead of silently scoring 0.
    """
    if category in _BONUS_CATEGORY_STAT:
        stat_col = _BONUS_CATEGORY_STAT[category]
        val = row.get(stat_col, 0) or 0
        return multiplier if val >= threshold else 0.0

    if category == "yd":
        rush = row.get("rush_yd", 0) or 0
        rec = row.get("rec_yd", 0) or 0
        return multiplier if (rush + rec) >= threshold else 0.0

    raise ValueError(
        f"Unknown bonus category '{category}' in rule key. "
        f"Known categories: {sorted(_BONUS_CATEGORY_STAT.keys())} + 'yd'. "
        f"Add new categories to _BONUS_CATEGORY_STAT in scoring_engine.py."
    )


def _find_matching_stats(prefix: str, row: dict) -> list[str]:
    """Find row keys matching the prefix as a complete stat-name token.

    Matches key == prefix exactly, or key starting with prefix + '_'
    (so def_pa matches def_pa_0 but not def_paadjusted).
    Returns exact match first if it exists, then prefix_ matches.
    """
    exact = None
    prefixed = []
    boundary = prefix + "_"
    for k in row:
        if k == prefix:
            exact = k
        elif k.startswith(boundary):
            prefixed.append(k)
    if exact is not None:
        return [exact] + prefixed
    return prefixed


