"""Tests for dynamic scoring engine — parse_scoring_rules, score_row, group_rules_by_unit."""

from scoring_engine import parse_scoring_rules, score_row, group_rules_by_unit


SAMPLE_SCORING = """# Sample scoring rules
pass_yd   | 0.05 | per_yd  | Passing yards
pass_td   | 4.00 | per_td  | Passing touchdown
pass_int  | -2.00| per_int | Interception thrown
rush_yd   | 0.10 | per_yd  | Rushing yards
rush_td   | 6.00 | per_td  | Rushing touchdown
rec       | 1.00 | per_rec | Reception
def_pa_0  | 30.00| game    | Shutout bonus
"""


# --- parse_scoring_rules tests ---


def test_parse_scoring_rules_count():
    rules = parse_scoring_rules(SAMPLE_SCORING)
    assert len(rules) == 7


def test_parse_scoring_rules_basic():
    rules = parse_scoring_rules(SAMPLE_SCORING)

    pass_yd = next(r for r in rules if r["stat_key"] == "pass_yd")
    assert pass_yd["points"] == 0.05
    assert pass_yd["unit"] == "per_yd"

    pass_int = next(r for r in rules if r["stat_key"] == "pass_int")
    assert pass_int["points"] == -2.0
    assert pass_int["unit"] == "per_int"


def test_parse_scoring_rules_skips_comments_and_blank():
    text = """# Header comment
pass_yd | 0.05 | per_yd | Yards

rush_td | 6.00 | per_td | Rush TD
# Another comment

rec | 1.00 | per_rec | PPR
"""
    rules = parse_scoring_rules(text)
    assert len(rules) == 3
    keys = [r["stat_key"] for r in rules]
    assert keys == ["pass_yd", "rush_td", "rec"]


def test_parse_scoring_rules_real_file():
    """Parse the actual scoring.txt from the project root."""
    from pathlib import Path

    scoring_path = Path(__file__).parent.parent / "scoring.txt"
    rules = parse_scoring_rules(scoring_path)
    assert len(rules) == 86

    # Spot-check a few rules
    pass_yd = next(r for r in rules if r["stat_key"] == "pass_yd")
    assert pass_yd["points"] == 0.05
    assert pass_yd["unit"] == "per_yd"

    def_pa_0 = next(r for r in rules if r["stat_key"] == "def_pa_0")
    assert def_pa_0["points"] == 30.0
    assert def_pa_0["unit"] == "game"

    # Verify game rules exist
    game_rules = [r for r in rules if r["unit"] == "game"]
    game_keys = {r["stat_key"] for r in game_rules}
    assert "def_pa_7_13" in game_keys
    assert "def_pa_35_plus" in game_keys
    assert "def_yd_lt_100" in game_keys
    assert "def_yd_100_199" in game_keys
    assert "bonus_rush_100" in game_keys
    assert "bonus_25cmp" in game_keys


# --- score_row tests ---


def test_score_row_basic_per_yd():
    rules = parse_scoring_rules("pass_yd | 0.05 | per_yd | Passing yards")
    row = {"pass_yd": 300}
    result = score_row(row, rules)
    assert result["fantasy_points"] == 15.0


def test_score_row_multiple_rules():
    rules = parse_scoring_rules(SAMPLE_SCORING)
    row = {
        "pass_yd": 250,
        "pass_td": 2,
        "pass_int": 0,
        "rush_yd": 80,
        "rush_td": 1,
        "rec": 4,
    }
    result = score_row(row, rules)
    breakdown = result["breakdown"]
    assert breakdown["pass_yd"] == 12.5    # 250 * 0.05
    assert breakdown["pass_td"] == 8.0     # 2 * 4.0
    assert breakdown["pass_int"] == 0.0    # 0 * -2.0
    assert breakdown["rush_yd"] == 8.0     # 80 * 0.10
    assert breakdown["rush_td"] == 6.0     # 1 * 6.0
    assert breakdown["rec"] == 4.0         # 4 * 1.0
    assert result["fantasy_points"] == 38.5


def test_score_row_missing_stat_defaults_to_zero():
    rules = parse_scoring_rules("pass_yd | 0.05 | per_yd | Yards")
    row = {}  # no pass_yd
    result = score_row(row, rules)
    assert result["fantasy_points"] == 0.0
    assert result["breakdown"]["pass_yd"] == 0.0


def test_score_row_negative_points():
    rules = parse_scoring_rules("pass_int | -2.00 | per_int | Interception")
    row = {"pass_int": 3}
    result = score_row(row, rules)
    assert result["fantasy_points"] == -6.0


def test_score_row_none_stat_treated_as_zero():
    rules = parse_scoring_rules("rush_yd | 0.10 | per_yd | Rush yards")
    row = {"rush_yd": None}
    result = score_row(row, rules)
    assert result["fantasy_points"] == 0.0


# --- Game rule threshold tests ---


def test_score_row_threshold_def_pa_direct_match():
    """def_pa_0 when column exists -> direct match."""
    rules = parse_scoring_rules("def_pa_0 | 30.00 | game | Shutout")
    row = {"def_pa_0": 1}
    result = score_row(row, rules)
    assert result["fantasy_points"] == 30.0


def test_score_row_threshold_def_pa_no_shutout():
    """def_pa_0 when flag is 0 -> 0 points."""
    rules = parse_scoring_rules("def_pa_0 | 30.00 | game | Shutout")
    row = {"def_pa_0": 0}
    result = score_row(row, rules)
    assert result["fantasy_points"] == 0.0


def test_score_row_threshold_def_pa_range():
    """def_pa_7_13 matches when def_pa is between 7 and 13."""
    rules = parse_scoring_rules("def_pa_7_13 | 10.00 | game | PA 7-13")
    row = {"def_pa": 10}
    result = score_row(row, rules)
    assert result["fantasy_points"] == 10.0


def test_score_row_threshold_def_pa_range_no_match():
    """def_pa_7_13 does not match when value is outside range."""
    rules = parse_scoring_rules("def_pa_7_13 | 10.00 | game | PA 7-13")
    row = {"def_pa": 21}
    result = score_row(row, rules)
    assert result["fantasy_points"] == 0.0


def test_score_row_threshold_def_pa_range_boundary_match():
    """def_pa_7_13 matches at boundaries (inclusive)."""
    rules = parse_scoring_rules("def_pa_7_13 | 10.00 | game | PA 7-13")
    # Lower boundary
    assert score_row({"def_pa": 7}, rules)["fantasy_points"] == 10.0
    # Upper boundary
    assert score_row({"def_pa": 13}, rules)["fantasy_points"] == 10.0


def test_score_row_threshold_plus():
    """def_pa_35_plus matches when def_pa >= 35."""
    rules = parse_scoring_rules("def_pa_35_plus | -5.00 | game | PA 35+")
    assert score_row({"def_pa": 35}, rules)["fantasy_points"] == -5.0
    assert score_row({"def_pa": 42}, rules)["fantasy_points"] == -5.0
    assert score_row({"def_pa": 34}, rules)["fantasy_points"] == 0.0


def test_score_row_threshold_lt():
    """def_yd_lt_100 matches when def_yd < 100."""
    rules = parse_scoring_rules("def_yd_lt_100 | 5.00 | game | < 100 yds allowed")
    assert score_row({"def_yd": 99}, rules)["fantasy_points"] == 5.0
    assert score_row({"def_yd": 0}, rules)["fantasy_points"] == 5.0
    assert score_row({"def_yd": 100}, rules)["fantasy_points"] == 0.0


def test_score_row_threshold_yd_range():
    """def_yd_100_199 matches when def_yd between 100 and 199 inclusive."""
    rules = parse_scoring_rules("def_yd_100_199 | 3.00 | game | 100-199 yds")
    assert score_row({"def_yd": 100}, rules)["fantasy_points"] == 3.0
    assert score_row({"def_yd": 150}, rules)["fantasy_points"] == 3.0
    assert score_row({"def_yd": 199}, rules)["fantasy_points"] == 3.0
    assert score_row({"def_yd": 200}, rules)["fantasy_points"] == 0.0
    assert score_row({"def_yd": 99}, rules)["fantasy_points"] == 0.0


def test_score_row_threshold_floor_rec_40():
    """rec_40 as game rule: direct column match."""
    rules = parse_scoring_rules("rec_40 | 1.00 | game | 40+ yard rec bonus")
    row = {"rec_40": 2}
    result = score_row(row, rules)
    assert result["fantasy_points"] == 2.0


# --- Bonus rule tests ---


def test_score_row_bonus_rush_100():
    """bonus_rush_100 matches when rush_yd >= 100."""
    rules = parse_scoring_rules("bonus_rush_100 | 3.00 | game | 100-199 rush yds")
    # rush_yd >= 100 -> award 3.0
    assert score_row({"rush_yd": 100}, rules)["fantasy_points"] == 3.0
    assert score_row({"rush_yd": 150}, rules)["fantasy_points"] == 3.0
    assert score_row({"rush_yd": 99}, rules)["fantasy_points"] == 0.0


def test_score_row_bonus_rush_200():
    """bonus_rush_200 matches when rush_yd >= 200."""
    rules = parse_scoring_rules("bonus_rush_200 | 5.00 | game | 200+ rush yds")
    assert score_row({"rush_yd": 200}, rules)["fantasy_points"] == 5.0
    assert score_row({"rush_yd": 199}, rules)["fantasy_points"] == 0.0


def test_score_row_bonus_rec_100():
    """bonus_rec_100 matches when rec_yd >= 100."""
    rules = parse_scoring_rules("bonus_rec_100 | 2.00 | game | 100-199 rec yds")
    assert score_row({"rec_yd": 100}, rules)["fantasy_points"] == 2.0
    assert score_row({"rec_yd": 50}, rules)["fantasy_points"] == 0.0


def test_score_row_bonus_pass_400():
    """bonus_pass_400 matches when pass_yd >= 400."""
    rules = parse_scoring_rules("bonus_pass_400 | 3.00 | game | 400+ pass yds")
    assert score_row({"pass_yd": 400}, rules)["fantasy_points"] == 3.0
    assert score_row({"pass_yd": 399}, rules)["fantasy_points"] == 0.0


def test_score_row_bonus_25cmp():
    """bonus_25cmp matches when pass_cmp >= 25 (alternate pattern: threshold first)."""
    rules = parse_scoring_rules("bonus_25cmp | 2.00 | game | 25+ completions")
    assert score_row({"pass_cmp": 25}, rules)["fantasy_points"] == 2.0
    assert score_row({"pass_cmp": 30}, rules)["fantasy_points"] == 2.0
    assert score_row({"pass_cmp": 24}, rules)["fantasy_points"] == 0.0


def test_score_row_bonus_20car():
    """bonus_20car matches when rush_att >= 20 (alternate pattern: threshold first)."""
    rules = parse_scoring_rules("bonus_20car | 2.00 | game | 20+ rush attempts")
    assert score_row({"rush_att": 20}, rules)["fantasy_points"] == 2.0
    assert score_row({"rush_att": 15}, rules)["fantasy_points"] == 0.0


def test_score_row_bonus_yd_100():
    """bonus_yd_100 matches when rush_yd + rec_yd >= 100."""
    rules = parse_scoring_rules("bonus_yd_100 | 2.00 | game | 100+ combined yds")
    assert score_row({"rush_yd": 60, "rec_yd": 40}, rules)["fantasy_points"] == 2.0
    assert score_row({"rush_yd": 100, "rec_yd": 0}, rules)["fantasy_points"] == 2.0
    assert score_row({"rush_yd": 50, "rec_yd": 49}, rules)["fantasy_points"] == 0.0


def test_score_row_bonus_yd_200():
    """bonus_yd_200 matches when rush_yd + rec_yd >= 200."""
    rules = parse_scoring_rules("bonus_yd_200 | 5.00 | game | 200+ combined yds")
    assert score_row({"rush_yd": 100, "rec_yd": 100}, rules)["fantasy_points"] == 5.0
    assert score_row({"rush_yd": 150, "rec_yd": 49}, rules)["fantasy_points"] == 0.0


# --- Integration test: multiple game rules together ---


def test_score_row_multiple_game_rules():
    """Row with both range and bonus game rules active."""
    rules_text = """rush_yd   | 0.10 | per_yd  | Rushing yards
rush_td   | 6.00 | per_td  | Rushing touchdown
bonus_rush_100 | 3.00 | game | 100+ rush yds
def_pa_7_13    | 10.00| game | PA 7-13
def_pa_35_plus | -5.00| game | PA 35+
"""
    rules = parse_scoring_rules(rules_text)
    row = {
        "rush_yd": 120,
        "rush_td": 2,
        "def_pa": 10,
    }
    result = score_row(row, rules)
    breakdown = result["breakdown"]
    # rush_yd: 120 * 0.10 = 12.0
    # rush_td: 2 * 6.0 = 12.0
    # bonus_rush_100: rush_yd >= 100 -> 3.0
    # def_pa_7_13: 10 in range -> 10.0
    # def_pa_35_plus: 10 < 35 -> 0.0
    assert breakdown["rush_yd"] == 12.0
    assert breakdown["rush_td"] == 12.0
    assert breakdown["bonus_rush_100"] == 3.0
    assert breakdown["def_pa_7_13"] == 10.0
    assert breakdown["def_pa_35_plus"] == 0.0
    assert result["fantasy_points"] == 37.0


def test_score_row_bonus_qb_1st():
    """bonus_qb_1st with per_1st unit (not game) — scored as regular per-stat rule."""
    rules = parse_scoring_rules("bonus_qb_1st | 1.00 | per_1st | First down bonus - QB")
    row = {"bonus_qb_1st": 3}
    result = score_row(row, rules)
    assert result["fantasy_points"] == 3.0
    assert result["breakdown"]["bonus_qb_1st"] == 3.0


def test_score_row_bonus_qb_1st_zero():
    """bonus_qb_1st returns 0 when stat is missing."""
    rules = parse_scoring_rules("bonus_qb_1st | 1.00 | per_1st | First down bonus - QB")
    row = {}
    result = score_row(row, rules)
    assert result["fantasy_points"] == 0.0


def test_score_row_bonus_qb_1st_from_real_file():
    """bonus_qb_1st from actual scoring.txt is per_1st, not game."""
    from pathlib import Path

    scoring_path = Path(__file__).parent.parent / "scoring.txt"
    rules = parse_scoring_rules(scoring_path)

    bonus_qb_1st = next(r for r in rules if r["stat_key"] == "bonus_qb_1st")
    assert bonus_qb_1st["unit"] == "per_1st"
    assert bonus_qb_1st["points"] == 1.0

    # It should score correctly as a per-stat rule
    row = {"bonus_qb_1st": 2}
    result = score_row(row, rules)
    assert result["breakdown"]["bonus_qb_1st"] == 2.0


# --- Error handling tests ---


def test_unknown_bonus_category_raises():
    """A bonus rule with unknown category must raise ValueError, not silently return 0."""
    import pytest
    from scoring_engine import _eval_game_rule

    rule = {"stat_key": "bonus_15sack", "points": 2.0, "unit": "game"}
    row = {"def_sack": 3}

    with pytest.raises(ValueError, match="Unknown bonus category"):
        _eval_game_rule(rule, row)


def test_unrecognized_bonus_format_raises():
    """A bonus rule that doesn't match either naming convention must raise."""
    import pytest
    from scoring_engine import _eval_game_rule

    rule = {"stat_key": "bonus_garbage_here", "points": 2.0, "unit": "game"}
    row = {}

    with pytest.raises(ValueError, match="Unrecognized bonus key format"):
        _eval_game_rule(rule, row)


# --- _find_matching_stats boundary guard test ---


def test_find_matching_stats_excludes_bare_suffix():
    """def_pa prefix must not match def_paadjusted (no underscore boundary)."""
    from scoring_engine import _find_matching_stats

    row = {"def_pa": 10, "def_paadjusted": 999}
    result = _find_matching_stats("def_pa", row)

    # Must find exact match 'def_pa'
    assert "def_pa" in result
    # Must NOT find 'def_paadjusted' — no underscore boundary
    assert "def_paadjusted" not in result


def test_find_matching_stats_includes_underscore_substats():
    """def_pa prefix must match def_pa_0, def_pa_7_13 (underscore boundary)."""
    from scoring_engine import _find_matching_stats

    row = {"def_pa": 10, "def_pa_0": 0, "def_pa_7_13": 1, "def_pa_adjusted": 5}
    result = _find_matching_stats("def_pa", row)

    assert "def_pa" in result
    assert "def_pa_0" in result
    assert "def_pa_7_13" in result
    # def_pa_adjusted has underscore boundary so it is matched
    # (nflreadpy naming conventions prevent this in practice)
    assert "def_pa_adjusted" in result
    # Exact match is always first
    assert result[0] == "def_pa"


# --- group_rules_by_unit tests ---


def test_group_rules_by_unit():
    rules = parse_scoring_rules(SAMPLE_SCORING)
    groups = group_rules_by_unit(rules)
    assert "per_yd" in groups
    assert "per_td" in groups
    assert "game" in groups
    assert len(groups["per_yd"]) == 2  # pass_yd, rush_yd
    assert len(groups["per_td"]) == 2  # pass_td, rush_td


def test_group_rules_by_unit_all_present():
    """Every rule is assigned to exactly one group."""
    rules = parse_scoring_rules(SAMPLE_SCORING)
    groups = group_rules_by_unit(rules)
    total_in_groups = sum(len(v) for v in groups.values())
    assert total_in_groups == len(rules)


# --- COLUMN_MAP and REVERSE_COLUMN_MAP tests ---


def test_column_map_maps_scoring_keys_to_nflreadpy():
    """COLUMN_MAP translates scoring.txt abbreviations to nflreadpy columns."""
    from scoring_engine import COLUMN_MAP

    assert COLUMN_MAP["pass_yd"] == "passing_yards"
    assert COLUMN_MAP["rush_yd"] == "rushing_yards"
    assert COLUMN_MAP["rec"] == "receptions"
    assert COLUMN_MAP["rec_yd"] == "receiving_yards"
    assert COLUMN_MAP["pass_td"] == "passing_tds"
    assert COLUMN_MAP["rush_td"] == "rushing_tds"
    assert COLUMN_MAP["rec_td"] == "receiving_tds"
    assert COLUMN_MAP["pass_int"] == "passing_interceptions"
    assert COLUMN_MAP["rush_att"] == "carries"
    assert COLUMN_MAP["pass_cmp"] == "completions"
    assert COLUMN_MAP["fum_lost"] == "fumbles_lost_total"
    assert COLUMN_MAP["def_sack"] == "def_sacks"
    assert COLUMN_MAP["def_int"] == "def_interceptions"


def test_reverse_column_map_is_inverse():
    """REVERSE_COLUMN_MAP maps nflreadpy columns back to some scoring key.

    Note: some nflreadpy columns (e.g. special_teams_tds) are mapped from
    multiple scoring.txt keys (st_td, stp_td). The reverse map picks one
    — this is fine because game rules don't differentiate them, and the
    forward COLUMN_MAP still resolves both correctly for per-unit lookups.
    """
    from scoring_engine import COLUMN_MAP, REVERSE_COLUMN_MAP

    # Every forward mapping's target must exist in reverse map
    for scoring_key, nfl_col in COLUMN_MAP.items():
        assert nfl_col in REVERSE_COLUMN_MAP, (
            f"nflreadpy column '{nfl_col}' (from '{scoring_key}') "
            f"missing from REVERSE_COLUMN_MAP"
        )

    # Reverse map should be at most the size of the forward map
    # (may be smaller due to collisions)
    assert len(REVERSE_COLUMN_MAP) <= len(COLUMN_MAP)

    # Verify that applying REVERSE then COLUMN yields the same nflreadpy column
    for nfl_col, scoring_key in REVERSE_COLUMN_MAP.items():
        assert COLUMN_MAP[scoring_key] == nfl_col


def test_positions_list():
    """POSITIONS list is the canonical position list."""
    from scoring_engine import POSITIONS

    assert POSITIONS == ["QB", "RB", "WR", "TE", "K", "DEF", "LB", "DB"]
    assert len(POSITIONS) == 8
    assert "DEF" in POSITIONS
    assert "LB" in POSITIONS
    assert "DB" in POSITIONS


def test_read_source_from_file():
    """read_source reads file content."""
    from pathlib import Path
    from scoring_engine import read_source

    scoring_path = Path(__file__).parent.parent / "scoring.txt"
    text = read_source(scoring_path)
    assert "pass_yd" in text
    assert "rush_td" in text


def test_read_source_inline_text():
    """read_source returns inline text when not a file path."""
    from scoring_engine import read_source

    text = read_source("pass_yd | 0.05 | per_yd")
    assert "pass_yd" in text
    assert "per_yd" in text
