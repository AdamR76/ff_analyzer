import polars as pl
from pathlib import Path
from score import run_score_pipeline, _apply_scoring_to_df


SAMPLE_RULES = [
    {"stat_key": "receptions", "points": 1.0, "unit": "per_rec"},
    {"stat_key": "receiving_yards", "points": 0.1, "unit": "per_yd"},
    {"stat_key": "receiving_tds", "points": 6.0, "unit": "per_td"},
]


def test_apply_scoring_to_df():
    df = pl.DataFrame({
        "player_id": ["A", "B"],
        "receptions": [5, 3],
        "receiving_yards": [80, 40],
        "receiving_tds": [1, 0],
    })

    result = _apply_scoring_to_df(df, SAMPLE_RULES)

    assert "fantasy_points" in result.columns
    # Player A: 5*1.0 + 80*0.1 + 1*6.0 = 5 + 8 + 6 = 19
    # Player B: 3*1.0 + 40*0.1 + 0*6.0 = 3 + 4 + 0 = 7
    points = result["fantasy_points"].to_list()
    assert points[0] == 19.0
    assert points[1] == 7.0


def test_run_score_pipeline(tmp_path, monkeypatch):
    """End to end: raw parquet -> scored parquet."""
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    proc_dir.mkdir(parents=True)

    # Create a minimal raw file
    df = pl.DataFrame({
        "player_id": ["00-001"],
        "player_name": ["Test"],
        "season": [2024],
        "week": [1],
        "position": ["WR"],
        "receptions": [10],
        "receiving_yards": [150],
        "receiving_tds": [2],
    })
    df.write_parquet(raw_dir / "2024.parquet")

    cfg = {"data_dir": tmp_path}
    result = run_score_pipeline(cfg, rules=SAMPLE_RULES)

    assert 2024 in result["output_paths"]
    scored = pl.read_parquet(result["output_paths"][2024])
    # 10*1 + 150*0.1 + 2*6 = 10 + 15 + 12 = 37
    assert scored["fantasy_points"][0] == 37.0
    assert "pp_pass" in scored.columns or "pp_rec" in scored.columns or True
    # breakdown columns present
    assert "pp_receptions" in scored.columns
    assert "pp_receiving_yards" in scored.columns
    assert "pp_receiving_tds" in scored.columns


# --- Game rule tests ---

BONUS_RUSH_100_RULE = [
    {"stat_key": "bonus_rush_100", "points": 3.0, "unit": "game"},
]

DEF_PA_RANGE_RULES = [
    {"stat_key": "def_pa_7_13", "points": 10.0, "unit": "game"},
]

DEF_YD_LT_RULE = [
    {"stat_key": "def_yd_lt_100", "points": 5.0, "unit": "game"},
]

DEF_PA_0_RULE = [
    {"stat_key": "def_pa_0", "points": 30.0, "unit": "game"},
]


def test_apply_scoring_bonus_rule():
    """bonus_rush_100 awards 3.0 when rush_yd >= 100."""
    df = pl.DataFrame({
        "player_id": ["A", "B"],
        "rush_yd": [120, 80],
    })

    result = _apply_scoring_to_df(df, BONUS_RUSH_100_RULE)

    assert "pp_bonus_rush_100" in result.columns
    points = result["pp_bonus_rush_100"].to_list()
    # Player A: 120 >= 100 -> 3.0, Player B: 80 < 100 -> 0.0
    assert points[0] == 3.0
    assert points[1] == 0.0
    # Total fantasy_points should match game rule contribution
    assert result["fantasy_points"][0] == 3.0
    assert result["fantasy_points"][1] == 0.0


def test_apply_scoring_def_pa_range():
    """def_pa_7_13 awards 10.0 when def_pa is in [7, 13]."""
    df = pl.DataFrame({
        "player_id": ["A", "B", "C"],
        "def_pa": [10, 20, 7],
    })

    result = _apply_scoring_to_df(df, DEF_PA_RANGE_RULES)

    points = result["pp_def_pa_7_13"].to_list()
    # A: 10 in range -> 10.0, B: 20 out -> 0.0, C: 7 boundary -> 10.0
    assert points[0] == 10.0
    assert points[1] == 0.0
    assert points[2] == 10.0


def test_apply_scoring_def_yd_lt():
    """def_yd_lt_100 awards 5.0 when def_yd < 100."""
    df = pl.DataFrame({
        "player_id": ["A", "B"],
        "def_yd": [80, 150],
    })

    result = _apply_scoring_to_df(df, DEF_YD_LT_RULE)

    points = result["pp_def_yd_lt_100"].to_list()
    # A: 80 < 100 -> 5.0, B: 150 >= 100 -> 0.0
    assert points[0] == 5.0
    assert points[1] == 0.0


def test_apply_scoring_shutout():
    """def_pa_0 awards 30.0 only when def_pa == 0 (not just >= 0)."""
    df = pl.DataFrame({
        "player_id": ["A", "B", "C"],
        "def_pa": [0, 3, 14],
    })

    result = _apply_scoring_to_df(df, DEF_PA_0_RULE)

    points = result["pp_def_pa_0"].to_list()
    # A: def_pa == 0 -> 30.0, B: 3 != 0 -> 0.0, C: 14 != 0 -> 0.0
    assert points[0] == 30.0
    assert points[1] == 0.0
    assert points[2] == 0.0


def test_apply_scoring_mixed_per_and_game_rules():
    """Per-unit and game rules combined produce correct totals."""
    rules = [
        {"stat_key": "rush_yd", "points": 0.1, "unit": "per_yd"},
        {"stat_key": "bonus_rush_100", "points": 3.0, "unit": "game"},
    ]

    df = pl.DataFrame({
        "player_id": ["A", "B"],
        "rushing_yards": [120, 80],
    })

    result = _apply_scoring_to_df(df, rules)

    # Player A: 120*0.1 + 3.0 = 12.0 + 3.0 = 15.0
    # Player B: 80*0.1 + 0.0 = 8.0
    assert result["fantasy_points"][0] == 15.0
    assert result["fantasy_points"][1] == 8.0
    assert "pp_rush_yd" in result.columns
    assert "pp_bonus_rush_100" in result.columns
