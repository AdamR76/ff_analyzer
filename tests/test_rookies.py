import pytest
from rookies import _round_baseline, _find_comparables, project_rookies, _get_hit_rate
import polars as pl
from pathlib import Path


def test_round_baseline_qb_first_round():
    result = _round_baseline("QB", 1)
    assert result["position"] == "QB"
    assert result["projected_points"] == 280
    assert result["projected_ppg"] == pytest.approx(280 / 17)
    assert result["source"] == "rookie_model"


def test_round_baseline_wr_fourth_plus():
    result = _round_baseline("WR", 5)
    assert result["projected_points"] == 80  # 4+ round fallback
    assert result["source"] == "rookie_model"


def test_find_comparables():
    """Find similar historical rookies by draft position."""
    rookie = {"position": "RB", "pick": 24, "round": 1}

    historical = pl.DataFrame({
        "player_id": ["A", "B", "C", "D", "E"],
        "player_name": ["RB1", "RB2", "RB3", "QB1", "RB4"],
        "position": ["RB", "RB", "RB", "QB", "RB"],
        "pick": [24, 30, 100, 25, 26],
        "round": [1, 1, 4, 1, 1],
        "fantasy_points": [200, 180, 60, 350, 190],
        "season": [2023, 2023, 2023, 2023, 2023],
    })

    result = _find_comparables(rookie, historical, top_k=3)

    assert len(result) == 3
    # Only RBs should be included (position filter)
    assert all(result["position"] == "RB")
    # Should include the exact pick match (A, pick=24)
    assert "A" in result["player_id"].to_list()


def test_find_comparables_combine_tiebreak():
    """When draft positions are similar, combine data breaks ties."""
    rookie = {"position": "WR", "pick": 10, "round": 1, "forty": 4.40, "vertical": 38}

    historical = pl.DataFrame({
        "player_id": ["W1", "W2", "W3", "W4"],
        "player_name": ["Fast", "Slow", "Fast2", "Med"],
        "position": ["WR", "WR", "WR", "WR"],
        "pick": [8, 9, 12, 11],
        "round": [1, 1, 1, 1],
        "forty": [4.40, 4.65, 4.41, 4.52],
        "vertical": [39, 32, 37, 35],
        "fantasy_points": [220, 140, 210, 180],
        "season": [2023, 2023, 2023, 2023],
    })

    result = _find_comparables(rookie, historical, top_k=2)
    assert len(result) == 2
    # W1 (forty=4.40) is exact match, should be closest
    assert result["player_id"][0] == "W1"


def test_project_rookies_basic(tmp_path):
    """project_rookies merges draft data with scored historical data."""
    scored_dir = tmp_path / "scored"
    scored_dir.mkdir()

    # Create a minimal scored parquet with historical rookies
    historical = pl.DataFrame({
        "player_id": ["00-001", "00-002"],
        "player_name": ["Hist RB1", "Hist RB2"],
        "position": ["RB", "RB"],
        "season": [2024, 2024],
        "week": [1, 1],
        "fantasy_points": [18.0, 12.0],
    })
    historical.write_parquet(scored_dir / "2024_scores.parquet")

    draft_data = pl.DataFrame({
        "season": [2026, 2026],
        "round": [1, 3],
        "pick": [12, 72],
        "team": ["CHI", "DAL"],
        "gsis_id": ["00-003", "00-004"],
        "pfr_player_name": ["Rookie RB", "Rookie WR"],
        "position": ["RB", "WR"],
    })

    result = project_rookies(scored_dir, draft_data)

    assert len(result) == 2
    # Columns match veteran projection schema
    for col in ["player_id", "player_name", "position", "team",
                "projected_points", "projected_ppg", "ceiling", "floor",
                "games_played_projection", "source"]:
        assert col in result.columns
    assert all(result["source"] == "rookie_model")
    assert all(result["games_played_projection"] == 17)


def test_project_rookies_with_comparables(tmp_path):
    """Comparable-player path: scored data with pick column triggers comparables."""
    scored_dir = tmp_path / "scored"
    scored_dir.mkdir()

    # Historical scored data WITH pick column (simulates draft-joined data)
    historical = pl.DataFrame({
        "player_id": ["00-H1", "00-H2", "00-H3", "00-H4", "00-H5"],
        "player_name": ["Hist A", "Hist B", "Hist C", "Hist D", "Hist E"],
        "position": ["RB", "RB", "RB", "RB", "RB"],
        "pick": [10, 15, 20, 25, 30],
        "round": [1, 1, 1, 1, 1],
        "fantasy_points": [15.0, 14.0, 13.0, 12.0, 11.0],
        "season": [2024, 2024, 2024, 2024, 2024],
        "week": [1, 1, 1, 1, 1],
    })
    historical.write_parquet(scored_dir / "2024_scores.parquet")

    # 2026 rookie with similar draft capital to historical
    draft_data = pl.DataFrame({
        "season": [2026],
        "round": [1],
        "pick": [12],
        "team": ["CHI"],
        "gsis_id": ["00-R1"],
        "pfr_player_name": ["Rookie RB1"],
        "position": ["RB"],
    })

    result = project_rookies(scored_dir, draft_data)

    assert len(result) == 1
    assert result["player_id"][0] == "00-R1"
    assert result["source"][0] == "rookie_model"
    # With 5 comparables (all RBs), should use comparable avg, not baseline
    # Baseline for RB round 1 is 240 pts (14.12 ppg). Comparables avg ~13 ppg.
    # RB round-1 hit rate is 0.80, so raw 13.0 ppg → 10.4 ppg adjusted.
    assert result["projected_ppg"][0] != pytest.approx(240 / 17)
    assert result["projected_points"][0] != pytest.approx(240)
    assert pytest.approx(result["projected_ppg"][0], rel=0.05) == 10.4


def test_project_rookies_deduplicates_existing_players(tmp_path):
    """Rookies already in scored data (veterans) are excluded."""
    scored_dir = tmp_path / "scored"
    scored_dir.mkdir()

    historical = pl.DataFrame({
        "player_id": ["00-EXISTING"],
        "player_name": ["Vet Player"],
        "position": ["QB"],
        "season": [2024],
        "week": [1],
        "fantasy_points": [22.0],
    })
    historical.write_parquet(scored_dir / "2024_scores.parquet")

    draft_data = pl.DataFrame({
        "season": [2026],
        "round": [1],
        "pick": [1],
        "team": ["CAR"],
        "gsis_id": ["00-EXISTING"],  # Same ID as veteran
        "pfr_player_name": ["Vet Player"],
        "position": ["QB"],
    })

    result = project_rookies(scored_dir, draft_data)
    assert len(result) == 0  # Already has data, skip rookie model


def test_get_hit_rate_wr_round_1():
    """WR round 1 hit rate is 0.65."""
    assert _get_hit_rate("WR", 1) == 0.65


def test_get_hit_rate_rb_round_4_plus():
    """RB round 4+ uses round-4 tier: 0.25."""
    assert _get_hit_rate("RB", 5) == 0.25


def test_get_hit_rate_idp_default():
    """K/DEF/LB/DB positions default to 1.0."""
    assert _get_hit_rate("LB", 1) == 1.0
    assert _get_hit_rate("K", 3) == 1.0


def test_get_hit_rate_unknown_position():
    """Unknown position gets 0.50 default."""
    assert _get_hit_rate("XX", 1) == 0.50


def test_round_baseline_raw_values():
    """_round_baseline returns raw values. Hit rate applied in project_rookies."""
    result = _round_baseline("WR", 1)
    assert result["projected_points"] == 220
    assert result["projected_ppg"] == pytest.approx(220 / 17)
    assert result["position"] == "WR"
