import polars as pl
from pathlib import Path
from rank import run_rank_pipeline, _simulate_snake_draft


def test_simulate_snake_draft_pick_1():
    """Pick 1 in 12-team snake draft: picks are 1, 24, 25, 48, 49..."""
    picks = _simulate_snake_draft(pick_position=1, num_teams=12, rounds=18)
    assert picks[0] == 1
    assert picks[1] == 24
    assert picks[2] == 25
    assert picks[3] == 48
    assert len(picks) == 18


def test_simulate_snake_draft_20_rounds():
    """Pick 1 in 12-team, 20-round snake draft."""
    picks = _simulate_snake_draft(pick_position=1, num_teams=12, rounds=20)
    assert len(picks) == 20
    assert picks[0] == 1
    assert picks[1] == 24
    assert picks[-1] == 240  # last pick of round 20


def test_simulate_snake_draft_pick_12():
    """Pick 12 in 12-team: picks are 12, 13, 36, 37..."""
    picks = _simulate_snake_draft(pick_position=12, num_teams=12, rounds=18)
    assert picks[0] == 12
    assert picks[1] == 13
    assert picks[2] == 36
    assert picks[3] == 37


def test_run_rank_pipeline(tmp_path):
    proj_dir = tmp_path / "projections"
    out_dir = tmp_path / "output"
    proj_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)

    projections = pl.DataFrame({
        "player_id": ["A", "B", "C", "D", "E", "F"],
        "player_name": ["QB1", "RB1", "WR1", "QB2", "RB2", "WR2"],
        "position": ["QB", "RB", "WR", "QB", "RB", "WR"],
        "projected_points": [400, 320, 300, 350, 280, 270],
        "projected_ppg": [23.5, 18.8, 17.6, 20.6, 16.5, 15.9],
        "ceiling": [28.0, 22.0, 21.0, 25.0, 20.0, 19.0],
        "floor": [18.0, 14.0, 13.0, 15.0, 12.0, 11.0],
        "games_played_projection": [17, 17, 17, 17, 17, 17],
    })
    projections.write_parquet(proj_dir / "2026_projections.parquet")

    roster = {
        "starters": {"QB": 1, "RB": 2, "WR": 1, "TE": 1, "K": 2, "DEF": 1},
        "flex": [
            {"type": "WRT", "eligible": ["RB", "WR", "TE"]},
            {"type": "WRT", "eligible": ["RB", "WR", "TE"]},
            {"type": "WRT", "eligible": ["RB", "WR", "TE"]},
            {"type": "WRTQ", "eligible": ["QB", "RB", "WR", "TE"]},
        ],
        "bench": 6,
        "total_roster": 18,
    }

    cfg = {
        "data_dir": tmp_path,
        "output_dir": out_dir,
        "num_teams": 12,
        "draft_rounds": 18,
        "draft_position": 5,
    }

    result = run_rank_pipeline(cfg, roster=roster)

    assert result["rankings"].exists()
    assert result["tiers"].exists()
    assert result["strategy"].exists()
    assert result.get("mock_draft") is not None
    assert result["mock_draft"].exists()

    rankings = pl.read_csv(result["rankings"])
    assert "vorp" in rankings.columns
    assert "tier" in rankings.columns
    assert "position_rank" in rankings.columns

    tiers = pl.read_csv(result["tiers"])
    assert "overall_rank" in tiers.columns
    assert "position" in tiers.columns
    assert "vorp" in tiers.columns
    assert "tier" in tiers.columns
    assert "position_rank" in tiers.columns

    strategy = pl.read_csv(result["strategy"])
    assert "round" in strategy.columns
    assert "overall_pick" in strategy.columns
    assert "best_qb" in strategy.columns
    assert "best_rb" in strategy.columns
    assert "best_wr" in strategy.columns
    assert "best_te" in strategy.columns
    assert len(strategy) == 54  # 18 rounds * 3 options
    assert "option" in strategy.columns
    assert strategy["option"].to_list() == [1, 2, 3] * 18

    mock_df = pl.read_csv(result["mock_draft"])
    assert "overall_pick" in mock_df.columns
    assert "team_slot" in mock_df.columns
    assert "player_name" in mock_df.columns
    assert "position" in mock_df.columns
    assert len(mock_df) > 0
