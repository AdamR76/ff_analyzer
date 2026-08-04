import polars as pl
from project import run_projection_pipeline, _project_position


def make_season_df(season, player_id, position, weekly_points):
    """Helper to create game-level scored data."""
    rows = []
    for week, pts in enumerate(weekly_points, 1):
        rows.append({
            "player_id": player_id,
            "player_name": f"Player {player_id}",
            "position": position,
            "season": season,
            "week": week,
            "fantasy_points": pts,
        })
    return pl.DataFrame(rows)


def test_project_position_weighted_avg():
    """3-year weighted avg: 2025=0.5, 2024=0.3, 2023=0.2."""
    df_2025 = make_season_df(2025, "A", "WR", [10, 12, 14, 10, 12, 14, 10, 12, 14, 10, 12, 14, 10, 12, 14, 10, 12])
    df_2024 = make_season_df(2024, "A", "WR", [8, 8, 10, 8, 8, 10, 8, 8, 10, 8, 8, 10, 8, 8, 10, 8, 8])
    df_2023 = make_season_df(2023, "A", "WR", [6, 6, 6, 6, 8, 6, 6, 6, 8, 6, 6, 6, 8, 6, 6, 6, 6])

    result = _project_position("WR", {"2025": df_2025, "2024": df_2024, "2023": df_2023},
                               weight_current=0.5, weight_prev=0.3, weight_oldest=0.2)

    assert "projected_points" in result.columns
    assert "projected_ppg" in result.columns
    assert "position" in result.columns
    assert len(result) == 1


def test_run_projection_pipeline(tmp_path):
    proc_dir = tmp_path / "processed"
    proj_dir = tmp_path / "projections"
    proc_dir.mkdir(parents=True)
    proj_dir.mkdir(parents=True)

    # Create 3 seasons of scored data for one player
    for season in [2023, 2024, 2025]:
        rows = []
        for week in range(1, 18):
            pts = 10.0 + (season - 2023) * 2  # slight upward trend
            rows.append({
                "player_id": "00-001",
                "player_name": "Test RB",
                "position": "RB",
                "season": season,
                "week": week,
                "fantasy_points": pts,
            })
        df = pl.DataFrame(rows)
        df.write_parquet(proc_dir / f"{season}_scores.parquet")

    cfg = {"data_dir": tmp_path}
    result = run_projection_pipeline(cfg)

    assert result["output_path"].exists()
    proj = pl.read_parquet(result["output_path"])
    assert "projected_points" in proj.columns
    assert "projected_ppg" in proj.columns
    assert proj["position"][0] == "RB"


def test_run_projection_pipeline_with_rookies(tmp_path):
    """Pipeline merges veteran and rookie projections."""
    from project import run_projection_pipeline

    proc_dir = tmp_path / "processed"
    proj_dir = tmp_path / "projections"
    proc_dir.mkdir(parents=True)
    proj_dir.mkdir(parents=True)

    # Create veteran data
    for season in [2023, 2024, 2025]:
        rows = []
        for week in range(1, 18):
            rows.append({
                "player_id": "00-VET",
                "player_name": "Vet RB",
                "position": "RB",
                "season": season,
                "week": week,
                "fantasy_points": 15.0,
            })
        pl.DataFrame(rows).write_parquet(proc_dir / f"{season}_scores.parquet")

    cfg = {"data_dir": tmp_path}
    result = run_projection_pipeline(cfg)

    assert result["output_path"].exists()
    proj = pl.read_parquet(result["output_path"])
    assert "source" in proj.columns
    # Veteran should be marked historical
    vets = proj.filter(pl.col("source") == "historical")
    assert len(vets) >= 1
    # Rookies should be present when draft data is available
    rookies = proj.filter(pl.col("source") == "rookie_model")
    assert len(rookies) >= 1
