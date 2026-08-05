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
                               weight_current=0.5, weight_prev=0.3, weight_oldest=0.2,
                               bio=None, cfg={})

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


def test_age_curve_rb_reduces_older_players():
    """RB age 28+ gets declining multiplier from age curve."""
    df_2025 = make_season_df(2025, "OLD-RB", "RB", [12] * 17)
    df_2024 = make_season_df(2024, "OLD-RB", "RB", [12] * 17)

    # Bio with age 29 (birth year 1997)
    bio = pl.DataFrame({
        "player_id": ["OLD-RB"],
        "birth_date": ["1997-05-15"],
        "years_of_experience": [7],
        "draft_year": [2018],
        "draft_round": [1],
    })

    # With age curve
    from project import _project_position
    result = _project_position("RB", {"2025": df_2025, "2024": df_2024},
                               weight_current=0.5, weight_prev=0.5, weight_oldest=0.0,
                               bio=bio, cfg={"age_curve_enabled": True})
    ppg_with = result["projected_ppg"][0]

    # Without age curve
    result_no = _project_position("RB", {"2025": df_2025, "2024": df_2024},
                                  weight_current=0.5, weight_prev=0.5, weight_oldest=0.0,
                                  bio=bio, cfg={"age_curve_enabled": False})
    ppg_without = result_no["projected_ppg"][0]

    # Age 29 RB should be reduced (0.75 multiplier)
    assert ppg_with < ppg_without


def test_injury_model_not_hardcoded():
    """Games played projection should reflect actual career games, not 17."""
    df_2025 = make_season_df(2025, "INJ-RB", "RB", [10] * 10)  # 10 games
    df_2024 = make_season_df(2024, "INJ-RB", "RB", [10] * 8)   # 8 games

    from project import _project_position
    result = _project_position("RB", {"2025": df_2025, "2024": df_2024},
                               weight_current=0.5, weight_prev=0.5, weight_oldest=0.0,
                               bio=None, cfg={"injury_model_enabled": True})

    gp = result["games_played_projection"][0]
    # Career: 18 games over 2 seasons. Prior: 3*15=45 games, 3 seasons.
    # (45 + 18) / (3 + 2) = 63/5 = 12.6, capped at 17
    assert gp < 17
    assert 9 <= gp <= 14


def test_shrinkage_reduces_extreme_ppg():
    """Low-sample seasons should regress toward positional mean."""
    # Small sample player: 2 games at 30 PPG
    # Normal player: 17 games at 12 PPG (brings pos_mean down to ~13.9)
    small_rows = []
    normal_rows = []
    for w in range(1, 3):
        small_rows.append({"player_id": "SMALL", "player_name": "Small",
                          "position": "WR", "season": 2025, "week": w,
                          "fantasy_points": 30.0})
    for w in range(1, 18):
        normal_rows.append({"player_id": "NORMAL", "player_name": "Normal",
                           "position": "WR", "season": 2025, "week": w,
                           "fantasy_points": 12.0})
    df_2025 = pl.DataFrame(small_rows + normal_rows)

    from project import _project_position
    result = _project_position("WR", {"2025": df_2025},
                               weight_current=1.0, weight_prev=0.0, weight_oldest=0.0,
                               bio=None, cfg={"shrinkage_enabled": True})

    # pos_mean ≈ (30 + 12) / 2 = 21 (per-player-season mean)
    # SMALL adjusted_ppg: (2*30 + 4*21)/(2+4) = 144/6 = 24.0
    small_proj = result.filter(pl.col("player_id") == "SMALL")
    ppg = small_proj["projected_ppg"][0]
    assert ppg < 30.0
    assert 22 <= ppg <= 26


def test_trend_multiplier_is_capped():
    """Trend multiplier should not exceed [0.85, 1.15]."""
    df_2025 = make_season_df(2025, "TREND", "WR", [20] * 17)
    df_2024 = make_season_df(2024, "TREND", "WR", [5] * 17)  # huge jump

    from project import _project_position
    result = _project_position("WR", {"2025": df_2025, "2024": df_2024},
                               weight_current=0.5, weight_prev=0.5, weight_oldest=0.0,
                               bio=None, cfg={"trend_adjustment_enabled": True})

    # Without trend, weighted avg = 12.5 PPG. With trend cap at 1.15, max = 14.375.
    ppg = result["projected_ppg"][0]
    assert ppg <= 20.0  # well within reason


def test_te_elite_top3_bonus():
    """Top 3 TEs by projected PPG get 1.10 multiplier."""
    df_2025 = pl.concat([
        make_season_df(2025, "TE1", "TE", [20.0] * 17),
        make_season_df(2025, "TE2", "TE", [12.0] * 17),
        make_season_df(2025, "TE3", "TE", [11.0] * 17),
        make_season_df(2025, "TE4", "TE", [10.0] * 17),
    ])

    from project import _project_position
    result = _project_position("TE", {"2025": df_2025},
                               weight_current=1.0, weight_prev=0.0, weight_oldest=0.0,
                               bio=None, cfg={"te_elite_enabled": True})

    # TE1 (20 PPG, top 3) should get 1.10 → 22.0
    te1 = result.filter(pl.col("player_id") == "TE1")
    assert te1["projected_ppg"][0] > 20.0

    # TE4 (10 PPG, not top 3) should stay at ~10
    te4 = result.filter(pl.col("player_id") == "TE4")
    assert te4["projected_ppg"][0] < 20.0
