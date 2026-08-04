"""Fetch stage — pull NFL player stats via nflreadpy, store as Parquet.

Functional, no classes. nflreadpy returns Polars DataFrames directly.
"""

from pathlib import Path
import nflreadpy as nfl
import polars as pl


def fetch_raw_data(cfg: dict, seasons: list[int] | None = None) -> dict:
    """Fetch player stats from nflreadpy, write per-season Parquet files.

    cfg: config dict from load_config()
    seasons: list of season years to fetch (default: 2023, 2024, 2025)

    Returns dict with keys:
        seasons: list[int]       — seasons fetched
        output_paths: dict[int, Path]  — season → parquet path
        row_counts: dict[int, int]     — season → row count
    """
    if seasons is None:
        seasons = [2023, 2024, 2025]

    raw_dir = cfg["data_dir"] / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    df = _call_nflreadpy(seasons)
    df = _filter_regular_season(df)

    output_paths = {}
    row_counts = {}

    for season in seasons:
        season_df = df.filter(pl.col("season") == season)
        path = raw_dir / f"{season}.parquet"
        season_df.write_parquet(path)
        output_paths[season] = path
        row_counts[season] = len(season_df)

    return {
        "seasons": seasons,
        "output_paths": output_paths,
        "row_counts": row_counts,
    }


def fetch_team_defense(cfg: dict, seasons: list[int] | None = None) -> dict:
    """Fetch team-level defense stats from nflreadpy, write per-season Parquet.

    nflreadpy.load_team_stats() returns team-level aggregates per game.
    We tag every row with position="DEF" so the scoring engine can apply
    defense rules (def_td, def_sack, def_int, def_ff, etc.).

    Stat columns available via load_team_stats (direct mapping):
        def_tds, def_sacks, def_interceptions, fumble_recovery_opp,
        def_tackles_for_loss, def_safeties, def_fumbles_forced,
        special_teams_tds

    Stats NOT available in team_stats (need schedule + pbp join):
        def_pa_*   — points allowed (opponent scores from schedule join)
        def_yd_*   — yards allowed (opponent yards from schedule join)
        def_3nout   — three-and-outs forced (pbp)
        def_4down_stop — fourth-down stops (pbp)
        def_blk_kick — blocked kicks (pbp)
        def_punt_force — forced punts (pbp)
        def_2pt_ret — defensive 2pt returns (pbp)

    cfg: config dict from load_config()
    seasons: list of season years to fetch (default: 2023, 2024, 2025)

    Returns dict with keys:
        seasons: list[int]
        output_paths: dict[int, Path]
        row_counts: dict[int, int]
    """
    if seasons is None:
        seasons = [2023, 2024, 2025]

    raw_dir = cfg["data_dir"] / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    df = _call_nflreadpy_team(seasons)
    df = _filter_regular_season(df)

    # Tag as team defense — every row gets position="DEF"
    # Use team abbreviation as player_id and team name as player_name
    df = df.with_columns([
        pl.lit("DEF").alias("position"),
        pl.col("team").alias("player_id"),
        pl.col("team").alias("player_name"),
    ])

    output_paths = {}
    row_counts = {}

    for season in seasons:
        season_df = df.filter(pl.col("season") == season)
        path = raw_dir / f"{season}_def.parquet"
        season_df.write_parquet(path)
        output_paths[season] = path
        row_counts[season] = len(season_df)

    return {
        "seasons": seasons,
        "output_paths": output_paths,
        "row_counts": row_counts,
    }


def _call_nflreadpy(seasons: list[int]) -> pl.DataFrame:
    """Call nflreadpy for game-level player stats. Extracted for testability."""
    return nfl.load_player_stats(seasons=seasons, summary_level="week")


def _call_nflreadpy_team(seasons: list[int]) -> pl.DataFrame:
    """Call nflreadpy for game-level team stats. Extracted for testability."""
    return nfl.load_team_stats(seasons=seasons, summary_level="week")


def _filter_regular_season(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only regular season weeks (exclude preseason and playoffs)."""
    return df.filter(pl.col("season_type") == "REG")
