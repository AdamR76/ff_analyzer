"""Fetch stage — pull NFL player stats via nflreadpy, store as Parquet.

Functional, no classes. nflreadpy returns Polars DataFrames directly.
"""

from pathlib import Path
import nflreadpy as nfl
import polars as pl


def fetch_raw_data(cfg: dict, seasons: list[int] | None = None) -> dict:
    """Fetch player stats from nflreadpy, write per-season Parquet files.

    Also joins PBP-derived stats (TD distance bonuses, pick_six, etc.)
    so the scoring engine has all columns it needs.

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

    # Join PBP-derived stats (TD distance, pick_six, rec_30_39, IDP bonuses)
    df = _join_pbp_aggregates(df, seasons)

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

    Also joins load_schedules() to add def_pa (points allowed) so the
    def_pa_* threshold rules fire.

    Stat columns available via load_team_stats (direct mapping):
        def_tds, def_sacks, def_interceptions, fumble_recovery_opp,
        def_tackles_for_loss, def_safeties, def_fumbles_forced,
        special_teams_tds

    Stats added via schedule join:
        def_pa   — points allowed (opponent scores from schedule)

    Stats still unavailable (need PBP):
        def_yd_*   — yards allowed (needs pbp)
        def_3nout   — three-and-outs forced (needs pbp)
        def_4down_stop — fourth-down stops (needs pbp)
        def_blk_kick — blocked kicks (needs pbp)
        def_2pt_ret — defensive 2pt returns (needs pbp)

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

    # Add points allowed from schedule data
    df = _add_def_pa(df, seasons)

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


def _add_def_pa(df: pl.DataFrame, seasons: list[int]) -> pl.DataFrame:
    """Join schedule data to add def_pa (points allowed per game).

    Creates two rows per game from schedule: one for each team
    with def_pa = opponent's score. Joins onto team defense data
    on (team, season, week).
    """
    try:
        sched = nfl.load_schedules(seasons=seasons)
    except Exception:
        return df

    sched = sched.filter(pl.col("game_type") == "REG").select([
        "season", "week", "home_team", "away_team",
        "home_score", "away_score",
    ])

    # Build points-allowed rows: one per team per game
    home_rows = sched.select([
        "season", "week",
        pl.col("home_team").alias("team"),
        pl.col("away_score").alias("def_pa"),
    ])
    away_rows = sched.select([
        "season", "week",
        pl.col("away_team").alias("team"),
        pl.col("home_score").alias("def_pa"),
    ])
    pa_df = pl.concat([home_rows, away_rows])

    # Join into team defense DataFrame
    df = df.join(pa_df, on=["season", "week", "team"], how="left")

    return df


def _join_pbp_aggregates(df: pl.DataFrame, seasons: list[int]) -> pl.DataFrame:
    """Join PBP-derived stats into player stats DataFrame.

    Aggregates play-by-play data to game-level columns (TD distance
    bonuses, pick_six, rec_30_39, IDP bonuses) and left-joins them
    onto the main player-game rows.
    """
    try:
        from pbp_aggregate import aggregate_pbp
        pbp_stats = aggregate_pbp(seasons)
    except Exception:
        # Best-effort: if PBP unavailable, continue without it
        return df

    # Join offense stats (TD distance, pick_six, rec_30_39)
    offense = pbp_stats.get("offense")
    if offense is not None and len(offense) > 0:
        df = df.join(
            offense,
            on=["player_id", "season", "week"],
            how="left",
        )

    # Join IDP stats (int_td_50, fum_td_50, blk_kick, stp_ff, stp_fum_rec)
    idp = pbp_stats.get("idp")
    if idp is not None and len(idp) > 0:
        df = df.join(
            idp,
            on=["player_id", "season", "week"],
            how="left",
        )

    return df


def _call_nflreadpy(seasons: list[int]) -> pl.DataFrame:
    """Call nflreadpy for game-level player stats. Extracted for testability."""
    return nfl.load_player_stats(seasons=seasons, summary_level="week")


def _call_nflreadpy_team(seasons: list[int]) -> pl.DataFrame:
    """Call nflreadpy for game-level team stats. Extracted for testability."""
    return nfl.load_team_stats(seasons=seasons, summary_level="week")


def _filter_regular_season(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only regular season weeks (exclude preseason and playoffs)."""
    return df.filter(pl.col("season_type") == "REG")
