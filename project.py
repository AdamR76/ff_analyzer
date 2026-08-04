"""Projection pipeline stage -- position-specific 3-year weighted models.

Reads scored parquet, applies per-position projection logic,
writes 2026 projections parquet. Functional, no classes.
"""

import polars as pl
from scoring_engine import POSITIONS, normalize_position
from rookies import project_rookies


def run_projection_pipeline(cfg: dict) -> dict:
    """Run 3-year weighted projections for all positions.

    cfg: config dict from load_config()

    Returns dict with keys:
        output_path: Path       -- path to 2026_projections.parquet
        player_count: int       -- number of players projected
    """
    data_dir = cfg["data_dir"]
    proc_dir = data_dir / "processed"
    proj_dir = data_dir / "projections"
    proj_dir.mkdir(parents=True, exist_ok=True)

    w_cur = cfg.get("weight_current", 0.50)
    w_prev = cfg.get("weight_prev", 0.30)
    w_old = cfg.get("weight_oldest", 0.20)

    # Load all scored seasons into dict
    season_dfs = {}
    for path in sorted(proc_dir.glob("*_scores.parquet")):
        season = int(path.stem.replace("_scores", ""))
        season_dfs[str(season)] = pl.read_parquet(path)

    # Project each position
    all_projections = []
    for pos in POSITIONS:
        pos_df = _project_position(pos, season_dfs, w_cur, w_prev, w_old)
        if pos_df is not None and len(pos_df) > 0:
            all_projections.append(pos_df)

    if all_projections:
        result = pl.concat(all_projections)
    else:
        result = pl.DataFrame()

    # Add source column to veterans
    if len(result) > 0:
        result = result.with_columns(pl.lit("historical").alias("source"))

    # Project rookies via comparable-player model
    try:
        import nflreadpy as nfl
        draft_2026 = nfl.load_draft_picks(seasons=[2026])
        if len(draft_2026) > 0:
            combine_2026 = _safe_load_combine()
            rookies_df = project_rookies(proc_dir, draft_2026, combine_2026)
            if len(rookies_df) > 0:
                result = pl.concat([result, rookies_df], how="diagonal_relaxed")
    except Exception as e:
        # Rookie projection is best-effort; pipeline continues without it
        import warnings
        warnings.warn(f"Rookie projection failed: {e}")

    # Filter out retired/inactive players
    result = _filter_active_players(result)

    out_path = proj_dir / "2026_projections.parquet"
    result.write_parquet(out_path)

    return {
        "output_path": out_path,
        "player_count": len(result),
    }


def _safe_load_combine():
    """Load combine data, returning None if unavailable (network, missing year)."""
    try:
        import nflreadpy as nfl
        return nfl.load_combine(seasons=[2026])
    except Exception:
        return None


def _filter_active_players(df: pl.DataFrame) -> pl.DataFrame:
    """Remove retired and inactive players from projections.

    Joins with nflreadpy.load_players() to get player status and last_season.
    Keeps: rookies (source="rookie_model") and players active in 2025+.
    Drops: players whose last season was 2024 or earlier, retired, cut.
    """
    if len(df) == 0:
        return df

    try:
        import nflreadpy as nfl
        players = nfl.load_players().select([
            pl.col("gsis_id").alias("player_id"),
            pl.col("last_season"),
        ])

        # Left join: keep all projections, add last_season from player db
        df = df.join(players, on="player_id", how="left")

        # Keep rookies (no player db entry, source guarantees they're current)
        # and players with last_season >= 2025
        df = df.filter(
            (pl.col("source") == "rookie_model")
            | pl.col("last_season").is_null()
            | (pl.col("last_season") >= 2025)
        )

        # Drop the join column
        df = df.drop("last_season")

    except Exception:
        # Best-effort: if player data unavailable, keep all projections
        pass

    return df


def _project_position(
    position: str,
    season_dfs: dict[str, pl.DataFrame],
    weight_current: float,
    weight_prev: float,
    weight_oldest: float,
) -> pl.DataFrame | None:
    """Project single position using weighted game-level averages.

    Returns DataFrame with: player_id, player_name (full display name),
    position, projected_points, projected_ppg, ceiling, floor,
    games_played_projection
    """
    # Collect all rows for this position across all seasons
    frames = []
    for _season_key, df in season_dfs.items():
        pos_rows = df.filter(pl.col("position") == position)
        if len(pos_rows) > 0:
            frames.append(pos_rows)

    if not frames:
        return None

    all_data = pl.concat(frames)

    # Determine target season: latest season in the data
    season = max(int(k) for k in season_dfs.keys())

    # Per-player per-season: games played, total points, ppg
    # Use player_display_name for full name (e.g. "Ja'Marr Chase" not "J.Chase")
    name_col = "player_display_name" if "player_display_name" in all_data.columns else "player_name"
    player_seasons = all_data.group_by(
        ["player_id", name_col, "season"]
    ).agg([
        pl.col("fantasy_points").sum().alias("total_points"),
        pl.len().alias("games_played"),
    ]).with_columns(
        (pl.col("total_points") / pl.col("games_played")).alias("ppg")
    )

    # Apply weights based on season
    player_seasons = player_seasons.with_columns(
        pl.when(pl.col("season") == season)
          .then(pl.lit(weight_current))
          .when(pl.col("season") == season - 1)
          .then(pl.lit(weight_prev))
          .when(pl.col("season") == season - 2)
          .then(pl.lit(weight_oldest))
          .otherwise(pl.lit(0.0))
          .alias("weight")
    )

    # Weighted average per player — compute projected_ppg first,
    # then derived columns in a second pass (Polars expressions in
    # a single with_columns are evaluated simultaneously).
    projections = player_seasons.group_by(
        ["player_id", name_col]
    ).agg([
        (pl.col("ppg") * pl.col("weight")).sum().alias("weighted_ppg_sum"),
        pl.col("weight").sum().alias("weight_sum"),
        pl.col("ppg").std(ddof=0).alias("ppg_std"),
        pl.col("ppg").max().alias("ppg_ceiling"),
        pl.col("ppg").min().alias("ppg_floor"),
        pl.col("games_played").sum().alias("total_games"),
        pl.col("games_played").mean().alias("avg_games_per_season"),
    ]).with_columns(
        (pl.col("weighted_ppg_sum") / pl.col("weight_sum")).alias("projected_ppg"),
    ).with_columns([
        (pl.col("projected_ppg") * 17).alias("projected_points"),
        (pl.col("projected_ppg") + pl.col("ppg_std")).alias("ceiling"),
        (pl.col("projected_ppg") - pl.col("ppg_std")).alias("floor"),
        pl.lit(17, dtype=pl.Int64).alias("games_played_projection"),
        pl.lit(position).alias("position"),
    ]).drop(["weighted_ppg_sum", "weight_sum", "ppg_std"])

    # Rename name column to canonical player_name so downstream
    # consumers (rank.py) get a consistent column name regardless
    # of whether we used player_display_name or player_name.
    if name_col != "player_name" and name_col in projections.columns:
        projections = projections.rename({name_col: "player_name"})

    # Add team column from most recent season (schema parity with rookies)
    if "team" in all_data.columns:
        latest_teams = (
            all_data.filter(pl.col("season") == season)
            .select(["player_id", "team"])
            .unique(subset=["player_id"])
        )
        projections = projections.join(latest_teams, on="player_id", how="left")

    # Position-specific adjustments
    projections = _apply_position_adjustments(position, projections)

    return projections


def _apply_position_adjustments(
    position: str, df: pl.DataFrame
) -> pl.DataFrame:
    """Apply position-specific adjustments to projections.

    DEF: heavy regression toward mean (70% pull to positional mean)
    K: moderate regression toward mean (50% pull)
    RB: age cliff penalty deferred (needs bio/age data from nflreadpy player info)
    WR: 3rd-year breakout bonus deferred
    TE: elite flag deferred
    QB: rushing baseline floor deferred
    """
    if position == "DEF":
        # Heavy regression: pull toward positional mean by 70%
        if len(df) > 0:
            pos_mean = df["projected_ppg"].mean()
            ceiling_mean = df["ceiling"].mean()
            floor_mean = df["floor"].mean()
            df = df.with_columns([
                (pl.col("projected_ppg") * 0.3 + pos_mean * 0.7).alias("projected_ppg"),
                (pl.col("ceiling") * 0.3 + ceiling_mean * 0.7).alias("ceiling"),
                (pl.col("floor") * 0.3 + floor_mean * 0.7).alias("floor"),
            ]).with_columns(
                (pl.col("projected_ppg") * 17).alias("projected_points"),
            )

    elif position == "K":
        # Kicker: regress heavily, value driven by offense volume
        if len(df) > 0:
            pos_mean = df["projected_ppg"].mean()
            ceiling_mean = df["ceiling"].mean()
            floor_mean = df["floor"].mean()
            df = df.with_columns([
                (pl.col("projected_ppg") * 0.5 + pos_mean * 0.5).alias("projected_ppg"),
                (pl.col("ceiling") * 0.5 + ceiling_mean * 0.5).alias("ceiling"),
                (pl.col("floor") * 0.5 + floor_mean * 0.5).alias("floor"),
            ]).with_columns(
                (pl.col("projected_ppg") * 17).alias("projected_points"),
            )

    # QB, WR, TE: no adjustment in baseline model (future: add trend/bonus factors)

    return df
