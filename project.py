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

    # Load player bio once for all positions
    bio = _load_player_bio()

    # Project each position
    all_projections = []
    for pos in POSITIONS:
        pos_df = _project_position(pos, season_dfs, w_cur, w_prev, w_old, bio, cfg)
        if pos_df is not None and len(pos_df) > 0:
            all_projections.append(pos_df)

    if all_projections:
        result = pl.concat(all_projections, how="diagonal_relaxed")
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


def _load_player_bio() -> pl.DataFrame | None:
    """Load player biographical data from nflreadpy.

    Returns DataFrame with: player_id, birth_date, years_of_experience,
    draft_year, draft_round.
    """
    try:
        import nflreadpy as nfl
        players = nfl.load_players()
        cols = ["gsis_id", "birth_date", "years_of_experience",
                "draft_year", "draft_round"]
        available = [c for c in cols if c in players.columns]
        df = players.select(available).rename({"gsis_id": "player_id"})
        return df
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
    bio: pl.DataFrame | None,
    cfg: dict,
) -> pl.DataFrame | None:
    """Project single position using weighted game-level averages.

    Returns DataFrame with: player_id, player_name, position,
    projected_points, projected_ppg, ceiling, floor,
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

    # Use player_display_name for full name (e.g. "Ja'Marr Chase" not "J.Chase")
    name_col = "player_display_name" if "player_display_name" in all_data.columns else "player_name"

    # ── Per-player per-season aggregation ──
    player_seasons = all_data.group_by(
        ["player_id", name_col, "season"]
    ).agg([
        pl.col("fantasy_points").sum().alias("total_points"),
        pl.len().alias("games_played"),
    ]).with_columns(
        (pl.col("total_points") / pl.col("games_played")).alias("ppg")
    )

    # ── Compute within-season quantiles from game-level data ──
    # (for better ceiling/floor — per-game fantasy_points distribution)
    game_quantiles = all_data.group_by(
        ["player_id", name_col, "season"]
    ).agg([
        pl.col("fantasy_points").quantile(0.75).alias("q75"),
        pl.col("fantasy_points").quantile(0.25).alias("q25"),
        pl.col("fantasy_points").std(ddof=0).alias("within_std"),
    ])

    # ── Positional mean PPG for shrinkage ──
    pos_mean_ppg = player_seasons["ppg"].mean()

    # ── Shrinkage: regress low-sample seasons toward positional mean ──
    if cfg.get("shrinkage_enabled", True):
        SHRINKAGE_GAMES = 4
        player_seasons = player_seasons.with_columns(
            ((pl.col("games_played") * pl.col("ppg")
              + pl.lit(SHRINKAGE_GAMES) * pl.lit(pos_mean_ppg))
             / (pl.col("games_played") + pl.lit(SHRINKAGE_GAMES)))
            .alias("ppg_shrunk")
        )
    else:
        player_seasons = player_seasons.with_columns(
            pl.col("ppg").alias("ppg_shrunk")
        )

    # ── Apply weights based on season ──
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

    # ── Trend: compute PPG slope across seasons ──
    if cfg.get("trend_adjustment_enabled", True):
        trends = player_seasons.group_by("player_id").agg([
            pl.col("season").sort().first().alias("first_season"),
            pl.col("season").sort().last().alias("last_season"),
            pl.col("ppg").sort_by("season").first().alias("first_ppg"),
            pl.col("ppg").sort_by("season").last().alias("last_ppg"),
        ]).with_columns(
            (pl.col("last_ppg") - pl.col("first_ppg")).alias("ppg_change"),
            ((pl.col("last_ppg") - pl.col("first_ppg"))
             / pl.when(pl.col("first_season") != pl.col("last_season"))
               .then(pl.col("last_season") - pl.col("first_season"))
               .otherwise(pl.lit(1)))
            .alias("ppg_slope"),
        ).with_columns(
            (pl.lit(1.0) + pl.col("ppg_slope") * 0.5).alias("trend_multiplier_raw")
        ).with_columns(
            pl.col("trend_multiplier_raw").clip(0.85, 1.15).alias("trend_multiplier")
        ).select(["player_id", "trend_multiplier"])
    else:
        trends = None

    # ── Weighted average per player ──
    projections = player_seasons.group_by(
        ["player_id", name_col]
    ).agg([
        (pl.col("ppg_shrunk") * pl.col("weight")).sum().alias("weighted_ppg_sum"),
        pl.col("weight").sum().alias("weight_sum"),
        pl.col("games_played").sum().alias("total_games"),
        pl.col("games_played").mean().alias("avg_games_per_season"),
        pl.col("games_played").count().alias("career_seasons"),
    ]).with_columns(
        (pl.col("weighted_ppg_sum") / pl.col("weight_sum")).alias("projected_ppg"),
    ).drop(["weighted_ppg_sum", "weight_sum"])

    # ── Injury model: data-driven games played projection ──
    if cfg.get("injury_model_enabled", True):
        SHRINKAGE_PRIOR = 3  # prior weight in seasons
        PRIOR_GAMES = 15     # prior mean
        projections = projections.with_columns(
            ((pl.lit(SHRINKAGE_PRIOR * PRIOR_GAMES) + pl.col("total_games"))
             / (pl.lit(SHRINKAGE_PRIOR) + pl.col("career_seasons")))
            .clip(1, 17)
            .alias("games_played_projection")
        )
    else:
        projections = projections.with_columns(
            pl.lit(17, dtype=pl.Int64).alias("games_played_projection")
        )

    # ── Ceiling/floor from weighted within-season quantiles ──
    # Join quantile data, compute weighted average q75/q25
    player_seasons_with_q = player_seasons.join(
        game_quantiles.select(["player_id", "season", "q75", "q25", "within_std"]),
        on=["player_id", "season"], how="left"
    )
    q_weights = player_seasons_with_q.with_columns(
        (pl.col("q75") * pl.col("weight")).alias("w_q75"),
        (pl.col("q25") * pl.col("weight")).alias("w_q25"),
    )
    q_agg = q_weights.group_by("player_id").agg([
        pl.col("w_q75").sum().alias("wq75_sum"),
        pl.col("w_q25").sum().alias("wq25_sum"),
        pl.col("weight").sum().alias("w_sum"),
    ]).with_columns([
        (pl.col("wq75_sum") / pl.col("w_sum")).alias("avg_q75"),
        (pl.col("wq25_sum") / pl.col("w_sum")).alias("avg_q25"),
    ]).select(["player_id", "avg_q75", "avg_q25"])

    projections = projections.join(q_agg, on="player_id", how="left")

    # Within-season band: 0.5 * (quantile_distance_from_mean) is reasonable
    projections = projections.with_columns([
        (pl.col("projected_ppg")
         + (pl.col("avg_q75") - pl.col("projected_ppg")).fill_null(0) * 0.5)
        .alias("ceiling"),
        (pl.col("projected_ppg")
         - (pl.col("projected_ppg") - pl.col("avg_q25")).fill_null(0) * 0.5)
        .alias("floor"),
    ])
    # Fallback for players without quantile data: use projected_ppg ± 20%
    projections = projections.with_columns([
        pl.col("ceiling").fill_null(pl.col("projected_ppg") * 1.2),
        pl.col("floor").fill_null(pl.col("projected_ppg") * 0.8),
    ])

    # ── Derived columns ──
    projections = projections.with_columns([
        (pl.col("projected_ppg") * pl.col("games_played_projection"))
        .alias("projected_points"),
        pl.lit(position).alias("position"),
    ])

    # ── Trend multiplier ──
    if trends is not None:
        projections = projections.join(trends, on="player_id", how="left")
        projections = projections.with_columns(
            pl.col("trend_multiplier").fill_null(1.0).alias("trend_multiplier")
        )
        projections = projections.with_columns([
            (pl.col("projected_ppg") * pl.col("trend_multiplier")).alias("projected_ppg"),
            (pl.col("ceiling") * pl.col("trend_multiplier")).alias("ceiling"),
            (pl.col("floor") * pl.col("trend_multiplier")).alias("floor"),
        ]).with_columns(
            (pl.col("projected_ppg") * pl.col("games_played_projection"))
            .alias("projected_points"),
        ).drop(["trend_multiplier"])
    else:
        projections = projections.drop("trend_multiplier") if "trend_multiplier" in projections.columns else projections

    # ── Clean up intermediate columns ──
    projections = projections.drop(["total_games", "avg_games_per_season",
                                     "career_seasons", "avg_q75", "avg_q25"])

    # Rename name column to canonical player_name
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

    # ── Join bio data for age/experience-based adjustments ──
    if bio is not None and len(bio) > 0:
        projections = projections.join(bio, on="player_id", how="left")

    # ── Position-specific adjustments ──
    projections = _apply_position_adjustments(
        position, projections, season_dfs, cfg
    )

    return projections


def _apply_position_adjustments(
    position: str,
    df: pl.DataFrame,
    season_dfs: dict[str, pl.DataFrame],
    cfg: dict,
) -> pl.DataFrame:
    """Apply position-specific adjustments to projections.

    DEF: heavy regression toward mean (70% pull to positional mean)
    K: moderate regression toward mean (50% pull)
    RB/WR/TE/QB: age curve, plus position-specific bonuses
    """
    if position == "DEF":
        return _regress_to_mean(df, 0.3)

    elif position == "K":
        return _regress_to_mean(df, 0.5)

    elif position in ("RB", "WR", "TE", "QB"):
        df = _apply_age_curve(df, position, cfg)

        if position == "WR" and cfg.get("wr_breakout_enabled", True):
            df = _apply_wr_breakout(df, season_dfs)

        if position == "TE" and cfg.get("te_elite_enabled", True):
            df = _apply_te_elite(df)

        if position == "QB" and cfg.get("qb_rushing_baseline_enabled", True):
            df = _apply_qb_rushing_baseline(df, season_dfs)

    # Recompute projected_points after all adjustments
    if "games_played_projection" in df.columns:
        df = df.with_columns(
            (pl.col("projected_ppg") * pl.col("games_played_projection"))
            .alias("projected_points"),
        )

    return df


def _regress_to_mean(df: pl.DataFrame, keep_weight: float) -> pl.DataFrame:
    """Regress PPG toward positional mean, keeping keep_weight of raw."""
    if len(df) == 0:
        return df
    pos_mean = df["projected_ppg"].mean()
    ceiling_mean = df["ceiling"].mean()
    floor_mean = df["floor"].mean()
    mean_wt = 1.0 - keep_weight
    df = df.with_columns([
        (pl.col("projected_ppg") * keep_weight + pos_mean * mean_wt).alias("projected_ppg"),
        (pl.col("ceiling") * keep_weight + ceiling_mean * mean_wt).alias("ceiling"),
        (pl.col("floor") * keep_weight + floor_mean * mean_wt).alias("floor"),
    ]).with_columns(
        (pl.col("projected_ppg") * pl.col("games_played_projection")).alias("projected_points"),
    )
    return df


# ═══════════════════════════════════════════════════════════════════════
# Age curve
# ═══════════════════════════════════════════════════════════════════════

# Position-specific age curves: multiplier at or above each age threshold.
# Ages not listed get multiplier 1.0 (no penalty).
_AGE_CURVES = {
    "RB": {27: 0.95, 28: 0.85, 29: 0.75, 30: 0.65, 31: 0.55},
    "WR": {29: 0.95, 30: 0.88, 31: 0.80, 32: 0.70, 33: 0.60},
    "TE": {30: 0.93, 31: 0.85, 32: 0.77, 33: 0.68, 34: 0.58},
    "QB": {35: 0.95, 36: 0.88, 37: 0.80, 38: 0.70, 39: 0.60},
}


def _apply_age_curve(df: pl.DataFrame, position: str, cfg: dict) -> pl.DataFrame:
    """Apply age-based multiplier to projections."""
    if not cfg.get("age_curve_enabled", True):
        return df
    if "birth_date" not in df.columns:
        return df
    if position not in _AGE_CURVES:
        return df

    curve = _AGE_CURVES[position]

    # Compute age as of Sept 1, 2026
    if "birth_date" not in df.columns:
        return df

    valid_bio = df.filter(pl.col("birth_date").is_not_null())
    if len(valid_bio) == 0:
        return df
    no_bio = df.filter(pl.col("birth_date").is_null())

    df = valid_bio.with_columns(
        pl.col("birth_date").str.slice(0, 4).cast(pl.Int32).alias("birth_year")
    ).with_columns(
        (pl.lit(2026) - pl.col("birth_year")).alias("age")
    )

    # Build age multiplier expression: start at 1.0, apply lowest applicable
    age_mult = pl.lit(1.0)
    for age in sorted(curve.keys(), reverse=True):
        mult = curve[age]
        age_mult = pl.when(pl.col("age") >= age).then(pl.lit(mult)).otherwise(age_mult)

    df = df.with_columns([
        (pl.col("projected_ppg") * age_mult).alias("projected_ppg"),
        (pl.col("ceiling") * age_mult).alias("ceiling"),
        (pl.col("floor") * age_mult).alias("floor"),
    ])

    # Re-join rows that had no bio data (no age curve applied)
    if len(no_bio) > 0:
        df = pl.concat([df, no_bio], how="diagonal_relaxed")

    # Clean up intermediate columns
    drop_cols = [c for c in ["birth_year", "age"] if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols)

    return df


# ═══════════════════════════════════════════════════════════════════════
# WR 3rd-year breakout
# ═══════════════════════════════════════════════════════════════════════


def _apply_wr_breakout(
    df: pl.DataFrame, season_dfs: dict[str, pl.DataFrame]
) -> pl.DataFrame:
    """Apply breakout bonus for WRs entering year 3 with ascending PPG.

    WRs drafted in 2024 (year 3 in 2026) whose year-2 PPG > year-1 PPG
    get a 1.08 multiplier. Flat trend gets 1.03.
    """
    if "draft_year" not in df.columns:
        return df

    year3_wrs = df.filter(
        (pl.col("position") == "WR") & (pl.col("draft_year") == 2024)
    )
    if len(year3_wrs) == 0:
        return df

    # Compute year-1 and year-2 PPG for each WR from season data
    year3_ids = set(year3_wrs["player_id"].to_list())
    breakout_mult = {}

    for pid in year3_ids:
        ppg_y1, ppg_y2 = _get_first_two_year_ppg(pid, season_dfs)
        if ppg_y2 is not None and ppg_y1 is not None and ppg_y2 > ppg_y1:
            breakout_mult[pid] = 1.08
        elif ppg_y2 is not None:
            breakout_mult[pid] = 1.03

    if not breakout_mult:
        return df

    # Apply per-player multiplier using when/then chain
    mult_expr = pl.lit(1.0)
    for pid, mult in breakout_mult.items():
        mult_expr = pl.when(pl.col("player_id") == pid).then(pl.lit(mult)).otherwise(mult_expr)

    df = df.with_columns([
        (pl.col("projected_ppg") * mult_expr).alias("projected_ppg"),
        (pl.col("ceiling") * mult_expr).alias("ceiling"),
        (pl.col("floor") * mult_expr).alias("floor"),
    ])

    return df


def _get_first_two_year_ppg(
    player_id: str, season_dfs: dict[str, pl.DataFrame]
) -> tuple[float | None, float | None]:
    """Get year-1 and year-2 PPG for a player from scored data.

    Returns (ppg_year1, ppg_year2) where year 1 = earliest season,
    year 2 = second season.
    """
    seasons = sorted(int(k) for k in season_dfs.keys())
    ppg_by_year = {}

    for s in seasons:
        sdf = season_dfs[str(s)]
        rows = sdf.filter(pl.col("player_id") == player_id)
        if len(rows) > 0:
            total = rows["fantasy_points"].sum()
            games = len(rows)
            if games > 0:
                ppg_by_year[s] = total / games

    sorted_years = sorted(ppg_by_year.keys())
    y1 = ppg_by_year.get(sorted_years[0]) if len(sorted_years) >= 1 else None
    y2 = ppg_by_year.get(sorted_years[1]) if len(sorted_years) >= 2 else None

    return y1, y2


# ═══════════════════════════════════════════════════════════════════════
# TE elite flag
# ═══════════════════════════════════════════════════════════════════════


def _apply_te_elite(df: pl.DataFrame) -> pl.DataFrame:
    """Apply elite premium to top-3 TEs by projected PPG."""
    te_df = df.filter(pl.col("position") == "TE")
    if len(te_df) <= 3:
        return df

    top_3_ids = set(
        te_df.sort("projected_ppg", descending=True)
        .head(3)["player_id"].to_list()
    )

    ELITE_MULT = 1.10
    is_elite = pl.col("player_id").is_in(top_3_ids)
    is_te = pl.col("position") == "TE"

    df = df.with_columns([
        pl.when(is_te & is_elite)
          .then(pl.col("projected_ppg") * ELITE_MULT)
          .otherwise(pl.col("projected_ppg"))
          .alias("projected_ppg"),
        pl.when(is_te & is_elite)
          .then(pl.col("ceiling") * ELITE_MULT)
          .otherwise(pl.col("ceiling"))
          .alias("ceiling"),
        pl.when(is_te & is_elite)
          .then(pl.col("floor") * ELITE_MULT)
          .otherwise(pl.col("floor"))
          .alias("floor"),
    ])

    return df


# ═══════════════════════════════════════════════════════════════════════
# QB rushing baseline
# ═══════════════════════════════════════════════════════════════════════


def _apply_qb_rushing_baseline(
    df: pl.DataFrame, season_dfs: dict[str, pl.DataFrame]
) -> pl.DataFrame:
    """Boost QB floor for rushing QBs. Rushing production is stickier
    than passing TDs."""
    if "rushing_yards" not in df.columns:
        # Compute per-QB average rush yards per game from season data
        rush_data = _compute_qb_rush_ppg(season_dfs)
        if rush_data is not None and len(rush_data) > 0:
            df = df.join(rush_data, on="player_id", how="left")
        else:
            return df

    rush_yd_col = "avg_rush_yd_pg"
    if rush_yd_col not in df.columns:
        return df

    # QBs averaging >= 20 rush yards/game over >= 8 games get a floor boost
    # Floor = max(current_projected_ppg, avg_rush_yd_pg * 0.10)
    df = df.with_columns(
        pl.when(
            (pl.col(rush_yd_col) >= 20)
            & (pl.col("position") == "QB")
            & pl.col(rush_yd_col).is_not_null()
        )
        .then(pl.max_horizontal(
            pl.col("projected_ppg"),
            pl.col(rush_yd_col) * 0.10,
        ))
        .otherwise(pl.col("projected_ppg"))
        .alias("projected_ppg"),
    )

    # Clean up intermediate columns
    drop_cols = [c for c in [rush_yd_col, "rush_games"] if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols)

    return df


def _compute_qb_rush_ppg(
    season_dfs: dict[str, pl.DataFrame]
) -> pl.DataFrame | None:
    """Compute per-QB average rushing yards per game from all seasons."""
    frames = []
    for sdf in season_dfs.values():
        qb_rows = sdf.filter(
            (pl.col("position") == "QB")
            & pl.col("rushing_yards").is_not_null()
        )
        if len(qb_rows) > 0:
            frames.append(qb_rows.select(["player_id", "rushing_yards"]))

    if not frames:
        return None

    all_qb = pl.concat(frames)
    return (
        all_qb.group_by("player_id")
        .agg([
            pl.col("rushing_yards").mean().alias("avg_rush_yd_pg"),
            pl.len().alias("rush_games"),
        ])
        .filter(pl.col("rush_games") >= 8)
    )
