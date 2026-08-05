"""Rookie projections — comparable-player model using draft capital.

No classes, pure functions, Polars DataFrames.
Uses nflreadpy draft picks + combine data; no external college stats.
"""

import math
import polars as pl
from pathlib import Path
import nflreadpy as nfl
from scoring_engine import normalize_position


# Round-based fallback projections (fantasy points, full season).
# Keyed as (position, round) → projected_points.
# Values: approximate historical rookie-year fantasy output by position/round.
# Round 4+ all map to the same "late-round" tier.
# LB and DB omitted — IDP rookies project as replacement-level.

_ROUND_BASELINES = {
    ("QB", 1): 280, ("QB", 2): 200, ("QB", 3): 140, ("QB", 4): 80,
    ("RB", 1): 240, ("RB", 2): 180, ("RB", 3): 140, ("RB", 4): 80,
    ("WR", 1): 220, ("WR", 2): 160, ("WR", 3): 120, ("WR", 4): 80,
    ("TE", 1): 180, ("TE", 2): 140, ("TE", 3): 100, ("TE", 4): 60,
}

# Position-specific rookie hit-rate multipliers.
# Derived from historical rookie-year fantasy production vs draft capital.
# Higher pick = higher hit rate. Round 4+ all use the round-4 tier.
# Multiplier is applied to raw projection: adjusted = raw * hit_rate.
# K, DEF, LB, DB default to 1.0 (not projected via rookie model).

_ROOKIE_HIT_RATES = {
    ("QB", 1): 0.55, ("QB", 2): 0.40, ("QB", 3): 0.30, ("QB", 4): 0.20,
    ("RB", 1): 0.80, ("RB", 2): 0.65, ("RB", 3): 0.45, ("RB", 4): 0.25,
    ("WR", 1): 0.65, ("WR", 2): 0.50, ("WR", 3): 0.35, ("WR", 4): 0.20,
    ("TE", 1): 0.55, ("TE", 2): 0.40, ("TE", 3): 0.25, ("TE", 4): 0.15,
}


def _get_hit_rate(
    position: str, round_num: int,
    computed_rates: dict | None = None,
) -> float:
    """Return rookie hit-rate multiplier for a position and draft round.

    Uses computed rates from historical data when available,
    falls back to hardcoded table.
    """
    if position in ("K", "DEF", "LB", "DB"):
        return 1.0
    tier = min(round_num, 4)
    if computed_rates:
        rate = computed_rates.get((position, tier))
        if rate is not None:
            return rate
    return _ROOKIE_HIT_RATES.get((position, tier), 0.50)


def _round_baseline(
    position: str, round_num: int,
    computed_baselines: dict | None = None,
) -> dict:
    """Return fallback projection dict for a rookie based on draft round.

    Rounds 4+ all use the round-4 baseline.
    Unknown positions (K, DEF, LB, DB) get small baseline.

    Uses computed baselines from historical data when available,
    falls back to hardcoded table.
    """
    tier = min(round_num, 4)
    if computed_baselines:
        pts = computed_baselines.get((position, tier))
        if pts is not None:
            return {
                "position": position,
                "projected_points": float(pts),
                "projected_ppg": pts / 17,
                "ceiling": pts / 17 * 1.2,
                "floor": pts / 17 * 0.7,
                "games_played_projection": 17,
                "source": "rookie_model",
            }

    points = _ROUND_BASELINES.get((position, tier), 60)

    return {
        "position": position,
        "projected_points": float(points),
        "projected_ppg": points / 17,
        "ceiling": points / 17 * 1.2,
        "floor": points / 17 * 0.7,
        "games_played_projection": 17,
        "source": "rookie_model",
    }


def _find_comparables(
    rookie: dict, historical_rookies: pl.DataFrame, top_k: int = 5
) -> pl.DataFrame:
    """Find K nearest historical rookies by draft position + combine similarity.

    rookie: dict with keys 'position', 'pick', 'round', optionally combine measurables
    historical_rookies: DataFrame of past rookies with fantasy_points from scored data
    top_k: number of comparables to return

    Filters to same position, scores by draft pick distance + combine similarity.
    Returns top_k closest, or fewer if not enough data.
    """
    position = rookie["position"]
    pos_df = historical_rookies.filter(pl.col("position") == position)

    if len(pos_df) == 0:
        return pos_df.head(0)

    # If historical data doesn't have pick info (e.g. game-level scored data
    # without draft join), return empty — caller falls back to baselines.
    if "pick" not in pos_df.columns:
        return pos_df.head(0)

    # Draft pick distance score (normalized 0-1)
    rookie_pick = rookie.get("pick", 100)
    if rookie_pick is None:
        rookie_pick = 100
    max_pick = 260  # max possible draft pick

    pos_df = pos_df.with_columns(
        (pl.col("pick").fill_null(100) - rookie_pick).abs().alias("pick_dist")
    )
    pos_df = pos_df.with_columns(
        (pl.col("pick_dist") / max_pick).alias("pick_score")
    )

    # Combine similarity (if rookie has measurables AND historical data has them)
    combine_cols = ["forty", "vertical", "bench", "broad_jump", "cone", "shuttle"]
    rookie_has_combine = any(rookie.get(c) is not None for c in combine_cols)

    if rookie_has_combine:
        combine_score = pl.lit(0.0)
        for col in combine_cols:
            r_val = rookie.get(col)
            if r_val is not None and col in pos_df.columns:
                col_vals = pos_df[col].fill_null(r_val)
                col_range = col_vals.max() - col_vals.min()
                if col_range and col_range > 0:
                    combine_score = combine_score + (
                        (col_vals - r_val).abs() / col_range
                    )
        pos_df = pos_df.with_columns(
            (pl.col("pick_score") * 0.6 + combine_score * 0.4).alias("similarity")
        )
    else:
        pos_df = pos_df.with_columns(
            pl.col("pick_score").alias("similarity")
        )

    return pos_df.sort("similarity", maintain_order=True).head(top_k)


def _join_historical_drafts(historical: pl.DataFrame) -> pl.DataFrame:
    """Join historical draft picks into scored data to enable comparable matching.

    Scored data from data/processed/ has game-level nflreadpy columns only
    (no pick/round). This loads historical draft picks and left-joins them
    on player_id = gsis_id, adding pick and round columns.

    If nflreadpy fails or returns empty, historical is returned unchanged.
    """
    try:
        draft_hist = nfl.load_draft_picks(seasons=[2023, 2024, 2025])
    except Exception:
        return historical

    if draft_hist is None or len(draft_hist) == 0:
        return historical

    # Select only the columns needed for the join: gsis_id, pick, round
    needed = ["gsis_id", "pick", "round"]
    available = [c for c in needed if c in draft_hist.columns]
    if len(available) < 2:
        return historical

    draft_info = draft_hist.select(available)
    historical = historical.join(
        draft_info,
        left_on="player_id",
        right_on="gsis_id",
        how="left",
    )
    # Drop the redundant gsis_id column from the join if present
    if "gsis_id" in historical.columns:
        historical = historical.drop("gsis_id")

    return historical


def _compute_round_baselines(historical: pl.DataFrame) -> dict | None:
    """Compute average rookie fantasy points by position and draft round.

    Returns dict {(position, round): avg_points} or None if insufficient data.
    """
    if "pick" not in historical.columns or "round" not in historical.columns:
        return None

    # Filter out rows without valid round/pick data (join misses)
    valid = historical.filter(
        pl.col("round").is_not_null() & pl.col("pick").is_not_null()
    )
    if len(valid) == 0:
        return None

    # Aggregate game-level data to player-season totals
    rookie_seasons = valid.group_by(
        ["player_id", "season"]
    ).agg(
        pl.col("fantasy_points").sum().alias("total_points"),
        pl.col("pick").first().alias("pick"),
        pl.col("round").first().alias("round"),
        pl.col("position").first().alias("position"),
    )

    # Group by position and draft round
    baselines = rookie_seasons.group_by(["position", "round"]).agg([
        pl.col("total_points").mean().alias("avg_points"),
        pl.len().alias("count"),
    ]).filter(pl.col("count") >= 3)  # minimum 3 players for reliable baseline

    if len(baselines) == 0:
        return None

    result = {}
    for row in baselines.iter_rows(named=True):
        pos = row["position"]
        round_val = row["round"]
        if round_val is None:
            continue
        round_num = min(int(round_val), 4)
        key = (pos, round_num)
        avg = float(row["avg_points"])
        # Sanity: full-season baseline should be >= 50 pts.
        # Lower values mean single-game data or incomplete seasons.
        if avg < 50:
            continue
        if key not in result:
            result[key] = avg

    return result if result else None


def _compute_hit_rates(
    historical: pl.DataFrame, baselines: dict
) -> dict | None:
    """Compute rookie hit rates from historical data.

    Hit rate = fraction of rookies at each (position, round) that scored
    above the position-round baseline.
    """
    if baselines is None:
        return None
    if "round" not in historical.columns:
        return None

    rookie_seasons = historical.group_by(
        ["player_id", "season"]
    ).agg(
        pl.col("fantasy_points").sum().alias("total_points"),
        pl.col("round").first().alias("round"),
        pl.col("position").first().alias("position"),
    )

    result = {}
    for (pos, round_num), baseline in baselines.items():
        group = rookie_seasons.filter(
            (pl.col("position") == pos) & (pl.col("round") == round_num)
        )
        if len(group) >= 3:
            hits = len(group.filter(pl.col("total_points") >= baseline))
            result[(pos, round_num)] = hits / len(group)

    return result if result else None


def project_rookies(
    scored_data_dir: Path,
    draft_data: pl.DataFrame,
    combine_data: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Project fantasy points for incoming rookies using comparable-player model.

    scored_data_dir: path to data/processed/ with *_scores.parquet files
    draft_data: Polars DataFrame from nflreadpy.load_draft_picks(seasons=[2026])
    combine_data: optional Polars DataFrame from nflreadpy.load_combine(seasons=[2026])

    Returns DataFrame with veteran-matching projection schema, all source="rookie_model".
    """
    # Load all scored historical data to find comparables
    historical_frames = []
    for path in sorted(scored_data_dir.glob("*_scores.parquet")):
        historical_frames.append(pl.read_parquet(path))

    if not historical_frames:
        # No historical data at all — fall back to round baselines
        return _project_from_baselines_only(draft_data)

    historical = pl.concat(historical_frames)

    # If scored data lacks draft capital (pick/round columns), join with
    # historical draft picks so the comparable-player model can activate.
    if "pick" not in historical.columns:
        historical = _join_historical_drafts(historical)

    # Get set of player_ids already in historical data (veterans)
    existing_ids = set(historical["player_id"].unique().to_list())

    # Compute data-driven baselines and hit rates from historical rookies
    computed_baselines = _compute_round_baselines(historical)
    computed_hit_rates = _compute_hit_rates(historical, computed_baselines)

    # Merge combine data into draft data if available
    if combine_data is not None and len(combine_data) > 0:
        # Match on player name (draft uses pfr_player_name, combine uses player_name)
        draft_data = draft_data.join(
            combine_data.select([
                "player_name", "pos", "forty", "vertical", "bench",
                "broad_jump", "cone", "shuttle", "ht", "wt"
            ]),
            left_on="pfr_player_name",
            right_on="player_name",
            how="left",
        )

    results = []
    for rookie in draft_data.iter_rows(named=True):
        player_id = rookie["gsis_id"]

        # Skip players already in historical data (veterans, not rookies)
        if player_id in existing_ids:
            continue

        position = normalize_position(rookie["position"])
        if not position:
            continue  # OL and other non-scoring positions
        round_num = rookie["round"]
        pick_num = rookie["pick"]

        # Build rookie dict for comparables search
        rookie_info = {
            "position": position,
            "round": round_num,
            "pick": pick_num,
        }
        # Add combine data if present
        for col in ["forty", "vertical", "bench", "broad_jump", "cone", "shuttle"]:
            val = rookie.get(col)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                rookie_info[col] = val

        # Find comparables
        comparables = _find_comparables(rookie_info, historical, top_k=5)

        if len(comparables) >= 2:
            # Average comparable rookie fantasy output
            avg_ppg = comparables["fantasy_points"].mean()
            std_ppg = comparables["fantasy_points"].std(ddof=0) if len(comparables) > 1 else 0.0
            projected_points = avg_ppg * 17
            projected_ppg = avg_ppg
            ceiling = avg_ppg + std_ppg
            floor = max(0.0, avg_ppg - std_ppg)
            source = "rookie_model"
        else:
            baseline = _round_baseline(position, round_num, computed_baselines)
            projected_points = baseline["projected_points"]
            projected_ppg = baseline["projected_ppg"]
            ceiling = baseline["ceiling"]
            floor = baseline["floor"]
            source = "rookie_model"

        # Apply position-specific hit-rate regression
        hit_rate = _get_hit_rate(position, round_num, computed_hit_rates)
        projected_points *= hit_rate
        projected_ppg *= hit_rate
        ceiling *= hit_rate
        floor *= hit_rate

        results.append({
            "player_id": player_id,
            "player_name": rookie["pfr_player_name"],
            "position": position,
            "team": rookie["team"],
            "projected_points": projected_points,
            "projected_ppg": projected_ppg,
            "ceiling": ceiling,
            "floor": floor,
            "games_played_projection": 17,
            "source": source,
        })

    return pl.DataFrame(results)


def _project_from_baselines_only(draft_data: pl.DataFrame) -> pl.DataFrame:
    """Fallback: project all rookies from round baselines when no historical data."""
    results = []
    for rookie in draft_data.iter_rows(named=True):
        position = normalize_position(rookie["position"])
        if not position:
            continue  # OL and other non-scoring positions
        baseline = _round_baseline(position, rookie["round"], computed_baselines=None)
        hit_rate = _get_hit_rate(position, rookie["round"], computed_rates=None)
        results.append({
            "player_id": rookie["gsis_id"],
            "player_name": rookie["pfr_player_name"],
            "position": position,
            "team": rookie["team"],
            "projected_points": baseline["projected_points"] * hit_rate,
            "projected_ppg": baseline["projected_ppg"] * hit_rate,
            "ceiling": baseline["ceiling"] * hit_rate,
            "floor": baseline["floor"] * hit_rate,
            "games_played_projection": 17,
            "source": "rookie_model",
        })
    return pl.DataFrame(results)


