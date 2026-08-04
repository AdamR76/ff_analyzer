"""VORP (Value Over Replacement) calculations — functional, no classes.

Computes replacement-level baselines from roster construction,
then calculates VORP and positional tiers.
"""

import polars as pl
from scoring_engine import POSITIONS


def compute_vorp(
    projections: pl.DataFrame,
    position: str,
    replacement_level: int,
) -> pl.DataFrame:
    """Compute VORP for a single position.

    projections: DataFrame with at least player_id, projected_points, position
    position: position string to filter
    replacement_level: rank N — the Nth-best player at this position
                       is the replacement baseline

    Returns DataFrame with added 'vorp' column.
    """
    pos_df = projections.filter(pl.col("position") == position)

    if len(pos_df) == 0:
        return pos_df.with_columns(pl.lit(0.0).alias("vorp"))

    # Sort by projected points descending
    pos_df = pos_df.sort("projected_points", descending=True)

    # Replacement level: the points of the Nth-best player.
    # If we have fewer players than the replacement rank, use the worst available.
    if len(pos_df) >= replacement_level:
        replacement_points = pos_df["projected_points"][replacement_level - 1]
    else:
        replacement_points = pos_df["projected_points"][-1]

    pos_df = pos_df.with_columns(
        (pl.col("projected_points") - replacement_points).alias("vorp")
    )

    return pos_df


def compute_tiers(
    df: pl.DataFrame,
    threshold: float = 2.0,
    per_position: bool = False,
) -> pl.DataFrame:
    """Assign tier numbers based on VORP gaps.

    A new tier starts when the VORP gap between consecutive players
    (sorted by VORP descending) exceeds the threshold.

    df: DataFrame with 'vorp' column (and 'position' if per_position=True)
    threshold: VORP gap that triggers a new tier
    per_position: if True, compute tiers independently within each
        position group (intra-position tiers). Default False computes
        global cross-position VORP tiers — useful for overall draft
        decisions but can intermix positions within the same tier.

    Returns DataFrame with added 'tier' column (starts at 1).
    """
    if per_position and "position" in df.columns:
        frames = []
        for _pos, group in df.group_by("position"):
            frames.append(
                _assign_tiers(group, threshold)
            )
        return pl.concat(frames) if frames else df

    return _assign_tiers(df, threshold)


def _assign_tiers(df: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """Assign tier numbers to a single sorted group."""
    df = df.sort("vorp", descending=True)
    vorp_values = df["vorp"].to_list()

    tiers = []
    current_tier = 1
    for i, vorp in enumerate(vorp_values):
        if i > 0:
            gap = vorp_values[i - 1] - vorp
            if gap > threshold:
                current_tier += 1
        tiers.append(current_tier)

    return df.with_columns(pl.Series("tier", tiers))


def _estimate_replacement_level(roster: dict, num_teams: int = 12) -> dict:
    """Estimate how many players at each position get drafted.

    Uses roster construction to determine draft demand per position,
    then estimates replacement level as drafted + buffer
    (accounts for bench depth and flex overlap).

    Returns dict: position → replacement rank (Nth best)
    """
    starters = roster["starters"]
    flex_slots = roster["flex"]
    bench = roster["bench"]

    # Count flex eligibility
    flex_by_pos = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}
    for slot in flex_slots:
        for pos in slot["eligible"]:
            flex_by_pos[pos] += 1

    levels = {}

    total_demand = sum(starters.values()) + sum(flex_by_pos.values())

    for pos in ["QB", "RB", "WR", "TE"]:
        start_count = starters.get(pos, 0)
        flex_count = flex_by_pos.get(pos, 0)
        # Approximate: each team drafts (starters + flex_share + bench_share)
        bench_share = (
            bench * (start_count + flex_count) / total_demand
            if total_demand > 0
            else 0.0
        )
        per_team = start_count + flex_count * 0.5 + bench_share * 0.3
        drafted = round(per_team * num_teams)
        # Replacement is drafted number + buffer (waiver wire players)
        levels[pos] = max(drafted + 4, start_count * num_teams + 4)

    for pos in ["K", "DEF", "LB", "DB"]:
        start_count = starters.get(pos, 0)
        levels[pos] = start_count * num_teams + 4

    return levels


def compute_all_vorp(
    projections: pl.DataFrame,
    roster: dict,
    num_teams: int = 12,
) -> pl.DataFrame:
    """Compute VORP for all positions using roster-driven replacement levels.

    projections: DataFrame with player_id, player_name, position, projected_points
    roster: parsed roster dict from roster.parse_roster()
    num_teams: number of teams in the league

    Returns DataFrame with 'vorp' column added for every player.
    """
    levels = _estimate_replacement_level(roster, num_teams)

    frames = []
    for pos, repl_level in levels.items():
        pos_df = compute_vorp(projections, pos, repl_level)
        if len(pos_df) > 0:
            frames.append(pos_df)

    if frames:
        return pl.concat(frames)
    return projections.with_columns(pl.lit(0.0).alias("vorp"))
