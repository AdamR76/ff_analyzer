"""Team context adjustment layer — post-projection multipliers.

Computes team-level factors (pace, pass rate, scoring environment) from
nflreadpy data and applies them to player projections.

Single public function: apply_team_context(df, train_years, cfg) -> pl.DataFrame
"""

import polars as pl
import nflreadpy as nfl


def compute_team_factors(
    train_years: list[int],
) -> dict[str, dict[str, float]]:
    """Compute team-level multipliers from nflreadpy data.

    Returns dict: team_abbr -> {volume, pass_share, env}
    Teams missing from all data sources get no entry (factor = 1.0).

    train_years: historical seasons used for computing tendencies.
                 Uses most recent year for team context.
    """
    recent_year = max(train_years)

    # ── Volume + pass share factors from ff_opportunity ──
    vol_factors, share_factors = _compute_play_factors(recent_year)

    # ── Scoring environment from Vegas lines ──
    env_factors = _compute_env_factors(recent_year)

    # ── Merge all factors ──
    all_teams = set(vol_factors.keys()) | set(share_factors.keys()) | set(env_factors.keys())
    result = {}
    for team in all_teams:
        pass_share = share_factors.get(team, 1.0)
        # RB run-share = inverse of pass share. run-heavy (low pass rate) → RB boost.
        run_share = 2.0 - pass_share  # simple inversion: 1.10 pass → 0.90 run
        run_share = max(0.90, min(1.10, run_share))
        result[team] = {
            "volume": vol_factors.get(team, 1.0),
            "pass_share": pass_share,
            "run_share": run_share,
            "env": env_factors.get(team, 1.0),
        }

    return result


def apply_team_context(
    df: pl.DataFrame,
    team_factors: dict[str, dict[str, float]],
) -> pl.DataFrame:
    """Apply team context multipliers to projected_ppg, ceiling, floor.

    Joins team factors on the 'team' column. Players without a team
    or on teams not in the factors dict get no adjustment (1.0).

    Pass share factor is position-aware:
    - QB/WR/TE: pass-heavy teams boost, run-heavy teams penalize
    - RB: inverted — run-heavy teams boost, pass-heavy penalize
    - K/DEF/LB/DB: no pass-share adjustment (1.0)

    Combined multiplier capped at [0.85, 1.15].
    """
    if "team" not in df.columns or not team_factors:
        return df

    has_position = "position" in df.columns

    # Build factor expressions per team using when/then chain
    vol_expr = pl.lit(1.0)
    pass_share_expr = pl.lit(1.0)
    run_share_expr = pl.lit(1.0)
    env_expr = pl.lit(1.0)

    for team, factors in team_factors.items():
        is_team = pl.col("team") == team
        vol_expr = pl.when(is_team).then(pl.lit(factors["volume"])).otherwise(vol_expr)
        pass_share_expr = pl.when(is_team).then(pl.lit(factors["pass_share"])).otherwise(pass_share_expr)
        run_share_expr = pl.when(is_team).then(pl.lit(factors["run_share"])).otherwise(run_share_expr)
        env_expr = pl.when(is_team).then(pl.lit(factors["env"])).otherwise(env_expr)

    # Position-aware share factor
    if has_position:
        is_rb = pl.col("position") == "RB"
        is_skill = pl.col("position").is_in(["QB", "WR", "TE"])
        share_expr = (
            pl.when(is_rb).then(run_share_expr)
            .when(is_skill).then(pass_share_expr)
            .otherwise(pl.lit(1.0))
        )
    else:
        share_expr = pass_share_expr

    combined = vol_expr * share_expr * env_expr
    combined = combined.clip(0.85, 1.15)

    df = df.with_columns([
        (pl.col("projected_ppg") * combined).alias("projected_ppg"),
        (pl.col("ceiling") * combined).alias("ceiling"),
        (pl.col("floor") * combined).alias("floor"),
    ])

    # Recompute projected_points
    if "games_played_projection" in df.columns:
        df = df.with_columns(
            (pl.col("projected_ppg") * pl.col("games_played_projection"))
            .alias("projected_points"),
        )

    return df


def _compute_play_factors(year: int) -> tuple[dict[str, float], dict[str, float]]:
    """Compute volume and pass-share factors from ff_opportunity team aggregates.

    Returns (volume_factors, share_factors) — dicts keyed by team abbreviation.
    """
    try:
        opp = nfl.load_ff_opportunity(seasons=[year])
    except Exception:
        return {}, {}

    if opp is None or len(opp) == 0:
        return {}, {}

    needed = ["posteam", "season", "week", "pass_attempt_team", "rush_attempt_team"]
    if not all(c in opp.columns for c in needed):
        return {}, {}

    # Per-team per-game plays
    team_games = opp.select(needed).unique(subset=["posteam", "season", "week"])

    # Pass attempts already per-game, but we need to sum per team
    team_totals = team_games.group_by("posteam").agg([
        pl.col("pass_attempt_team").sum().alias("total_pass"),
        pl.col("rush_attempt_team").sum().alias("total_rush"),
        pl.len().alias("games"),
    ])

    team_totals = team_totals.with_columns([
        (pl.col("total_pass") + pl.col("total_rush")).alias("total_plays"),
        (pl.col("total_pass") / (pl.col("total_pass") + pl.col("total_rush"))).alias("pass_rate"),
        ((pl.col("total_pass") + pl.col("total_rush")) / pl.col("games")).alias("plays_per_game"),
    ])

    # League averages
    league_plays = team_totals["plays_per_game"].mean()
    league_pass_rate = team_totals["pass_rate"].mean()

    if league_plays is None or league_pass_rate is None:
        return {}, {}

    # Volume factor
    team_totals = team_totals.with_columns(
        (pl.col("plays_per_game") / league_plays).clip(0.92, 1.08).alias("volume_factor"),
    )

    # Pass share factor (to be applied position-dependently by caller)
    team_totals = team_totals.with_columns(
        (pl.col("pass_rate") / league_pass_rate).clip(0.90, 1.10).alias("pass_share_factor"),
    )

    vol_factors = {}
    share_factors = {}
    for row in team_totals.iter_rows(named=True):
        team = row["posteam"]
        vol_factors[team] = float(row["volume_factor"])
        share_factors[team] = float(row["pass_share_factor"])

    return vol_factors, share_factors


def _compute_env_factors(year: int) -> dict[str, float]:
    """Compute scoring environment factors from Vegas total lines.

    Returns dict: team_abbr -> env_factor.
    """
    try:
        sched = nfl.load_schedules(seasons=[year])
    except Exception:
        return {}

    if sched is None or len(sched) == 0:
        return {}

    if "total_line" not in sched.columns:
        return {}

    reg = sched.filter(pl.col("game_type") == "REG")

    # Build rows for each team's implied total
    home_rows = reg.select([
        pl.col("home_team").alias("team"),
        pl.col("total_line"),
    ])
    away_rows = reg.select([
        pl.col("away_team").alias("team"),
        pl.col("total_line"),
    ])
    all_teams = pl.concat([home_rows, away_rows])

    # Implied team total ≈ total_line / 2 (rough split)
    team_totals = all_teams.group_by("team").agg(
        (pl.col("total_line").mean() / 2.0).alias("implied_total"),
    )

    league_avg = team_totals["implied_total"].mean()
    if league_avg is None or league_avg == 0:
        return {}

    team_totals = team_totals.with_columns(
        (pl.col("implied_total") / league_avg).clip(0.90, 1.10).alias("env_factor"),
    )

    env_factors = {}
    for row in team_totals.iter_rows(named=True):
        env_factors[row["team"]] = float(row["env_factor"])

    return env_factors


def compute_position_share_factor(
    position: str, pass_share_factor: float, league_pass_rate: float | None = None
) -> float:
    """Convert raw pass-share factor to position-appropriate multiplier.

    QB/WR/TE: pass-heavy teams = more targets → use pass_share_factor as-is
    RB: run-heavy teams = more carries → invert the factor
    K/DEF/LB/DB: no adjustment → 1.0

    If league_pass_rate is provided, uses it to invert the RB factor.
    Otherwise uses simple inversion of pass_share_factor.
    """
    if position in ("K", "DEF", "LB", "DB"):
        return 1.0

    if position == "RB":
        # Invert: run-heavy = lower pass rate = higher RB value
        if league_pass_rate is not None:
            team_run_rate = pass_share_factor * league_pass_rate
            league_run_rate = 1.0 - league_pass_rate
            team_run_share = 1.0 - team_run_rate
            raw = team_run_share / league_run_rate if league_run_rate > 0 else 1.0
        else:
            raw = 2.0 - pass_share_factor  # simple inversion
        return max(0.90, min(1.10, raw))

    # QB, WR, TE: pass share as-is
    return pass_share_factor
