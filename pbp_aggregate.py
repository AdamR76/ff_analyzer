"""PBP aggregation stage — pre-aggregate play-by-play to game-level stats.

Converts nflreadpy.load_pbp() play-level data into game-level columns
that the scoring engine can consume. No changes to scoring_engine.py needed
— just feed it the right column names.

Single public function: aggregate_pbp(seasons) -> dict[str, pl.DataFrame]
"""

import polars as pl
import nflreadpy as nfl


def aggregate_pbp(
    seasons: list[int] | None = None,
) -> dict[str, pl.DataFrame]:
    """Pre-aggregate PBP data into game-level stat DataFrames.

    Returns dict with keys:
        'offense' — player-game rows: TD distance, pick_six, rec_30_39
        'defense' — defteam-game rows: def_yd, def_3nout, def_4down_stop,
                    def_blk_kick, def_2pt_ret, st_ff, st_fum_rec
        'idp'     — IDP-player-game rows: idp_int_td_50, idp_fum_td_50,
                    idp_blk_kick, stp_ff, stp_fum_rec

    Each DataFrame has (player_id, season, week) or (team, season, week)
    as join keys for merging into the main player/team DataFrames.
    """
    if seasons is None:
        seasons = [2023, 2024, 2025]

    pbp = nfl.load_pbp(seasons=seasons)
    pbp = pbp.filter(pl.col("season_type") == "REG")

    return {
        "offense": _aggregate_offense(pbp),
        "defense": _aggregate_defense(pbp),
        "idp": _aggregate_idp(pbp),
    }


# ═══════════════════════════════════════════════════════════════════════
# Offense: TD distance bonuses, pick_six, rec_30_39
# ═══════════════════════════════════════════════════════════════════════


def _aggregate_offense(pbp: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-player per-game offensive stats from PBP.

    Returns DataFrame with columns:
        player_id, season, week,
        pass_td_40, pass_td_50, rush_td_40, rush_td_50,
        rec_td_40, rec_td_50, rec_30_39, pick_six
    """
    td_frames = []

    # Pass TD distance bonuses — credited to passer
    pass_td = pbp.filter(pl.col("pass_touchdown") == 1)
    td_frames.append(_count_td_distance(pass_td, "passer_player_id", "pass_td"))

    # Rush TD distance bonuses — credited to rusher
    rush_td = pbp.filter(pl.col("rush_touchdown") == 1)
    td_frames.append(_count_td_distance(rush_td, "rusher_player_id", "rush_td"))

    # Rec TD distance bonuses — credited to receiver
    rec_td = pbp.filter(
        (pl.col("pass_touchdown") == 1) & pl.col("receiver_player_id").is_not_null()
    )
    td_frames.append(_count_td_distance(rec_td, "receiver_player_id", "rec_td"))

    # rec_30_39: completions of 30-39 yards (excluding TDs — those already
    # score under rec_td and rec_td_40/50 rules)
    rec_30_39 = (
        pbp.filter(
            (pl.col("play_type") == "pass")
            & (pl.col("yards_gained") >= 30)
            & (pl.col("yards_gained") < 40)
            & (pl.col("touchdown") == 0)
            & pl.col("receiver_player_id").is_not_null()
        )
        .group_by(["receiver_player_id", "season", "week"])
        .agg(pl.len().alias("rec_30_39"))
        .rename({"receiver_player_id": "player_id"})
    )

    # pick_six: interception returned for TD — penalty on passer
    pick_six = (
        pbp.filter(
            (pl.col("interception") == 1)
            & (pl.col("return_touchdown") == 1)
        )
        .group_by(["passer_player_id", "season", "week"])
        .agg(pl.len().alias("pick_six"))
        .rename({"passer_player_id": "player_id"})
    )

    # Merge all offense stats into one player-game DataFrame
    all_offense = pl.concat(td_frames + [rec_30_39, pick_six], how="diagonal_relaxed")

    if len(all_offense) == 0:
        return _empty_offense_frame()

    # Aggregate: sum all columns per player-game (handles overlapping keys
    # from concat — same player might appear in multiple TD frames)
    stat_cols = [c for c in all_offense.columns if c not in ("player_id", "season", "week")]
    all_offense = all_offense.group_by(["player_id", "season", "week"]).agg(
        [pl.col(c).sum() for c in stat_cols]
    )

    return all_offense


def _count_td_distance(
    td_plays: pl.DataFrame, player_col: str, prefix: str
) -> pl.DataFrame:
    """Count TD plays by distance band (40+, 50+) per player per game."""
    td_plays = td_plays.with_columns([
        (pl.col("yards_gained") >= 40).cast(pl.Int32).alias(f"{prefix}_40"),
        (pl.col("yards_gained") >= 50).cast(pl.Int32).alias(f"{prefix}_50"),
    ])

    return (
        td_plays.group_by([player_col, "season", "week"])
        .agg([
            pl.col(f"{prefix}_40").sum(),
            pl.col(f"{prefix}_50").sum(),
        ])
        .rename({player_col: "player_id"})
    )


def _empty_offense_frame() -> pl.DataFrame:
    """Return empty offense frame with correct schema."""
    return pl.DataFrame(
        schema={
            "player_id": pl.Utf8, "season": pl.Int32, "week": pl.Int32,
            "pass_td_40": pl.Int32, "pass_td_50": pl.Int32,
            "rush_td_40": pl.Int32, "rush_td_50": pl.Int32,
            "rec_td_40": pl.Int32, "rec_td_50": pl.Int32,
            "rec_30_39": pl.UInt32, "pick_six": pl.UInt32,
        }
    )


# ═══════════════════════════════════════════════════════════════════════
# Defense: yards allowed, 3-and-outs, 4th-down stops, blocked kicks, ST
# ═══════════════════════════════════════════════════════════════════════


def _aggregate_defense(pbp: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-team per-game defensive stats from PBP.

    Returns DataFrame with columns:
        team, season, week,
        def_yd, def_3nout, def_4down_stop,
        def_blk_kick, def_2pt_ret, st_ff, st_fum_rec
    """
    frames = []

    # Yards allowed: sum opponent offensive yards per defteam per game
    def_yd = (
        pbp.filter(pl.col("defteam").is_not_null())
        .group_by(["defteam", "season", "week"])
        .agg(
            (pl.col("passing_yards").sum() + pl.col("rushing_yards").sum()).alias("def_yd")
        )
        .rename({"defteam": "team"})
    )
    frames.append(def_yd)

    # 3-and-outs: drives with <= 3 real plays, 0 first downs, no score
    frames.append(_count_three_and_outs(pbp))

    # 4th-down stops
    def_4down = (
        pbp.filter(
            (pl.col("down") == 4) & (pl.col("fourth_down_failed") == 1)
        )
        .group_by(["defteam", "season", "week"])
        .agg(pl.len().alias("def_4down_stop"))
        .rename({"defteam": "team"})
    )
    frames.append(def_4down)

    # Blocked kicks (defense)
    def_blk = _count_blocked_kicks(pbp)
    if len(def_blk) > 0:
        frames.append(def_blk)

    # Defensive 2pt returns
    def_2pt = (
        pbp.filter(pl.col("defensive_two_point_conv") == 1)
        .group_by(["defteam", "season", "week"])
        .agg(pl.len().alias("def_2pt_ret"))
        .rename({"defteam": "team"})
    )
    frames.append(def_2pt)

    # ST forced fumbles and recoveries (team-level, coverage unit)
    st_plays = pbp.filter(pl.col("special_teams_play") == 1.0)
    if len(st_plays) > 0:
        st_ff = (
            st_plays.filter(pl.col("fumble_forced") == 1)
            .group_by(["defteam", "season", "week"])
            .agg(pl.len().alias("st_ff"))
            .rename({"defteam": "team"})
        )
        frames.append(st_ff)

        st_fr = (
            st_plays.filter(pl.col("fumble") == 1)
            .group_by(["defteam", "season", "week"])
            .agg(pl.len().alias("st_fum_rec"))
            .rename({"defteam": "team"})
        )
        frames.append(st_fr)

    # Merge all defense stats
    all_def = pl.concat(frames, how="diagonal_relaxed")
    if len(all_def) == 0:
        return _empty_defense_frame()

    stat_cols = [c for c in all_def.columns if c not in ("team", "season", "week")]
    return all_def.group_by(["team", "season", "week"]).agg(
        [pl.col(c).sum() for c in stat_cols]
    )


def _count_three_and_outs(pbp: pl.DataFrame) -> pl.DataFrame:
    """Count 3-and-outs forced per team per game.

    A 3-and-out: drive had <= 3 real plays, 0 first downs, no TD, no FG made.
    Credited to the defteam that was on the field.
    """
    # Exclude non-plays and clock kills from play count
    real = pbp.filter(
        ~pl.col("play_type").is_in(["no_play", "qb_kneel", "qb_spike"])
        & pl.col("defteam").is_not_null()
    )

    drive_summary = real.group_by(["game_id", "drive", "defteam", "season", "week"]).agg([
        pl.len().alias("play_count"),
        pl.col("first_down").sum().alias("first_downs"),
        pl.col("touchdown").max().alias("had_td"),
        pl.col("field_goal_result")
        .filter(pl.col("field_goal_result") == "made")
        .count()
        .alias("fg_made_count"),
    ])

    three_and_out = drive_summary.filter(
        (pl.col("play_count") <= 3)
        & (pl.col("first_downs") == 0)
        & (pl.col("had_td") == 0)
        & (pl.col("fg_made_count") == 0)
    )

    return (
        three_and_out.group_by(["defteam", "season", "week"])
        .agg(pl.len().alias("def_3nout"))
        .rename({"defteam": "team"})
    )


def _count_blocked_kicks(pbp: pl.DataFrame) -> pl.DataFrame:
    """Count blocked kicks (FG, punt, XP) per team per game."""
    fg_blocked = pbp.filter(pl.col("field_goal_result") == "blocked")
    punt_blocked = pbp.filter(pl.col("punt_blocked") == 1.0)
    xp_blocked = pbp.filter(pl.col("extra_point_result") == "blocked")

    blocked = pl.concat([fg_blocked, punt_blocked, xp_blocked])
    if len(blocked) == 0:
        return _empty_defense_frame().select(["team", "season", "week", "def_blk_kick"])

    return (
        blocked.group_by(["defteam", "season", "week"])
        .agg(pl.len().alias("def_blk_kick"))
        .rename({"defteam": "team"})
    )


def _empty_defense_frame() -> pl.DataFrame:
    """Return empty defense frame with correct schema."""
    return pl.DataFrame(
        schema={
            "team": pl.Utf8, "season": pl.Int32, "week": pl.Int32,
            "def_yd": pl.Int32, "def_3nout": pl.UInt32, "def_4down_stop": pl.UInt32,
            "def_blk_kick": pl.UInt32, "def_2pt_ret": pl.UInt32,
            "st_ff": pl.UInt32, "st_fum_rec": pl.UInt32,
        }
    )


# ═══════════════════════════════════════════════════════════════════════
# IDP: individual defensive player bonuses
# ═══════════════════════════════════════════════════════════════════════


def _aggregate_idp(pbp: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-IDP-player per-game stats from PBP.

    Returns DataFrame with columns:
        player_id, season, week,
        idp_int_td_50, idp_fum_td_50, idp_blk_kick, stp_ff, stp_fum_rec
    """
    frames = []

    # idp_int_td_50: INT returned for TD, 50+ yards
    int_td_50 = (
        pbp.filter(
            (pl.col("interception") == 1)
            & (pl.col("return_touchdown") == 1)
            & (pl.col("return_yards") >= 50)
        )
        .group_by(["interception_player_id", "season", "week"])
        .agg(pl.len().alias("idp_int_td_50"))
        .rename({"interception_player_id": "player_id"})
    )
    frames.append(int_td_50)

    # idp_fum_td_50: fumble returned for TD, 50+ yards
    fum_td_50 = (
        pbp.filter(
            (pl.col("fumble") == 1)
            & (pl.col("return_touchdown") == 1)
            & (pl.col("return_yards") >= 50)
        )
        .group_by(["fumble_recovery_1_player_id", "season", "week"])
        .agg(pl.len().alias("idp_fum_td_50"))
        .rename({"fumble_recovery_1_player_id": "player_id"})
    )
    frames.append(fum_td_50)

    # idp_blk_kick: player who blocked the kick
    blocked = pl.concat([
        pbp.filter(pl.col("field_goal_result") == "blocked"),
        pbp.filter(pl.col("punt_blocked") == 1.0),
        pbp.filter(pl.col("extra_point_result") == "blocked"),
    ])
    if len(blocked) > 0:
        blk_player = (
            blocked.filter(pl.col("blocked_player_id").is_not_null())
            .group_by(["blocked_player_id", "season", "week"])
            .agg(pl.len().alias("idp_blk_kick"))
            .rename({"blocked_player_id": "player_id"})
        )
        frames.append(blk_player)

    # ST player forced fumbles and recoveries
    st_plays = pbp.filter(pl.col("special_teams_play") == 1.0)
    if len(st_plays) > 0:
        # stp_ff: credit forced fumble player
        stp_ff = (
            st_plays.filter(
                (pl.col("fumble_forced") == 1)
                & pl.col("forced_fumble_player_1_player_id").is_not_null()
            )
            .group_by(["forced_fumble_player_1_player_id", "season", "week"])
            .agg(pl.len().alias("stp_ff"))
            .rename({"forced_fumble_player_1_player_id": "player_id"})
        )
        frames.append(stp_ff)

        # stp_fum_rec: credit fumble recovery player
        stp_fr = (
            st_plays.filter(
                (pl.col("fumble") == 1)
                & pl.col("fumble_recovery_1_player_id").is_not_null()
            )
            .group_by(["fumble_recovery_1_player_id", "season", "week"])
            .agg(pl.len().alias("stp_fum_rec"))
            .rename({"fumble_recovery_1_player_id": "player_id"})
        )
        frames.append(stp_fr)

    # Also: forced fumble player #2
    if len(st_plays) > 0:
        stp_ff2 = (
            st_plays.filter(
                (pl.col("fumble_forced") == 1)
                & pl.col("forced_fumble_player_2_player_id").is_not_null()
            )
            .group_by(["forced_fumble_player_2_player_id", "season", "week"])
            .agg(pl.len().alias("stp_ff"))
            .rename({"forced_fumble_player_2_player_id": "player_id"})
        )
        frames.append(stp_ff2)

    all_idp = pl.concat(frames, how="diagonal_relaxed")
    if len(all_idp) == 0:
        return _empty_idp_frame()

    stat_cols = [c for c in all_idp.columns if c not in ("player_id", "season", "week")]
    return all_idp.group_by(["player_id", "season", "week"]).agg(
        [pl.col(c).sum() for c in stat_cols]
    )


def _empty_idp_frame() -> pl.DataFrame:
    """Return empty IDP frame with correct schema."""
    return pl.DataFrame(
        schema={
            "player_id": pl.Utf8, "season": pl.Int32, "week": pl.Int32,
            "idp_int_td_50": pl.UInt32, "idp_fum_td_50": pl.UInt32,
            "idp_blk_kick": pl.UInt32, "stp_ff": pl.UInt32, "stp_fum_rec": pl.UInt32,
        }
    )
