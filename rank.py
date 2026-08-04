"""Rank pipeline stage — VORP rankings, tiers, draft strategy.

Reads projections, computes VORP, produces CSV outputs.
Functional, no classes.
"""

import polars as pl
from roster import parse_roster
from vorp import compute_vorp, compute_tiers, _estimate_replacement_level
from scoring_engine import POSITIONS


def run_rank_pipeline(cfg: dict, roster: dict | None = None) -> dict:
    """Run full ranking pipeline: VORP -> tiers -> strategy.

    cfg: config dict from load_config()
    roster: pre-parsed roster dict (loads from roster file if None)

    Returns dict with keys:
        rankings: Path   — output/rankings.csv
        tiers: Path      — output/tiers.csv
        strategy: Path   — output/strategy.csv
    """
    if roster is None:
        roster = parse_roster(cfg.get("roster_file", "roster.txt"))

    proj_path = cfg["data_dir"] / "projections" / "2026_projections.parquet"
    projections = pl.read_parquet(proj_path)

    out_dir = cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute replacement levels
    repl_levels = _estimate_replacement_level(roster, cfg["num_teams"])

    # Compute VORP for each position
    all_vorp = []
    for pos in POSITIONS:
        vorp_df = compute_vorp(projections, pos, repl_levels.get(pos, 12))
        if len(vorp_df) > 0:
            all_vorp.append(vorp_df)

    ranked = pl.concat(all_vorp) if all_vorp else pl.DataFrame()

    # Overall rank by VORP
    ranked = ranked.sort("vorp", descending=True).with_row_index("overall_rank", offset=1)

    # Position rank
    ranked = ranked.with_columns(
        pl.col("vorp").rank("ordinal", descending=True).over("position").alias("position_rank")
    )

    # Tiers (gap > 2 VORP starts new tier)
    ranked = compute_tiers(ranked, threshold=2.0)

    # Write rankings CSV (top 300)
    rankings_path = out_dir / "rankings.csv"
    ranked.head(300).write_csv(rankings_path)

    # Write positional tiers CSV
    tiers_path = out_dir / "tiers.csv"
    tiers_out = ranked.select([
        "overall_rank", "player_id", "player_name", "position",
        "projected_points", "projected_ppg", "vorp", "tier", "position_rank"
    ]).sort(["position", "vorp"], descending=[False, True])
    tiers_out.write_csv(tiers_path)

    # Strategy: snake draft simulation (only when draft_position is set)
    result = {
        "rankings": rankings_path,
        "tiers": tiers_path,
    }
    if cfg.get("draft_position"):
        strategy_path = out_dir / "strategy.csv"
        strategy_df = _build_strategy(ranked, cfg)
        strategy_df.write_csv(strategy_path)
        result["strategy"] = strategy_path

    return result


def _simulate_snake_draft(
    pick_position: int, num_teams: int = 12, rounds: int = 20
) -> list[int]:
    """Compute overall pick numbers for a given draft slot in snake format.

    Snake draft: round 1 is 1->12, round 2 is 12->1, alternates.
    Returns list of overall pick numbers for this position.
    """
    picks = []
    for rnd in range(1, rounds + 1):
        if rnd % 2 == 1:  # odd round: forward
            pick = (rnd - 1) * num_teams + pick_position
        else:  # even round: reverse
            pick = rnd * num_teams - pick_position + 1
        picks.append(pick)
    return picks


def _build_strategy(
    ranked: pl.DataFrame, cfg: dict
) -> pl.DataFrame:
    """Simulate who's likely available at each of your picks.

    At each pick, show the top available player by VORP at each position.
    """
    picks = _simulate_snake_draft(
        cfg["draft_position"], cfg["num_teams"], cfg["draft_rounds"]
    )

    rows = []
    # Simple model: at each pick, assume all higher-ranked players are taken
    # (best-case availability = you get your pick's rank exactly)
    for i, overall_pick in enumerate(picks):
        round_num = i + 1
        still_available = ranked.filter(pl.col("overall_rank") >= overall_pick)

        best_qb = still_available.filter(pl.col("position") == "QB").head(1)
        best_rb = still_available.filter(pl.col("position") == "RB").head(1)
        best_wr = still_available.filter(pl.col("position") == "WR").head(1)
        best_te = still_available.filter(pl.col("position") == "TE").head(1)
        best_k = still_available.filter(pl.col("position") == "K").head(1)
        best_def = still_available.filter(pl.col("position") == "DEF").head(1)
        best_lb = still_available.filter(pl.col("position") == "LB").head(1)
        best_db = still_available.filter(pl.col("position") == "DB").head(1)

        rows.append({
            "round": round_num,
            "overall_pick": overall_pick,
            "best_qb": best_qb["player_name"][0] if len(best_qb) > 0 else "",
            "best_qb_vorp": best_qb["vorp"][0] if len(best_qb) > 0 else 0.0,
            "best_rb": best_rb["player_name"][0] if len(best_rb) > 0 else "",
            "best_rb_vorp": best_rb["vorp"][0] if len(best_rb) > 0 else 0.0,
            "best_wr": best_wr["player_name"][0] if len(best_wr) > 0 else "",
            "best_wr_vorp": best_wr["vorp"][0] if len(best_wr) > 0 else 0.0,
            "best_te": best_te["player_name"][0] if len(best_te) > 0 else "",
            "best_te_vorp": best_te["vorp"][0] if len(best_te) > 0 else 0.0,
            "best_k": best_k["player_name"][0] if len(best_k) > 0 else "",
            "best_k_vorp": best_k["vorp"][0] if len(best_k) > 0 else 0.0,
            "best_def": best_def["player_name"][0] if len(best_def) > 0 else "",
            "best_def_vorp": best_def["vorp"][0] if len(best_def) > 0 else 0.0,
            "best_lb": best_lb["player_name"][0] if len(best_lb) > 0 else "",
            "best_lb_vorp": best_lb["vorp"][0] if len(best_lb) > 0 else 0.0,
            "best_db": best_db["player_name"][0] if len(best_db) > 0 else "",
            "best_db_vorp": best_db["vorp"][0] if len(best_db) > 0 else 0.0,
        })

    return pl.DataFrame(rows)
