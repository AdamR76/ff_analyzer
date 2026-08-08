#!/usr/bin/env python3
"""Backtest module — validate projections against actual fantasy results.

Trains on historical seasons, projects a held-out target year, then compares
projected rankings against actual fantasy finishes.

Usage:
    python backtest.py --train-years 2022,2023,2024 --target-year 2025
    python backtest.py  # uses defaults: 2023,2024,2025 -> 2026
"""

import polars as pl
from pathlib import Path
from config import load_config, parse_args
from fetch import fetch_raw_data, fetch_team_defense
from score import run_score_pipeline
from project import run_projection_pipeline
from scoring_engine import parse_scoring_rules, POSITIONS


def run_backtest(cfg: dict) -> dict:
    """Run backtest: train on train_years, project target_year, compare.

    Returns dict with keys:
        results: pl.DataFrame  — per-position + overall correlations
        output_path: Path      — path to backtest CSV
    """
    train_years = cfg["train_years"]
    target_year = cfg["target_year"]

    print(f"=== Backtest: {train_years} -> {target_year} ===\n")

    # Stage 1: Fetch raw data for train years only (target year fetched separately
    # so it doesn't contaminate projections)
    print("[1/4] Fetching training data...")
    fetch_raw_data(cfg, seasons=train_years)
    fetch_team_defense(cfg, seasons=train_years)
    print(f"  Fetched {train_years}")

    # Stage 2: Score train years
    print("\n[2/4] Scoring training data...")
    rules = parse_scoring_rules(cfg["scoring_file"])
    print(f"  Loaded {len(rules)} scoring rules")
    run_score_pipeline(cfg, rules=rules)

    # Stage 3: Project target year (only sees train_years in processed/)
    print(f"\n[3/4] Projecting {target_year}...")
    proj_result = run_projection_pipeline(cfg, train_seasons=train_years)
    print(f"  {proj_result['player_count']} players projected")

    # Stage 4: Fetch + score target year for ground truth
    print(f"\n[4/4] Fetching + scoring {target_year} actuals...")
    fetch_raw_data(cfg, seasons=[target_year])
    fetch_team_defense(cfg, seasons=[target_year])
    run_score_pipeline(cfg, rules=rules)

    # ── Compare projections vs actuals ──
    print("\n=== Comparing Projections vs Actuals ===\n")

    proj_path = cfg["data_dir"] / "projections" / f"{target_year}_projections.parquet"
    projections = pl.read_parquet(proj_path)

    actual_path = cfg["data_dir"] / "processed" / f"{target_year}_scores.parquet"
    actual_scores = pl.read_parquet(actual_path)

    results = _compute_rank_correlation(projections, actual_scores)

    # Write output
    out_dir = cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"backtest_{target_year}.csv"
    results.write_csv(out_path)

    # Print summary
    _print_summary(results, projections, actual_scores, out_path)

    return {
        "results": results,
        "output_path": out_path,
    }


def _compute_rank_correlation(
    projections: pl.DataFrame, actual_scores: pl.DataFrame
) -> pl.DataFrame:
    """Compute Spearman rank correlation between projected and actual points.

    Joins on player_id, computes rank correlation overall and per position.
    Excludes rookies (source="rookie_model") and players not in actual data.
    """
    # Aggregate actual scores to season totals per player
    actual_agg = actual_scores.group_by(["player_id"]).agg([
        pl.col("fantasy_points").sum().alias("actual_points"),
        pl.col("position").first().alias("position"),
        pl.col("player_name").first().alias("player_name"),
    ])

    # Inner join: only players in both projections and actuals
    joined = projections.join(
        actual_agg.select(["player_id", "actual_points", "player_name"]),
        on="player_id",
        how="inner",
        suffix="_actual",
    )

    # Exclude rookies (no actual history to validate against)
    if "source" in joined.columns:
        joined = joined.filter(pl.col("source") != "rookie_model")

    if len(joined) == 0:
        return pl.DataFrame(schema={
            "position": pl.Utf8, "n_players": pl.Int64,
            "spearman_r": pl.Float64, "p_value": pl.Float64,
            "mean_abs_error": pl.Float64,
        })

    # Use position from actual data for grouping
    if "position" in joined.columns:
        joined = joined.rename({"position": "projected_position"})

    # Get position from actual_agg
    joined = joined.join(
        actual_agg.select(["player_id", "position"]),
        on="player_id", how="left",
    )

    # Compute overall correlation
    overall = _spearman_row(joined, "OVERALL")

    # Compute per-position correlations
    rows = [overall]
    for pos in POSITIONS:
        pos_data = joined.filter(pl.col("position") == pos)
        if len(pos_data) >= 5:  # Need minimum sample for meaningful correlation
            rows.append(_spearman_row(pos_data, pos))

    return pl.DataFrame(rows)


def _spearman_row(df: pl.DataFrame, position: str) -> dict:
    """Compute Spearman rank correlation for a DataFrame subset."""
    n = len(df)
    if n < 3:
        return {
            "position": position, "n_players": n,
            "spearman_r": None, "p_value": None,
            "mean_abs_error": None,
        }

    # Rank projected_points and actual_points
    proj_vals = df["projected_points"].to_list()
    actual_vals = df["actual_points"].to_list()

    # Compute Spearman rho = Pearson r on ranks
    rho = _spearman_rho(proj_vals, actual_vals)

    # Mean absolute error
    mae = sum(abs(p - a) for p, a in zip(proj_vals, actual_vals)) / n

    return {
        "position": position,
        "n_players": n,
        "spearman_r": round(rho, 4) if rho is not None else None,
        "p_value": None,  # Simplified: no p-value computation
        "mean_abs_error": round(mae, 2),
    }


def _spearman_rho(x: list[float], y: list[float]) -> float | None:
    """Compute Spearman rank correlation coefficient."""
    n = len(x)
    if n < 2:
        return None

    # Rank the values (1 = lowest)
    def rank_list(vals: list[float]) -> list[float]:
        sorted_pairs = sorted(enumerate(vals), key=lambda kv: kv[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and sorted_pairs[j][1] == sorted_pairs[i][1]:
                j += 1
            avg_rank = (i + j + 2) / 2.0  # 1-based average rank
            for k in range(i, j):
                ranks[sorted_pairs[k][0]] = avg_rank
            i = j
        return ranks

    rx = rank_list(x)
    ry = rank_list(y)

    # Pearson correlation on ranks
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    std_x = (sum((rx[i] - mean_rx) ** 2 for i in range(n))) ** 0.5
    std_y = (sum((ry[i] - mean_ry) ** 2 for i in range(n))) ** 0.5

    if std_x == 0 or std_y == 0:
        return None

    return cov / (std_x * std_y)


def _print_summary(
    results: pl.DataFrame, projections: pl.DataFrame, actual_scores: pl.DataFrame,
    out_path: Path | None = None,
) -> None:
    """Print human-readable backtest summary."""
    overall = results.filter(pl.col("position") == "OVERALL")
    if len(overall) > 0:
        r = overall["spearman_r"][0]
        n = overall["n_players"][0]
        mae = overall["mean_abs_error"][0]
        print(f"OVERALL: n={n}, Spearman r={r:.4f}, MAE={mae:.1f} pts")

    print()
    for pos in POSITIONS:
        row = results.filter(pl.col("position") == pos)
        if len(row) > 0 and row["spearman_r"][0] is not None:
            r = row["spearman_r"][0]
            n = row["n_players"][0]
            mae = row["mean_abs_error"][0]
            print(f"  {pos:4s}: n={n:3d}, Spearman r={r:+.4f}, MAE={mae:7.1f} pts")

    # Top 5 over-projected and under-projected players
    actual_agg = actual_scores.group_by(["player_id", "player_name"]).agg(
        pl.col("fantasy_points").sum().alias("actual_points")
    )
    joined = projections.join(
        actual_agg, on="player_id", how="inner"
    ).with_columns(
        (pl.col("projected_points") - pl.col("actual_points")).alias("error")
    ).sort("error", descending=True)

    if len(joined) > 0:
        print("\nMost over-projected (projected >> actual):")
        for row in joined.head(5).iter_rows(named=True):
            print(f"  {row['player_name']:25s} proj={row['projected_points']:7.1f}  "
                  f"actual={row['actual_points']:7.1f}  error=+{row['error']:.1f}")

        print("\nMost under-projected (actual >> projected):")
        for row in joined.tail(5).sort("error").head(5).iter_rows(named=True):
            print(f"  {row['player_name']:25s} proj={row['projected_points']:7.1f}  "
                  f"actual={row['actual_points']:7.1f}  error={row['error']:.1f}")

    if out_path:
        print(f"\nResults saved to: {out_path}")


def main(argv: list[str] | None = None) -> dict:
    """CLI entry point for backtest."""
    overrides = parse_args(argv)
    cfg = load_config()

    # Apply overrides
    if "train_years" in overrides:
        cfg["train_years"] = overrides["train_years"]
    if "target_year" in overrides:
        cfg["target_year"] = overrides["target_year"]
    if "scoring_file" in overrides:
        cfg["scoring_file"] = overrides["scoring_file"]
    if "roster_file" in overrides:
        cfg["roster_file"] = overrides["roster_file"]

    return run_backtest(cfg)


if __name__ == "__main__":
    main()
