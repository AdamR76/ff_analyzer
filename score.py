"""Score pipeline stage -- apply scoring rules to raw stats.

Reads raw parquet, applies dynamic scoring engine, writes scored parquet.
Functional, no classes.
"""

import polars as pl
from scoring_engine import (
    parse_scoring_rules,
    group_rules_by_unit,
    COLUMN_MAP,
    REVERSE_COLUMN_MAP,
    normalize_position,
    IDP_POSITIONS,
)


def run_score_pipeline(cfg: dict, rules: list[dict] | None = None) -> dict:
    """Apply scoring rules to all raw season parquet files.

    cfg: config dict from load_config()
    rules: pre-parsed rules (optional, loads from scoring.txt if None)

    Returns dict with keys:
        output_paths: dict[int, Path]  -- season -> scored parquet path
        row_counts: dict[int, int]      -- season -> row count
    """
    if rules is None:
        rules = parse_scoring_rules(cfg["scoring_file"])

    raw_dir = cfg["data_dir"] / "raw"
    proc_dir = cfg["data_dir"] / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}
    row_counts = {}

    # Only match 4-digit season files (e.g. 2024.parquet). Skips
    # *_def.parquet files which have non-numeric stems.
    for raw_path in sorted(raw_dir.glob("[0-9][0-9][0-9][0-9].parquet")):
        season = int(raw_path.stem)
        df = pl.read_parquet(raw_path)

        # Normalize nflreadpy granular positions to canonical positions
        # (e.g. CB -> DB, DE -> LB) so downstream filtering works correctly.
        if "position" in df.columns:
            df = df.with_columns(
                pl.col("position").map_elements(
                    normalize_position, return_dtype=pl.Utf8
                ).alias("position")
            )
            # Drop rows with empty position (OL, P, etc.)
            df = df.filter(pl.col("position") != "")

        df = _apply_scoring_to_df(df, rules)

        out_path = proc_dir / f"{season}_scores.parquet"
        df.write_parquet(out_path)
        output_paths[season] = out_path
        row_counts[season] = len(df)

    return {
        "output_paths": output_paths,
        "row_counts": row_counts,
    }


def _apply_scoring_to_df(df: pl.DataFrame, rules: list[dict]) -> pl.DataFrame:
    """Apply all scoring rules to a Polars DataFrame.

    Uses column arithmetic for per-unit rules (vectorized, fast).
    Handles threshold rules via expressions that evaluate per-row.

    Returns DataFrame with original columns plus:
        fantasy_points: float  -- total fantasy points
        pp_{stat_key}: float   -- points from each rule (prefix 'pp_')
    """
    groups = group_rules_by_unit(rules)
    point_exprs = []
    has_position = "position" in df.columns

    for unit, unit_rules in groups.items():
        for rule in unit_rules:
            key = rule["stat_key"]
            multiplier = rule["points"]
            col_name = f"pp_{key}"

            if unit == "game":
                # Game rules: threshold patterns. Handled separately
                # in _apply_game_rules with row-by-row evaluation.
                # TODO: Replace row loop with DuckDB/vectorized
                # expressions for O(n*m) perf improvement on
                # larger datasets (57k rows fine for MVP).
                continue

            # All per_* units: simple multiplication.
            # Map scoring.txt key to actual DataFrame column name.
            df_key = COLUMN_MAP.get(key, key)
            if df_key in df.columns:
                base_expr = pl.col(df_key).fill_null(0) * multiplier

                # Per-position filtering:
                #   def_* rules → team defense (position="DEF") only
                #   idp_* rules → individual defenders (LB/DB) only
                #   all other rules → all positions
                if has_position and key.startswith("def_"):
                    expr = pl.when(pl.col("position") == "DEF").then(base_expr).otherwise(0.0).alias(col_name)
                elif has_position and key.startswith("idp_"):
                    expr = pl.when(pl.col("position").is_in(["LB", "DB"])).then(base_expr).otherwise(0.0).alias(col_name)
                else:
                    expr = base_expr.alias(col_name)

                point_exprs.append(expr)

    # Derive combined columns that scoring.txt rules reference but
    # nflreadpy does not provide directly.
    if "def_tackles_solo" in df.columns and "def_tackle_assists" in df.columns:
        df = df.with_columns(
            (pl.col("def_tackles_solo").fill_null(0)
             + pl.col("def_tackle_assists").fill_null(0))
            .alias("idp_tackle")
        )

    if point_exprs:
        df = df.with_columns(point_exprs)

    # Apply game rules using row-by-row logic via scoring_engine
    game_rules = [r for r in rules if r["unit"] == "game"]
    if game_rules:
        df = _apply_game_rules(df, game_rules)

    # Compute total fantasy points once, after all per-unit and game rules
    pp_cols = [c for c in df.columns if c.startswith("pp_")]
    if pp_cols:
        df = df.with_columns(
            pl.sum_horizontal([pl.col(c) for c in pp_cols]).alias("fantasy_points")
        )

    return df


def _apply_game_rules(df: pl.DataFrame, game_rules: list[dict]) -> pl.DataFrame:
    """Apply game-unit threshold rules using row-by-row evaluation.

    Delegates to scoring_engine._eval_game_rule so all 6 pattern
    branches (direct column, bonus, less-than, plus, range, floor)
    are covered by a single source of truth.

    Translates nflreadpy column names back to scoring.txt keys so
    the game-rule engine can find stat columns by their short names
    (e.g. rushing_yards -> rush_yd for bonus_rush_100 lookups).

    TODO: Replace O(n*m) Python row loop with DuckDB or vectorized
    Polars expressions once data volume grows beyond MVP (57k rows
    is acceptable for now).
    """
    from scoring_engine import _eval_game_rule

    # Build a column rename map for this DataFrame: translate
    # nflreadpy columns back to scoring.txt keys where known.
    rename_map = {}
    for nfl_col in df.columns:
        scoring_key = REVERSE_COLUMN_MAP.get(nfl_col)
        if scoring_key is not None and scoring_key != nfl_col:
            rename_map[nfl_col] = scoring_key

    df_mapped = df.rename(rename_map) if rename_map else df

    for rule in game_rules:
        key = rule["stat_key"]
        col_name = f"pp_{key}"

        # Evaluate rule for each row
        points = []
        for row in df_mapped.iter_rows(named=True):
            points.append(_eval_game_rule(rule, row))

        df = df.with_columns(pl.Series(col_name, points, dtype=pl.Float64))

    return df
