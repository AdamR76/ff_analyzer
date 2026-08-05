# CLAUDE.md

Fantasy football draft analyzer. Four-stage pipeline + PBP aggregation: fetch → pbp → score → project → rank.

## Tech Stack

- **Python 3.12** (venv at `venv/`)
- **Polars** (`polars>=1.43`) — primary dataframe library (not pandas)
- **DuckDB** (`duckdb>=1.5`) — in-process SQL (available, not yet used)
- **PyArrow** (`pyarrow>=25.0`) — columnar data interchange
- **Pydantic** (`pydantic>=2.13`) — data models (available, not yet used)
- **nflreadpy** (`nflreadpy>=0.1.5`) — all NFL data: player stats, team stats, PBP, drafts, combine, players

## Commands

```bash
source venv/bin/activate        # Always activate first

# Full pipeline
python run_pipeline.py --pick 5

# Individual stages
python fetch.py                 # Stage 1: nflreadpy → data/raw/*.parquet
python score.py                 # Stage 2: data/raw/* → data/processed/*_scores.parquet
python project.py               # Stage 3: data/processed/* → data/projections/2026_projections.parquet
python rank.py --pick 5         # Stage 4: data/projections/* → output/*.csv

# Tests
python -m pytest tests/ -v
python -m pytest tests/test_score.py -v
```

## Pipeline

```
Stage 1: Fetch          fetch.py           nflreadpy → data/raw/*.parquet + *_def.parquet
Stage 1.5: PBP          pbp_aggregate.py   nflreadpy.load_pbp() → game-level stat columns
                                            (TD distance, pick_six, def_3nout, def_4down_stop,
                                             def_blk_kick, rec_30_39, st_ff, stp_ff, etc.)
                                            Joined into raw parquets before scoring.
Stage 2: Score          score.py           data/raw/* → data/processed/*_scores.parquet
Stage 3: Project        project.py         data/processed/* → data/projections/2026_projections.parquet
Stage 4: Rank           rank.py            data/projections/* → output/*.csv
```

### Stage 1 — Fetch (`fetch.py`)

Pulls `nfl.load_player_stats()`, `nfl.load_team_stats()`, and PBP aggregates for 2023-2025. Joins schedules for `def_pa` (points allowed). Regular season only (`season_type == "REG"`). Writes per-season Parquet to `data/raw/`.

### Stage 1.5 — PBP Aggregate (`pbp_aggregate.py`)

Pre-aggregates `nfl.load_pbp()` to game-level stats. Produces three DataFrames:
- `offense` — player-game: TD distance bonuses, `pick_six`, `rec_30_39`
- `defense` — team-game: `def_yd`, `def_3nout`, `def_4down_stop`, `def_blk_kick`, `def_2pt_ret`, `st_ff`, `st_fum_rec`
- `idp` — IDP-player-game: `idp_int_td_50`, `idp_fum_td_50`, `idp_blk_kick`, `stp_ff`, `stp_fum_rec`

Offense and IDP joined into player stats; defense available for future team defense integration.

### Stage 2 — Score (`score.py`)

Reads raw Parquet files, applies scoring rules from `scoring.txt` via `scoring_engine.py`. Vectorized column arithmetic for `per_*` rules; row-by-row evaluation for `game` threshold rules. Normalizes positions (CB/FS→DB, DE/DT→LB, FB→RB). Writes scored Parquet.

### Stage 3 — Project (`project.py`)

Three-year weighted PPG projection (50%/30%/20%). Position-specific adjustments:
- **Age curves** — declining multiplier by position (RB:27+, WR:29+, TE:30+, QB:35+)
- **Injury model** — Bayesian games-played projection from career history
- **Partial-season shrinkage** — low-sample PPGs regressed to positional mean
- **Trend adjustment** — ascending/declining PPG trajectory, capped ±15%
- **WR 3rd-year breakout** — bonus for year-3 WRs (draft_year=2024)
- **TE elite flag** — 1.10× premium for top-3 TEs
- **QB rushing baseline** — floor boost for QBs with ≥20 rush yd/game

Rookies projected via comparable-player model (`rookies.py`) using draft capital + combine data. Data-driven hit rates and round baselines computed from historical data; fall back to hardcoded tables when insufficient data.

All adjustments controlled by env-var feature flags: `FF_AGE_CURVE`, `FF_INJURY_MODEL`, `FF_SHRINKAGE`, `FF_TREND_ADJUST`, `FF_WR_BREAKOUT`, `FF_TE_ELITE`, `FF_QB_RUSHING_BASELINE` (all default `true`).

### Stage 4 — Rank (`rank.py`)

Computes VORP (Value Over Replacement) from roster-driven replacement levels. Assigns tiers based on VORP gaps (threshold: 2.0). Generates snake draft strategy when `--pick N` is given. Writes `output/rankings.csv`, `output/tiers.csv`, `output/strategy.csv`.

## Project Structure

| File | Purpose |
|---|---|
| `run_pipeline.py` | Main pipeline runner (stages 1-4) |
| `config.py` | Configuration loader from env vars / CLI args |
| `fetch.py` | Stage 1: data ingestion from nflreadpy |
| `pbp_aggregate.py` | Stage 1.5: PBP → game-level stat aggregation |
| `score.py` | Stage 2: scoring engine wrapper |
| `scoring_engine.py` | Dynamic stat-to-points rule evaluator |
| `project.py` | Stage 3: 3-year weighted projections + adjustments |
| `rookies.py` | Rookie comparable-player projection model |
| `rank.py` | Stage 4: VORP rankings, tiers, draft strategy |
| `roster.py` | Roster file parser |
| `vorp.py` | VORP calculations, tier assignment |
| `scoring.txt` | League scoring rules (86 rules, pipe-delimited) |
| `roster.txt` | League roster construction (single CSV line) |
| `requirements.txt` | Python dependencies |
| `tests/` | pytest test suite (102 tests) |

## Data Files

### `scoring.txt`

Pipe-delimited fantasy scoring rules: `stat_key | points | unit | description`.

`scoring_engine.py.COLUMN_MAP` bridges stat keys to nflreadpy column names. Keys not in COLUMN_MAP use identity passthrough (PBP-derived columns keep their stat key name).

### `roster.txt`

Single comma-separated line. Position tokens: `QB, RB, WR, TE, K, DEF, LB, DB`. Flex tokens: `WRT` (RB/WR/TE), `WRTQ` (QB/RB/WR/TE superflex). `BN` = bench, `IR` = injured reserve.

## Configuration

All via `config.py` — env vars with sensible defaults. Key vars:

| Variable | Default | Purpose |
|----------|---------|---------|
| `FF_WEIGHT_CURRENT` | `0.50` | Most recent season weight |
| `FF_WEIGHT_PREV` | `0.30` | Second season weight |
| `FF_WEIGHT_OLDEST` | `0.20` | Oldest season weight |
| `FF_NUM_TEAMS` | `12` | Teams in league |
| `FF_DRAFT_POSITION` | (none) | Draft slot (also `--pick N`) |
| `FF_SCORING_FILE` | `scoring.txt` | Custom scoring rules path |
| `FF_ROSTER_FILE` | `roster.txt` | Custom roster config path |
| `FF_AGE_CURVE` / `FF_INJURY_MODEL` / etc. | `true` | Projection feature flags |

## Conventions

- **Polars over pandas** — polars is the standard; never use pandas.
- **Functional style** — pure functions, no classes. Plain dicts for config.
- **Position normalization** — `scoring_engine.normalize_position()` maps nflreadpy granular positions to 8 canonical positions: QB, RB, WR, TE, K, DEF, LB, DB.
- **Scoring engine is dynamic** — all rules from `scoring.txt` at runtime. No hardcoded stat-to-points mapping. To change scoring, edit `scoring.txt` or the `COLUMN_MAP` bridge in `scoring_engine.py`.
- **PBP aggregation is pre-processing** — PBP data is aggregated to game-level BEFORE scoring. The scoring engine itself needs no changes to handle PBP-derived stats.
- **nflreadpy returns Polars DataFrames** — no conversion needed.
