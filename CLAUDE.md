# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Fantasy football draft analyzer. Uses prior-year NFL statistics to predict player performance for the upcoming fantasy draft. Consumes league scoring rules, roster sizes, draft rounds, and draft position to generate rankings/recommendations.

## Tech Stack

- **Python 3.12** (venv at `venv/`)
- **Polars** (`polars>=1.43`) — primary dataframe/analytics library (not pandas)
- **DuckDB** (`duckdb>=1.5`) — in-process analytical SQL database
- **PyArrow** (`pyarrow>=25.0`) — columnar data interchange
- **Pydantic** (`pydantic>=2.13`) — data models and settings management
- **nflreadpy** (`nflreadpy>=0.1.5`) — NFL play-by-play and stat data source
- **PostgreSQL** — optional, use only if data volume warrants it

## Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python run_pipeline.py --pick 5

# Run individual stages
python fetch.py
python score.py
python project.py
python rank.py --pick 5

# Run all tests
python -m pytest tests/ -v

# Run single test file
python -m pytest tests/test_config.py -v
```

## Pipeline Architecture

Four-stage pipeline run by `run_pipeline.py`:

```
Stage 1: Fetch          fetch.py         nflreadpy -> data/raw/*.parquet
Stage 2: Score          score.py         data/raw/* -> data/processed/*_scores.parquet
Stage 3: Project        project.py       data/processed/* -> data/projections/2026_projections.parquet
Stage 4: Rank           rank.py          data/projections/* -> output/*.csv
```

### Stage 1 — Fetch (`fetch.py`)

Pulls player stats via `nflreadpy.load_player_stats()` for 2023-2025 seasons. Filters to regular season only (`season_type == "REG"`). Writes per-season Parquet files to `data/raw/{season}.parquet`.

### Stage 2 — Score (`score.py`)

Reads raw Parquet files, applies scoring rules from `scoring.txt` via `scoring_engine.py`. Vectorized column arithmetic for per-unit rules; row-by-row evaluation for game-unit threshold rules. Writes scored Parquet to `data/processed/{season}_scores.parquet`.

### Stage 3 — Project (`project.py`)

Three-year weighted projection model. Computes per-player weighted averages (current: 50%, prev: 30%, oldest: 20%) of fantasy points per game for veterans. Rookies are projected via comparable-player model using nflreadpy draft capital + combine data (`rookies.py`). Applies position-specific adjustments (DEF/K regression, future RB age cliff, WR breakout, TE elite flag). Writes to `data/projections/2026_projections.parquet`.

### Stage 4 — Rank (`rank.py`)

Computes VORP (Value Over Replacement) from roster-driven replacement levels. Assigns tiers based on VORP gaps. Generates snake draft strategy when `--pick N` is given. Writes to `output/rankings.csv`, `output/tiers.csv`, `output/strategy.csv`.

## Project Structure

| File | Purpose |
|---|---|
| `run_pipeline.py` | Main pipeline runner (stages 1-4) |
| `config.py` | Configuration loader from env vars / CLI args |
| `fetch.py` | Stage 1: data ingestion from nflreadpy |
| `score.py` | Stage 2: scoring engine wrapper |
| `scoring_engine.py` | Dynamic stat-to-points rule evaluator |
| `project.py` | Stage 3: 3-year weighted projections + rookie model |
| `rookies.py` | Rookie comparable-player projection model |
| `rank.py` | Stage 4: VORP rankings, tiers, draft strategy |
| `roster.py` | Roster file parser |
| `vorp.py` | VORP calculations, tier assignment |
| `scoring.txt` | League scoring rules (pipe-delimited) |
| `roster.txt` | League roster construction spec |
| `requirements.txt` | Python dependencies |
| `tests/` | pytest test suite |

## Data Files

### `scoring.txt`

Pipe-delimited fantasy scoring rules. Format:

```
stat_key | points | unit | description
```

**Units** (actual, from file):

| Unit | Meaning |
|---|---|
| `per_yd` | Per yard gained |
| `per_td` | Per touchdown scored |
| `per_rec` | Per reception |
| `per_cmp` | Per completion |
| `per_att` | Per attempt |
| `per_1st` | Per first down |
| `per_int` | Per interception |
| `per_sack` | Per sack |
| `per_fum` | Per fumble (lost/recovered) |
| `per_ff` | Per forced fumble |
| `per_tfl` | Per tackle for loss |
| `per_safety` | Per safety |
| `per_block` | Per blocked kick |
| `per_punt` | Per forced punt |
| `per_3nout` | Per 3-and-out forced |
| `per_stop` | Per 4th down stop |
| `per_pat` | Per PAT made |
| `per_miss` | Per kick missed |
| `per_2pt` | Per 2-point conversion |
| `game` | Once per game (flat bonus/penalty) |

**Threshold-based stats** (e.g. `def_pa_7_13`, `fg_miss_0_19`, `rec_40`) encode range in the key name. Ranges matched by convention — no per-row range config. Unit is still `game` for these.

Parse with `csv` module (`delimiter='|'`). Skip `#` comments and blank lines.

### `outline.txt`

Original project spec — tech choice and objective summary.

## Notes

- **Polars over pandas** — polars is already installed and preferred for this project.
- **nflreadpy** — provides `nflreadpy.import_season()` and related functions. Check its API before designing data ingestion.
- When adding a database, document the schema and connection details here.
