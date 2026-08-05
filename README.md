# FF Analyzer

Fantasy football draft analyzer. Uses prior-year NFL statistics to project player performance, compute VORP rankings, and generate a snake draft strategy tailored to your league.

## Quick Start

```bash
# Python 3.12+ required
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run full pipeline (fetch → score → project → rank)
python run_pipeline.py --pick 5

# Run individual stages
python fetch.py
python score.py
python project.py
python rank.py --pick 5
```

First run downloads 3 seasons of NFL data via [nflreadpy](https://github.com/nflverse/nflreadpy) (~30 seconds). Subsequent runs use cached Parquet files.

## How It Works

```
nflreadpy  ──►  Fetch  ──►  PBP Aggregate  ──►  Score  ──►  Project  ──►  Rank  ──►  CSV output
  (NFL data)    raw.parquet   game-level stats    scored.parquet   projections      rankings/tiers/strategy
```

### Stage 1 — Fetch
Pulls player stats, team defense stats, and play-by-play data from nflreadpy for the last 3 seasons. Filters to regular season only. Writes `data/raw/{season}.parquet`.

### Stage 1.5 — PBP Aggregate
Pre-aggregates play-by-play data into game-level stats (TD distance bonuses, pick-six penalties, defense situational stats, blocked kicks, special teams subs). Joins into raw parquets so the scoring engine can apply all rules. Module: `pbp_aggregate.py`.

### Stage 2 — Score
Applies league scoring rules from `scoring.txt` to every player, every game. Computes fantasy points per game for each player-season. Scoring engine is fully dynamic — no stat-to-points mapping is hardcoded. Writes `data/processed/{season}_scores.parquet`.

### Stage 3 — Project
Three-year weighted projection model (current: 50%, previous: 30%, oldest: 20%). Position-specific adjustments:
- **Age curves** — declining multipliers for aging players (RB: 27+, WR: 29+, TE: 30+, QB: 35+)
- **Injury model** — Bayesian games-played projection using career history (not hardcoded 17)
- **Partial-season shrinkage** — low-sample PPGs regressed toward positional mean
- **Trend adjustment** — ascending/descending PPG trajectory factored in (capped ±15%)
- **WR 3rd-year breakout** — bonus for year-3 WRs with ascending performance
- **TE elite flag** — premium for top-3 TEs
- **QB rushing baseline** — floor boost for rushing QBs

Rookies projected via comparable-player model using draft capital and combine data, with data-driven hit rates and round baselines computed from historical data. Writes `data/projections/2026_projections.parquet`.

### Stage 4 — Rank
Computes VORP (Value Over Replacement) from roster-driven replacement levels. Assigns tiers based on VORP gaps. Generates snake draft strategy showing best available players at each of your picks. Writes `output/rankings.csv`, `output/tiers.csv`, `output/strategy.csv`.

## Configuration

### Scoring rules (`scoring.txt`)

Pipe-delimited format. Fully customizable — no stat-to-points mapping is hardcoded.

```
stat_key | points | unit | description
```

| Unit | Meaning |
|------|---------|
| `per_yd` | Per yard gained |
| `per_td` | Per touchdown |
| `per_rec` | Per reception |
| `per_cmp` | Per completion |
| `per_att` | Per attempt |
| `per_int` | Per interception |
| `per_sack` | Per sack |
| `per_fum` | Per fumble |
| `per_ff` | Per forced fumble |
| `per_tfl` | Per tackle for loss |
| `per_safety` | Per safety |
| `per_block` | Per blocked kick |
| `per_3nout` | Per 3-and-out forced |
| `per_stop` | Per 4th down stop |
| `per_pat` | Per PAT made |
| `per_miss` | Per kick missed |
| `per_2pt` | Per 2-point conversion |
| `game` | Flat bonus/penalty once per game |

Threshold-based stats (e.g. `def_pa_7_13`, `fg_miss_0_19`, `rec_40`) encode ranges in the key name. TD distance bonuses (`pass_td_40`, `rush_td_50`, etc.) and defense situational stats (`def_3nout`, `def_4down_stop`) are derived from play-by-play data.

Lines starting with `#` are comments. Blank lines are ignored.

Use `--scoring` to point at a custom file:
```bash
python run_pipeline.py --scoring my_league_scoring.txt
```

### Roster (`roster.txt`)

Single comma-separated line of position slots:

```
QB,RB,RB,WR,TE,WRT,WRT,WRT,WRTQ,K,K,DEF,LB,DB,BN,BN,BN,BN,BN,BN,BN,IR,IR
```

Flex slots (`WRT` = RB/WR/TE, `WRTQ` = QB/RB/WR/TE superflex) and bench slots (`BN`) determine replacement levels for VORP calculations. Use `--roster` to point at a custom file:
```bash
python run_pipeline.py --roster my_roster.txt
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FF_NUM_TEAMS` | `12` | Number of teams in league |
| `FF_DRAFT_ROUNDS` | `20` | Total draft rounds |
| `FF_DRAFT_POSITION` | (none) | Your draft slot (1-12). Also settable via `--pick N` |
| `FF_WEIGHT_CURRENT` | `0.50` | Weight for most recent season |
| `FF_WEIGHT_PREV` | `0.30` | Weight for second-most recent season |
| `FF_WEIGHT_OLDEST` | `0.20` | Weight for oldest season |
| `FF_SCORING_FILE` | `scoring.txt` | Path to scoring rules file |
| `FF_ROSTER_FILE` | `roster.txt` | Path to roster config file |
| `FF_DATA_DIR` | `data` | Directory for raw/processed/projection data |
| `FF_OUTPUT_DIR` | `output` | Directory for CSV output |
| `FF_AGE_CURVE` | `true` | Enable age-based projection adjustments |
| `FF_WR_BREAKOUT` | `true` | Enable WR 3rd-year breakout bonus |
| `FF_TE_ELITE` | `true` | Enable TE elite tier premium |
| `FF_QB_RUSHING_BASELINE` | `true` | Enable QB rushing floor boost |
| `FF_INJURY_MODEL` | `true` | Enable data-driven games-played projection |
| `FF_TREND_ADJUST` | `true` | Enable PPG trend adjustment |
| `FF_SHRINKAGE` | `true` | Enable partial-season PPG shrinkage |

### CLI flags

| Flag | Description |
|------|-------------|
| `--pick N` | Your draft position (1-based) |
| `--scoring FILE` | Path to custom scoring rules file |
| `--roster FILE` | Path to custom roster config file |

## Output

All output lands in `output/`:

| File | Contents |
|------|----------|
| `rankings.csv` | Top 300 players ranked by VORP (overall rank, position rank, projected points, PPG, tier) |
| `tiers.csv` | Same data sorted by position → VORP for tier-based drafting |
| `strategy.csv` | Per-round snake draft guide — best available QB/RB/WR/TE/K/DEF/LB/DB at each of your picks |

Strategy file is only generated when `--pick N` or `FF_DRAFT_POSITION` is set.

## Development

```bash
# Run all tests
python -m pytest tests/ -v

# Run single test file
python -m pytest tests/test_scoring_engine.py -v
```

### Project structure

```
ff_analyzer/
├── run_pipeline.py      # Main entry point (stages 1-4)
├── config.py            # Env var + CLI arg loader
├── fetch.py             # Stage 1: nflreadpy → Parquet
├── pbp_aggregate.py     # Stage 1.5: play-by-play → game-level stats
├── score.py             # Stage 2: scoring engine wrapper
├── scoring_engine.py    # Dynamic rule parser + evaluator
├── project.py           # Stage 3: 3-year weighted projections
├── rookies.py           # Rookie comparable-player model
├── rank.py              # Stage 4: VORP, tiers, draft strategy
├── roster.py            # Roster file parser
├── vorp.py              # VORP calculation + tier assignment
├── scoring.txt          # League scoring rules
├── roster.txt           # League roster config
├── requirements.txt     # Python dependencies
├── data/                # Raw, processed, and projection Parquet files
│   ├── raw/
│   ├── processed/
│   └── projections/
├── output/              # CSV output (rankings, tiers, strategy)
└── tests/               # pytest test suite
```
