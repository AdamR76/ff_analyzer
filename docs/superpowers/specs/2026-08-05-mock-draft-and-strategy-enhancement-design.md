# Mock Draft + Enhanced Strategy CSV

**Date**: 2026-08-05
**Status**: design-approved

## Context

User picks 5th in 12-team snake draft (20 rounds). Current `strategy.csv` shows only 1 best available player per position per round — not enough when that player gets taken before your pick. Need to see alternatives. Also want a full mock draft simulating all 12 teams to see how the draft might unfold.

## Changes

### 1. Enhanced Strategy CSV

**File**: `rank.py` → `_build_strategy()`

Same 19 columns. New `option` column (1, 2, 3). 3 rows per round = 60 rows total.

| Column | Change |
|--------|--------|
| `option` | **New** — 1=best available, 2=2nd best, 3=3rd best |
| All existing | Unchanged, but now populated from head(3) not head(1) |

When fewer than 3 players at a position, fall back to `""` / `0.0` for missing rows (existing pattern, unchanged).

### 2. Mock Draft CSV

**File**: `mock_draft.py` (new module)

`simulate_mock_draft(ranked, roster, num_teams, rounds)` → `pl.DataFrame`

240 rows (12 teams × 20 rounds). Columns:

| Column | Type | Description |
|--------|------|-------------|
| `overall_pick` | int | 1-240 |
| `round` | int | 1-20 |
| `team_slot` | int | 1-12 |
| `player_name` | str | |
| `player_id` | str | |
| `position` | str | QB/RB/WR/TE/K/LB/DB |
| `vorp` | f64 | |
| `tier` | int | |
| `projected_points` | f64 | |

**Draft logic — position-aware fill per team**:

Each of 12 teams maintains state: unfilled starters, flex slots, bench count, drafted players.

Priority cascade per pick:
1. **Mandatory starters** — among unfilled starter positions, pick position with highest best-available VORP (biggest value gap if you wait). Draft that player.
2. **WRT flex** — best available RB/WR/TE by VORP.
3. **WRTQ superflex** — best available QB/RB/WR/TE by VORP.
4. **Bench** — best available player overall by VORP.

Tiebreaker: higher absolute VORP, then alphabetical position name.

**Snake order**: odd rounds 1→12, even rounds 12→1.

**Private helpers in `mock_draft.py`**:
- `_get_snake_order(round_num, num_teams)` → `list[int]`
- `_init_team_state(roster)` → `dict`
- `_needs_starter(team_state)` → `list[str]`
- `_needs_flex(team_state)` → `list[dict]`
- `_pick_for_team(team_state, available, roster)` → `(dict, pl.DataFrame)`
- `_best_available_at_position(available, position)` → `float`

**Edge cases**:
- DEF has zero projected players → VORP gap = -inf, skipped
- Position with < 3 players in strategy → `""` / `0.0` fallback
- Available pool exhausted before all slots filled → draft ends early

### 3. Default Pick 5

**File**: `config.py` → `load_config()`

`draft_position` defaults to `5` instead of `None`. `--pick N` CLI flag still overrides. Strategy CSV and mock draft CSV always generated.

### 4. Pipeline Integration

**File**: `rank.py` → `run_rank_pipeline()`

Mock draft called unconditionally (no draft_position guard — simulates all 12 teams regardless of user slot).

Result dict gains `mock_draft` key.

**File**: `run_pipeline.py` → `main()`

Add print line for mock draft path.

## Test Plan

### `tests/test_mock_draft.py` (new)

- `test_get_snake_order` — round 1 forward, round 2 reverse
- `test_needs_starter` — returns unfilled positions correctly
- `test_pick_for_team_priority` — mandatory starters before flex before bench
- `test_pick_for_team_empty_position` — DEF with no players skipped gracefully
- `test_simulate_mock_draft_row_count` — 240 rows
- `test_simulate_mock_draft_no_duplicates` — each player_id appears once
- `test_simulate_mock_draft_snake_order` — pick 1 = team 1, pick 240 = team 1

### `tests/test_rank.py` (modify)

- `test_run_rank_pipeline` — assert `mock_draft` in result, file exists
- Assert strategy has `option` column, 54 rows (18 rounds × 3 in test config)
- Assert mock draft has correct columns and 216 rows (12 × 18)

## Verification

```bash
source venv/bin/activate
python -m pytest tests/ -v           # all tests pass
python run_pipeline.py               # generates with pick 5 default
python run_pipeline.py --pick 3      # overrides to pick 3
head -20 output/strategy.csv         # 3 rows per round with option col
head -30 output/mock_draft.csv       # full draft log, 240 rows
```

## Files Changed

| File | Change |
|------|--------|
| `mock_draft.py` | **New** — `simulate_mock_draft()` + 6 private helpers |
| `rank.py` | Modify `_build_strategy()` (head(3) + option col), add mock draft call |
| `config.py` | Default `draft_position` to 5 |
| `run_pipeline.py` | Print mock draft path |
| `tests/test_mock_draft.py` | **New** — 7 unit tests |
| `tests/test_rank.py` | Update assertions for 3-option strategy + mock draft |
