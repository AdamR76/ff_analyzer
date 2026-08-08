"""Tests for team_context.py — team-level adjustment factors."""

import polars as pl
import pytest
from team_context import (
    compute_team_factors,
    apply_team_context,
    compute_position_share_factor,
)


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests: compute_position_share_factor
# ═══════════════════════════════════════════════════════════════════════════


def test_pass_share_qb_gets_pass_factor():
    """QB gets pass share factor as-is."""
    assert compute_position_share_factor("QB", 1.05) == 1.05


def test_pass_share_wr_gets_pass_factor():
    """WR gets pass share factor as-is."""
    assert compute_position_share_factor("WR", 0.95) == 0.95


def test_pass_share_te_gets_pass_factor():
    """TE gets pass share factor as-is."""
    assert compute_position_share_factor("TE", 1.10) == 1.10


def test_pass_share_rb_inverted():
    """RB gets inverted factor: high pass rate = low RB value."""
    # pass_share=1.10 → run_share = 2.0 - 1.10 = 0.90
    result = compute_position_share_factor("RB", 1.10)
    assert result == pytest.approx(0.90)


def test_pass_share_rb_inverted_low_pass():
    """RB gets boost when pass rate is low (run-heavy team)."""
    # pass_share=0.90 → run_share = 2.0 - 0.90 = 1.10
    result = compute_position_share_factor("RB", 0.90)
    assert result == pytest.approx(1.10)


def test_pass_share_k_no_adjustment():
    """K gets no pass share adjustment."""
    assert compute_position_share_factor("K", 1.05) == 1.0


def test_pass_share_def_no_adjustment():
    """DEF gets no pass share adjustment."""
    assert compute_position_share_factor("DEF", 0.90) == 1.0


def test_pass_share_lb_no_adjustment():
    """LB gets no pass share adjustment."""
    assert compute_position_share_factor("LB", 1.10) == 1.0


def test_pass_share_db_no_adjustment():
    """DB gets no pass share adjustment."""
    assert compute_position_share_factor("DB", 0.95) == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests: apply_team_context
# ═══════════════════════════════════════════════════════════════════════════


def make_proj_df(players: list[dict]) -> pl.DataFrame:
    """Helper: build projection DataFrame."""
    rows = []
    for p in players:
        rows.append({
            "player_id": p["id"],
            "player_name": p.get("name", p["id"]),
            "position": p["position"],
            "team": p["team"],
            "projected_ppg": p["ppg"],
            "projected_points": p["ppg"] * 17,
            "ceiling": p["ppg"] * 1.2,
            "floor": p["ppg"] * 0.8,
            "games_played_projection": 17,
        })
    return pl.DataFrame(rows)


def test_apply_boosts_qb_on_pass_heavy_team():
    """QB on pass-heavy team gets >1.0 multiplier."""
    df = make_proj_df([
        {"id": "a", "position": "QB", "team": "KC", "ppg": 20.0},
    ])
    factors = {"KC": {"volume": 1.03, "pass_share": 1.08, "run_share": 0.92, "env": 1.02}}
    result = apply_team_context(df, factors)
    new_ppg = result["projected_ppg"][0]
    assert new_ppg > 20.0


def test_apply_penalizes_rb_on_pass_heavy_team():
    """RB on pass-heavy team gets <1.0 multiplier (run_share < 1.0)."""
    df = make_proj_df([
        {"id": "a", "position": "RB", "team": "KC", "ppg": 15.0},
    ])
    factors = {"KC": {"volume": 1.03, "pass_share": 1.08, "run_share": 0.92, "env": 1.02}}
    result = apply_team_context(df, factors)
    new_ppg = result["projected_ppg"][0]
    # RB gets run_share (0.92) — should decrease
    assert new_ppg < 15.0


def test_apply_boosts_rb_on_run_heavy_team():
    """RB on run-heavy team gets >1.0 multiplier."""
    df = make_proj_df([
        {"id": "a", "position": "RB", "team": "BAL", "ppg": 15.0},
    ])
    factors = {"BAL": {"volume": 1.01, "pass_share": 0.92, "run_share": 1.08, "env": 1.01}}
    result = apply_team_context(df, factors)
    new_ppg = result["projected_ppg"][0]
    # RB gets run_share (1.08) — should increase
    assert new_ppg > 15.0


def test_apply_no_adjustment_for_k():
    """K gets volume and env but no pass share adjustment (share=1.0)."""
    df = make_proj_df([
        {"id": "a", "position": "K", "team": "KC", "ppg": 8.0},
    ])
    factors = {"KC": {"volume": 1.05, "pass_share": 1.10, "run_share": 0.90, "env": 1.03}}
    result = apply_team_context(df, factors)
    new_ppg = result["projected_ppg"][0]
    # K: volume(1.05) × share(1.0) × env(1.03) = 1.0815
    expected = 8.0 * 1.05 * 1.0 * 1.03
    assert new_ppg == pytest.approx(expected)


def test_apply_missing_team_no_adjustment():
    """Player on team not in factors gets no adjustment (1.0)."""
    df = make_proj_df([
        {"id": "a", "position": "QB", "team": "UNK", "ppg": 20.0},
    ])
    factors = {"KC": {"volume": 1.05, "pass_share": 1.05, "run_share": 0.95, "env": 1.05}}
    result = apply_team_context(df, factors)
    assert result["projected_ppg"][0] == pytest.approx(20.0)


def test_apply_combined_capped_at_085():
    """Combined multiplier cannot go below 0.85."""
    df = make_proj_df([
        {"id": "a", "position": "RB", "team": "BAD", "ppg": 15.0},
    ])
    # All factors at minimum → combined should floor at 0.85
    factors = {"BAD": {"volume": 0.92, "pass_share": 0.90, "run_share": 0.90, "env": 0.90}}
    result = apply_team_context(df, factors)
    ratio = result["projected_ppg"][0] / 15.0
    assert ratio == pytest.approx(0.85)


def test_apply_combined_capped_at_115():
    """Combined multiplier cannot exceed 1.15."""
    df = make_proj_df([
        {"id": "a", "position": "WR", "team": "ELITE", "ppg": 15.0},
    ])
    # All factors at maximum → combined should cap at 1.15
    factors = {"ELITE": {"volume": 1.08, "pass_share": 1.10, "run_share": 0.90, "env": 1.10}}
    result = apply_team_context(df, factors)
    ratio = result["projected_ppg"][0] / 15.0
    assert ratio == pytest.approx(1.15)


def test_apply_no_team_column():
    """DataFrame without team column returned unchanged."""
    df = pl.DataFrame({
        "player_id": ["a"],
        "position": ["QB"],
        "projected_ppg": [20.0],
        "projected_points": [340.0],
        "ceiling": [24.0],
        "floor": [16.0],
        "games_played_projection": [17],
    })
    factors = {"KC": {"volume": 1.05, "pass_share": 1.05, "run_share": 0.95, "env": 1.05}}
    result = apply_team_context(df, factors)
    assert result["projected_ppg"][0] == pytest.approx(20.0)


def test_apply_empty_factors():
    """Empty factors dict returns DataFrame unchanged."""
    df = make_proj_df([
        {"id": "a", "position": "QB", "team": "KC", "ppg": 20.0},
    ])
    result = apply_team_context(df, {})
    assert result["projected_ppg"][0] == pytest.approx(20.0)


def test_apply_recomputes_projected_points():
    """After PPG adjustment, projected_points = ppg × games_played."""
    df = make_proj_df([
        {"id": "a", "position": "QB", "team": "KC", "ppg": 20.0},
    ])
    factors = {"KC": {"volume": 1.05, "pass_share": 1.05, "run_share": 0.95, "env": 1.02}}
    result = apply_team_context(df, factors)
    expected_points = result["projected_ppg"][0] * result["games_played_projection"][0]
    assert result["projected_points"][0] == pytest.approx(expected_points)


def test_apply_multiple_positions():
    """Different positions on same team get different share factors."""
    df = make_proj_df([
        {"id": "a", "position": "QB", "team": "KC", "ppg": 20.0},
        {"id": "b", "position": "RB", "team": "KC", "ppg": 15.0},
        {"id": "c", "position": "WR", "team": "KC", "ppg": 12.0},
        {"id": "d", "position": "K", "team": "KC", "ppg": 8.0},
    ])
    factors = {"KC": {"volume": 1.05, "pass_share": 1.08, "run_share": 0.92, "env": 1.02}}

    result = apply_team_context(df, factors)

    # QB and WR should increase (pass_share > 1.0)
    qb_ratio = result.filter(pl.col("player_id") == "a")["projected_ppg"][0] / 20.0
    wr_ratio = result.filter(pl.col("player_id") == "c")["projected_ppg"][0] / 12.0
    assert qb_ratio > 1.0
    assert wr_ratio > 1.0

    # RB should decrease (run_share = 0.92)
    rb_ratio = result.filter(pl.col("player_id") == "b")["projected_ppg"][0] / 15.0
    assert rb_ratio < 1.0

    # K: share = 1.0, only volume and env
    k_ratio = result.filter(pl.col("player_id") == "d")["projected_ppg"][0] / 8.0
    assert k_ratio == pytest.approx(1.05 * 1.0 * 1.02)
