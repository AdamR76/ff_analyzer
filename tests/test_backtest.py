"""Tests for backtest.py — rank correlation and projection validation."""

import polars as pl
import pytest
from backtest import (
    _compute_rank_correlation,
    _spearman_rho,
    _spearman_row,
)


# ═══════════════════════════════════════════════════════════════════════════
# Spearman rho unit tests
# ═══════════════════════════════════════════════════════════════════════════


def test_spearman_perfect_positive():
    """Identical rankings → ρ = 1.0."""
    x = [10.0, 20.0, 30.0, 40.0, 50.0]
    y = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _spearman_rho(x, y) == pytest.approx(1.0)


def test_spearman_perfect_negative():
    """Inverted rankings → ρ = -1.0."""
    x = [10.0, 20.0, 30.0, 40.0, 50.0]
    y = [50.0, 40.0, 30.0, 20.0, 10.0]
    assert _spearman_rho(x, y) == pytest.approx(-1.0)


def test_spearman_ties():
    """Tied values get average rank."""
    x = [10.0, 10.0, 30.0, 40.0, 50.0]
    y = [10.0, 10.0, 30.0, 40.0, 50.0]
    assert _spearman_rho(x, y) == pytest.approx(1.0)


def test_spearman_few_points():
    """Less than 2 points → None."""
    assert _spearman_rho([1.0], [1.0]) is None
    assert _spearman_rho([], []) is None


def test_spearman_no_variance():
    """All same values → None (zero variance)."""
    assert _spearman_rho([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]) is None


def test_spearman_known():
    """Hand-computed case: x=[1,2,3,4,5], y=[2,4,1,5,3] → ρ = 0.3."""
    # Ranks x: 1,2,3,4,5  Ranks y: 2,4,1,5,3
    # d^2 = 1+4+4+1+4 = 14, n=5, ρ = 1 - 6*14/(5*24) = 1 - 84/120 = 0.3
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 1.0, 5.0, 3.0]
    assert _spearman_rho(x, y) == pytest.approx(0.3)


# ═══════════════════════════════════════════════════════════════════════════
# _spearman_row tests
# ═══════════════════════════════════════════════════════════════════════════


def test_spearman_row():
    """Produces correct dict structure."""
    df = pl.DataFrame({
        "projected_points": [100.0, 200.0, 300.0, 400.0, 500.0],
        "actual_points": [100.0, 200.0, 300.0, 400.0, 500.0],
    })
    result = _spearman_row(df, "QB")
    assert result["position"] == "QB"
    assert result["n_players"] == 5
    assert result["spearman_r"] == pytest.approx(1.0)
    assert result["mean_abs_error"] == pytest.approx(0.0)


def test_spearman_row_small_sample():
    """Fewer than 3 players → None correlation."""
    df = pl.DataFrame({
        "projected_points": [100.0, 200.0],
        "actual_points": [100.0, 200.0],
    })
    result = _spearman_row(df, "QB")
    assert result["spearman_r"] is None


# ═══════════════════════════════════════════════════════════════════════════
# _compute_rank_correlation integration tests
# ═══════════════════════════════════════════════════════════════════════════


def make_projections(players: list[dict]) -> pl.DataFrame:
    """Helper: build projections DataFrame from list of player dicts."""
    rows = []
    for p in players:
        rows.append({
            "player_id": p["id"],
            "player_name": p.get("name", p["id"]),
            "position": p.get("position", "QB"),
            "projected_points": p["proj"],
            "projected_ppg": p["proj"] / 17,
            "ceiling": p["proj"] / 17 * 1.2,
            "floor": p["proj"] / 17 * 0.8,
            "games_played_projection": 17,
            "source": p.get("source", "historical"),
        })
    return pl.DataFrame(rows)


def make_actuals(players: list[dict]) -> pl.DataFrame:
    """Helper: build actual scores DataFrame from list of player dicts."""
    rows = []
    for p in players:
        rows.append({
            "player_id": p["id"],
            "player_name": p.get("name", p["id"]),
            "position": p.get("position", "QB"),
            "fantasy_points": p["actual"],
        })
    return pl.DataFrame(rows)


def test_correlation_perfect():
    """Identical projected and actual → ρ = 1.0 overall."""
    proj = make_projections([
        {"id": "a", "proj": 500}, {"id": "b", "proj": 400},
        {"id": "c", "proj": 300}, {"id": "d", "proj": 200},
        {"id": "e", "proj": 100},
    ])
    actual = make_actuals([
        {"id": "a", "actual": 500}, {"id": "b", "actual": 400},
        {"id": "c", "actual": 300}, {"id": "d", "actual": 200},
        {"id": "e", "actual": 100},
    ])
    results = _compute_rank_correlation(proj, actual)
    overall = results.filter(pl.col("position") == "OVERALL")
    assert overall["spearman_r"][0] == pytest.approx(1.0)
    assert overall["n_players"][0] == 5


def test_correlation_reversed():
    """Inverted rankings → ρ = -1.0."""
    proj = make_projections([
        {"id": "a", "proj": 100}, {"id": "b", "proj": 200},
        {"id": "c", "proj": 300}, {"id": "d", "proj": 400},
        {"id": "e", "proj": 500},
    ])
    actual = make_actuals([
        {"id": "a", "actual": 500}, {"id": "b", "actual": 400},
        {"id": "c", "actual": 300}, {"id": "d", "actual": 200},
        {"id": "e", "actual": 100},
    ])
    results = _compute_rank_correlation(proj, actual)
    overall = results.filter(pl.col("position") == "OVERALL")
    assert overall["spearman_r"][0] == pytest.approx(-1.0)


def test_join_excludes_rookies():
    """Players with source='rookie_model' excluded from correlation."""
    proj = make_projections([
        {"id": "a", "proj": 500, "source": "historical"},
        {"id": "b", "proj": 400, "source": "rookie_model"},
        {"id": "c", "proj": 300, "source": "historical"},
    ])
    actual = make_actuals([
        {"id": "a", "actual": 500},
        {"id": "b", "actual": 400},
        {"id": "c", "actual": 300},
    ])
    results = _compute_rank_correlation(proj, actual)
    overall = results.filter(pl.col("position") == "OVERALL")
    # Only 'a' and 'c' should be included (rookie 'b' excluded)
    assert overall["n_players"][0] == 2


def test_join_excludes_missing_actuals():
    """Players only in projections but not actuals excluded."""
    proj = make_projections([
        {"id": "a", "proj": 500},
        {"id": "b", "proj": 400},  # retired, not in actuals
        {"id": "c", "proj": 300},
    ])
    actual = make_actuals([
        {"id": "a", "actual": 500},
        {"id": "c", "actual": 300},
    ])
    results = _compute_rank_correlation(proj, actual)
    overall = results.filter(pl.col("position") == "OVERALL")
    assert overall["n_players"][0] == 2


def test_join_excludes_actuals_not_in_projections():
    """Players only in actuals but not projections excluded."""
    proj = make_projections([
        {"id": "a", "proj": 500},
    ])
    actual = make_actuals([
        {"id": "a", "actual": 500},
        {"id": "z", "actual": 100},  # waiver wire pickup, not projected
    ])
    results = _compute_rank_correlation(proj, actual)
    overall = results.filter(pl.col("position") == "OVERALL")
    assert overall["n_players"][0] == 1


def test_per_position_breakdown():
    """Each position gets its own correlation row."""
    proj = make_projections([
        {"id": "qb1", "proj": 400, "position": "QB"},
        {"id": "qb2", "proj": 350, "position": "QB"},
        {"id": "qb3", "proj": 300, "position": "QB"},
        {"id": "qb4", "proj": 250, "position": "QB"},
        {"id": "qb5", "proj": 200, "position": "QB"},
        {"id": "rb1", "proj": 300, "position": "RB"},
        {"id": "rb2", "proj": 280, "position": "RB"},
        {"id": "rb3", "proj": 260, "position": "RB"},
        {"id": "rb4", "proj": 240, "position": "RB"},
        {"id": "rb5", "proj": 220, "position": "RB"},
        {"id": "wr1", "proj": 250, "position": "WR"},
        {"id": "wr2", "proj": 230, "position": "WR"},
        {"id": "wr3", "proj": 210, "position": "WR"},
        {"id": "wr4", "proj": 190, "position": "WR"},
        {"id": "wr5", "proj": 170, "position": "WR"},
    ])
    actual = make_actuals([
        {"id": "qb1", "actual": 400, "position": "QB"},
        {"id": "qb2", "actual": 350, "position": "QB"},
        {"id": "qb3", "actual": 300, "position": "QB"},
        {"id": "qb4", "actual": 250, "position": "QB"},
        {"id": "qb5", "actual": 200, "position": "QB"},
        {"id": "rb1", "actual": 300, "position": "RB"},
        {"id": "rb2", "actual": 280, "position": "RB"},
        {"id": "rb3", "actual": 260, "position": "RB"},
        {"id": "rb4", "actual": 240, "position": "RB"},
        {"id": "rb5", "actual": 220, "position": "RB"},
        {"id": "wr1", "actual": 250, "position": "WR"},
        {"id": "wr2", "actual": 230, "position": "WR"},
        {"id": "wr3", "actual": 210, "position": "WR"},
        {"id": "wr4", "actual": 190, "position": "WR"},
        {"id": "wr5", "actual": 170, "position": "WR"},
    ])
    results = _compute_rank_correlation(proj, actual)

    overall = results.filter(pl.col("position") == "OVERALL")
    assert overall["spearman_r"][0] == pytest.approx(1.0)

    qb_row = results.filter(pl.col("position") == "QB")
    assert qb_row["spearman_r"][0] == pytest.approx(1.0)
    assert qb_row["n_players"][0] == 5

    rb_row = results.filter(pl.col("position") == "RB")
    assert rb_row["spearman_r"][0] == pytest.approx(1.0)

    wr_row = results.filter(pl.col("position") == "WR")
    assert wr_row["spearman_r"][0] == pytest.approx(1.0)


def test_small_position_filtered():
    """Positions with fewer than 5 players are excluded from per-position results."""
    proj = make_projections([
        {"id": "qb1", "proj": 400, "position": "QB"},
        {"id": "qb2", "proj": 350, "position": "QB"},
        {"id": "qb3", "proj": 300, "position": "QB"},
        {"id": "qb4", "proj": 250, "position": "QB"},
        {"id": "qb5", "proj": 200, "position": "QB"},
        {"id": "te1", "proj": 150, "position": "TE"},  # only 1 TE
    ])
    actual = make_actuals([
        {"id": "qb1", "actual": 400, "position": "QB"},
        {"id": "qb2", "actual": 350, "position": "QB"},
        {"id": "qb3", "actual": 300, "position": "QB"},
        {"id": "qb4", "actual": 250, "position": "QB"},
        {"id": "qb5", "actual": 200, "position": "QB"},
        {"id": "te1", "actual": 150, "position": "TE"},
    ])
    results = _compute_rank_correlation(proj, actual)

    # Overall includes all 6
    overall = results.filter(pl.col("position") == "OVERALL")
    assert overall["n_players"][0] == 6

    # QB has 5 players (>= 5 minimum)
    qb_row = results.filter(pl.col("position") == "QB")
    assert len(qb_row) == 1

    # TE has only 1 player (< 5) — excluded from per-position
    te_row = results.filter(pl.col("position") == "TE")
    assert len(te_row) == 0


def test_mean_absolute_error():
    """MAE correctly computed."""
    proj = make_projections([
        {"id": "a", "proj": 500}, {"id": "b", "proj": 400},
        {"id": "c", "proj": 300},
    ])
    actual = make_actuals([
        {"id": "a", "actual": 480}, {"id": "b", "actual": 420},
        {"id": "c", "actual": 310},
    ])
    results = _compute_rank_correlation(proj, actual)
    overall = results.filter(pl.col("position") == "OVERALL")
    # |500-480| + |400-420| + |300-310| = 20+20+10 = 50; 50/3 = 16.67
    assert overall["mean_abs_error"][0] == pytest.approx(16.67, abs=0.01)
