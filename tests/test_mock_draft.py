"""Tests for mock_draft.py — position-aware snake draft simulation."""
import polars as pl
from mock_draft import (
    simulate_mock_draft,
    _get_snake_order,
    _init_team_state,
    _needs_starter,
    _needs_flex,
    _pick_for_team,
)


def test_get_snake_order_round_1():
    """Round 1 (odd): forward 1..12."""
    assert _get_snake_order(1, 12) == list(range(1, 13))


def test_get_snake_order_round_2():
    """Round 2 (even): reverse 12..1."""
    assert _get_snake_order(2, 12) == list(range(12, 0, -1))


def test_get_snake_order_round_3():
    """Round 3 (odd): forward 1..12."""
    assert _get_snake_order(3, 12) == list(range(1, 13))


def test_init_team_state():
    """Team state initialized from roster with correct counts."""
    roster = {
        "starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1},
        "flex": [
            {"type": "WRT", "eligible": ["RB", "WR", "TE"]},
        ],
        "bench": 3,
        "total_roster": 9,
    }
    state = _init_team_state(roster)
    assert state["needed_starters"] == ["QB", "RB", "RB", "WR", "WR", "TE"]
    assert len(state["needed_flex"]) == 1
    assert state["needed_flex"][0]["type"] == "WRT"
    assert state["needed_bench"] == 3
    assert state["drafted"] == []


def test_needs_starter_empty():
    """Empty list when all starters filled."""
    state = {"needed_starters": [], "needed_flex": [], "needed_bench": 2}
    assert _needs_starter(state) == []


def test_needs_starter_some():
    """Returns remaining starters."""
    state = {"needed_starters": ["RB", "WR"], "needed_flex": [], "needed_bench": 2}
    assert _needs_starter(state) == ["RB", "WR"]


def test_needs_flex():
    """Returns unfilled flex slots."""
    state = {
        "needed_starters": [],
        "needed_flex": [
            {"type": "WRT", "eligible": ["RB", "WR", "TE"]},
        ],
        "needed_bench": 2,
    }
    result = _needs_flex(state)
    assert len(result) == 1
    assert result[0]["type"] == "WRT"


def test_needs_flex_empty():
    """Empty when flex slots all filled."""
    state = {"needed_starters": [], "needed_flex": [], "needed_bench": 2}
    assert _needs_flex(state) == []


def test_pick_for_team_fills_starter_first():
    """Team fills mandatory starter before bench."""
    roster = {
        "starters": {"QB": 1, "RB": 1},
        "flex": [],
        "bench": 1,
        "total_roster": 3,
    }
    state = _init_team_state(roster)

    available = pl.DataFrame({
        "player_id": ["q1", "r1", "r2"],
        "player_name": ["QB1", "RB1", "RB2"],
        "position": ["QB", "RB", "RB"],
        "vorp": [100.0, 80.0, 60.0],
        "tier": [1, 1, 2],
        "projected_points": [400.0, 320.0, 280.0],
        "overall_rank": [1, 2, 3],
    })

    new_state, remaining = _pick_for_team(state, available, roster)

    assert len(new_state["drafted"]) == 1
    assert new_state["drafted"][0]["player_name"] == "QB1"
    assert new_state["drafted"][0]["position"] == "QB"
    assert len(remaining) == 2


def test_pick_for_team_fills_bench_after_starters():
    """After starters filled, fills bench with BPA."""
    roster = {
        "starters": {"QB": 1},
        "flex": [],
        "bench": 1,
        "total_roster": 2,
    }
    state = _init_team_state(roster)

    available = pl.DataFrame({
        "player_id": ["q1", "r1"],
        "player_name": ["QB1", "RB1"],
        "position": ["QB", "RB"],
        "vorp": [100.0, 80.0],
        "tier": [1, 1],
        "projected_points": [400.0, 320.0],
        "overall_rank": [1, 2],
    })

    state, available = _pick_for_team(state, available, roster)
    assert state["drafted"][0]["player_name"] == "QB1"

    state, available = _pick_for_team(state, available, roster)
    assert state["drafted"][1]["player_name"] == "RB1"


def test_pick_for_team_skips_empty_position():
    """When position has no players, it's skipped."""
    roster = {
        "starters": {"QB": 1, "DEF": 1},
        "flex": [],
        "bench": 1,
        "total_roster": 3,
    }
    state = _init_team_state(roster)

    available = pl.DataFrame({
        "player_id": ["q1", "r1"],
        "player_name": ["QB1", "RB1"],
        "position": ["QB", "RB"],
        "vorp": [100.0, 80.0],
        "tier": [1, 1],
        "projected_points": [400.0, 320.0],
        "overall_rank": [1, 2],
    })

    state, available = _pick_for_team(state, available, roster)
    assert state["drafted"][0]["player_name"] == "QB1"

    state, available = _pick_for_team(state, available, roster)
    assert len(state["drafted"]) == 2


def test_simulate_mock_draft_row_count():
    """Produces num_teams * rounds rows."""
    ranked = pl.DataFrame({
        "player_id": [f"p{i}" for i in range(300)],
        "player_name": [f"Player {i}" for i in range(300)],
        "position": (["QB"] * 30 + ["RB"] * 60 + ["WR"] * 80 +
                     ["TE"] * 30 + ["K"] * 20 + ["DEF"] * 20 +
                     ["LB"] * 30 + ["DB"] * 30),
        "vorp": [float(300 - i) for i in range(300)],
        "tier": [1] * 50 + [2] * 50 + [3] * 200,
        "projected_points": [float(400 - i) for i in range(300)],
        "overall_rank": list(range(1, 301)),
    })

    roster = {
        "starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1, "LB": 1, "DB": 1},
        "flex": [
            {"type": "WRT", "eligible": ["RB", "WR", "TE"]},
            {"type": "WRT", "eligible": ["RB", "WR", "TE"]},
            {"type": "WRTQ", "eligible": ["QB", "RB", "WR", "TE"]},
        ],
        "bench": 7,
        "total_roster": 20,
    }

    result = simulate_mock_draft(ranked, roster, num_teams=12, rounds=20)
    assert len(result) == 240


def test_simulate_mock_draft_no_duplicates():
    """Each player drafted at most once."""
    ranked = pl.DataFrame({
        "player_id": [f"p{i}" for i in range(300)],
        "player_name": [f"Player {i}" for i in range(300)],
        "position": (["QB"] * 30 + ["RB"] * 60 + ["WR"] * 80 +
                     ["TE"] * 30 + ["K"] * 20 + ["DEF"] * 20 +
                     ["LB"] * 30 + ["DB"] * 30),
        "vorp": [float(300 - i) for i in range(300)],
        "tier": [1] * 50 + [2] * 50 + [3] * 200,
        "projected_points": [float(400 - i) for i in range(300)],
        "overall_rank": list(range(1, 301)),
    })

    roster = {
        "starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1, "LB": 1, "DB": 1},
        "flex": [
            {"type": "WRT", "eligible": ["RB", "WR", "TE"]},
            {"type": "WRT", "eligible": ["RB", "WR", "TE"]},
            {"type": "WRTQ", "eligible": ["QB", "RB", "WR", "TE"]},
        ],
        "bench": 7,
        "total_roster": 20,
    }

    result = simulate_mock_draft(ranked, roster, num_teams=12, rounds=20)
    ids = result["player_id"].to_list()
    assert len(ids) == len(set(ids))


def test_simulate_mock_draft_snake_order():
    """Pick 1 = team 1, pick 12 = team 12, pick 13 = team 12, pick 24 = team 1."""
    ranked = pl.DataFrame({
        "player_id": [f"p{i}" for i in range(300)],
        "player_name": [f"Player {i}" for i in range(300)],
        "position": (["QB"] * 30 + ["RB"] * 60 + ["WR"] * 80 +
                     ["TE"] * 30 + ["K"] * 20 + ["DEF"] * 20 +
                     ["LB"] * 30 + ["DB"] * 30),
        "vorp": [float(300 - i) for i in range(300)],
        "tier": [1] * 300,
        "projected_points": [float(400 - i) for i in range(300)],
        "overall_rank": list(range(1, 301)),
    })

    roster = {
        "starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1},
        "flex": [],
        "bench": 2,
        "total_roster": 8,
    }

    result = simulate_mock_draft(ranked, roster, num_teams=12, rounds=4)

    assert result["team_slot"][0] == 1    # overall_pick 1
    assert result["team_slot"][11] == 12  # overall_pick 12
    assert result["team_slot"][12] == 12  # overall_pick 13
    assert result["team_slot"][23] == 1   # overall_pick 24


def test_simulate_mock_draft_columns():
    """Output has expected columns."""
    ranked = pl.DataFrame({
        "player_id": ["p1", "p2"],
        "player_name": ["QB1", "RB1"],
        "position": ["QB", "RB"],
        "vorp": [100.0, 80.0],
        "tier": [1, 1],
        "projected_points": [400.0, 320.0],
        "overall_rank": [1, 2],
    })

    roster = {
        "starters": {"QB": 1},
        "flex": [],
        "bench": 1,
        "total_roster": 2,
    }

    result = simulate_mock_draft(ranked, roster, num_teams=12, rounds=1)
    expected_cols = [
        "overall_pick", "round", "team_slot", "player_name",
        "player_id", "position", "vorp", "tier", "projected_points",
    ]
    for col in expected_cols:
        assert col in result.columns
