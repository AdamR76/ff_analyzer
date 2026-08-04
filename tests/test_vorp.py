import polars as pl
from vorp import compute_vorp, compute_tiers, compute_all_vorp, _estimate_replacement_level


def test_estimate_replacement_level():
    """Replacement level: Nth-best drafted at each position."""
    # With 12 teams, superflex -> ~24 QBs drafted
    roster = {
        "starters": {"QB": 1, "RB": 2, "WR": 1, "TE": 1, "K": 2, "DEF": 1},
        "flex": [
            {"type": "WRT", "eligible": ["RB", "WR", "TE"]},
            {"type": "WRT", "eligible": ["RB", "WR", "TE"]},
            {"type": "WRT", "eligible": ["RB", "WR", "TE"]},
            {"type": "WRTQ", "eligible": ["QB", "RB", "WR", "TE"]},
        ],
        "bench": 6,
        "total_roster": 18,
    }

    levels = _estimate_replacement_level(roster, num_teams=12)

    # Superflex means ~24 QBs drafted -> replacement ~QB28
    assert levels["QB"] >= 20
    # 2 RB starters + flex competition -> ~36+ RBs drafted
    assert levels["RB"] >= 30
    # Deep replacement for WR and TE in this format
    assert levels["WR"] >= 36
    assert levels["TE"] >= 15


def test_compute_vorp():
    # 30 QBs — replacement_level=24 means 24th-best is the baseline
    qbs = []
    for i in range(30):
        rank = i + 1
        pts = 450 - rank * 10  # QB1=440, QB24=200, QB30=140
        qbs.append({
            "player_id": f"QB{rank}",
            "player_name": f"QB{rank}",
            "position": "QB",
            "projected_points": pts,
        })
    df = pl.DataFrame(qbs)

    result = compute_vorp(df, "QB", replacement_level=24)

    # VORP = projected - replacement (replacement = QB24 at ~200 pts)
    assert "vorp" in result.columns
    assert result.sort("vorp", descending=True)["player_id"][0] == "QB1"
    # QB30 has 140 pts, replacement at QB24 ~200 -> negative VORP
    vorp_qb30 = result.filter(pl.col("player_id") == "QB30")["vorp"][0]
    assert vorp_qb30 < 0
    # QB1 should have strongly positive VORP
    vorp_qb1 = result.filter(pl.col("player_id") == "QB1")["vorp"][0]
    assert vorp_qb1 > 0


def test_compute_tiers():
    df = pl.DataFrame({
        "player_id": ["A", "B", "C", "D", "E"],
        "position": ["QB", "QB", "QB", "QB", "QB"],
        "vorp": [120, 115, 80, 78, 50],
    })

    result = compute_tiers(df, threshold=15.0)

    assert "tier" in result.columns
    # A and B in same tier (gap = 5 < 15)
    assert result["tier"][0] == result["tier"][1]
    # B to C gap = 35 > 15 -> new tier
    assert result["tier"][1] != result["tier"][2]


def test_compute_vorp_replacement_beyond_data():
    """When replacement_level exceeds len(data), use last player's points."""
    df = pl.DataFrame({
        "player_id": ["X", "Y", "Z"],
        "player_name": ["QB1", "QB2", "QB3"],
        "position": ["QB", "QB", "QB"],
        "projected_points": [400, 380, 360],
    })

    result = compute_vorp(df, "QB", replacement_level=50)
    # Replacement should be last player (360), so vorp for QB1 = 400-360 = 40
    assert result["vorp"][0] == 40.0


def test_compute_vorp_empty_position():
    """Empty position returns empty DataFrame with vorp column."""
    df = pl.DataFrame({
        "player_id": ["A"],
        "player_name": ["QB1"],
        "position": ["QB"],
        "projected_points": [400],
    })

    result = compute_vorp(df, "RB", replacement_level=24)
    assert len(result) == 0
    assert "vorp" in result.columns


def test_compute_vorp_other_position_unchanged():
    """VORP only computed for the specified position."""
    df = pl.DataFrame({
        "player_id": ["A", "B"],
        "player_name": ["QB1", "RB1"],
        "position": ["QB", "RB"],
        "projected_points": [400, 300],
    })

    result = compute_vorp(df, "QB", replacement_level=24)
    # Only QB rows returned
    assert len(result) == 1
    assert result["position"][0] == "QB"


def test_compute_tiers_single_player():
    """Single player gets tier 1."""
    df = pl.DataFrame({
        "player_id": ["A"],
        "position": ["QB"],
        "vorp": [100],
    })

    result = compute_tiers(df, threshold=5.0)
    assert result["tier"][0] == 1


def test_compute_tiers_default_threshold():
    """Default threshold works."""
    df = pl.DataFrame({
        "player_id": ["A", "B"],
        "position": ["QB", "QB"],
        "vorp": [100, 97],
    })

    result = compute_tiers(df)
    # gap = 3 > 2.0 -> new tier
    assert result["tier"][0] == 1
    assert result["tier"][1] == 2


def test_compute_all_vorp():
    """Full pipeline: projections + roster -> VORP for all positions."""
    # Generate enough players per position so worst are below replacement.
    # 4-team league to keep replacement levels manageable.
    players = []
    for pos, count, start_pts in [("QB", 18, 400), ("RB", 28, 350)]:
        for i in range(count):
            rank = i + 1
            pts = start_pts - rank * 12
            players.append({
                "player_id": f"{pos}{rank}",
                "player_name": f"{pos}{rank}",
                "position": pos,
                "projected_points": pts,
            })
    df = pl.DataFrame(players)

    roster = {
        "starters": {"QB": 1, "RB": 2, "WR": 1, "TE": 1, "K": 2, "DEF": 1},
        "flex": [
            {"type": "WRT", "eligible": ["RB", "WR", "TE"]},
            {"type": "WRT", "eligible": ["RB", "WR", "TE"]},
            {"type": "WRT", "eligible": ["RB", "WR", "TE"]},
            {"type": "WRTQ", "eligible": ["QB", "RB", "WR", "TE"]},
        ],
        "bench": 6,
        "total_roster": 18,
    }

    result = compute_all_vorp(df, roster, num_teams=4)

    assert "vorp" in result.columns
    assert len(result) == len(df)
    # Top QB should have positive VORP
    assert result.filter(pl.col("player_id") == "QB1")["vorp"][0] > 0
    # Top RB should have positive VORP
    assert result.filter(pl.col("player_id") == "RB1")["vorp"][0] > 0
    # Worst RB should have negative VORP (below replacement)
    worst_rb_vorp = result.filter(pl.col("position") == "RB").sort("projected_points")["vorp"][0]
    assert worst_rb_vorp < 0
    # Worst QB should have negative VORP
    worst_qb_vorp = result.filter(pl.col("position") == "QB").sort("projected_points")["vorp"][0]
    assert worst_qb_vorp < 0


def test_estimate_replacement_level_with_parse_roster():
    """Integration: replacement levels from parsed roster match expectations."""
    from roster import parse_roster

    roster = parse_roster("QB,RB,RB,WR,TE,WRT,WRT,WRT,WRTQ,K,K,DEF,BN,BN,BN,BN,BN,BN,IR,IR")
    levels = _estimate_replacement_level(roster, num_teams=12)

    # This is the actual league format
    assert levels["QB"] >= 20  # superflex drives QB demand
    assert levels["RB"] >= 30
    assert levels["WR"] >= 36
    assert levels["TE"] >= 15
    assert levels["K"] >= 24  # 2 starters * 12 teams + 4
    assert levels["DEF"] >= 16  # 1 starter * 12 teams + 4


def test_compute_tiers_per_position():
    """per_position=True computes tiers independently per position."""
    df = pl.DataFrame({
        "player_id": ["QB1", "QB2", "QB3", "RB1", "RB2", "RB3"],
        "player_name": ["QB1", "QB2", "QB3", "RB1", "RB2", "RB3"],
        "position": ["QB", "QB", "QB", "RB", "RB", "RB"],
        "vorp": [120, 80, 78, 100, 50, 20],
    })

    result = compute_tiers(df, threshold=15.0, per_position=True)

    assert "tier" in result.columns
    # QB: QB1-QB2 gap=40 > 15 → different tiers (two tiers: QB1, QB2+QB3)
    qb_tiers = result.filter(pl.col("position") == "QB").sort("vorp", descending=True)
    assert qb_tiers["tier"][0] == 1  # QB1 alone in tier 1
    assert qb_tiers["tier"][1] == 2  # QB2+QB3 gap=2 < 15 → same tier
    assert qb_tiers["tier"][2] == 2

    # RB: RB1-RB2 gap=50 > 15, RB2-RB3 gap=30 > 15 → all different
    rb_tiers = result.filter(pl.col("position") == "RB").sort("vorp", descending=True)
    assert rb_tiers["tier"][0] == 1
    assert rb_tiers["tier"][1] == 2
    assert rb_tiers["tier"][2] == 3


def test_compute_tiers_per_position_no_position_col():
    """per_position=True without position column falls back to global."""
    df = pl.DataFrame({
        "player_id": ["A", "B"],
        "vorp": [100, 95],
    })
    result = compute_tiers(df, threshold=15.0, per_position=True)
    assert "tier" in result.columns
    # gap=5 < 15 → same tier
    assert result["tier"][0] == result["tier"][1]
