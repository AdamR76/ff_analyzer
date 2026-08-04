from roster import parse_roster


ROSTER_CSV = "QB,RB,RB,WR,TE,WRT,WRT,WRT,WRTQ,K,K,DEF,BN,BN,BN,BN,BN,BN,IR,IR"


def test_parse_roster_starters():
    result = parse_roster(ROSTER_CSV)

    assert result["starters"] == {
        "QB": 1,
        "RB": 2,
        "WR": 1,
        "TE": 1,
        "K": 2,
        "DEF": 1,
    }


def test_parse_roster_flex():
    result = parse_roster(ROSTER_CSV)

    assert len(result["flex"]) == 4
    # 3 WRT slots
    wrt_slots = [f for f in result["flex"] if f["type"] == "WRT"]
    assert len(wrt_slots) == 3
    for slot in wrt_slots:
        assert set(slot["eligible"]) == {"RB", "WR", "TE"}

    # 1 WRTQ (superflex)
    wrtq_slots = [f for f in result["flex"] if f["type"] == "WRTQ"]
    assert len(wrtq_slots) == 1
    assert set(wrtq_slots[0]["eligible"]) == {"QB", "RB", "WR", "TE"}


def test_parse_roster_bench_and_ir():
    result = parse_roster(ROSTER_CSV)

    assert result["bench"] == 6
    assert result["total_roster"] == 18  # excludes IR
    assert result["ir"] == 2


def test_parse_roster_from_file(tmp_path):
    roster_file = tmp_path / "roster.txt"
    roster_file.write_text(ROSTER_CSV)

    result = parse_roster(roster_file)
    assert result["starters"]["QB"] == 1
    assert result["total_roster"] == 18
