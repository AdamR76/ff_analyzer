from pathlib import Path
from config import load_config, parse_args


def test_load_config_defaults():
    """load_config with no args returns expected defaults."""
    cfg = load_config()

    assert cfg["num_teams"] == 12
    assert cfg["draft_rounds"] == 20
    assert cfg["draft_position"] == 5
    assert cfg["weight_current"] == 0.50
    assert cfg["weight_prev"] == 0.30
    assert cfg["weight_oldest"] == 0.20
    assert isinstance(cfg["scoring_file"], Path)
    assert isinstance(cfg["roster_file"], Path)
    assert isinstance(cfg["data_dir"], Path)
    assert isinstance(cfg["output_dir"], Path)


def test_parse_args_pick():
    result = parse_args(["--pick", "7"])
    assert result == {"pick": 7}


def test_parse_args_multiple():
    result = parse_args(["--pick", "3", "--scoring", "custom.txt"])
    assert result == {"pick": 3, "scoring_file": Path("custom.txt")}


def test_parse_args_empty():
    result = parse_args([])
    assert result == {}


def test_load_config_with_draft_position():
    cfg = load_config(draft_position=5)
    assert cfg["draft_position"] == 5


def test_load_config_env_override(monkeypatch):
    monkeypatch.setenv("FF_NUM_TEAMS", "10")
    monkeypatch.setenv("FF_WEIGHT_CURRENT", "0.60")
    cfg = load_config()
    assert cfg["num_teams"] == 10
    assert cfg["weight_current"] == 0.60


def test_parse_args_unknown_flag_warns():
    """Unrecognized flags produce a warning."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = parse_args(["--unknown", "value"])
        assert result == {}
        assert len(w) >= 1
        assert "Unrecognized argument" in str(w[0].message)
        assert "--unknown" in str(w[0].message)


def test_parse_args_unknown_flag_skip_value():
    """Unknown flag skips its value arg."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = parse_args(["--unknown", "something", "--pick", "5"])
        assert result == {"pick": 5}
        assert len(w) >= 1
