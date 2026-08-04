import polars as pl
from pathlib import Path
from fetch import fetch_raw_data, _filter_regular_season


def test_filter_regular_season():
    """_filter_regular_season keeps only REG weeks."""
    df = pl.DataFrame({
        "season_type": ["REG", "REG", "POST", "REG", "POST"],
        "season": [2024, 2024, 2024, 2024, 2024],
        "week": [1, 2, 18, 3, 19],
    })
    result = _filter_regular_season(df)
    assert result["season_type"].unique().to_list() == ["REG"]
    assert len(result) == 3


def test_fetch_raw_data_output_paths(tmp_path, monkeypatch):
    """fetch_raw_data creates parquet files per season."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)

    import fetch as f

    original = getattr(f, "_call_nflreadpy", None)

    def mock_nflreadpy(seasons):
        return pl.DataFrame({
            "player_id": ["00-001"],
            "player_name": ["Test Player"],
            "season": [2024],
            "week": [1],
            "season_type": ["REG"],
            "receptions": [5],
            "receiving_yards": [60],
        })

    f._call_nflreadpy = mock_nflreadpy
    try:
        cfg = {"data_dir": tmp_path / "data", "num_teams": 12}
        result = f.fetch_raw_data(cfg, seasons=[2024])
        assert 2024 in result["output_paths"]
        assert result["output_paths"][2024].exists()
        assert result["row_counts"][2024] > 0
    finally:
        if original:
            f._call_nflreadpy = original


def test_fetch_team_defense(tmp_path, monkeypatch):
    """fetch_team_defense creates _def.parquet files with position=DEF."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)

    import fetch as f

    original = getattr(f, "_call_nflreadpy_team", None)

    def mock_team_stats(seasons):
        return pl.DataFrame({
            "season": [2024, 2024],
            "week": [1, 1],
            "season_type": ["REG", "REG"],
            "team": ["KC", "SF"],
            "def_tds": [1, 0],
            "def_sacks": [3, 1],
            "def_interceptions": [2, 0],
        })

    f._call_nflreadpy_team = mock_team_stats
    try:
        cfg = {"data_dir": tmp_path / "data"}
        result = f.fetch_team_defense(cfg, seasons=[2024])

        assert 2024 in result["output_paths"]
        path = result["output_paths"][2024]
        assert path.exists()
        assert "_def" in path.name

        df = pl.read_parquet(path)
        assert "position" in df.columns
        assert (df["position"] == "DEF").all()
        assert "player_id" in df.columns
        assert df["player_id"][0] == "KC"
        assert df["player_name"][0] == "KC"
    finally:
        if original:
            f._call_nflreadpy_team = original
