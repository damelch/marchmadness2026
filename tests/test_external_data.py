"""Tests for external data scrapers (Barttorvik, ESPN BPI)."""

import pandas as pd
import pytest


class TestBarttorvik:
    """Test Barttorvik data loading and conversion."""

    def _make_sample_barttorvik_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"Team": "Duke", "Conf": "ACC", "AdjOE": 120.5, "AdjDE": 90.2,
             "Barthag": 0.975, "AdjT": 68.5, "WAB": 10.2},
            {"Team": "Kansas", "Conf": "B12", "AdjOE": 118.0, "AdjDE": 92.1,
             "Barthag": 0.950, "AdjT": 67.0, "WAB": 8.5},
            {"Team": "Gonzaga", "Conf": "WCC", "AdjOE": 115.5, "AdjDE": 95.0,
             "Barthag": 0.890, "AdjT": 70.0, "WAB": 5.1},
        ])

    def test_barttorvik_to_team_stats(self):
        from data.scrapers.barttorvik import barttorvik_to_team_stats

        bt_df = self._make_sample_barttorvik_df()
        result = barttorvik_to_team_stats(bt_df)

        assert len(result) == 3
        assert "TeamName" in result.columns
        assert "Barthag" in result.columns
        assert "WAB" in result.columns

    def test_barttorvik_to_team_stats_with_id_map(self):
        from data.scrapers.barttorvik import barttorvik_to_team_stats

        bt_df = self._make_sample_barttorvik_df()
        id_map = {"Duke": 1181, "Kansas": 1242}
        result = barttorvik_to_team_stats(bt_df, team_id_map=id_map)

        assert len(result) == 2  # Gonzaga not in map
        assert set(result["TeamID"]) == {1181, 1242}
        duke = result[result["TeamID"] == 1181].iloc[0]
        assert duke["Barthag"] == pytest.approx(0.975)
        assert duke["WAB"] == pytest.approx(10.2)

    def test_barttorvik_values_reasonable(self):
        bt_df = self._make_sample_barttorvik_df()
        for _, row in bt_df.iterrows():
            assert 0 <= row["Barthag"] <= 1.0
            assert 80 <= row["AdjOE"] <= 130
            assert 80 <= row["AdjDE"] <= 130

    def test_load_barttorvik_missing_file(self):
        from data.scrapers.barttorvik import load_barttorvik

        with pytest.raises(FileNotFoundError):
            load_barttorvik("nonexistent_file.csv")


class TestESPNBPI:
    """Test ESPN BPI data loading and conversion."""

    def _make_sample_bpi_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"Team": "Duke", "ESPN_ID": 150, "BPI": 25.8, "BPIOff": 12.9,
             "BPIDef": 12.9, "BPIRank": 1, "SOR": 0.999, "QualityWins": 14},
            {"Team": "Michigan", "ESPN_ID": 130, "BPI": 24.7, "BPIOff": 12.1,
             "BPIDef": 12.6, "BPIRank": 2, "SOR": 0.998, "QualityWins": 16},
            {"Team": "Arizona", "ESPN_ID": 12, "BPI": 23.7, "BPIOff": 11.8,
             "BPIDef": 11.9, "BPIRank": 3, "SOR": 0.997, "QualityWins": 15},
        ])

    def test_bpi_to_team_stats(self):
        from data.scrapers.espn_bpi import bpi_to_team_stats

        bpi_df = self._make_sample_bpi_df()
        result = bpi_to_team_stats(bpi_df)

        assert len(result) == 3
        assert "BPI" in result.columns
        assert "BPIOff" in result.columns
        assert "BPIDef" in result.columns

    def test_bpi_to_team_stats_with_id_map(self):
        from data.scrapers.espn_bpi import bpi_to_team_stats

        bpi_df = self._make_sample_bpi_df()
        id_map = {"Duke": 1181, "Arizona": 1112}
        result = bpi_to_team_stats(bpi_df, team_id_map=id_map)

        assert len(result) == 2
        assert set(result["TeamID"]) == {1181, 1112}
        duke = result[result["TeamID"] == 1181].iloc[0]
        assert duke["BPI"] == pytest.approx(25.8)
        assert duke["BPIOff"] == pytest.approx(12.9)

    def test_bpi_values_reasonable(self):
        bpi_df = self._make_sample_bpi_df()
        for _, row in bpi_df.iterrows():
            assert -30 <= row["BPI"] <= 40
            assert -20 <= row["BPIOff"] <= 20
            assert -20 <= row["BPIDef"] <= 20
            assert 0 <= row["SOR"] <= 1.0


class TestFeatureIntegration:
    """Test that new features work in the model pipeline."""

    def test_feature_columns_include_barttorvik(self):
        from data.feature_engineering import FEATURE_COLUMNS

        assert "BarthagDiff" in FEATURE_COLUMNS
        assert "WABDiff" in FEATURE_COLUMNS

    def test_feature_columns_include_bpi(self):
        from data.feature_engineering import FEATURE_COLUMNS

        assert "BPIDiff" in FEATURE_COLUMNS
        assert "BPIOffDiff" in FEATURE_COLUMNS
        assert "BPIDefDiff" in FEATURE_COLUMNS

    def test_feature_columns_count(self):
        from data.feature_engineering import FEATURE_COLUMNS

        # 13 original + 5 new = 18
        assert len(FEATURE_COLUMNS) == 18
