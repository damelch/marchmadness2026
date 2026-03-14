"""Tests for KenPom data integration."""

import pytest
import pandas as pd
from data.kenpom import (
    load_kenpom,
    kenpom_to_team_stats,
    kenpom_predict_matchup,
)


class TestLoadKenpom:
    def test_loads_csv(self):
        df = load_kenpom("data/kenpom_2026.csv")
        assert len(df) == 365
        assert "Team" in df.columns
        assert "NetRtg" in df.columns

    def test_numeric_columns(self):
        df = load_kenpom("data/kenpom_2026.csv")
        assert df["NetRtg"].dtype in (float, "float64")
        assert df["ORtg"].dtype in (float, "float64")
        assert df["DRtg"].dtype in (float, "float64")

    def test_duke_is_first(self):
        df = load_kenpom("data/kenpom_2026.csv")
        assert df.iloc[0]["Team"] == "Duke"
        assert df.iloc[0]["NetRtg"] > 35

    def test_win_pct_computed(self):
        df = load_kenpom("data/kenpom_2026.csv")
        assert "WinPct" in df.columns
        # Duke 31-2 = 0.939
        assert abs(df.iloc[0]["WinPct"] - 31 / 33) < 0.01


class TestKenpomToTeamStats:
    def test_without_id_map(self):
        """Without Kaggle mapping, uses rank as ID."""
        df = load_kenpom("data/kenpom_2026.csv")
        stats = kenpom_to_team_stats(df)
        assert len(stats) == 365
        assert "TeamID" in stats.columns
        assert "AdjO" in stats.columns
        assert "AdjD" in stats.columns
        assert "AdjEM" in stats.columns
        assert "SOS" in stats.columns

    def test_columns_match_predictor(self):
        """Output should have all columns the Predictor expects."""
        df = load_kenpom("data/kenpom_2026.csv")
        stats = kenpom_to_team_stats(df)
        required = ["TeamID", "AdjO", "AdjD", "AdjEM", "AdjT", "SOS", "WinPct"]
        for col in required:
            assert col in stats.columns, f"Missing column: {col}"


class TestKenpomPredict:
    def test_favorite_wins(self):
        """Duke (#1) should be heavily favored vs worst team."""
        df = load_kenpom("data/kenpom_2026.csv")
        p = kenpom_predict_matchup("Duke", df.iloc[-1]["Team"], df)
        assert p > 0.95

    def test_symmetry(self):
        """P(A beats B) + P(B beats A) = 1."""
        df = load_kenpom("data/kenpom_2026.csv")
        p1 = kenpom_predict_matchup("Duke", "Michigan", df)
        p2 = kenpom_predict_matchup("Michigan", "Duke", df)
        assert abs(p1 + p2 - 1.0) < 1e-10

    def test_close_matchup(self):
        """Top 2 teams should be close to 50/50."""
        df = load_kenpom("data/kenpom_2026.csv")
        p = kenpom_predict_matchup("Duke", "Michigan", df)
        assert 0.40 < p < 0.60

    def test_unknown_team(self):
        """Unknown team returns 0.5."""
        df = load_kenpom("data/kenpom_2026.csv")
        p = kenpom_predict_matchup("Nonexistent U", "Duke", df)
        assert p == 0.5
