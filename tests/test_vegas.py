"""Tests for Vegas betting lines scraper and utilities."""

import pandas as pd
import pytest


class TestAmericanToImpliedProb:
    """Test American odds to implied probability conversion."""

    def test_negative_odds(self):
        from data.scrapers.vegas_lines import _american_to_implied_prob

        # -150 means bet 150 to win 100
        prob = _american_to_implied_prob(-150)
        assert prob == pytest.approx(0.6, abs=0.01)

    def test_positive_odds(self):
        from data.scrapers.vegas_lines import _american_to_implied_prob

        # +200 means bet 100 to win 200
        prob = _american_to_implied_prob(200)
        assert prob == pytest.approx(1 / 3, abs=0.01)

    def test_even_money(self):
        from data.scrapers.vegas_lines import _american_to_implied_prob

        prob = _american_to_implied_prob(100)
        assert prob == pytest.approx(0.5)

    def test_heavy_favorite(self):
        from data.scrapers.vegas_lines import _american_to_implied_prob

        prob = _american_to_implied_prob(-1000)
        assert prob > 0.9

    def test_zero_returns_zero(self):
        from data.scrapers.vegas_lines import _american_to_implied_prob

        assert _american_to_implied_prob(0) == 0.0

    def test_invalid_returns_zero(self):
        from data.scrapers.vegas_lines import _american_to_implied_prob

        assert _american_to_implied_prob("abc") == 0.0
        assert _american_to_implied_prob(None) == 0.0


class TestLoadHistoricalVegas:
    """Test loading historical Vegas lines from CSV."""

    def test_missing_file_returns_empty(self, tmp_path):
        from data.scrapers.vegas_lines import load_historical_vegas

        df = load_historical_vegas(tmp_path / "nonexistent.csv")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_valid_csv(self, tmp_path):
        from data.scrapers.vegas_lines import load_historical_vegas

        csv_path = tmp_path / "vegas.csv"
        csv_path.write_text(
            "Season,WTeamID,LTeamID,Spread,OverUnder\n"
            "2023,1181,1242,-5.5,145.0\n"
            "2023,1242,1181,3.0,140.0\n"
        )
        df = load_historical_vegas(csv_path)
        assert len(df) == 2
        assert df["Spread"].dtype == float
        assert df["OverUnder"].dtype == float
        assert df["Season"].iloc[0] == 2023


class TestMergeVegasWithMatchups:
    """Test merging Vegas lines into matchup features."""

    def test_empty_vegas_adds_zero_columns(self):
        from data.scrapers.vegas_lines import merge_vegas_with_matchups

        matchups = pd.DataFrame({
            "Season": [2023, 2023],
            "TeamA": [1181, 1242],
            "TeamB": [1242, 1181],
            "SeedDiff": [-3, 3],
        })
        result = merge_vegas_with_matchups(matchups, pd.DataFrame())
        assert "VegasSpread" in result.columns
        assert "VegasOU" in result.columns
        assert (result["VegasSpread"] == 0.0).all()

    def test_matched_merge(self):
        from data.scrapers.vegas_lines import merge_vegas_with_matchups

        matchups = pd.DataFrame({
            "Season": [2023],
            "TeamA": [1181],
            "TeamB": [1242],
            "SeedDiff": [-3],
        })
        vegas = pd.DataFrame({
            "Season": [2023],
            "WTeamID": [1181],
            "LTeamID": [1242],
            "Spread": [-5.5],
            "OverUnder": [145.0],
        })
        result = merge_vegas_with_matchups(matchups, vegas)
        assert result["VegasSpread"].iloc[0] == pytest.approx(-5.5)
        assert result["VegasOU"].iloc[0] == pytest.approx(145.0)

    def test_reversed_ordering_flip(self):
        from data.scrapers.vegas_lines import merge_vegas_with_matchups

        matchups = pd.DataFrame({
            "Season": [2023],
            "TeamA": [1242],
            "TeamB": [1181],
            "SeedDiff": [3],
        })
        vegas = pd.DataFrame({
            "Season": [2023],
            "WTeamID": [1181],
            "LTeamID": [1242],
            "Spread": [-5.5],
            "OverUnder": [145.0],
        })
        result = merge_vegas_with_matchups(matchups, vegas)
        # Spread should be flipped since team ordering is reversed
        assert result["VegasSpread"].iloc[0] == pytest.approx(5.5)


class TestSafeFloat:
    """Test the _safe_float helper."""

    def test_valid_numbers(self):
        from data.scrapers.vegas_lines import _safe_float

        assert _safe_float(3.14) == pytest.approx(3.14)
        assert _safe_float("2.5") == pytest.approx(2.5)
        assert _safe_float(0) == 0.0

    def test_invalid_returns_default(self):
        from data.scrapers.vegas_lines import _safe_float

        assert _safe_float(None) == 0.0
        assert _safe_float("abc") == 0.0
        assert _safe_float("abc", default=-1.0) == -1.0


class TestFetchCurrentOdds:
    """Test live odds fetching (without actually hitting the API)."""

    def test_no_api_key_returns_empty(self, monkeypatch):
        from data.scrapers.vegas_lines import fetch_current_odds

        monkeypatch.delenv("ODDS_API_KEY", raising=False)
        df = fetch_current_odds(api_key=None)
        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestParseOddsEvents:
    """Test parsing of Odds API event data."""

    def test_parse_single_event(self):
        from data.scrapers.vegas_lines import _parse_odds_events

        events = [{
            "home_team": "Duke",
            "away_team": "Kansas",
            "bookmakers": [{
                "title": "DraftKings",
                "markets": [
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Duke", "point": -5.5},
                            {"name": "Kansas", "point": 5.5},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "point": 145.0},
                            {"name": "Under", "point": 145.0},
                        ],
                    },
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Duke", "price": -200},
                            {"name": "Kansas", "price": 170},
                        ],
                    },
                ],
            }],
        }]

        rows = _parse_odds_events(events)
        # 1 bookmaker row + 1 consensus row
        assert len(rows) == 2
        bm_row = rows[0]
        assert bm_row["HomeTeam"] == "Duke"
        assert bm_row["Spread"] == pytest.approx(-5.5)
        assert bm_row["Total"] == pytest.approx(145.0)
        assert bm_row["ImpliedProbHome"] > 0.5  # Duke is favored

    def test_empty_events(self):
        from data.scrapers.vegas_lines import _parse_odds_events

        assert _parse_odds_events([]) == []
