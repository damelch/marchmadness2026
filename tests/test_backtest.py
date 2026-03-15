"""Tests for the backtesting framework."""

import numpy as np
import pandas as pd
import pytest


class TestBacktestResult:
    """Test BacktestResult dataclass properties."""

    def test_avg_days_survived(self):
        from models.backtest import BacktestResult

        r = BacktestResult(
            season=2023, strategy="top_seeds", n_entries=3,
            days_survived=[5, 3, 7], final_alive=1,
        )
        assert r.avg_days_survived == pytest.approx(5.0)

    def test_avg_days_survived_empty(self):
        from models.backtest import BacktestResult

        r = BacktestResult(
            season=2023, strategy="top_seeds", n_entries=0,
            days_survived=[], final_alive=0,
        )
        assert r.avg_days_survived == 0.0

    def test_max_days_survived(self):
        from models.backtest import BacktestResult

        r = BacktestResult(
            season=2023, strategy="top_seeds", n_entries=3,
            days_survived=[2, 9, 4], final_alive=1,
        )
        assert r.max_days_survived == 9

    def test_survival_rate(self):
        from models.backtest import BacktestResult

        # 9 total days in schedule; 2 out of 3 survived all 9
        r = BacktestResult(
            season=2023, strategy="optimizer", n_entries=3,
            days_survived=[9, 9, 3], final_alive=2,
        )
        assert r.survival_rate == pytest.approx(2 / 3)


class TestReconstructBracket:
    """Test bracket reconstruction from Kaggle seed data."""

    def _make_seeds(self) -> pd.DataFrame:
        rows = []
        team_id = 1000
        for region in ["W", "X", "Y", "Z"]:
            for seed in range(1, 17):
                rows.append({
                    "Season": 2023,
                    "Seed": f"{region}{seed:02d}",
                    "TeamID": team_id,
                })
                team_id += 1
        return pd.DataFrame(rows)

    def _make_teams(self) -> pd.DataFrame:
        return pd.DataFrame({
            "TeamID": list(range(1000, 1064)),
            "TeamName": [f"Team{i}" for i in range(1000, 1064)],
        })

    def test_basic_reconstruction(self):
        from models.backtest import reconstruct_bracket

        seeds_df = self._make_seeds()
        teams_df = self._make_teams()
        bracket = reconstruct_bracket(2023, seeds_df, teams_df)
        assert len(bracket.teams) == 64

    def test_missing_season(self):
        from models.backtest import reconstruct_bracket

        seeds_df = self._make_seeds()
        teams_df = self._make_teams()
        bracket = reconstruct_bracket(2020, seeds_df, teams_df)
        assert len(bracket.teams) == 0


class TestGetActualWinners:
    """Test extracting winners from tournament results."""

    def test_basic_winners(self):
        from models.backtest import get_actual_winners

        results_df = pd.DataFrame({
            "Season": [2023, 2023, 2023],
            "DayNum": [136, 136, 138],  # R64, R64, R32
            "WTeamID": [1001, 1002, 1001],
            "WScore": [80, 75, 70],
            "LTeamID": [1064, 1063, 1002],
            "LScore": [70, 65, 60],
        })
        winners = get_actual_winners(2023, results_df)
        assert 1 in winners  # Round 1
        assert 1001 in winners[1]
        assert 1002 in winners[1]
        assert 2 in winners  # Round 2
        assert 1001 in winners[2]

    def test_empty_season(self):
        from models.backtest import get_actual_winners

        results_df = pd.DataFrame({
            "Season": [2023],
            "DayNum": [136],
            "WTeamID": [1001],
            "WScore": [80],
            "LTeamID": [1064],
            "LScore": [70],
        })
        winners = get_actual_winners(2020, results_df)
        assert len(winners) == 0


class TestStrategies:
    """Test individual pick strategies."""

    def test_top_seeds_picks_best(self):
        from models.backtest import _strategy_top_seeds

        available = {101: 1, 102: 5, 103: 12, 104: 3}
        picks = _strategy_top_seeds(available, n_picks=2)
        assert len(picks) == 2
        assert picks[0] == 101  # 1 seed
        assert picks[1] == 104  # 3 seed

    def test_random_correct_count(self):
        from models.backtest import _strategy_random

        available = {101: 1, 102: 5, 103: 12}
        rng = np.random.default_rng(42)
        picks = _strategy_random(available, n_picks=2, rng=rng)
        assert len(picks) == 2
        assert all(p in available for p in picks)

    def test_random_no_duplicates(self):
        from models.backtest import _strategy_random

        available = {101: 1, 102: 5, 103: 12, 104: 3}
        rng = np.random.default_rng(42)
        picks = _strategy_random(available, n_picks=3, rng=rng)
        assert len(picks) == len(set(picks))

    def test_contrarian_picks_worst_seeds_without_ownership(self):
        from models.backtest import _strategy_contrarian

        available = {101: 1, 102: 5, 103: 12, 104: 3}
        picks = _strategy_contrarian(available, n_picks=2)
        # Without ownership, picks highest seeds (most contrarian)
        assert 103 in picks  # 12 seed

    def test_contrarian_picks_lowest_ownership(self):
        from models.backtest import _strategy_contrarian

        available = {101: 1, 102: 5, 103: 8}
        ownership = {101: 0.15, 102: 0.02, 103: 0.05}
        win_probs = {101: 0.95, 102: 0.60, 103: 0.55}
        picks = _strategy_contrarian(
            available, n_picks=2,
            ownership=ownership, win_probs=win_probs,
        )
        assert len(picks) == 2
        # 102 has lowest ownership and passes win prob threshold
        assert 102 in picks


class TestBacktestSchedule:
    """Test the hardcoded backtest schedule."""

    def test_schedule_has_9_days(self):
        from models.backtest import _BACKTEST_SCHEDULE

        assert len(_BACKTEST_SCHEDULE) == 9

    def test_schedule_covers_all_rounds(self):
        from models.backtest import _BACKTEST_SCHEDULE

        rounds = {d["round"] for d in _BACKTEST_SCHEDULE}
        assert rounds == {1, 2, 3, 4, 5, 6}

    def test_day_1_is_double_pick(self):
        from models.backtest import _BACKTEST_SCHEDULE

        day1 = _BACKTEST_SCHEDULE[0]
        assert day1["num_picks"] == 2
        assert day1["round"] == 1
