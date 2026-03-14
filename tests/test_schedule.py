"""Tests for contest schedule module."""

import pytest
from contest.schedule import ContestDay, ContestSchedule
from simulation.engine import TournamentBracket


def _make_bracket():
    bracket = TournamentBracket()
    team_id = 100
    for region in ["W", "X", "Y", "Z"]:
        for seed in range(1, 17):
            bracket.set_seed(team_id, seed, region, name=f"{region}{seed}")
            team_id += 1
    return bracket


class TestContestDay:
    def test_single_pick_day(self):
        day = ContestDay(3, "R32 Saturday", "2026-03-21", 2, 1, ["W", "X"])
        assert not day.is_double_pick
        assert day.num_picks == 1

    def test_double_pick_day(self):
        day = ContestDay(1, "R64 Thursday", "2026-03-19", 1, 2, ["W", "X"])
        assert day.is_double_pick
        assert day.num_picks == 2


class TestContestSchedule:
    def test_default_schedule(self):
        sched = ContestSchedule.default()
        assert sched.total_days() == 9
        assert sum(d.num_picks for d in sched.days) == 12

    def test_from_config(self):
        config = {
            "contest": {
                "days": [
                    {"day": 1, "label": "R64 Thu", "round": 1, "picks": 2, "regions": ["W", "X"]},
                    {"day": 2, "label": "R64 Fri", "round": 1, "picks": 2, "regions": ["Y", "Z"]},
                    {"day": 3, "label": "R32 Sat", "round": 2, "picks": 1, "regions": ["W", "X"]},
                ]
            }
        }
        sched = ContestSchedule.from_config(config)
        assert sched.total_days() == 3
        assert sched.get_day(1).num_picks == 2

    def test_get_day_invalid(self):
        sched = ContestSchedule.default()
        with pytest.raises(ValueError):
            sched.get_day(99)

    def test_remaining_days(self):
        sched = ContestSchedule.default()
        remaining = sched.get_remaining_days(3)
        assert all(d.day_num > 3 for d in remaining)
        assert len(remaining) == 6  # days 4-9

    def test_total_picks_remaining(self):
        sched = ContestSchedule.default()
        total = sched.total_picks_remaining(1)
        assert total == 12  # all picks
        # From day 8 onward: 1 + 1 = 2
        assert sched.total_picks_remaining(8) == 2

    def test_double_pick_days(self):
        sched = ContestSchedule.default()
        double_days = [d for d in sched.days if d.is_double_pick]
        assert len(double_days) == 3  # Day 1, 2, 7

    def test_repr(self):
        sched = ContestSchedule.default()
        r = repr(sched)
        assert "9 days" in r
        assert "12 total picks" in r


class TestGameFiltering:
    def test_day1_gets_16_games(self):
        bracket = _make_bracket()
        sched = ContestSchedule.default()
        games = sched.get_games_for_day(1, bracket)
        assert len(games) == 16  # W + X regions, 8 games each

    def test_day2_gets_16_games(self):
        bracket = _make_bracket()
        sched = ContestSchedule.default()
        games = sched.get_games_for_day(2, bracket)
        assert len(games) == 16  # Y + Z regions

    def test_day1_and_day2_disjoint(self):
        bracket = _make_bracket()
        sched = ContestSchedule.default()
        games1 = sched.get_games_for_day(1, bracket)
        games2 = sched.get_games_for_day(2, bracket)

        teams1 = {a for a, b, _ in games1} | {b for a, b, _ in games1}
        teams2 = {a for a, b, _ in games2} | {b for a, b, _ in games2}
        assert teams1.isdisjoint(teams2)

    def test_day7_gets_4_games(self):
        """Elite 8 combined day has all 4 regions."""
        bracket = _make_bracket()
        sched = ContestSchedule.default()
        games = sched.get_games_for_day(7, bracket)
        assert len(games) == 4  # One E8 game per region

    def test_all_r64_games_covered(self):
        """Days 1+2 together cover all 32 R64 games."""
        bracket = _make_bracket()
        sched = ContestSchedule.default()
        games1 = sched.get_games_for_day(1, bracket)
        games2 = sched.get_games_for_day(2, bracket)
        all_r64 = bracket.get_round_matchups(1)
        assert len(games1) + len(games2) == len(all_r64)
