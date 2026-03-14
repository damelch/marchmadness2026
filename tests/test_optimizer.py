"""Tests for survivor pool optimizer."""

import pytest
from optimizer.ownership import estimate_ownership
from optimizer.survival import survival_probability, leverage_score, single_entry_ev
from optimizer.kelly import optimal_entries


class TestOwnership:
    def test_sums_to_one(self):
        teams = {1: 1, 2: 2, 3: 5, 4: 12}
        probs = {1: 0.95, 2: 0.85, 3: 0.65, 4: 0.65}
        own = estimate_ownership(teams, probs, round_num=1)
        assert abs(sum(own.values()) - 1.0) < 1e-10

    def test_favorites_get_more_ownership(self):
        teams = {1: 1, 2: 16}
        probs = {1: 0.99, 2: 0.01}
        own = estimate_ownership(teams, probs, round_num=1, pool_sophistication=0.0)
        assert own[1] > own[2]

    def test_sophistication_affects_concentration(self):
        teams = {1: 1, 2: 2, 3: 8, 4: 9}
        probs = {1: 0.95, 2: 0.90, 3: 0.51, 4: 0.49}

        casual = estimate_ownership(teams, probs, round_num=1, pool_sophistication=0.0)
        sharp = estimate_ownership(teams, probs, round_num=1, pool_sophistication=1.0)

        # Casual pool concentrates more on favorites
        assert casual[1] > sharp[1]
        # Sharp pool gives more to underdogs
        assert sharp[4] > casual[4]


class TestSurvival:
    def test_single_round(self):
        assert abs(survival_probability([0.9]) - 0.9) < 1e-10

    def test_multi_round(self):
        probs = [0.9, 0.8, 0.7]
        expected = 0.9 * 0.8 * 0.7
        assert abs(survival_probability(probs) - expected) < 1e-10

    def test_perfect_survival(self):
        assert survival_probability([1.0, 1.0, 1.0]) == 1.0

    def test_zero_survival(self):
        assert survival_probability([0.9, 0.0, 0.8]) == 0.0


class TestLeverage:
    def test_high_prob_low_ownership(self):
        """High win prob + low ownership = high leverage."""
        high_lev = leverage_score(0.9, 0.1, 0.5)
        low_lev = leverage_score(0.9, 0.9, 0.5)
        assert high_lev > low_lev

    def test_leverage_increases_with_win_prob(self):
        lev_high = leverage_score(0.9, 0.3, 0.5)
        lev_low = leverage_score(0.5, 0.3, 0.5)
        assert lev_high > lev_low


class TestKelly:
    def test_negative_ev(self):
        result = optimal_entries(50, 30, 1000)  # EV < cost
        assert result["n_entries"] == 0

    def test_positive_ev(self):
        result = optimal_entries(50, 150, 5000)
        assert result["n_entries"] > 0
        assert result["total_ev"] > result["total_cost"]

    def test_bankroll_constraint(self):
        result = optimal_entries(50, 150, 100, kelly_multiplier=1.0)
        assert result["total_cost"] <= 100

    def test_max_entries_constraint(self):
        result = optimal_entries(10, 100, 10000, max_entries=3)
        assert result["n_entries"] <= 3
