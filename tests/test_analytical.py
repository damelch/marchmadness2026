"""Tests for analytical (closed-form) EV calculations."""

import pytest
from optimizer.analytical import (
    exact_pick_ev,
    exact_round_ev,
    exact_day_ev,
    field_survival_rate,
    optimal_multi_entry,
    optimal_day_picks,
    _find_opponent,
)


def _make_matchup():
    """Two-game round: team 1 vs 2 (1 favored), team 3 vs 4 (3 favored)."""
    win_probs = {1: 0.9, 2: 0.1, 3: 0.7, 4: 0.3}
    ownership = {1: 0.5, 2: 0.05, 3: 0.35, 4: 0.10}
    return win_probs, ownership


class TestFieldSurvival:
    def test_basic(self):
        wp, own = _make_matchup()
        fsr = field_survival_rate(wp, own)
        # = 0.5*0.9 + 0.05*0.1 + 0.35*0.7 + 0.10*0.3 = 0.45+0.005+0.245+0.03 = 0.73
        assert abs(fsr - 0.73) < 0.01


class TestExactPickEV:
    def test_positive_ev(self):
        wp, own = _make_matchup()
        ev = exact_pick_ev(1, wp, own, pool_size=100, prize_pool=5000)
        assert ev > 0

    def test_low_ownership_boosts_ev(self):
        """Lower ownership increases EV for same win probability."""
        # Teams 1 and 3 have same win prob but different ownership
        wp = {1: 0.8, 2: 0.2, 3: 0.8, 4: 0.2}
        own = {1: 0.7, 2: 0.05, 3: 0.2, 4: 0.05}
        ev_high_own = exact_pick_ev(1, wp, own, 100, 5000)
        ev_low_own = exact_pick_ev(3, wp, own, 100, 5000)
        # Team 3 has lower ownership, so when it wins, fewer opponents also survive
        assert ev_low_own > ev_high_own

    def test_zero_win_prob(self):
        wp = {1: 1.0, 2: 0.0}
        own = {1: 0.5, 2: 0.5}
        ev = exact_pick_ev(2, wp, own, 100, 5000)
        assert ev == 0.0

    def test_scales_with_prize(self):
        wp, own = _make_matchup()
        ev_small = exact_pick_ev(1, wp, own, 100, 1000)
        ev_big = exact_pick_ev(1, wp, own, 100, 10000)
        assert abs(ev_big / ev_small - 10.0) < 0.01


class TestExactRoundEV:
    def test_multi_entry(self):
        wp, own = _make_matchup()
        result = exact_round_ev([1, 3], wp, own, 100, 5000)
        assert result["total_ev"] > 0
        assert len(result["ev_per_entry"]) == 2
        assert result["joint_survival"] > 0

    def test_joint_survival_higher_than_individual(self):
        wp, own = _make_matchup()
        result = exact_round_ev([1, 3], wp, own, 100, 5000)
        # P(at least one survives) >= max(individual survival probs)
        assert result["joint_survival"] >= max(result["per_entry_survival"])


class TestOptimalMultiEntry:
    def test_picks_viable_teams(self):
        """Should pick teams with reasonable win probability."""
        wp = {1: 0.9, 2: 0.1, 3: 0.85, 4: 0.15, 5: 0.7, 6: 0.3}
        own = {1: 0.4, 2: 0.05, 3: 0.3, 4: 0.05, 5: 0.15, 6: 0.05}
        teams = {1: 1, 2: 16, 3: 2, 4: 15, 5: 4, 6: 13}

        picks = optimal_multi_entry(3, teams, wp, own, 100, 5000)
        assert len(picks) == 3
        # All picks should be viable (win prob >= 0.3)
        for p in picks:
            assert wp[p] >= 0.3

    def test_respects_used_teams(self):
        wp = {1: 0.9, 2: 0.1, 3: 0.7, 4: 0.3}
        own = {1: 0.5, 2: 0.1, 3: 0.3, 4: 0.1}
        teams = {1: 1, 2: 16, 3: 4, 4: 13}

        picks = optimal_multi_entry(2, teams, wp, own, 100, 5000,
                                     used_teams_per_entry=[{1}, set()])
        assert picks[0] != 1  # Entry 0 can't use team 1


class TestFindOpponent:
    def test_finds_correct_opponent(self):
        wp = {1: 0.9, 2: 0.1, 3: 0.7, 4: 0.3}
        assert _find_opponent(1, wp) == 2
        assert _find_opponent(3, wp) == 4


class TestExactDayEV:
    def test_single_pick_matches_exact_pick_ev(self):
        wp, own = _make_matchup()
        ev_day = exact_day_ev([1], wp, own, 100, 5000, num_picks_per_opponent=1)
        ev_pick = exact_pick_ev(1, wp, own, 100, 5000)
        assert abs(ev_day - ev_pick) < 0.01

    def test_double_pick_survival_lower(self):
        """Double-pick survival probability is lower (both must win)."""
        wp, own = _make_matchup()
        # P(survive single) = 0.9, P(survive double) = 0.9 * 0.7 = 0.63
        p_single = wp[1]
        p_double = wp[1] * wp[3]
        assert p_double < p_single

    def test_double_pick_positive(self):
        wp, own = _make_matchup()
        ev = exact_day_ev([1, 3], wp, own, 100, 5000, num_picks_per_opponent=2)
        assert ev > 0

    def test_double_pick_zero_if_any_zero(self):
        wp = {1: 0.9, 2: 0.1, 3: 0.0, 4: 1.0}
        own = {1: 0.5, 2: 0.1, 3: 0.3, 4: 0.1}
        ev = exact_day_ev([1, 3], wp, own, 100, 5000, num_picks_per_opponent=2)
        assert ev == 0.0


class TestOptimalDayPicks:
    def test_single_pick_day(self):
        wp = {1: 0.9, 2: 0.1, 3: 0.85, 4: 0.15, 5: 0.7, 6: 0.3}
        own = {1: 0.4, 2: 0.05, 3: 0.3, 4: 0.05, 5: 0.15, 6: 0.05}
        teams = {1: 1, 2: 16, 3: 2, 4: 15, 5: 4, 6: 13}

        picks = optimal_day_picks(2, teams, wp, own, 100, 5000, num_picks=1)
        assert len(picks) == 2
        assert all(len(ps) == 1 for ps in picks)

    def test_double_pick_day(self):
        wp = {1: 0.9, 2: 0.1, 3: 0.85, 4: 0.15, 5: 0.7, 6: 0.3, 7: 0.8, 8: 0.2}
        own = {t: 1/8 for t in range(1, 9)}
        teams = {t: t for t in range(1, 9)}
        matchup_pairs = [(1, 2), (3, 4), (5, 6), (7, 8)]

        picks = optimal_day_picks(
            2, teams, wp, own, 100, 5000,
            num_picks=2, matchup_pairs=matchup_pairs,
        )
        assert len(picks) == 2
        assert all(len(ps) == 2 for ps in picks)
        # No pair should contain opponents
        for ps in picks:
            assert (ps[0], ps[1]) not in matchup_pairs
            assert (ps[1], ps[0]) not in matchup_pairs

    def test_double_pick_viable_teams(self):
        """All picked teams should have decent win prob."""
        wp = {1: 0.9, 2: 0.1, 3: 0.85, 4: 0.15, 5: 0.7, 6: 0.3, 7: 0.8, 8: 0.2}
        own = {t: 1/8 for t in range(1, 9)}
        teams = {t: t for t in range(1, 9)}
        matchup_pairs = [(1, 2), (3, 4), (5, 6), (7, 8)]

        picks = optimal_day_picks(
            1, teams, wp, own, 100, 5000,
            num_picks=2, matchup_pairs=matchup_pairs,
        )
        for ps in picks:
            for t in ps:
                assert wp[t] >= 0.3
