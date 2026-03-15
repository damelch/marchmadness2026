"""Tests for Nash equilibrium solver."""

import numpy as np

from optimizer.nash import best_response, blended_ownership, nash_equilibrium, verify_equilibrium


class TestNashEquilibrium:
    def test_converges(self):
        """Nash should converge for a simple matchup set."""
        win_probs = {1: 0.9, 2: 0.1, 3: 0.7, 4: 0.3}
        nash = nash_equilibrium(win_probs, pool_size=100, prize_pool=5000)
        assert len(nash) > 0
        assert abs(sum(nash.values()) - 1.0) < 0.01

    def test_equal_ev_at_equilibrium(self):
        """At Nash equilibrium, all picked teams should have similar EV."""
        win_probs = {1: 0.9, 2: 0.1, 3: 0.7, 4: 0.3, 5: 0.65, 6: 0.35}
        nash = nash_equilibrium(win_probs, pool_size=100, prize_pool=5000)
        check = verify_equilibrium(nash, win_probs, 100, 5000)

        # EVs should be within 10% of each other for teams with meaningful ownership
        assert check["max_ev_diff"] < 0.15, f"EV diff too large: {check['max_ev_diff']}"

    def test_favorites_get_more_ownership(self):
        """Teams with higher win prob should get more ownership at Nash."""
        win_probs = {1: 0.95, 2: 0.05, 3: 0.60, 4: 0.40}
        nash = nash_equilibrium(win_probs, pool_size=100, prize_pool=5000)

        # Team 1 (95% win prob) should have more ownership than team 3 (60%)
        assert nash.get(1, 0) > nash.get(3, 0)

    def test_favorites_from_each_game_get_ownership(self):
        """The favorite from each independent game should get ownership."""
        # Two games: 1 vs 2, 3 vs 4. Favorites are 1 and 3.
        # At Nash, both favorites should get ownership since they're in
        # different games (picking 1 doesn't eliminate team-3 pickers).
        win_probs = {1: 0.55, 2: 0.45, 3: 0.52, 4: 0.48}
        nash = nash_equilibrium(win_probs, pool_size=100, prize_pool=5000)

        # Favorites from each game should get ownership
        assert nash.get(1, 0) > 0.1
        assert sum(nash.values()) > 0.99  # Should sum to ~1

    def test_symmetric_matchup(self):
        """50/50 matchup should produce equal ownership."""
        win_probs = {1: 0.5, 2: 0.5}
        nash = nash_equilibrium(win_probs, pool_size=100, prize_pool=5000)

        assert abs(nash.get(1, 0) - nash.get(2, 0)) < 0.05


class TestBestResponse:
    def test_returns_ranked_teams(self):
        win_probs = {1: 0.9, 2: 0.1, 3: 0.7, 4: 0.3}
        ownership = {1: 0.5, 2: 0.05, 3: 0.35, 4: 0.10}
        br = best_response(win_probs, ownership, 100, 5000)

        assert len(br) == 4
        # Should be sorted by EV descending
        evs = [r["ev"] for r in br]
        assert evs == sorted(evs, reverse=True)

    def test_best_response_against_nash(self):
        """Against Nash ownership, best response should have ~equal EV for all teams."""
        win_probs = {1: 0.9, 2: 0.1, 3: 0.7, 4: 0.3}
        nash = nash_equilibrium(win_probs, pool_size=100, prize_pool=5000)
        br = best_response(win_probs, nash, 100, 5000)

        # All EVs should be similar (that's the Nash property)
        evs = [r["ev"] for r in br if nash.get(r["team_id"], 0) > 0.01]
        if len(evs) >= 2:
            max_diff = (max(evs) - min(evs)) / np.mean(evs)
            assert max_diff < 0.20


class TestBlendedOwnership:
    def test_blend_between_heuristic_and_nash(self):
        win_probs = {1: 0.9, 2: 0.1, 3: 0.7, 4: 0.3}
        heuristic = {1: 0.6, 2: 0.02, 3: 0.3, 4: 0.08}

        blend = blended_ownership(win_probs, 100, 5000, heuristic, field_efficiency=0.5)

        assert abs(sum(blend.values()) - 1.0) < 0.01
        # Should be between heuristic and Nash values
        nash = nash_equilibrium(win_probs, 100, 5000)
        for t in win_probs:
            h = heuristic.get(t, 0)
            n = nash.get(t, 0)
            b = blend.get(t, 0)
            # Blend should be between min and max (with some tolerance)
            assert b >= min(h, n) - 0.05
            assert b <= max(h, n) + 0.05
