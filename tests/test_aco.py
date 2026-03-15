"""Tests for Ant Colony Optimization portfolio optimizer."""

from optimizer.aco import (
    _compute_single_heuristic,
    aco_optimize,
)


def _make_matchup():
    """4-game round: 8 teams in 4 matchups."""
    win_probs = {
        1: 0.95, 2: 0.05,   # Game 1: 1-seed vs 16-seed
        3: 0.85, 4: 0.15,   # Game 2: 2-seed vs 15-seed
        5: 0.75, 6: 0.25,   # Game 3: 3-seed vs 14-seed
        7: 0.65, 8: 0.35,   # Game 4: 4-seed vs 13-seed
    }
    ownership = {
        1: 0.30, 2: 0.01, 3: 0.25, 4: 0.02,
        5: 0.20, 6: 0.03, 7: 0.15, 8: 0.04,
    }
    available_teams = {
        1: 1, 2: 16, 3: 2, 4: 15, 5: 3, 6: 14, 7: 4, 8: 13,
    }
    matchup_pairs = [(1, 2), (3, 4), (5, 6), (7, 8)]
    return win_probs, ownership, available_teams, matchup_pairs


class TestACOSinglePick:
    """Tests for single-pick day ACO."""

    def test_returns_correct_format(self):
        """ACO returns list[list[int]] with inner lists of length 1."""
        wp, own, avail, mp = _make_matchup()
        result = aco_optimize(
            n_entries=3, available_teams=avail, win_probs=wp,
            ownership=own, pool_size=1000, prize_pool=50000,
            num_picks=1, matchup_pairs=mp,
            n_ants=10, n_iterations=5,
        )
        assert len(result) == 3
        for pick_set in result:
            assert isinstance(pick_set, list)
            assert len(pick_set) == 1
            assert pick_set[0] in avail

    def test_respects_used_teams(self):
        """No entry picks a team from its used set."""
        wp, own, avail, mp = _make_matchup()
        used = [
            {1, 3},   # Entry 0 already used teams 1, 3
            {5},       # Entry 1 already used team 5
            set(),     # Entry 2 has no restrictions
        ]
        result = aco_optimize(
            n_entries=3, available_teams=avail, win_probs=wp,
            ownership=own, pool_size=1000, prize_pool=50000,
            num_picks=1, used_teams_per_entry=used,
            matchup_pairs=mp, n_ants=10, n_iterations=5,
        )
        assert result[0][0] not in {1, 3}
        assert result[1][0] != 5

    def test_portfolio_diversified(self):
        """With 4 entries and 4 games, ACO should diversify across games."""
        wp, own, avail, mp = _make_matchup()
        result = aco_optimize(
            n_entries=4, available_teams=avail, win_probs=wp,
            ownership=own, pool_size=1000, prize_pool=50000,
            num_picks=1, matchup_pairs=mp,
            n_ants=20, n_iterations=20,
        )
        picks = [ps[0] for ps in result]
        # Not all the same team
        assert len(set(picks)) > 1

    def test_single_entry_picks_high_ev(self):
        """With 1 entry, ACO should pick one of the highest-EV teams."""
        wp, own, avail, mp = _make_matchup()
        result = aco_optimize(
            n_entries=1, available_teams=avail, win_probs=wp,
            ownership=own, pool_size=1000, prize_pool=50000,
            num_picks=1, matchup_pairs=mp,
            n_ants=15, n_iterations=20,
        )
        chosen = result[0][0]
        # Should pick a high-win-prob team (not a 16-seed)
        assert wp[chosen] >= 0.5

    def test_deterministic_with_seed(self):
        """Same seed produces same picks."""
        wp, own, avail, mp = _make_matchup()
        kwargs = dict(
            n_entries=3, available_teams=avail, win_probs=wp,
            ownership=own, pool_size=1000, prize_pool=50000,
            num_picks=1, matchup_pairs=mp,
            n_ants=15, n_iterations=10, seed=123,
        )
        r1 = aco_optimize(**kwargs)
        r2 = aco_optimize(**kwargs)
        assert r1 == r2

    def test_beats_or_matches_greedy(self):
        """ACO score should be >= greedy score (since greedy is seeded)."""
        from optimizer.analytical import _portfolio_score, optimal_day_picks

        wp, own, avail, mp = _make_matchup()
        n_entries = 3

        greedy = optimal_day_picks(
            n_entries, avail, wp, own, 1000, 50000, 1, matchup_pairs=mp,
        )
        greedy_flat = [ps[0] for ps in greedy]
        greedy_score = _portfolio_score(greedy_flat, wp, own, 1000, 50000)

        aco_result = aco_optimize(
            n_entries=n_entries, available_teams=avail, win_probs=wp,
            ownership=own, pool_size=1000, prize_pool=50000,
            num_picks=1, matchup_pairs=mp,
            n_ants=20, n_iterations=40,
        )
        aco_flat = [ps[0] for ps in aco_result]
        aco_score = _portfolio_score(aco_flat, wp, own, 1000, 50000)

        # ACO should be at least as good (it starts with greedy solution)
        assert aco_score >= greedy_score - 0.01  # Small tolerance for float

    def test_handles_few_teams(self):
        """Works when available teams < n_entries."""
        wp = {1: 0.9, 2: 0.1}
        own = {1: 0.6, 2: 0.4}
        avail = {1: 1, 2: 16}
        mp = [(1, 2)]

        result = aco_optimize(
            n_entries=3, available_teams=avail, win_probs=wp,
            ownership=own, pool_size=100, prize_pool=5000,
            num_picks=1, matchup_pairs=mp,
            n_ants=10, n_iterations=5,
        )
        assert len(result) == 3
        # All picks should be valid team IDs
        for ps in result:
            assert ps[0] in avail


class TestACODoublePick:
    """Tests for double-pick day ACO."""

    def test_returns_correct_format(self):
        """ACO returns list[list[int]] with inner lists of length 2."""
        wp, own, avail, mp = _make_matchup()
        result = aco_optimize(
            n_entries=2, available_teams=avail, win_probs=wp,
            ownership=own, pool_size=1000, prize_pool=50000,
            num_picks=2, matchup_pairs=mp,
            n_ants=10, n_iterations=5,
        )
        assert len(result) == 2
        for pick_set in result:
            assert isinstance(pick_set, list)
            assert len(pick_set) == 2

    def test_no_opponent_pairs(self):
        """Double-pick never has both sides of same game."""
        wp, own, avail, mp = _make_matchup()
        opp_map = {}
        for a, b in mp:
            opp_map[a] = b
            opp_map[b] = a

        result = aco_optimize(
            n_entries=3, available_teams=avail, win_probs=wp,
            ownership=own, pool_size=1000, prize_pool=50000,
            num_picks=2, matchup_pairs=mp,
            n_ants=15, n_iterations=10,
        )
        for pick_set in result:
            a, b = pick_set
            assert opp_map.get(a) != b, f"Picked both sides: {a} vs {b}"


class TestHeuristic:
    """Tests for heuristic computation."""

    def test_heuristic_positive(self):
        """All heuristic values should be positive."""
        wp, own, avail, mp = _make_matchup()
        viable = list(avail.keys())
        h = _compute_single_heuristic(
            viable, avail, wp, own, 1000, 50000, 3, {},
        )
        for v in h.values():
            assert v > 0

    def test_heuristic_favors_high_ev(self):
        """Teams with higher win prob and lower ownership should have higher heuristic."""
        wp, own, avail, mp = _make_matchup()
        viable = list(avail.keys())
        h = _compute_single_heuristic(
            viable, avail, wp, own, 1000, 50000, 3, {},
        )
        # Team 1 (95% win, 30% owned) vs Team 2 (5% win, 1% owned)
        # Team 1 should have much higher heuristic despite higher ownership
        assert h[1] > h[2]
