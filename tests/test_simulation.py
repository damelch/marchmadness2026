"""Tests for the tournament simulation engine."""

import numpy as np

from simulation.engine import TournamentBracket, simulate_tournament, team_advancement_probs


def _make_test_bracket() -> TournamentBracket:
    """Create a test bracket with 64 teams."""
    bracket = TournamentBracket()
    team_id = 100
    for region in ["W", "X", "Y", "Z"]:
        for seed in range(1, 17):
            bracket.set_seed(team_id, seed, region, name=f"{region}{seed}")
            team_id += 1
    return bracket


class TestBracketStructure:
    def test_63_slots(self):
        bracket = TournamentBracket()
        assert len(bracket.slots) == 63

    def test_round_counts(self):
        bracket = TournamentBracket()
        for round_num, expected_count in [(1, 32), (2, 16), (3, 8), (4, 4), (5, 2), (6, 1)]:
            count = sum(1 for s in bracket.slots if s.round_num == round_num)
            assert count == expected_count, f"Round {round_num}: expected {expected_count}, got {count}"

    def test_64_teams_placed(self):
        bracket = _make_test_bracket()
        assert len(bracket.teams) == 64

    def test_round1_matchups_filled(self):
        bracket = _make_test_bracket()
        matchups = bracket.get_round_matchups(1)
        assert len(matchups) == 32
        for a, b, _ in matchups:
            assert a is not None
            assert b is not None

    def test_seed_matchup_order(self):
        """1 plays 16, 8 plays 9, etc. in round 1."""
        bracket = _make_test_bracket()
        matchups = bracket.get_round_matchups(1)

        # Collect seed matchups per region
        for a, b, _ in matchups[:8]:  # First region
            seed_a = bracket.teams[a]["seed"]
            seed_b = bracket.teams[b]["seed"]
            assert seed_a + seed_b == 17, f"Seeds {seed_a} vs {seed_b} don't sum to 17"


class TestSimulation:
    def test_basic_simulation(self):
        bracket = _make_test_bracket()
        from data.seed_history import get_seed_win_prob

        def predict(a, b):
            return get_seed_win_prob(bracket.teams[a]["seed"], bracket.teams[b]["seed"])

        results = simulate_tournament(bracket, predict, n_sims=100, rng_seed=42)
        assert results.shape == (100, 63)

    def test_deterministic(self):
        bracket = _make_test_bracket()
        from data.seed_history import get_seed_win_prob

        def predict(a, b):
            return get_seed_win_prob(bracket.teams[a]["seed"], bracket.teams[b]["seed"])

        r1 = simulate_tournament(bracket, predict, n_sims=50, rng_seed=42)
        r2 = simulate_tournament(bracket, predict, n_sims=50, rng_seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_advancement_probs_reasonable(self):
        """1-seeds should advance more often than 16-seeds."""
        bracket = _make_test_bracket()
        from data.seed_history import get_seed_win_prob

        def predict(a, b):
            return get_seed_win_prob(bracket.teams[a]["seed"], bracket.teams[b]["seed"])

        results = simulate_tournament(bracket, predict, n_sims=1000, rng_seed=42)
        probs = team_advancement_probs(results, bracket)

        # Find a 1-seed and 16-seed
        one_seed = probs[probs["Seed"] == 1].iloc[0]
        sixteen_seed = probs[probs["Seed"] == 16].iloc[0]

        assert one_seed["R32"] > sixteen_seed["R32"]
        assert one_seed["R32"] > 0.9  # 1-seeds should win R64 >90%

    def test_all_winners_are_valid_teams(self):
        bracket = _make_test_bracket()
        from data.seed_history import get_seed_win_prob

        def predict(a, b):
            return get_seed_win_prob(bracket.teams[a]["seed"], bracket.teams[b]["seed"])

        results = simulate_tournament(bracket, predict, n_sims=100, rng_seed=42)
        valid_teams = set(bracket.teams.keys()) | {0}
        for sim in range(100):
            for slot in range(63):
                assert results[sim, slot] in valid_teams
