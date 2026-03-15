"""Tests for dynamic programming multi-round planner."""

import numpy as np

from data.seed_history import get_seed_win_prob
from optimizer.dp import (
    compute_advancement_probs,
    compute_future_values,
    compute_round_win_probs,
    team_scarcity,
)
from simulation.engine import TournamentBracket, simulate_tournament


def _make_bracket_and_sim():
    """Create test bracket with simulation results."""
    bracket = TournamentBracket()
    team_id = 100
    for region in ["W", "X", "Y", "Z"]:
        for seed in range(1, 17):
            bracket.set_seed(team_id, seed, region, name=f"{region}{seed}")
            team_id += 1

    def predict_fn(a, b):
        return get_seed_win_prob(
            bracket.teams[a]["seed"], bracket.teams[b]["seed"]
        )

    sim_results = simulate_tournament(bracket, predict_fn, n_sims=500, rng_seed=42)
    return bracket, predict_fn, sim_results


class TestRoundWinProbs:
    def test_round1_exact(self):
        bracket, predict_fn, sim_results = _make_bracket_and_sim()
        probs = compute_round_win_probs(bracket, predict_fn, sim_results)

        # Round 1 should have exact probabilities
        assert 1 in probs
        # 1-seed vs 16-seed should be ~0.99
        for team_id, info in bracket.teams.items():
            if info["seed"] == 1:
                assert probs[1][team_id] > 0.9
                break

    def test_later_rounds_decrease(self):
        bracket, predict_fn, sim_results = _make_bracket_and_sim()
        probs = compute_round_win_probs(bracket, predict_fn, sim_results)

        # A 1-seed's win prob should generally decrease in later rounds
        for team_id, info in bracket.teams.items():
            if info["seed"] == 1:
                r1 = probs.get(1, {}).get(team_id, 0)
                r3 = probs.get(3, {}).get(team_id, 0)
                # P(win in S16) < P(win in R64) because opponents are tougher
                assert r3 < r1
                break


class TestAdvancementProbs:
    def test_everyone_in_round1(self):
        bracket, predict_fn, sim_results = _make_bracket_and_sim()
        adv = compute_advancement_probs(sim_results, bracket)

        for team_id in bracket.teams:
            assert adv[team_id][1] == 1.0  # Everyone starts in round 1

    def test_1_seed_advances_more(self):
        bracket, predict_fn, sim_results = _make_bracket_and_sim()
        adv = compute_advancement_probs(sim_results, bracket)

        one_seed = None
        sixteen_seed = None
        for team_id, info in bracket.teams.items():
            if info["seed"] == 1 and one_seed is None:
                one_seed = team_id
            if info["seed"] == 16 and sixteen_seed is None:
                sixteen_seed = team_id

        # 1-seed should advance further more often
        assert adv[one_seed][2] > adv[sixteen_seed][2]


class TestTeamScarcity:
    def test_only_option_is_max_scarcity(self):
        available = {3: [99]}
        probs = {3: {99: 0.8}}
        scar = team_scarcity(99, 3, available, probs)
        assert scar == 1.0

    def test_many_options_low_scarcity(self):
        available = {1: [1, 2, 3, 4, 5, 6, 7, 8]}
        probs = {1: {t: 0.6 for t in range(1, 9)}}
        scar = team_scarcity(1, 1, available, probs)
        assert scar < 0.5


class TestFutureValues:
    def test_1_seeds_higher_future_value(self):
        """1-seeds should generally have higher future value than 16-seeds."""
        bracket, predict_fn, sim_results = _make_bracket_and_sim()
        probs = compute_round_win_probs(bracket, predict_fn, sim_results)
        adv = compute_advancement_probs(sim_results, bracket)

        fv = compute_future_values(bracket, probs, adv, current_day=1)

        one_seed_fv = []
        sixteen_seed_fv = []
        for team_id, info in bracket.teams.items():
            if info["seed"] == 1:
                one_seed_fv.append(fv.get(team_id, 0))
            if info["seed"] == 16:
                sixteen_seed_fv.append(fv.get(team_id, 0))

        assert np.mean(one_seed_fv) > np.mean(sixteen_seed_fv)

    def test_future_values_non_negative(self):
        bracket, predict_fn, sim_results = _make_bracket_and_sim()
        probs = compute_round_win_probs(bracket, predict_fn, sim_results)
        adv = compute_advancement_probs(sim_results, bracket)

        fv = compute_future_values(bracket, probs, adv, current_day=1)

        for v in fv.values():
            assert v >= 0

    def test_future_values_with_schedule(self):
        """Future values work with explicit ContestSchedule (9 days)."""
        from contest.schedule import ContestSchedule
        bracket, predict_fn, sim_results = _make_bracket_and_sim()
        probs = compute_round_win_probs(bracket, predict_fn, sim_results)
        adv = compute_advancement_probs(sim_results, bracket)
        schedule = ContestSchedule.default()

        fv = compute_future_values(bracket, probs, adv, current_day=1, schedule=schedule)

        # Should still have non-negative values
        for v in fv.values():
            assert v >= 0

        # 1-seeds should still be more valuable
        one_seed_fv = []
        sixteen_seed_fv = []
        for team_id, info in bracket.teams.items():
            if info["seed"] == 1:
                one_seed_fv.append(fv.get(team_id, 0))
            if info["seed"] == 16:
                sixteen_seed_fv.append(fv.get(team_id, 0))
        assert np.mean(one_seed_fv) > np.mean(sixteen_seed_fv)
