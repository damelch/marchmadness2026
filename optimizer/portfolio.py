"""Approach B: Correlation-aware portfolio optimization using Monte Carlo."""

import numpy as np
from itertools import product
from simulation.engine import TournamentBracket


def evaluate_portfolio(
    picks: list[int],
    sim_results: np.ndarray,
    bracket: TournamentBracket,
    round_num: int,
    ownership: dict[int, float],
    pool_size: int,
    prize_pool: float,
    rng: np.random.Generator | None = None,
    n_opponent_samples: int = 1000,
) -> dict:
    """Evaluate expected payout for a set of entry picks in a single round.

    Args:
        picks: List of team_ids (one per entry) for this round
        sim_results: (n_sims, 63) tournament sim results
        bracket: Tournament bracket
        round_num: Current round
        ownership: team_id -> ownership fraction
        pool_size: Total pool size
        prize_pool: Total prize
        rng: Random generator for opponent sampling
        n_opponent_samples: How many opponent scenarios to sample per sim

    Returns:
        Dict with ev_per_entry, total_ev, joint_survival
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_sims = sim_results.shape[0]
    n_entries = len(picks)
    n_opponents = pool_size - n_entries

    # Get round slots
    round_slots = [i for i, s in enumerate(bracket.slots) if s.round_num == round_num]

    # For each sim, determine which of our picks won
    entry_survived = np.zeros((n_sims, n_entries), dtype=bool)
    for sim in range(n_sims):
        winners = set(sim_results[sim, round_slots])
        for e_idx, team_id in enumerate(picks):
            entry_survived[sim, e_idx] = team_id in winners

    # Estimate opponent survival for each sim
    # P(opponent survives) = sum over teams: ownership[t] * I(t won in this sim)
    opp_survival_prob = np.zeros(n_sims)
    teams = list(ownership.keys())
    team_probs = np.array([ownership.get(t, 0) for t in teams])
    team_probs = team_probs / team_probs.sum()

    for sim in range(n_sims):
        winners = set(sim_results[sim, round_slots])
        p_survive = sum(
            ownership.get(t, 0) for t in teams if t in winners
        )
        opp_survival_prob[sim] = p_survive

    # Expected opponents surviving per sim
    expected_opp_surviving = n_opponents * opp_survival_prob

    # Calculate payout per sim
    ev_per_entry = np.zeros(n_entries)
    for sim in range(n_sims):
        our_surviving = entry_survived[sim].sum()
        total_surviving = our_surviving + expected_opp_surviving[sim]
        if total_surviving == 0 or our_surviving == 0:
            continue
        payout_per = prize_pool / total_surviving
        for e_idx in range(n_entries):
            if entry_survived[sim, e_idx]:
                ev_per_entry[e_idx] += payout_per

    ev_per_entry /= n_sims

    return {
        "ev_per_entry": ev_per_entry.tolist(),
        "total_ev": float(ev_per_entry.sum()),
        "joint_survival": float(entry_survived.any(axis=1).mean()),
        "per_entry_survival": [float(entry_survived[:, i].mean()) for i in range(n_entries)],
    }


def optimize_portfolio_greedy(
    n_entries: int,
    candidate_teams: list[int],
    sim_results: np.ndarray,
    bracket: TournamentBracket,
    round_num: int,
    ownership: dict[int, float],
    pool_size: int,
    prize_pool: float,
    used_teams_per_entry: list[set[int]] | None = None,
    min_win_prob: float = 0.3,
) -> list[int]:
    """Greedy portfolio optimization with local search.

    1. Start with highest-EV team for entry 1
    2. For each subsequent entry, find the team that maximizes marginal EV
    3. Run pairwise swap improvement

    Args:
        n_entries: Number of entries
        candidate_teams: Teams available to pick this round
        sim_results: Pre-simulated tournament outcomes
        bracket: Tournament bracket
        round_num: Current round number
        ownership: Ownership estimates
        pool_size: Pool size
        prize_pool: Prize pool
        used_teams_per_entry: Already-used teams per entry (for reuse constraint)
        min_win_prob: Min win prob threshold

    Returns:
        List of team_ids, one per entry
    """
    if used_teams_per_entry is None:
        used_teams_per_entry = [set() for _ in range(n_entries)]

    n_sims = sim_results.shape[0]
    round_slots = [i for i, s in enumerate(bracket.slots) if s.round_num == round_num]

    # Pre-compute: for each candidate team, which sims it wins
    team_wins = {}
    for team_id in candidate_teams:
        wins = np.zeros(n_sims, dtype=bool)
        for sim in range(n_sims):
            if team_id in set(sim_results[sim, round_slots]):
                wins[sim] = True
        team_wins[team_id] = wins

    # Filter by min_win_prob
    viable = [t for t in candidate_teams if team_wins[t].mean() >= min_win_prob]
    if not viable:
        viable = candidate_teams

    # Greedy assignment
    picks = []
    for entry_idx in range(n_entries):
        used = used_teams_per_entry[entry_idx]
        best_team = None
        best_ev = -1

        for team_id in viable:
            if team_id in used:
                continue

            trial_picks = picks + [team_id]
            result = evaluate_portfolio(
                trial_picks, sim_results, bracket, round_num,
                ownership, pool_size, prize_pool,
            )
            total_ev = result["total_ev"]
            if total_ev > best_ev:
                best_ev = total_ev
                best_team = team_id

        if best_team is None:
            best_team = viable[0]

        picks.append(best_team)

    # Local search: try pairwise swaps
    improved = True
    max_iterations = 10
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        for e_idx in range(n_entries):
            current = picks[e_idx]
            used = used_teams_per_entry[e_idx]

            current_result = evaluate_portfolio(
                picks, sim_results, bracket, round_num,
                ownership, pool_size, prize_pool,
            )
            current_ev = current_result["total_ev"]

            for team_id in viable:
                if team_id in used or team_id == current:
                    continue

                trial = picks.copy()
                trial[e_idx] = team_id
                trial_result = evaluate_portfolio(
                    trial, sim_results, bracket, round_num,
                    ownership, pool_size, prize_pool,
                )
                if trial_result["total_ev"] > current_ev * 1.001:  # 0.1% improvement threshold
                    picks[e_idx] = team_id
                    current_ev = trial_result["total_ev"]
                    improved = True

    return picks


def future_value_estimate(
    team_id: int,
    sim_results: np.ndarray,
    bracket: TournamentBracket,
    current_round: int,
    ownership_estimator,
    total_rounds: int = 6,
) -> float:
    """Estimate the future value of saving a team for later rounds.

    Future value = sum over future rounds of P(team available and best option).
    Higher future value = should save this team for later.
    """
    n_sims = sim_results.shape[0]
    future_val = 0.0

    for future_round in range(current_round + 1, total_rounds + 1):
        round_slots = [i for i, s in enumerate(bracket.slots) if s.round_num == future_round]
        if not round_slots:
            continue

        # Count how often this team appears in the future round
        appearances = 0
        for sim in range(n_sims):
            for slot_idx in round_slots:
                slot = bracket.slots[slot_idx]
                # Check if team is in this game in this sim (would need forward sim)
                if sim_results[sim, slot_idx] == team_id:
                    appearances += 1
                    break

        # P(team reaches this round) ≈ appearances / n_sims
        p_reaches = appearances / n_sims

        # Discount by round distance
        discount = 0.8 ** (future_round - current_round)
        future_val += p_reaches * discount

    return future_val
