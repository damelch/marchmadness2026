"""Portfolio optimization using analytical EV + MC for correlations.

MC simulation is now only used for:
1. Estimating P(team reaches round R) — tournament outcome correlations
2. Validating analytical EV estimates

Single-round EV calculations use optimizer.analytical (exact, instant).
"""

import numpy as np

from optimizer.analytical import exact_round_ev, optimal_multi_entry
from simulation.engine import TournamentBracket


def evaluate_portfolio(
    picks: list[int],
    win_probs: dict[int, float],
    ownership: dict[int, float],
    pool_size: int,
    prize_pool: float,
    matchup_pairs: list[tuple[int, int]] | None = None,
) -> dict:
    """Evaluate expected payout for a set of entry picks using exact math.

    This replaces the old MC-based evaluate_portfolio. Instant and exact.
    """
    return exact_round_ev(picks, win_probs, ownership, pool_size, prize_pool, matchup_pairs)


def evaluate_portfolio_mc(
    picks: list[int],
    sim_results: np.ndarray,
    bracket: TournamentBracket,
    round_num: int,
    ownership: dict[int, float],
    pool_size: int,
    prize_pool: float,
    rng: np.random.Generator | None = None,
) -> dict:
    """MC-based portfolio evaluation (kept for validation/comparison).

    Use evaluate_portfolio() (analytical) for production. This is useful for
    verifying analytical results or when tournament correlations matter.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_sims = sim_results.shape[0]
    n_entries = len(picks)
    n_opponents = pool_size - n_entries

    round_slots = [i for i, s in enumerate(bracket.slots) if s.round_num == round_num]

    # For each sim, determine which of our picks won
    entry_survived = np.zeros((n_sims, n_entries), dtype=bool)
    for sim in range(n_sims):
        winners = set(sim_results[sim, round_slots])
        for e_idx, team_id in enumerate(picks):
            entry_survived[sim, e_idx] = team_id in winners

    # Opponent survival per sim
    opp_survival_prob = np.zeros(n_sims)
    teams = list(ownership.keys())

    for sim in range(n_sims):
        winners = set(sim_results[sim, round_slots])
        opp_survival_prob[sim] = sum(
            ownership.get(t, 0) for t in teams if t in winners
        )

    expected_opp_surviving = n_opponents * opp_survival_prob

    # Payout per sim
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
    available_teams: dict[int, int],
    win_probs: dict[int, float],
    ownership: dict[int, float],
    pool_size: int,
    prize_pool: float,
    used_teams_per_entry: list[set[int]] | None = None,
    min_win_prob: float = 0.3,
    future_values: dict[int, float] | None = None,
) -> list[int]:
    """Portfolio optimization using analytical EV (no MC needed).

    Delegates to optimizer.analytical.optimal_multi_entry which uses
    exact closed-form EV calculations for instant evaluation.
    """
    return optimal_multi_entry(
        n_entries, available_teams, win_probs, ownership,
        pool_size, prize_pool, used_teams_per_entry,
        min_win_prob, future_values,
    )


def future_value_estimate(
    team_id: int,
    sim_results: np.ndarray,
    bracket: TournamentBracket,
    current_round: int,
    total_rounds: int = 6,
) -> float:
    """Estimate future value using simulation data.

    Kept for backward compatibility. Prefer optimizer.dp.compute_future_values
    for the full DP-based approach.
    """
    n_sims = sim_results.shape[0]
    future_val = 0.0

    for future_round in range(current_round + 1, total_rounds + 1):
        round_slots = [i for i, s in enumerate(bracket.slots) if s.round_num == future_round]
        if not round_slots:
            continue

        appearances = 0
        for sim in range(n_sims):
            for slot_idx in round_slots:
                if sim_results[sim, slot_idx] == team_id:
                    appearances += 1
                    break

        p_reaches = appearances / n_sims
        discount = 0.8 ** (future_round - current_round)
        future_val += p_reaches * discount

    return future_val
