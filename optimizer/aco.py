"""Ant Colony Optimization for multi-entry survivor pool portfolio construction.

Uses pheromone-guided probabilistic search to find diverse, high-EV portfolios
that greedy + local swap search may miss. Seeded with the greedy solution so
ACO can only improve, never regress.

Single-pick days: pheromone on individual teams.
Double-pick days: pheromone on (team_a, team_b) pairs to capture synergy.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from optimizer.analytical import (
    _portfolio_score,
    exact_day_ev,
    exact_pick_ev,
)
from optimizer.constants import (
    ACO_ALPHA,
    ACO_BETA,
    ACO_ELITE_WEIGHT,
    ACO_MAX_PHEROMONE,
    ACO_MIN_PHEROMONE,
    ACO_N_ANTS,
    ACO_N_ITERATIONS,
    ACO_RHO,
    ACO_TOP_K,
    CONCENTRATION_PENALTY,
    FV_WEIGHT,
    SEED_FV_MULTIPLIER,
)


def aco_optimize(
    n_entries: int,
    available_teams: dict[int, int],
    win_probs: dict[int, float],
    ownership: dict[int, float],
    pool_size: int,
    prize_pool: float,
    num_picks: int = 1,
    used_teams_per_entry: list[set[int]] | None = None,
    min_win_prob: float = 0.3,
    future_values: dict[int, float] | None = None,
    matchup_pairs: list[tuple[int, int]] | None = None,
    # ACO parameters (defaults from constants.py)
    n_ants: int = ACO_N_ANTS,
    n_iterations: int = ACO_N_ITERATIONS,
    alpha: float = ACO_ALPHA,
    beta: float = ACO_BETA,
    rho: float = ACO_RHO,
    elite_weight: float = ACO_ELITE_WEIGHT,
    seed: int | None = 42,
) -> list[list[int]]:
    """Optimize picks for N entries using Ant Colony Optimization.

    Returns same format as optimal_day_picks():
        Single-pick: [[team_a], [team_b], ...]
        Double-pick: [[team_a, team_b], [team_c, team_d], ...]

    Args:
        n_entries: Number of entries to optimize
        available_teams: team_id -> seed mapping
        win_probs: team_id -> P(win)
        ownership: team_id -> ownership fraction
        pool_size: Total pool size
        prize_pool: Prize money
        num_picks: Picks per entry (1 or 2)
        used_teams_per_entry: Already-used teams per entry
        min_win_prob: Minimum win probability threshold
        future_values: team_id -> future value penalty
        matchup_pairs: List of (teamA, teamB) game pairs
        n_ants: Ants per generation
        n_iterations: Number of generations
        alpha: Pheromone importance exponent
        beta: Heuristic importance exponent
        rho: Evaporation rate (0-1)
        elite_weight: Extra pheromone deposit for best-ever solution
        seed: Random seed for reproducibility
    """
    if used_teams_per_entry is None:
        used_teams_per_entry = [set() for _ in range(n_entries)]
    if future_values is None:
        future_values = {}

    rng = np.random.default_rng(seed)

    if num_picks == 1:
        return _aco_single_pick(
            n_entries, available_teams, win_probs, ownership, pool_size,
            prize_pool, used_teams_per_entry, min_win_prob, future_values,
            matchup_pairs, n_ants, n_iterations, alpha, beta, rho,
            elite_weight, rng,
        )
    else:
        return _aco_double_pick(
            n_entries, available_teams, win_probs, ownership, pool_size,
            prize_pool, used_teams_per_entry, min_win_prob, future_values,
            matchup_pairs, n_ants, n_iterations, alpha, beta, rho,
            elite_weight, rng,
        )


# ---------------------------------------------------------------------------
# Single-pick ACO
# ---------------------------------------------------------------------------


def _greedy_seed_single(
    n_entries: int,
    viable: list[int],
    heuristic: dict[int, float],
    used_teams_per_entry: list[set[int]],
) -> list[int]:
    """Fast greedy assignment for ACO seeding (no swap search)."""
    viable_sorted = sorted(viable, key=lambda t: heuristic.get(t, 0), reverse=True)
    picks: list[int] = []
    used_this_round: set[int] = set()

    for entry_idx in range(n_entries):
        used = used_teams_per_entry[entry_idx]
        chosen = None
        for t in viable_sorted:
            if t not in used and t not in used_this_round:
                chosen = t
                break
        if chosen is None:
            for t in viable_sorted:
                if t not in used:
                    chosen = t
                    break
        if chosen is None:
            chosen = viable_sorted[0]
        picks.append(chosen)
        used_this_round.add(chosen)

    return picks


def _greedy_seed_double(
    n_entries: int,
    valid_pairs: list[tuple[int, int]],
    pair_heuristic: dict[tuple[int, int], float],
    used_teams_per_entry: list[set[int]],
) -> list[list[int]]:
    """Fast greedy assignment for double-pick ACO seeding (no swap search)."""
    sorted_pairs = sorted(valid_pairs, key=lambda p: pair_heuristic.get(p, 0), reverse=True)
    picks: list[list[int]] = []
    used_count: dict[int, int] = {}

    for entry_idx in range(n_entries):
        used = used_teams_per_entry[entry_idx]
        chosen = None
        for pair in sorted_pairs[:60]:
            a, b = pair
            if a in used or b in used:
                continue
            overlap = used_count.get(a, 0) + used_count.get(b, 0)
            if chosen is None or overlap == 0:
                chosen = pair
                if overlap == 0:
                    break
        if chosen is None:
            chosen = sorted_pairs[0]
        picks.append(list(chosen))
        for t in chosen:
            used_count[t] = used_count.get(t, 0) + 1

    return picks


def _aco_single_pick(
    n_entries: int,
    available_teams: dict[int, int],
    win_probs: dict[int, float],
    ownership: dict[int, float],
    pool_size: int,
    prize_pool: float,
    used_teams_per_entry: list[set[int]],
    min_win_prob: float,
    future_values: dict[int, float],
    matchup_pairs: list[tuple[int, int]] | None,
    n_ants: int,
    n_iterations: int,
    alpha: float,
    beta: float,
    rho: float,
    elite_weight: float,
    rng: np.random.Generator,
) -> list[list[int]]:
    """ACO for single-pick days."""
    # Filter viable teams
    viable = [t for t in available_teams if win_probs.get(t, 0) >= min_win_prob]
    if not viable:
        viable = list(available_teams.keys())

    if len(viable) == 0:
        return [[] for _ in range(n_entries)]

    # Compute heuristic: EV adjusted by future value
    heuristic = _compute_single_heuristic(
        viable, available_teams, win_probs, ownership, pool_size,
        prize_pool, n_entries, future_values,
    )

    # Initialize pheromone uniformly
    pheromone = {t: 1.0 for t in viable}

    # Get greedy seed solution (fast — no swap search, just heuristic-sorted assignment)
    greedy_flat = _greedy_seed_single(n_entries, viable, heuristic, used_teams_per_entry)
    greedy_score = _score_single_portfolio(
        greedy_flat, win_probs, ownership, pool_size, prize_pool,
    )

    best_ever_picks = greedy_flat[:]
    best_ever_score = greedy_score

    for _iteration in range(n_iterations):
        # Generate solutions from all ants
        ant_solutions: list[list[int]] = []
        ant_scores: list[float] = []

        for ant_idx in range(n_ants):
            if ant_idx == 0 and _iteration == 0:
                # First ant of first generation = greedy solution
                solution = greedy_flat[:]
            else:
                solution = _construct_single_solution(
                    n_entries, viable, used_teams_per_entry, pheromone,
                    heuristic, alpha, beta, rng,
                )

            score = _score_single_portfolio(
                solution, win_probs, ownership, pool_size, prize_pool,
            )
            ant_solutions.append(solution)
            ant_scores.append(score)

            if score > best_ever_score:
                best_ever_score = score
                best_ever_picks = solution[:]

        # Update pheromone
        _update_pheromone_single(
            pheromone, ant_solutions, ant_scores,
            best_ever_picks, best_ever_score,
            rho, elite_weight,
        )

    return [[t] for t in best_ever_picks]


def _compute_single_heuristic(
    viable: list[int],
    available_teams: dict[int, int],
    win_probs: dict[int, float],
    ownership: dict[int, float],
    pool_size: int,
    prize_pool: float,
    n_entries: int,
    future_values: dict[int, float],
) -> dict[int, float]:
    """Compute heuristic desirability for each team."""
    team_evs = {
        t: exact_pick_ev(t, win_probs, ownership, pool_size, prize_pool, n_entries)
        for t in viable
    }
    max_ev = max(team_evs.values()) if team_evs else 1.0
    fv_vals = [future_values.get(t, 0.0) for t in viable]
    max_fv = max(fv_vals) if fv_vals else 1.0

    fv_scale = (max_ev / max_fv) * FV_WEIGHT if max_fv > 0 else 0.0

    heuristic = {}
    for t in viable:
        ev = team_evs[t]
        seed = available_teams.get(t, 8)
        seed_mult = SEED_FV_MULTIPLIER.get(seed, 1.0)
        fv_penalty = future_values.get(t, 0.0) * fv_scale * seed_mult
        heuristic[t] = max(ev - fv_penalty, 1e-10)  # Floor to avoid zero

    return heuristic


def _construct_single_solution(
    n_entries: int,
    viable: list[int],
    used_teams_per_entry: list[set[int]],
    pheromone: dict[int, float],
    heuristic: dict[int, float],
    alpha: float,
    beta: float,
    rng: np.random.Generator,
) -> list[int]:
    """One ant constructs a full portfolio for single-pick day."""
    picks: list[int] = []
    used_this_round: dict[int, int] = {}  # team -> count used

    for entry_idx in range(n_entries):
        used = used_teams_per_entry[entry_idx]

        # Compute selection probabilities
        candidates = [t for t in viable if t not in used]
        if not candidates:
            candidates = viable[:]

        probs = np.zeros(len(candidates))
        for i, t in enumerate(candidates):
            tau = pheromone.get(t, 1.0)
            eta = heuristic.get(t, 1e-10)

            # Concentration penalty: reduce attractiveness if already used by other entries
            concentration = 1.0 / (1.0 + CONCENTRATION_PENALTY * used_this_round.get(t, 0))

            probs[i] = (tau ** alpha) * (eta ** beta) * concentration

        # Normalize
        total = probs.sum()
        if total <= 0:
            probs = np.ones(len(candidates)) / len(candidates)
        else:
            probs /= total

        # Select
        idx = rng.choice(len(candidates), p=probs)
        chosen = candidates[idx]
        picks.append(chosen)
        used_this_round[chosen] = used_this_round.get(chosen, 0) + 1

    return picks


def _score_single_portfolio(
    picks: list[int],
    win_probs: dict[int, float],
    ownership: dict[int, float],
    pool_size: int,
    prize_pool: float,
) -> float:
    """Score a single-pick portfolio using existing analytical scoring."""
    return _portfolio_score(picks, win_probs, ownership, pool_size, prize_pool)


def _update_pheromone_single(
    pheromone: dict[int, float],
    ant_solutions: list[list[int]],
    ant_scores: list[float],
    best_ever_picks: list[int],
    best_ever_score: float,
    rho: float,
    elite_weight: float,
) -> None:
    """Update pheromone trails for single-pick ACO."""
    # Evaporate
    for t in pheromone:
        pheromone[t] *= (1.0 - rho)

    # Deposit from top-K ants
    if ant_scores:
        indexed = sorted(enumerate(ant_scores), key=lambda x: x[1], reverse=True)
        top_k = indexed[:ACO_TOP_K]
        ref_score = best_ever_score if best_ever_score > 0 else 1.0

        for ant_idx, score in top_k:
            deposit = score / ref_score
            for t in ant_solutions[ant_idx]:
                pheromone[t] = pheromone.get(t, 1.0) + deposit

    # Elitist deposit for best-ever
    if best_ever_score > 0:
        deposit = elite_weight * (best_ever_score / max(best_ever_score, 1.0))
        for t in best_ever_picks:
            pheromone[t] = pheromone.get(t, 1.0) + deposit

    # Clamp pheromone
    for t in pheromone:
        pheromone[t] = max(ACO_MIN_PHEROMONE, min(ACO_MAX_PHEROMONE, pheromone[t]))


# ---------------------------------------------------------------------------
# Double-pick ACO
# ---------------------------------------------------------------------------


def _aco_double_pick(
    n_entries: int,
    available_teams: dict[int, int],
    win_probs: dict[int, float],
    ownership: dict[int, float],
    pool_size: int,
    prize_pool: float,
    used_teams_per_entry: list[set[int]],
    min_win_prob: float,
    future_values: dict[int, float],
    matchup_pairs: list[tuple[int, int]] | None,
    n_ants: int,
    n_iterations: int,
    alpha: float,
    beta: float,
    rho: float,
    elite_weight: float,
    rng: np.random.Generator,
) -> list[list[int]]:
    """ACO for double-pick days using pair-level pheromone."""
    # Filter viable teams
    viable = [t for t in available_teams if win_probs.get(t, 0) >= min_win_prob]
    if len(viable) < 2:
        viable = list(available_teams.keys())

    if len(viable) < 2:
        return [[viable[0], viable[0]] if viable else [] for _ in range(n_entries)]

    # Build opponent map
    opp_map: dict[int, int] = {}
    if matchup_pairs:
        for a, b in matchup_pairs:
            opp_map[a] = b
            opp_map[b] = a

    # Generate valid pairs (not from same game)
    valid_pairs = [
        (a, b) for a, b in combinations(viable, 2)
        if opp_map.get(a) != b and opp_map.get(b) != a
    ]

    if not valid_pairs:
        # Fallback: allow any pairs
        valid_pairs = list(combinations(viable, 2))

    if not valid_pairs:
        return [[viable[0]] * 2 for _ in range(n_entries)]

    # Compute heuristic for each pair
    pair_heuristic = _compute_pair_heuristic(
        valid_pairs, available_teams, win_probs, ownership, pool_size,
        prize_pool, n_entries, future_values, matchup_pairs,
    )

    # Initialize pheromone on pairs
    pheromone: dict[tuple[int, int], float] = {p: 1.0 for p in valid_pairs}

    # Get greedy seed solution (fast — no swap search)
    greedy_picks = _greedy_seed_double(n_entries, valid_pairs, pair_heuristic, used_teams_per_entry)
    greedy_score = _score_double_portfolio(
        greedy_picks, win_probs, ownership, pool_size, prize_pool,
        n_entries, matchup_pairs, future_values,
    )

    best_ever_picks = [ps[:] for ps in greedy_picks]
    best_ever_score = greedy_score

    for _iteration in range(n_iterations):
        ant_solutions: list[list[list[int]]] = []
        ant_scores: list[float] = []

        for ant_idx in range(n_ants):
            if ant_idx == 0 and _iteration == 0:
                solution = [ps[:] for ps in greedy_picks]
            else:
                solution = _construct_double_solution(
                    n_entries, valid_pairs, used_teams_per_entry,
                    pheromone, pair_heuristic, alpha, beta, rng,
                )

            score = _score_double_portfolio(
                solution, win_probs, ownership, pool_size, prize_pool,
                n_entries, matchup_pairs, future_values,
            )
            ant_solutions.append(solution)
            ant_scores.append(score)

            if score > best_ever_score:
                best_ever_score = score
                best_ever_picks = [ps[:] for ps in solution]

        # Update pheromone
        _update_pheromone_double(
            pheromone, ant_solutions, ant_scores,
            best_ever_picks, best_ever_score,
            rho, elite_weight,
        )

    return best_ever_picks


def _compute_pair_heuristic(
    valid_pairs: list[tuple[int, int]],
    available_teams: dict[int, int],
    win_probs: dict[int, float],
    ownership: dict[int, float],
    pool_size: int,
    prize_pool: float,
    n_entries: int,
    future_values: dict[int, float],
    matchup_pairs: list[tuple[int, int]] | None,
) -> dict[tuple[int, int], float]:
    """Compute heuristic desirability for each valid pair."""
    pair_evs = {}
    for pair in valid_pairs:
        pair_evs[pair] = exact_day_ev(
            list(pair), win_probs, ownership, pool_size, prize_pool,
            num_picks_per_opponent=2, n_our_entries=n_entries,
            matchup_pairs=matchup_pairs,
        )

    max_ev = max(pair_evs.values()) if pair_evs else 1.0
    all_fv = [sum(future_values.get(t, 0.0) for t in p) for p in valid_pairs]
    max_fv = max(all_fv) if all_fv else 1.0
    fv_scale = (max_ev / max_fv) * FV_WEIGHT if max_fv > 0 else 0.0

    heuristic = {}
    for pair in valid_pairs:
        ev = pair_evs[pair]
        fv_penalty = 0.0
        for t in pair:
            seed = available_teams.get(t, 8)
            seed_mult = SEED_FV_MULTIPLIER.get(seed, 1.0)
            fv_penalty += future_values.get(t, 0.0) * fv_scale * seed_mult
        heuristic[pair] = max(ev - fv_penalty, 1e-10)

    return heuristic


def _construct_double_solution(
    n_entries: int,
    valid_pairs: list[tuple[int, int]],
    used_teams_per_entry: list[set[int]],
    pheromone: dict[tuple[int, int], float],
    pair_heuristic: dict[tuple[int, int], float],
    alpha: float,
    beta: float,
    rng: np.random.Generator,
) -> list[list[int]]:
    """One ant constructs a full double-pick portfolio."""
    picks: list[list[int]] = []
    used_teams_count: dict[int, int] = {}  # team -> count across entries

    for entry_idx in range(n_entries):
        used = used_teams_per_entry[entry_idx]

        # Filter pairs: neither team in entry's used set
        candidates = [
            p for p in valid_pairs
            if p[0] not in used and p[1] not in used
        ]
        if not candidates:
            candidates = valid_pairs[:]

        probs = np.zeros(len(candidates))
        for i, pair in enumerate(candidates):
            tau = pheromone.get(pair, 1.0)
            eta = pair_heuristic.get(pair, 1e-10)

            # Concentration penalty for teams already used by other entries
            overlap = sum(used_teams_count.get(t, 0) for t in pair)
            concentration = 1.0 / (1.0 + CONCENTRATION_PENALTY * overlap)

            probs[i] = (tau ** alpha) * (eta ** beta) * concentration

        total = probs.sum()
        if total <= 0:
            probs = np.ones(len(candidates)) / len(candidates)
        else:
            probs /= total

        idx = rng.choice(len(candidates), p=probs)
        chosen = candidates[idx]
        picks.append(list(chosen))
        for t in chosen:
            used_teams_count[t] = used_teams_count.get(t, 0) + 1

    return picks


def _score_double_portfolio(
    picks: list[list[int]],
    win_probs: dict[int, float],
    ownership: dict[int, float],
    pool_size: int,
    prize_pool: float,
    n_entries: int,
    matchup_pairs: list[tuple[int, int]] | None,
    future_values: dict[int, float] | None,
) -> float:
    """Score a double-pick portfolio: total EV - concentration penalty - FV."""
    if future_values is None:
        future_values = {}

    total_ev = sum(
        exact_day_ev(
            ps, win_probs, ownership, pool_size, prize_pool,
            num_picks_per_opponent=2, n_our_entries=n_entries,
            matchup_pairs=matchup_pairs,
        )
        for ps in picks
    )

    # Concentration penalty
    if len(picks) > 1:
        from collections import Counter
        team_counts = Counter(t for ps in picks for t in ps)
        n = len(picks)
        penalty = 0.0
        for count in team_counts.values():
            if count > 1:
                exposure = count / n
                penalty += exposure ** 2 * total_ev * CONCENTRATION_PENALTY
        total_ev -= penalty

    # FV penalty (lightweight)
    max_ev = abs(total_ev) if total_ev != 0 else 1.0
    fv_vals = [sum(future_values.get(t, 0.0) for t in ps) for ps in picks]
    max_fv = max(fv_vals) if fv_vals else 1.0
    fv_scale = (max_ev / max_fv) * FV_WEIGHT if max_fv > 0 else 0.0
    fv_cost = sum(v * fv_scale for v in fv_vals)

    return total_ev - fv_cost


def _update_pheromone_double(
    pheromone: dict[tuple[int, int], float],
    ant_solutions: list[list[list[int]]],
    ant_scores: list[float],
    best_ever_picks: list[list[int]],
    best_ever_score: float,
    rho: float,
    elite_weight: float,
) -> None:
    """Update pheromone trails for double-pick ACO."""
    # Evaporate
    for pair in pheromone:
        pheromone[pair] *= (1.0 - rho)

    # Deposit from top-K ants
    if ant_scores:
        indexed = sorted(enumerate(ant_scores), key=lambda x: x[1], reverse=True)
        top_k = indexed[:ACO_TOP_K]
        ref_score = best_ever_score if best_ever_score > 0 else 1.0

        for ant_idx, score in top_k:
            deposit = score / ref_score
            for ps in ant_solutions[ant_idx]:
                pair_key = tuple(sorted(ps))
                if pair_key in pheromone:
                    pheromone[pair_key] += deposit

    # Elitist deposit
    if best_ever_score > 0:
        deposit = elite_weight
        for ps in best_ever_picks:
            pair_key = tuple(sorted(ps))
            if pair_key in pheromone:
                pheromone[pair_key] += deposit

    # Clamp
    for pair in pheromone:
        pheromone[pair] = max(ACO_MIN_PHEROMONE, min(ACO_MAX_PHEROMONE, pheromone[pair]))
