"""Nash equilibrium solver for survivor pool ownership.

At Nash equilibrium, every team that gets positive ownership has the same EV.
No player can improve their expected payout by unilaterally changing their pick.

Uses replicator dynamics / multiplicative weights to converge to equilibrium.
"""

import math
import numpy as np
from optimizer.analytical import exact_pick_ev


def nash_equilibrium(
    win_probs: dict[int, float],
    pool_size: int,
    prize_pool: float,
    max_iter: int = 2000,
    tol: float = 1e-8,
    learning_rate: float = 0.5,
    min_ownership: float = 1e-6,
) -> dict[int, float]:
    """Compute Nash equilibrium ownership distribution.

    At equilibrium, all teams with positive ownership have equal EV.
    Teams with zero ownership have EV <= the equilibrium EV.

    Uses multiplicative weights update (replicator dynamics):
        own[t] <- own[t] * (EV[t] / avg_EV) ^ lr
        normalize to sum to 1

    Args:
        win_probs: team_id -> P(win this round)
        pool_size: Total entries in pool
        prize_pool: Total prize money
        max_iter: Maximum iterations
        tol: Convergence tolerance (max ownership change)
        learning_rate: Speed of adjustment (lower = more stable)
        min_ownership: Floor to prevent teams from going to exactly 0

    Returns:
        Dict of team_id -> equilibrium ownership fraction
    """
    teams = list(win_probs.keys())
    n_teams = len(teams)

    if n_teams == 0:
        return {}

    # Initialize with uniform ownership
    ownership = {t: 1.0 / n_teams for t in teams}

    for iteration in range(max_iter):
        # Compute EV for each team at current ownership
        evs = {}
        for t in teams:
            evs[t] = exact_pick_ev(t, win_probs, ownership, pool_size, prize_pool)

        # Average EV (weighted by ownership)
        avg_ev = sum(ownership[t] * evs[t] for t in teams)
        if avg_ev <= 0:
            break

        # Multiplicative weights update
        new_ownership = {}
        for t in teams:
            ratio = evs[t] / avg_ev if avg_ev > 0 else 1.0
            new_ownership[t] = ownership[t] * (ratio ** learning_rate)
            new_ownership[t] = max(new_ownership[t], min_ownership)

        # Normalize
        total = sum(new_ownership.values())
        new_ownership = {t: v / total for t, v in new_ownership.items()}

        # Check convergence
        max_change = max(abs(new_ownership[t] - ownership[t]) for t in teams)
        ownership = new_ownership

        if max_change < tol:
            break

    # Clean up: remove teams with negligible ownership
    threshold = 1.0 / (n_teams * 100)
    cleaned = {t: v for t, v in ownership.items() if v > threshold}
    total = sum(cleaned.values())
    cleaned = {t: v / total for t, v in cleaned.items()}

    return cleaned


def best_response(
    win_probs: dict[int, float],
    field_ownership: dict[int, float],
    pool_size: int,
    prize_pool: float,
    n_our_entries: int = 1,
) -> list[dict]:
    """Compute best response: our optimal play given the field's strategy.

    Returns all teams ranked by EV against the given field ownership.
    This is what you should pick if you believe the field plays according
    to field_ownership.

    Args:
        win_probs: team_id -> P(win)
        field_ownership: How the rest of the pool is picking
        pool_size: Total pool size
        prize_pool: Prize money
        n_our_entries: Number of our entries

    Returns:
        List of {team_id, ev, win_prob, ownership, ev_edge} sorted by EV desc
    """
    results = []
    for team_id in win_probs:
        ev = exact_pick_ev(
            team_id, win_probs, field_ownership, pool_size, prize_pool, n_our_entries
        )
        results.append({
            "team_id": team_id,
            "ev": ev,
            "win_prob": win_probs[team_id],
            "ownership": field_ownership.get(team_id, 0),
        })

    # Sort by EV descending
    results.sort(key=lambda x: x["ev"], reverse=True)

    # Add edge relative to average
    avg_ev = np.mean([r["ev"] for r in results]) if results else 0
    for r in results:
        r["ev_edge"] = r["ev"] - avg_ev

    return results


def blended_ownership(
    win_probs: dict[int, float],
    pool_size: int,
    prize_pool: float,
    heuristic_ownership: dict[int, float],
    field_efficiency: float = 0.5,
) -> dict[int, float]:
    """Blend Nash equilibrium with heuristic ownership.

    field_efficiency = 0.0: field plays pure heuristic (casual pool)
    field_efficiency = 1.0: field plays Nash (perfectly sharp pool)
    field_efficiency = 0.5: blend (typical pool)

    The blended ownership represents our BELIEF about how the field picks.
    We then compute best_response against this belief.
    """
    nash_own = nash_equilibrium(win_probs, pool_size, prize_pool)

    blended = {}
    all_teams = set(nash_own.keys()) | set(heuristic_ownership.keys())

    for t in all_teams:
        nash_val = nash_own.get(t, 0)
        heur_val = heuristic_ownership.get(t, 0)
        blended[t] = field_efficiency * nash_val + (1 - field_efficiency) * heur_val

    # Normalize
    total = sum(blended.values())
    if total > 0:
        blended = {t: v / total for t, v in blended.items()}

    return blended


def verify_equilibrium(
    ownership: dict[int, float],
    win_probs: dict[int, float],
    pool_size: int,
    prize_pool: float,
) -> dict:
    """Verify that an ownership distribution is (approximately) a Nash equilibrium.

    At equilibrium, all teams with positive ownership should have equal EV.

    Returns:
        Dict with max_ev_diff, is_equilibrium, team_evs
    """
    team_evs = {}
    for t in ownership:
        if ownership[t] > 0.001:  # Only check teams with meaningful ownership
            team_evs[t] = exact_pick_ev(t, win_probs, ownership, pool_size, prize_pool)

    if not team_evs:
        return {"max_ev_diff": 0, "is_equilibrium": True, "team_evs": {}}

    evs = list(team_evs.values())
    max_ev = max(evs)
    min_ev = min(evs)
    avg_ev = np.mean(evs)

    # Relative difference
    max_diff = (max_ev - min_ev) / avg_ev if avg_ev > 0 else 0

    return {
        "max_ev_diff": max_diff,
        "is_equilibrium": max_diff < 0.05,  # Within 5% is "close enough"
        "team_evs": team_evs,
        "avg_ev": avg_ev,
    }
