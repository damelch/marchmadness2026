"""Kelly Criterion for optimal number of survivor pool entries."""

import math


def kelly_fraction(
    ev_per_entry: float,
    entry_cost: float,
    variance: float | None = None,
) -> float:
    """Calculate Kelly fraction for a single entry.

    f* = edge / odds
    where edge = EV / cost - 1

    Args:
        ev_per_entry: Expected payout per entry
        entry_cost: Cost per entry
        variance: Optional variance of payout (for fractional Kelly)

    Returns:
        Optimal fraction of bankroll per entry
    """
    if entry_cost <= 0:
        return 0.0

    edge = ev_per_entry / entry_cost - 1.0
    if edge <= 0:
        return 0.0

    # If we have variance info, use it for more precise Kelly
    if variance is not None and variance > 0:
        return edge * entry_cost / variance

    # Simple Kelly: f = edge / (payout_odds)
    # For survivor pools, odds ≈ prize / (entry_cost * expected_winners)
    return edge


def optimal_entries(
    entry_cost: float,
    ev_per_entry: float,
    bankroll: float,
    kelly_multiplier: float = 0.5,
    max_entries: int | None = None,
    diminishing_factor: float = 0.95,
) -> dict:
    """Calculate optimal number of entries to purchase.

    Uses half-Kelly by default for safety. Accounts for diminishing
    returns as more entries are added (entries are correlated).

    Args:
        entry_cost: Cost per entry
        ev_per_entry: Expected value per entry (from simulation)
        bankroll: Total available bankroll
        kelly_multiplier: Fraction of full Kelly (0.5 = half-Kelly)
        max_entries: Maximum allowed entries (contest rules)
        diminishing_factor: Each additional entry has this fraction of marginal EV

    Returns:
        Dict with n_entries, total_cost, total_ev, roi
    """
    if entry_cost <= 0 or ev_per_entry <= entry_cost:
        return {
            "n_entries": 0,
            "total_cost": 0,
            "total_ev": 0,
            "roi": 0,
            "reasoning": "Negative EV - don't enter",
        }

    # Calculate entries one at a time with diminishing returns
    n = 0
    total_cost = 0
    total_ev = 0
    marginal_ev = ev_per_entry

    while True:
        n += 1
        new_cost = total_cost + entry_cost
        new_ev = total_ev + marginal_ev

        # Check bankroll constraint (with Kelly fraction)
        max_from_bankroll = math.floor(bankroll * kelly_multiplier / entry_cost)
        if n > max_from_bankroll:
            n -= 1
            break

        # Check max entries constraint
        if max_entries is not None and n > max_entries:
            n -= 1
            break

        # Check if marginal EV still positive
        if marginal_ev < entry_cost:
            n -= 1
            break

        total_cost = new_cost
        total_ev = new_ev

        # Apply diminishing returns for next entry
        marginal_ev *= diminishing_factor

    total_cost = n * entry_cost
    # Recalculate total EV with diminishing returns
    total_ev = 0
    ev = ev_per_entry
    for _ in range(n):
        total_ev += ev
        ev *= diminishing_factor

    roi = (total_ev - total_cost) / total_cost if total_cost > 0 else 0

    return {
        "n_entries": n,
        "total_cost": total_cost,
        "total_ev": total_ev,
        "roi": roi,
        "edge_per_entry": ev_per_entry / entry_cost - 1,
        "kelly_fraction": kelly_multiplier,
        "reasoning": f"{n} entries at ${entry_cost} = ${total_cost}. EV=${total_ev:.2f}, ROI={roi:.1%}",
    }
