"""Core survival probability math for survivor pool optimization."""

import math


def survival_probability(round_win_probs: list[float]) -> float:
    """P(surviving all rounds) = product of per-round win probabilities."""
    return math.prod(round_win_probs)


def opponent_survival_rate(
    ownership_by_round: list[dict[int, float]],
    win_probs_by_round: list[dict[int, float]],
) -> float:
    """Estimate P(a random opponent survives all rounds).

    For each round, opponent picks team T with prob ownership[T],
    and survives if T wins with prob win_probs[T].
    P(opp survives round r) = sum_T ownership[T] * win_probs[T]
    """
    per_round = []
    for ownership, win_probs in zip(ownership_by_round, win_probs_by_round):
        round_surv = sum(
            ownership.get(t, 0) * win_probs.get(t, 0.5)
            for t in ownership
        )
        per_round.append(round_surv)
    return math.prod(per_round)


def expected_survivors(
    pool_size: int,
    opp_survival: float,
    n_our_entries: int,
    our_survival_probs: list[float],
) -> float:
    """Expected total survivors in the pool.

    Args:
        pool_size: Total entries including ours
        opp_survival: P(random opponent survives all rounds)
        n_our_entries: Number of our entries
        our_survival_probs: P(survive) for each of our entries
    """
    n_opponents = pool_size - n_our_entries
    expected_opp = n_opponents * opp_survival
    expected_ours = sum(our_survival_probs)
    return expected_opp + expected_ours


def single_entry_ev(
    survival_prob: float,
    opp_survival_rate: float,
    pool_size: int,
    prize_pool: float,
    n_our_entries: int = 1,
) -> float:
    """Expected payout for a single entry.

    E[payout] = P(survive) * prize / E[survivors | I survive]
    E[survivors | I survive] ≈ 1 + (pool_size - 1) * opp_survival
    """
    expected_others = (pool_size - n_our_entries) * opp_survival_rate
    expected_total = expected_others + 1  # +1 for this entry surviving
    return survival_prob * prize_pool / max(expected_total, 1)


def pick_ev(
    team_win_prob: float,
    team_ownership: float,
    future_survival: float,
    opp_future_survival: float,
    pool_size: int,
    prize_pool: float,
) -> float:
    """EV of picking a specific team in the current round.

    Args:
        team_win_prob: P(this team wins their game)
        team_ownership: Fraction of pool picking this team
        future_survival: P(our entry survives all future rounds | survive this round)
        opp_future_survival: P(opponent survives future rounds | survive this round)
        pool_size: Total pool entries
        prize_pool: Total prize money

    Returns:
        Expected payout contribution from this pick
    """
    # If we pick this team and it wins:
    #   - We survive this round
    #   - Opponents who picked this team (ownership fraction) also survive
    #   - Opponents who picked other teams survive iff their team won

    # P(opponent survives this round) = ownership * 1 + (1-ownership) * other_teams_win_rate
    # Simplified: sum of ownership[t] * win_prob[t] for all t
    # But we don't have all teams here, just this one's stats

    # Simple model: after this round,
    # expected opponents surviving = pool_size * (opp_round_survival) * opp_future_survival
    team_ownership * 1.0  # those who picked our team and it won
    # Plus others who picked a winning team (we approximate this)

    our_full_survival = team_win_prob * future_survival

    return single_entry_ev(
        our_full_survival,
        opp_future_survival * 0.7,  # rough estimate of opponent making it
        pool_size,
        prize_pool,
    )


def leverage_score(
    win_prob: float,
    ownership: float,
    field_survival_rate: float,
) -> float:
    """Calculate leverage of a pick.

    Leverage measures how much this pick improves your position relative to the field.
    High leverage = high win prob, low ownership (contrarian winner).

    leverage = win_prob * (1 - ownership) / field_survival_rate
    """
    if field_survival_rate == 0:
        return 0.0
    return win_prob * (1 - ownership) / field_survival_rate
