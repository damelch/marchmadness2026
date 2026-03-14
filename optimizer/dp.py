"""Dynamic programming for multi-round survivor pool planning.

Computes future value of each team to decide whether to use them now
or save them for later rounds where they may be more valuable.
"""

import math
import numpy as np
from simulation.engine import TournamentBracket
from optimizer.analytical import exact_pick_ev


def compute_round_win_probs(
    bracket: TournamentBracket,
    predict_fn,
    sim_results: np.ndarray | None = None,
) -> dict[int, dict[int, float]]:
    """Compute P(team wins) in each round they could play.

    For round 1: exact from model.
    For later rounds: estimated from simulation (which opponents they'd face).

    Returns:
        Dict of round_num -> {team_id -> P(win in that round)}
    """
    round_probs = {}

    # Round 1: exact
    round_probs[1] = {}
    matchups = bracket.get_round_matchups(1)
    for a, b, _ in matchups:
        if a and b:
            p = predict_fn(a, b)
            round_probs[1][a] = p
            round_probs[1][b] = 1.0 - p

    # Later rounds: use simulation results if available
    if sim_results is not None:
        n_sims = sim_results.shape[0]
        all_teams = list(bracket.teams.keys())

        for round_num in range(2, 7):
            round_probs[round_num] = {}
            round_slots = [
                i for i, s in enumerate(bracket.slots) if s.round_num == round_num
            ]

            for team_id in all_teams:
                # P(team wins in this round) = (sims where team wins a game in this round) / n_sims
                wins = sum(
                    1 for sim in range(n_sims)
                    for slot in round_slots
                    if sim_results[sim, slot] == team_id
                )
                round_probs[round_num][team_id] = wins / n_sims

    return round_probs


def compute_advancement_probs(
    sim_results: np.ndarray,
    bracket: TournamentBracket,
) -> dict[int, dict[int, float]]:
    """P(team reaches round R) for each team and round.

    Returns:
        Dict of team_id -> {round_num -> P(reaches that round)}
    """
    n_sims = sim_results.shape[0]
    all_teams = list(bracket.teams.keys())

    adv_probs = {t: {1: 1.0} for t in all_teams}  # Everyone reaches round 1

    for round_num in range(1, 7):
        round_slots = [
            i for i, s in enumerate(bracket.slots) if s.round_num == round_num
        ]

        for team_id in all_teams:
            wins = sum(
                1 for sim in range(n_sims)
                for slot in round_slots
                if sim_results[sim, slot] == team_id
            )
            # "Reaches round R+1" = "wins in round R"
            if round_num + 1 <= 7:
                adv_probs[team_id][round_num + 1] = wins / n_sims

    return adv_probs


def team_scarcity(
    team_id: int,
    round_num: int,
    available_teams_by_round: dict[int, list[int]],
    round_win_probs: dict[int, dict[int, float]],
    min_prob: float = 0.3,
) -> float:
    """How replaceable is this team in a given round.

    Scarcity = 1 / (number of viable alternatives with similar or better EV)
    High scarcity = few alternatives = more valuable to save.

    Args:
        team_id: Team to evaluate
        round_num: Round to evaluate for
        available_teams_by_round: Which teams play in each round
        round_win_probs: Win probabilities per round
        min_prob: Minimum viable win probability
    """
    probs = round_win_probs.get(round_num, {})
    team_wp = probs.get(team_id, 0)

    if team_wp < min_prob:
        return 0.0  # Not viable in this round anyway

    # Count alternatives: teams with win_prob >= min_prob in this round
    available = available_teams_by_round.get(round_num, [])
    n_viable = sum(1 for t in available if probs.get(t, 0) >= min_prob)

    if n_viable <= 1:
        return 1.0  # Only option

    # Scarcity based on relative strength
    # A team that's much better than alternatives is scarcer
    alt_probs = sorted(
        [probs.get(t, 0) for t in available if t != team_id and probs.get(t, 0) >= min_prob],
        reverse=True,
    )

    if not alt_probs:
        return 1.0

    # How much better is this team than the best alternative?
    best_alt = alt_probs[0]
    advantage = team_wp - best_alt

    # Scarcity: high if few alternatives, higher if we're much better
    base_scarcity = 1.0 / n_viable
    advantage_bonus = max(0, advantage) * 2  # 10% better = 0.2 bonus

    return min(base_scarcity + advantage_bonus, 1.0)


def compute_future_values(
    bracket: TournamentBracket,
    round_win_probs: dict[int, dict[int, float]],
    adv_probs: dict[int, dict[int, float]],
    current_round: int,
    ownership_by_round: dict[int, dict[int, float]] | None = None,
    pool_size: int = 100,
    prize_pool: float = 5000,
    total_rounds: int = 6,
    discount: float = 0.85,
) -> dict[int, float]:
    """Compute future value of each team across remaining rounds.

    Future value represents the opportunity cost of using a team now:
    if we use them now, we can't use them in a future round where they
    might be more valuable (scarcer, higher leverage).

    Uses backward induction:
        FV[t] = sum over future rounds r:
            P(t reaches r) * scarcity(t, r) * EV_contribution(t, r) * discount^(r - current)

    Returns:
        Dict of team_id -> future_value (higher = save for later)
    """
    all_teams = list(bracket.teams.keys())

    # Build available teams by round (teams that play in each round)
    available_by_round = {}
    for r in range(1, total_rounds + 1):
        matchups = bracket.get_round_matchups(r)
        teams_in_round = set()
        for a, b, _ in matchups:
            if a:
                teams_in_round.add(a)
            if b:
                teams_in_round.add(b)
        available_by_round[r] = list(teams_in_round)

    future_values = {}

    for team_id in all_teams:
        fv = 0.0
        for future_round in range(current_round + 1, total_rounds + 1):
            # P(team reaches this round)
            p_reaches = adv_probs.get(team_id, {}).get(future_round, 0)
            if p_reaches < 0.01:
                continue

            # Win probability in that round
            wp = round_win_probs.get(future_round, {}).get(team_id, 0)
            if wp < 0.1:
                continue

            # Scarcity in that round
            scar = team_scarcity(
                team_id, future_round, available_by_round, round_win_probs
            )

            # EV contribution estimate
            if ownership_by_round and future_round in ownership_by_round:
                own = ownership_by_round[future_round]
            else:
                # Rough estimate: ownership proportional to win prob
                own = {t: round_win_probs.get(future_round, {}).get(t, 0.5)
                       for t in available_by_round.get(future_round, [])}
                total_own = sum(own.values())
                if total_own > 0:
                    own = {t: v / total_own for t, v in own.items()}

            ev = exact_pick_ev(team_id, round_win_probs.get(future_round, {}),
                               own, pool_size, prize_pool)

            # Weighted future value
            d = discount ** (future_round - current_round)
            fv += p_reaches * scar * ev * d

        future_values[team_id] = fv

    return future_values


def dp_optimal_picks(
    n_entries: int,
    bracket: TournamentBracket,
    round_win_probs: dict[int, dict[int, float]],
    adv_probs: dict[int, dict[int, float]],
    ownership: dict[int, float],
    pool_size: int,
    prize_pool: float,
    current_round: int,
    used_teams_per_entry: list[set[int]] | None = None,
    min_win_prob: float = 0.3,
) -> dict:
    """Generate optimal picks considering future rounds via DP.

    Returns:
        Dict with picks, future_values, adjusted_evs, reasoning
    """
    from optimizer.analytical import optimal_multi_entry

    # Compute future values
    future_vals = compute_future_values(
        bracket, round_win_probs, adv_probs, current_round,
        pool_size=pool_size, prize_pool=prize_pool,
    )

    # Get current round teams
    matchups = bracket.get_round_matchups(current_round)
    available = {}
    current_win_probs = round_win_probs.get(current_round, {})
    for a, b, _ in matchups:
        if a and a in bracket.teams:
            available[a] = bracket.teams[a]["seed"]
        if b and b in bracket.teams:
            available[b] = bracket.teams[b]["seed"]

    # Optimize with future value adjustment
    picks = optimal_multi_entry(
        n_entries, available, current_win_probs, ownership,
        pool_size, prize_pool, used_teams_per_entry,
        min_win_prob, future_vals,
    )

    # Also compute picks WITHOUT future value for comparison
    picks_no_fv = optimal_multi_entry(
        n_entries, available, current_win_probs, ownership,
        pool_size, prize_pool, used_teams_per_entry,
        min_win_prob, None,
    )

    # Build reasoning
    reasoning = []
    for i, (pick, pick_nofv) in enumerate(zip(picks, picks_no_fv)):
        fv = future_vals.get(pick, 0)
        ev = exact_pick_ev(pick, current_win_probs, ownership, pool_size, prize_pool, n_entries)
        info = bracket.teams.get(pick, {})

        if pick != pick_nofv:
            nofv_info = bracket.teams.get(pick_nofv, {})
            nofv_fv = future_vals.get(pick_nofv, 0)
            reasoning.append(
                f"Entry {i+1}: Switched from ({nofv_info.get('seed','?')}) {nofv_info.get('name',pick_nofv)} "
                f"(FV={nofv_fv:.2f}) to ({info.get('seed','?')}) {info.get('name',pick)} "
                f"(FV={fv:.2f}) — saving the higher-FV team for later"
            )
        else:
            reasoning.append(
                f"Entry {i+1}: ({info.get('seed','?')}) {info.get('name',pick)} "
                f"(EV=${ev:.2f}, FV={fv:.2f})"
            )

    return {
        "picks": picks,
        "picks_without_fv": picks_no_fv,
        "future_values": {t: round(v, 4) for t, v in future_vals.items() if v > 0.01},
        "reasoning": reasoning,
    }
