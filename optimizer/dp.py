"""Dynamic programming for multi-round survivor pool planning.

Computes future value of each team to decide whether to use them now
or save them for later days where they may be more valuable.

Updated to work with day-based contest schedule (9 days, 12 picks).
"""

import math
import numpy as np
from simulation.engine import TournamentBracket
from optimizer.analytical import exact_pick_ev, exact_day_ev


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
    alt_probs = sorted(
        [probs.get(t, 0) for t in available if t != team_id and probs.get(t, 0) >= min_prob],
        reverse=True,
    )

    if not alt_probs:
        return 1.0

    best_alt = alt_probs[0]
    advantage = team_wp - best_alt

    base_scarcity = 1.0 / n_viable
    advantage_bonus = max(0, advantage) * 2

    return min(base_scarcity + advantage_bonus, 1.0)


def compute_future_values(
    bracket: TournamentBracket,
    round_win_probs: dict[int, dict[int, float]],
    adv_probs: dict[int, dict[int, float]],
    current_day: int,
    schedule=None,
    ownership_by_round: dict[int, dict[int, float]] | None = None,
    pool_size: int = 100,
    prize_pool: float = 5000,
    discount: float = 0.85,
) -> dict[int, float]:
    """Compute future value of each team across remaining contest days.

    Future value represents the opportunity cost of using a team now:
    if we use them now, we can't use them in a future day where they
    might be more valuable (scarcer, higher leverage).

    Uses backward induction over contest days (not rounds).

    Args:
        bracket: Tournament bracket
        round_win_probs: round_num -> {team_id -> P(win)}
        adv_probs: team_id -> {round_num -> P(reaches)}
        current_day: Current contest day number
        schedule: ContestSchedule instance (uses default if None)
        ownership_by_round: Optional ownership per round
        pool_size: Pool size
        prize_pool: Prize money
        discount: Discount factor per day
    """
    if schedule is None:
        from contest.schedule import ContestSchedule
        schedule = ContestSchedule.default()

    all_teams = list(bracket.teams.keys())

    # Build available teams by round
    available_by_round = {}
    for r in range(1, 7):
        matchups = bracket.get_round_matchups(r)
        teams_in_round = set()
        for a, b, _ in matchups:
            if a:
                teams_in_round.add(a)
            if b:
                teams_in_round.add(b)
        available_by_round[r] = list(teams_in_round)

    future_values = {}
    remaining_days = schedule.get_remaining_days(current_day)

    for team_id in all_teams:
        fv = 0.0
        for day_idx, future_day in enumerate(remaining_days):
            r = future_day.round_num

            # P(team reaches this round)
            p_reaches = adv_probs.get(team_id, {}).get(r, 0)
            if p_reaches < 0.01:
                continue

            # Win probability in that round
            wp = round_win_probs.get(r, {}).get(team_id, 0)
            if wp < 0.1:
                continue

            # Check team plays in this day's regions
            team_region = bracket.teams.get(team_id, {}).get("region", "")
            if team_region and team_region not in future_day.regions and r <= 4:
                continue  # Team's region not on this day

            # Scarcity in that round
            scar = team_scarcity(
                team_id, r, available_by_round, round_win_probs
            )

            # On double-pick days, teams are more valuable because you
            # need 2 viable picks (scarcer resource)
            if future_day.is_double_pick:
                scar = min(scar * 1.3, 1.0)

            # Ownership estimate
            if ownership_by_round and r in ownership_by_round:
                own = ownership_by_round[r]
            else:
                own = {t: round_win_probs.get(r, {}).get(t, 0.5)
                       for t in available_by_round.get(r, [])}
                total_own = sum(own.values())
                if total_own > 0:
                    own = {t: v / total_own for t, v in own.items()}

            ev = exact_pick_ev(team_id, round_win_probs.get(r, {}),
                               own, pool_size, prize_pool)

            d = discount ** (day_idx + 1)
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
    current_day: int,
    schedule=None,
    used_teams_per_entry: list[set[int]] | None = None,
    min_win_prob: float = 0.3,
    num_picks: int = 1,
    matchup_pairs: list[tuple[int, int]] | None = None,
) -> dict:
    """Generate optimal picks considering future days via DP.

    Args:
        current_day: Contest day number (1-9)
        schedule: ContestSchedule instance
        num_picks: Picks required this day (1 or 2)
        matchup_pairs: Game matchups for this day

    Returns:
        Dict with picks (list of pick sets), future_values, reasoning
    """
    from optimizer.analytical import optimal_day_picks

    if schedule is None:
        from contest.schedule import ContestSchedule
        schedule = ContestSchedule.default()

    # Map day to round for win probs
    day = schedule.get_day(current_day)
    round_num = day.round_num

    # Compute future values
    future_vals = compute_future_values(
        bracket, round_win_probs, adv_probs, current_day,
        schedule=schedule,
        pool_size=pool_size, prize_pool=prize_pool,
    )

    # Get available teams for this day (filtered by region)
    matchups = bracket.get_day_matchups(round_num, day.regions)
    available = {}
    current_win_probs = round_win_probs.get(round_num, {})
    for a, b, _ in matchups:
        if a and a in bracket.teams:
            available[a] = bracket.teams[a]["seed"]
        if b and b in bracket.teams:
            available[b] = bracket.teams[b]["seed"]

    if matchup_pairs is None:
        matchup_pairs = [(a, b) for a, b, _ in matchups if a and b]

    # Optimize with future value adjustment
    picks = optimal_day_picks(
        n_entries, available, current_win_probs, ownership,
        pool_size, prize_pool, num_picks,
        used_teams_per_entry, min_win_prob, future_vals, matchup_pairs,
    )

    # Also compute picks WITHOUT future value for comparison
    picks_no_fv = optimal_day_picks(
        n_entries, available, current_win_probs, ownership,
        pool_size, prize_pool, num_picks,
        used_teams_per_entry, min_win_prob, None, matchup_pairs,
    )

    # Build reasoning
    reasoning = []
    for i, (pick_set, pick_set_nofv) in enumerate(zip(picks, picks_no_fv)):
        names = []
        for t in pick_set:
            info = bracket.teams.get(t, {})
            names.append(f"({info.get('seed', '?')}) {info.get('name', t)}")
        pick_str = " + ".join(names)

        fv_total = sum(future_vals.get(t, 0) for t in pick_set)

        if num_picks == 1:
            ev = exact_pick_ev(
                pick_set[0], current_win_probs, ownership, pool_size, prize_pool, n_entries,
            )
        else:
            ev = exact_day_ev(
                pick_set, current_win_probs, ownership, pool_size, prize_pool,
                num_picks_per_opponent=num_picks, n_our_entries=n_entries,
                matchup_pairs=matchup_pairs,
            )

        if pick_set != pick_set_nofv:
            nofv_names = []
            for t in pick_set_nofv:
                info = bracket.teams.get(t, {})
                nofv_names.append(f"({info.get('seed', '?')}) {info.get('name', t)}")
            nofv_str = " + ".join(nofv_names)
            reasoning.append(
                f"Entry {i+1}: Switched from {nofv_str} to {pick_str} "
                f"(FV={fv_total:.2f}) — saving higher-FV teams for later"
            )
        else:
            reasoning.append(
                f"Entry {i+1}: {pick_str} (EV=${ev:.2f}, FV={fv_total:.2f})"
            )

    return {
        "picks": picks,
        "picks_without_fv": picks_no_fv,
        "future_values": {t: round(v, 4) for t, v in future_vals.items() if v > 0.01},
        "reasoning": reasoning,
    }
