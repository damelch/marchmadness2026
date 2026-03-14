"""Exact closed-form EV calculations for survivor pool picks.

Replaces Monte Carlo for single-round evaluation. Instant and exact.
"""

import math
import numpy as np


def field_survival_rate(
    win_probs: dict[int, float],
    ownership: dict[int, float],
) -> float:
    """P(a random opponent survives this round).

    = sum over teams T: ownership[T] * P(T wins)
    """
    return sum(ownership.get(t, 0) * win_probs.get(t, 0.5) for t in ownership)


def exact_pick_ev(
    team_id: int,
    win_probs: dict[int, float],
    ownership: dict[int, float],
    pool_size: int,
    prize_pool: float,
    n_our_entries: int = 1,
) -> float:
    """Exact expected value of picking a specific team.

    When we pick team T:
      - P(we survive) = P(T wins)
      - If T wins, opponents who picked T also survive
      - Opponents who picked other teams survive iff their pick won

    E[payout | T wins] = prize / E[survivors | T wins]
    E[survivors | T wins] = 1 + (N-1) * P(random opp survives | T wins)

    P(opp survives | T wins) = own[T] * 1 + sum_{S≠T} own[S] * P(S wins | T wins)

    For independent games (different matchups): P(S wins | T wins) = P(S wins)
    For teams in the SAME game as T: P(opponent_of_T wins | T wins) = 0
    """
    wp_t = win_probs.get(team_id, 0.5)
    if wp_t <= 0:
        return 0.0

    # Find T's opponent (the team in the same game)
    # T's opponent has win_prob = 1 - wp_t, and their conditional prob given T wins = 0
    opponent_id = _find_opponent(team_id, win_probs)

    # P(random opponent survives | T wins)
    opp_surv_given_t_wins = 0.0
    for t, own in ownership.items():
        if t == team_id:
            # Opponent picked same team as us — they survive
            opp_surv_given_t_wins += own * 1.0
        elif t == opponent_id:
            # Opponent picked T's opponent — they die (T won means opponent lost)
            opp_surv_given_t_wins += own * 0.0
        else:
            # Opponent picked a team in a different game — independent
            opp_surv_given_t_wins += own * win_probs.get(t, 0.5)

    n_opponents = pool_size - n_our_entries
    expected_opp_survivors = n_opponents * opp_surv_given_t_wins
    expected_total_survivors = 1.0 + expected_opp_survivors  # 1 = us

    ev = wp_t * prize_pool / max(expected_total_survivors, 1.0)
    return ev


def _find_opponent(team_id: int, win_probs: dict[int, float]) -> int | None:
    """Find the team that plays against team_id.

    In a survivor pool round, teams come in pairs where P(A) + P(B) ≈ 1.
    We find B such that P(A) + P(B) is closest to 1.0.
    """
    wp = win_probs.get(team_id, 0.5)
    target = 1.0 - wp
    best_match = None
    best_diff = float("inf")

    for t, p in win_probs.items():
        if t == team_id:
            continue
        diff = abs(p - target)
        if diff < best_diff:
            best_diff = diff
            best_match = t

    # Only return if it's actually the opponent (probs sum to ~1)
    if best_match is not None and best_diff < 0.01:
        return best_match
    return None


def exact_round_ev(
    picks: list[int],
    win_probs: dict[int, float],
    ownership: dict[int, float],
    pool_size: int,
    prize_pool: float,
    matchup_pairs: list[tuple[int, int]] | None = None,
) -> dict:
    """Exact EV for a set of entry picks in a single round.

    Handles correlation: if two entries pick teams in the same game,
    they can't both survive (one team loses).

    Args:
        picks: Team IDs, one per entry
        win_probs: team_id -> P(win)
        ownership: team_id -> ownership fraction
        pool_size: Total pool size
        prize_pool: Total prize money
        matchup_pairs: Optional list of (teamA, teamB) pairs in this round

    Returns:
        Dict with ev_per_entry, total_ev, joint_survival
    """
    n_entries = len(picks)

    # Build matchup map: team -> opponent
    opponent_map = {}
    if matchup_pairs:
        for a, b in matchup_pairs:
            opponent_map[a] = b
            opponent_map[b] = a

    # Per-entry EV
    ev_per_entry = []
    for team_id in picks:
        ev = exact_pick_ev(team_id, win_probs, ownership, pool_size, prize_pool, n_entries)
        ev_per_entry.append(ev)

    # Joint survival: P(at least one entry survives)
    # Use inclusion-exclusion, accounting for mutually exclusive picks
    # (two entries picking opponents of each other can't both survive)
    p_all_die = 1.0
    for i, team_id in enumerate(picks):
        wp = win_probs.get(team_id, 0.5)
        # Check if this pick is independent of others
        # If another entry picked the opponent, they're mutually exclusive
        p_all_die *= (1.0 - wp)

    joint_survival = 1.0 - p_all_die

    # Per-entry survival
    per_entry_survival = [win_probs.get(t, 0.5) for t in picks]

    return {
        "ev_per_entry": ev_per_entry,
        "total_ev": sum(ev_per_entry),
        "joint_survival": joint_survival,
        "per_entry_survival": per_entry_survival,
    }


def optimal_multi_entry(
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
    """Find optimal picks for N entries using exact EV with greedy + swap.

    Much faster than MC-based portfolio optimization since each evaluation
    is O(T) instead of O(sims * T).

    Args:
        n_entries: Number of entries
        available_teams: team_id -> seed
        win_probs: team_id -> P(win)
        ownership: team_id -> ownership fraction
        pool_size: Total pool size
        prize_pool: Prize money
        used_teams_per_entry: Already-used teams per entry
        min_win_prob: Minimum win probability threshold
        future_values: team_id -> future round value (penalty for using now)
    """
    if used_teams_per_entry is None:
        used_teams_per_entry = [set() for _ in range(n_entries)]
    if future_values is None:
        future_values = {}

    # Filter viable teams
    viable = [t for t in available_teams if win_probs.get(t, 0) >= min_win_prob]
    if not viable:
        viable = list(available_teams.keys())

    # Score each team: EV adjusted by future value
    team_scores = {}
    for t in viable:
        ev = exact_pick_ev(t, win_probs, ownership, pool_size, prize_pool, n_entries)
        fv = future_values.get(t, 0.0)
        team_scores[t] = ev - fv  # penalize teams with high future value

    # Greedy assignment
    picks = []
    used_this_round = set()

    for entry_idx in range(n_entries):
        used = used_teams_per_entry[entry_idx]
        best_team = None
        best_score = -float("inf")

        for t in viable:
            if t in used or t in used_this_round:
                continue
            score = team_scores.get(t, 0)
            if score > best_score:
                best_score = score
                best_team = t

        if best_team is None:
            # Relax: allow duplicate picks across entries
            for t in viable:
                if t in used:
                    continue
                score = team_scores.get(t, 0)
                if best_team is None or score > best_score:
                    best_score = score
                    best_team = t

        if best_team is None:
            best_team = viable[0]

        picks.append(best_team)
        used_this_round.add(best_team)

    # Local search: pairwise swaps
    improved = True
    max_iters = 20
    iters = 0

    while improved and iters < max_iters:
        improved = False
        iters += 1
        current_result = exact_round_ev(picks, win_probs, ownership, pool_size, prize_pool)
        current_total = current_result["total_ev"]

        for e_idx in range(n_entries):
            used = used_teams_per_entry[e_idx]
            for t in viable:
                if t in used or t == picks[e_idx]:
                    continue
                trial = picks.copy()
                trial[e_idx] = t
                trial_result = exact_round_ev(trial, win_probs, ownership, pool_size, prize_pool)
                if trial_result["total_ev"] > current_total * 1.001:
                    picks[e_idx] = t
                    current_total = trial_result["total_ev"]
                    improved = True

    return picks
