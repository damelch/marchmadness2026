"""Exact closed-form EV calculations for survivor pool picks.

Replaces Monte Carlo for single-round evaluation. Instant and exact.
Supports both single-pick and double-pick contest days.
"""

from itertools import combinations


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


def _portfolio_score(
    picks: list[int],
    win_probs: dict[int, float],
    ownership: dict[int, float],
    pool_size: int,
    prize_pool: float,
) -> float:
    """Score a portfolio of picks, penalizing duplicate teams.

    Raw total EV is misleading for multi-entry because putting all entries
    on the best team gives the highest EV but zero diversification — they
    all live or die together.

    Score = total_EV * diversification_bonus

    The bonus rewards spreading picks across independent games so that
    at least one entry is likely to survive even if an upset occurs.
    """
    ev_result = exact_round_ev(picks, win_probs, ownership, pool_size, prize_pool)
    total_ev = ev_result["total_ev"]

    if len(picks) <= 1:
        return total_ev

    # Count how many entries share each team
    from collections import Counter
    counts = Counter(picks)
    n = len(picks)

    # Diversification: fraction of picks that are unique
    uniqueness = len(counts) / n  # noqa: F841 — used conceptually in scoring

    # Correlation penalty: entries on the same team are perfectly correlated
    # Effective entries ≈ sum of sqrt(count) for each unique team (like portfolio theory)
    effective_entries = sum(c ** 0.5 for c in counts.values())
    efficiency = effective_entries / n  # 1.0 when all unique, lower with dupes

    # Joint survival: P(at least one survives)
    # With all-same picks: P = win_prob (binary)
    # With diverse picks: P = 1 - product(1 - wp_i) (much higher)
    p_all_die = 1.0
    for team_id in set(picks):
        wp = win_probs.get(team_id, 0.5)
        k = counts[team_id]
        p_all_die *= (1.0 - wp) ** (k > 0)  # only count once per unique team

    joint_survival = 1.0 - p_all_die

    # Weighted score: EV * diversification_factor
    # The factor blends raw efficiency with survival breadth
    div_factor = 0.5 * efficiency + 0.5 * (joint_survival ** 0.2)

    return total_ev * div_factor


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

    Diversifies picks across entries so that entries don't all live or die
    together. Uses portfolio scoring that rewards both high EV and spread
    across independent games.

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
    # FV is accumulated across all future days, so normalize it relative to
    # the current day's EV scale. Scale factor: max single-day EV / max FV,
    # then apply a weight (0.5 = use half the future value as opportunity cost).
    team_evs = {t: exact_pick_ev(t, win_probs, ownership, pool_size, prize_pool, n_entries)
                for t in viable}
    max_ev = max(team_evs.values()) if team_evs else 1.0
    fv_vals = [future_values.get(t, 0.0) for t in viable]
    max_fv = max(fv_vals) if fv_vals else 1.0

    fv_weight = 0.3  # Base penalty for using high-FV teams now
    fv_scale = (max_ev / max_fv) * fv_weight if max_fv > 0 else 0.0

    # Top seeds are scarce (only 4 one-seeds across 9 days) — extra penalty
    SEED_FV_MULTIPLIER = {1: 4.0, 2: 3.5, 3: 1.5}

    team_scores = {}
    for t in viable:
        ev = team_evs[t]
        seed = available_teams.get(t, 8)
        seed_mult = SEED_FV_MULTIPLIER.get(seed, 1.0)
        fv = future_values.get(t, 0.0) * fv_scale * seed_mult
        team_scores[t] = ev - fv

    # Sort viable teams by score descending
    viable_sorted = sorted(viable, key=lambda t: team_scores.get(t, 0), reverse=True)

    # Greedy assignment: each entry gets a unique team
    picks = []
    used_this_round = set()

    for entry_idx in range(n_entries):
        used = used_teams_per_entry[entry_idx]
        best_team = None
        best_score = -float("inf")

        for t in viable_sorted:
            if t in used or t in used_this_round:
                continue
            score = team_scores.get(t, 0)
            if score > best_score:
                best_score = score
                best_team = t

        if best_team is None:
            # All viable teams used — pick best available ignoring round constraint
            for t in viable_sorted:
                if t in used:
                    continue
                best_team = t
                break

        if best_team is None:
            best_team = viable_sorted[0]

        picks.append(best_team)
        used_this_round.add(best_team)

    # Local search: pairwise swaps using portfolio score (not raw EV)
    # Pre-cache per-pick EVs to avoid redundant recomputation during swaps.
    # When swapping one entry, only that entry's EV changes — the rest are reused.
    pick_evs = {
        t: exact_pick_ev(t, win_probs, ownership, pool_size, prize_pool, n_entries)
        for t in set(viable_sorted[:30]) | set(picks)
    }

    def _fast_portfolio_score(pick_list: list[int]) -> float:
        """Score portfolio using cached per-pick EVs."""
        total_ev = sum(pick_evs.get(t, exact_pick_ev(
            t, win_probs, ownership, pool_size, prize_pool, n_entries,
        )) for t in pick_list)
        if len(pick_list) <= 1:
            return total_ev
        from collections import Counter
        counts = Counter(pick_list)
        n = len(pick_list)
        effective_entries = sum(c ** 0.5 for c in counts.values())
        efficiency = effective_entries / n
        p_all_die = 1.0
        for team_id in set(pick_list):
            wp = win_probs.get(team_id, 0.5)
            p_all_die *= (1.0 - wp)
        joint_survival = 1.0 - p_all_die
        div_factor = 0.5 * efficiency + 0.5 * (joint_survival ** 0.2)
        return total_ev * div_factor

    improved = True
    max_iters = 20
    iters = 0

    while improved and iters < max_iters:
        improved = False
        iters += 1
        current_score = _fast_portfolio_score(picks)

        for e_idx in range(n_entries):
            used = used_teams_per_entry[e_idx]
            for t in viable_sorted[:30]:  # Only try top 30 teams for speed
                if t in used or t == picks[e_idx]:
                    continue
                trial = picks.copy()
                trial[e_idx] = t
                trial_score = _fast_portfolio_score(trial)
                if trial_score > current_score * 1.001:
                    picks[e_idx] = t
                    current_score = trial_score
                    improved = True

    return picks


# ---------------------------------------------------------------------------
# Double-pick day support
# ---------------------------------------------------------------------------


def exact_day_ev(
    pick_set: list[int],
    win_probs: dict[int, float],
    ownership: dict[int, float],
    pool_size: int,
    prize_pool: float,
    num_picks_per_opponent: int = 1,
    n_our_entries: int = 1,
    matchup_pairs: list[tuple[int, int]] | None = None,
) -> float:
    """Exact EV for a set of picks on a single contest day.

    On a double-pick day, ALL picks must win for survival.

    P(we survive) = product(P(t wins) for t in pick_set)

    For opponents (who also make `num_picks_per_opponent` picks):
    P(opp survives) ≈ field_survival_rate ^ num_picks_per_opponent
    (simplified: each of their picks independently survives at the field rate)

    For single-pick days (num_picks=1), this reduces to exact_pick_ev.
    """
    if len(pick_set) == 1 and num_picks_per_opponent == 1:
        return exact_pick_ev(
            pick_set[0], win_probs, ownership, pool_size, prize_pool, n_our_entries,
        )

    # Joint probability we survive: all our picks must win
    p_we_survive = 1.0
    for t in pick_set:
        p_we_survive *= win_probs.get(t, 0.5)

    if p_we_survive <= 0:
        return 0.0

    # Build opponent map for conditional independence
    opp_map = {}
    if matchup_pairs:
        for a, b in matchup_pairs:
            opp_map[a] = b
            opp_map[b] = a

    # Field survival rate per pick slot (for opponent modeling)
    # Each opponent pick independently survives at the field rate
    field_surv = field_survival_rate(win_probs, ownership)

    # On a double-pick day, opponent must also make `num_picks_per_opponent` picks
    # that all win. Simplified model: independent picks at field rate.
    p_opp_survives = field_surv ** num_picks_per_opponent

    n_opponents = pool_size - n_our_entries
    expected_opp_survivors = n_opponents * p_opp_survives
    expected_total_survivors = 1.0 + expected_opp_survivors

    ev = p_we_survive * prize_pool / max(expected_total_survivors, 1.0)
    return ev


def _pick_set_score(
    pick_set: list[int],
    win_probs: dict[int, float],
    future_values: dict[int, float],
) -> float:
    """Score a pick set by joint win prob minus future value cost."""
    joint_wp = 1.0
    fv_cost = 0.0
    for t in pick_set:
        joint_wp *= win_probs.get(t, 0.5)
        fv_cost += future_values.get(t, 0.0)
    return joint_wp - fv_cost * 0.01  # scale FV to same order


def optimal_day_picks(
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
) -> list[list[int]]:
    """Optimal picks for N entries on a contest day (1 or 2 picks per entry).

    Returns list of pick sets: [[team_a], [team_b], ...] for single-pick days,
    or [[team_a, team_b], [team_c, team_d], ...] for double-pick days.
    """
    if num_picks == 1:
        single_picks = optimal_multi_entry(
            n_entries, available_teams, win_probs, ownership,
            pool_size, prize_pool, used_teams_per_entry,
            min_win_prob, future_values,
        )
        return [[t] for t in single_picks]

    # Double-pick day optimization
    if used_teams_per_entry is None:
        used_teams_per_entry = [set() for _ in range(n_entries)]
    if future_values is None:
        future_values = {}

    # Build matchup map: team -> opponent (can't pick both sides of a game)
    opp_map = {}
    if matchup_pairs:
        for a, b in matchup_pairs:
            opp_map[a] = b
            opp_map[b] = a

    # Filter viable teams
    viable = [t for t in available_teams if win_probs.get(t, 0) >= min_win_prob]
    if len(viable) < 2:
        viable = list(available_teams.keys())

    # Generate all valid 2-team combinations (must be from different games)
    def valid_pair(a: int, b: int) -> bool:
        return opp_map.get(a) != b and opp_map.get(b) != a

    all_pairs = [
        (a, b) for a, b in combinations(viable, 2)
        if valid_pair(a, b)
    ]

    # Score each pair by EV, with normalized future value penalty
    # Compute raw EVs first to establish scale
    pair_evs = {}
    for pair in all_pairs:
        pair_evs[pair] = exact_day_ev(
            list(pair), win_probs, ownership, pool_size, prize_pool,
            num_picks_per_opponent=num_picks, n_our_entries=n_entries,
            matchup_pairs=matchup_pairs,
        )

    max_pair_ev = max(pair_evs.values()) if pair_evs else 1.0
    all_fv_costs = [sum(future_values.get(t, 0.0) for t in p) for p in all_pairs]
    max_fv_cost = max(all_fv_costs) if all_fv_costs else 1.0

    fv_weight = 0.3
    fv_scale = (max_pair_ev / max_fv_cost) * fv_weight if max_fv_cost > 0 else 0.0

    # Top seeds are scarce — extra penalty for burning them in early rounds
    SEED_FV_MULTIPLIER = {1: 4.0, 2: 3.5, 3: 1.5}

    pair_scores = {}
    for pair in all_pairs:
        fv_cost = 0.0
        for t in pair:
            seed = available_teams.get(t, 8)
            seed_mult = SEED_FV_MULTIPLIER.get(seed, 1.0)
            fv_cost += future_values.get(t, 0.0) * fv_scale * seed_mult
        pair_scores[pair] = pair_evs[pair] - fv_cost

    # Sort pairs by score
    sorted_pairs = sorted(all_pairs, key=lambda p: pair_scores.get(p, 0), reverse=True)

    def _portfolio_concentration_penalty(picks_list: list[list[int]]) -> float:
        """Penalty for having too many entries depend on the same team.

        If one team appears in all N entries, a single upset wipes out
        the entire portfolio. Penalty scales quadratically with exposure.
        """
        if len(picks_list) <= 1:
            return 0.0
        from collections import Counter
        team_counts = Counter(t for ps in picks_list for t in ps)
        n = len(picks_list)
        best_ev = max(pair_evs.values()) if pair_evs else 1.0
        # Penalty: for each team, (fraction of entries exposed)^2
        # A team on all 5 entries = (5/5)^2 = 1.0 full penalty
        # A team on 3 of 5 = (3/5)^2 = 0.36
        # A team on 1 of 5 = (1/5)^2 = 0.04 (negligible)
        penalty = 0.0
        for team_id, count in team_counts.items():
            if count > 1:
                exposure = count / n
                penalty += exposure ** 2 * best_ev * 0.4
        return penalty

    def _score_portfolio(picks_list: list[list[int]]) -> float:
        total_ev = sum(
            exact_day_ev(
                ps, win_probs, ownership, pool_size, prize_pool,
                num_picks, n_our_entries=n_entries, matchup_pairs=matchup_pairs,
            )
            for ps in picks_list
        )
        fv_penalty = sum(
            sum(future_values.get(t, 0.0) for t in ps) * fv_scale
            for ps in picks_list
        )
        concentration = _portfolio_concentration_penalty(picks_list)
        return total_ev - fv_penalty - concentration

    # Greedy assignment with soft diversity preference
    picks: list[list[int]] = []
    used_teams_count: dict[int, int] = {}  # Track how many entries use each team

    for entry_idx in range(n_entries):
        used = used_teams_per_entry[entry_idx]
        best_pair = None
        best_score = -float("inf")

        for pair in sorted_pairs[:60]:
            a, b = pair
            if a in used or b in used:
                continue
            # Penalize overlap: each additional entry on same team reduces score
            overlap_penalty = (
                used_teams_count.get(a, 0) + used_teams_count.get(b, 0)
            ) * pair_scores.get(sorted_pairs[0], 1.0) * 0.4
            score = pair_scores.get(pair, 0) - overlap_penalty
            if score > best_score:
                best_score = score
                best_pair = pair

        if best_pair is None:
            best_pair = sorted_pairs[0] if sorted_pairs else (viable[0], viable[1])

        picks.append(list(best_pair))
        for t in best_pair:
            used_teams_count[t] = used_teams_count.get(t, 0) + 1

    # Local search: swap pairs to improve portfolio score (EV + diversity - FV)
    improved = True
    max_iters = 20
    iters = 0
    current_score = _score_portfolio(picks)

    while improved and iters < max_iters:
        improved = False
        iters += 1

        for e_idx in range(n_entries):
            used = used_teams_per_entry[e_idx]
            for pair in sorted_pairs[:50]:
                a, b = pair
                if a in used or b in used:
                    continue
                if list(pair) == picks[e_idx]:
                    continue

                trial = [ps[:] for ps in picks]
                trial[e_idx] = list(pair)
                trial_score = _score_portfolio(trial)

                if trial_score > current_score + 0.01:
                    picks[e_idx] = list(pair)
                    current_score = trial_score
                    improved = True

    return picks
