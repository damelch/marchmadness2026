"""Coverage guarantee analysis for mid-tournament pick advice.

Computes safety sets, risk levels, and future flexibility.
The key feature is worst-case lookahead: for every possible outcome
of today's games, we check whether the entry will still have at least
one unused alive team on every future contest day. This ensures the
advice guarantees future availability regardless of upsets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from contest.schedule import ContestSchedule
from simulation.engine import TournamentBracket


@dataclass
class FutureDayRisk:
    """Risk assessment for a specific future contest day."""

    day_num: int
    label: str
    picks_needed: int
    worst_case_available: int  # min available across all outcome scenarios
    blocked_scenarios: int     # how many outcome combos leave 0 available
    total_scenarios: int


@dataclass
class CoverageAnalysis:
    """Coverage analysis for a single entry."""

    entry_id: int
    available_teams: list[dict] = field(default_factory=list)
    available_by_region: dict[str, list[dict]] = field(default_factory=dict)
    picks_remaining: int = 0
    risk_level: str = "safe"      # "safe" | "at_risk" | "critical"
    risk_reason: str = ""
    safety_set: list[int] = field(default_factory=list)
    uncovered_matchups: list[tuple[int, int]] = field(default_factory=list)
    future_risks: list[FutureDayRisk] = field(default_factory=list)


def compute_coverage(
    bracket: TournamentBracket,
    alive_teams: set[int],
    used_teams: set[int],
    current_day: int,
    schedule: ContestSchedule,
    win_probs: dict[int, float] | None = None,
) -> CoverageAnalysis:
    """Compute coverage analysis for an entry.

    Includes worst-case lookahead: simulates all possible outcomes of
    today's games and checks that every future day still has available
    teams in every scenario.

    Args:
        bracket: Tournament bracket (possibly with resolved results).
        alive_teams: Set of team IDs still in the tournament.
        used_teams: Set of team IDs already used by this entry.
        current_day: Current contest day number.
        schedule: Contest schedule mapping days to rounds/regions.
        win_probs: Optional win probabilities for ranking; uses seed if absent.

    Returns:
        CoverageAnalysis with available teams, safety set, and risk level.
    """
    if win_probs is None:
        win_probs = {}

    day = schedule.get_day(current_day)
    available_ids = alive_teams - used_teams
    picks_remaining = schedule.total_picks_remaining(current_day)

    # Build available team info list
    available_teams = []
    available_by_region: dict[str, list[dict]] = {}
    for team_id in sorted(available_ids):
        info = bracket.teams.get(team_id, {})
        wp = win_probs.get(team_id, _seed_win_prob(info.get("seed", 8)))
        team_info = {
            "id": team_id,
            "name": info.get("name", str(team_id)),
            "seed": info.get("seed", 0),
            "region": info.get("region", "?"),
            "win_prob": wp,
        }
        available_teams.append(team_info)
        region = info.get("region", "?")
        available_by_region.setdefault(region, []).append(team_info)

    # Sort by win_prob descending within each region
    available_teams.sort(key=lambda t: t["win_prob"], reverse=True)
    for region_teams in available_by_region.values():
        region_teams.sort(key=lambda t: t["win_prob"], reverse=True)

    # Get today's matchups and build safety set
    matchups = bracket.get_day_matchups(day.round_num, day.regions)
    safety_set: list[int] = []
    uncovered: list[tuple[int, int]] = []

    for team_a, team_b, _slot_idx in matchups:
        if team_a is None or team_b is None:
            continue
        if team_a not in alive_teams or team_b not in alive_teams:
            # Game already resolved or bye
            alive_side = (
                team_a if team_a in alive_teams else
                team_b if team_b in alive_teams else None
            )
            if alive_side and alive_side in available_ids:
                safety_set.append(alive_side)
            continue

        a_available = team_a in available_ids
        b_available = team_b in available_ids

        if a_available and b_available:
            # Both available — pick the one with higher win prob
            wp_a = win_probs.get(team_a, _seed_win_prob(
                bracket.teams.get(team_a, {}).get("seed", 8)))
            wp_b = win_probs.get(team_b, _seed_win_prob(
                bracket.teams.get(team_b, {}).get("seed", 8)))
            safety_set.append(team_a if wp_a >= wp_b else team_b)
        elif a_available:
            safety_set.append(team_a)
        elif b_available:
            safety_set.append(team_b)
        else:
            # Neither available — this matchup is uncovered
            uncovered.append((team_a, team_b))

    # Worst-case future lookahead
    future_risks = _compute_future_risks(
        bracket, alive_teams, used_teams, current_day, schedule, matchups,
    )

    # Risk level (now incorporating future lookahead)
    n_available = len(available_ids)
    risk_level, risk_reason = _assess_risk(
        n_available, picks_remaining, uncovered, schedule, current_day,
        bracket, alive_teams, used_teams, future_risks,
    )

    return CoverageAnalysis(
        entry_id=-1,  # caller sets this
        available_teams=available_teams,
        available_by_region=available_by_region,
        picks_remaining=picks_remaining,
        risk_level=risk_level,
        risk_reason=risk_reason,
        safety_set=safety_set,
        uncovered_matchups=uncovered,
        future_risks=future_risks,
    )


def _compute_future_risks(
    bracket: TournamentBracket,
    alive_teams: set[int],
    used_teams: set[int],
    current_day: int,
    schedule: ContestSchedule,
    today_matchups: list[tuple[int | None, int | None, int]],
) -> list[FutureDayRisk]:
    """Simulate all outcomes of today's games and check future availability.

    For each possible combination of winners in today's games, compute
    which teams will be alive after today. Then for each future day,
    check how many unused alive teams the entry has in the relevant regions.
    """
    future_days = schedule.get_remaining_days(current_day)
    if not future_days:
        return []

    # Build list of live matchups (both teams alive) for outcome enumeration
    live_matchups: list[tuple[int, int]] = []
    for team_a, team_b, _slot in today_matchups:
        if (
            team_a is not None and team_b is not None
            and team_a in alive_teams and team_b in alive_teams
        ):
            live_matchups.append((team_a, team_b))

    if not live_matchups:
        return []

    # Cap scenario enumeration to avoid explosion (2^N matchups)
    # With 16 games on R64 day, 2^16 = 65536 — manageable
    # With more, sample worst-case paths instead
    n_matchups = len(live_matchups)
    if n_matchups > 20:
        # Too many to enumerate; fall back to simpler region-based check
        return _future_risks_by_region(
            bracket, alive_teams, used_teams, future_days,
        )

    # Enumerate all 2^N outcome combinations
    # Each matchup has 2 outcomes: team_a wins or team_b wins
    # outcome[i] = 0 means live_matchups[i][0] wins, 1 means [1] wins
    risks: list[FutureDayRisk] = []

    for future_day in future_days:
        day_regions = set(future_day.regions)
        worst_available = float("inf")
        blocked = 0
        total = 0

        for outcomes in product(range(2), repeat=n_matchups):
            total += 1
            # Compute alive teams after this outcome
            eliminated_this_round: set[int] = set()
            for i, outcome in enumerate(outcomes):
                loser = live_matchups[i][1 - outcome]  # the one who didn't win
                eliminated_this_round.add(loser)

            future_alive = alive_teams - eliminated_this_round
            future_available = future_alive - used_teams

            # Count available in the relevant regions for this future day
            if day_regions == {"FF"}:
                # Final Four / Championship: any alive unused team works
                n_avail = len(future_available)
            else:
                n_avail = sum(
                    1 for t in future_available
                    if bracket.teams.get(t, {}).get("region", "") in day_regions
                )

            if n_avail < worst_available:
                worst_available = n_avail
            if n_avail < future_day.num_picks:
                blocked += 1

        risks.append(FutureDayRisk(
            day_num=future_day.day_num,
            label=future_day.label,
            picks_needed=future_day.num_picks,
            worst_case_available=int(worst_available),
            blocked_scenarios=blocked,
            total_scenarios=total,
        ))

    return risks


def _future_risks_by_region(
    bracket: TournamentBracket,
    alive_teams: set[int],
    used_teams: set[int],
    future_days: list,
) -> list[FutureDayRisk]:
    """Simplified future risk when too many matchups to enumerate."""
    available_ids = alive_teams - used_teams
    risks = []
    for future_day in future_days:
        day_regions = set(future_day.regions)
        if day_regions == {"FF"}:
            n_avail = len(available_ids)
        else:
            n_avail = sum(
                1 for t in available_ids
                if bracket.teams.get(t, {}).get("region", "") in day_regions
            )
        # Worst case: each matchup could halve available teams
        # Rough estimate: divide by 2 for each round between now and then
        risks.append(FutureDayRisk(
            day_num=future_day.day_num,
            label=future_day.label,
            picks_needed=future_day.num_picks,
            worst_case_available=max(0, n_avail // 2),
            blocked_scenarios=-1,  # -1 means "estimated, not enumerated"
            total_scenarios=-1,
        ))
    return risks


def _assess_risk(
    n_available: int,
    picks_remaining: int,
    uncovered: list[tuple[int, int]],
    schedule: ContestSchedule,
    current_day: int,
    bracket: TournamentBracket,
    alive_teams: set[int],
    used_teams: set[int],
    future_risks: list[FutureDayRisk],
) -> tuple[str, str]:
    """Determine risk level and reason."""
    if n_available == 0:
        return "critical", "no available teams remaining"

    if uncovered:
        names = []
        for a, b in uncovered:
            na = bracket.teams.get(a, {}).get("name", str(a))
            nb = bracket.teams.get(b, {}).get("name", str(b))
            names.append(f"{na} vs {nb}")
        return "critical", f"uncovered matchups: {', '.join(names)}"

    if n_available < picks_remaining:
        return "critical", f"{n_available} available < {picks_remaining} picks remaining"

    # Check future lookahead results
    for risk in future_risks:
        if risk.worst_case_available < risk.picks_needed:
            pct = ""
            if risk.total_scenarios > 0:
                pct = f" ({risk.blocked_scenarios}/{risk.total_scenarios} scenarios blocked)"
            return "critical", (
                f"Day {risk.day_num} ({risk.label}): worst case only "
                f"{risk.worst_case_available} available, need {risk.picks_needed}{pct}"
            )

    # Check if any future day is tight (worst case barely enough)
    for risk in future_risks:
        if 0 < risk.worst_case_available < 2 * risk.picks_needed:
            return "at_risk", (
                f"Day {risk.day_num} ({risk.label}): worst case "
                f"{risk.worst_case_available} available for {risk.picks_needed} picks"
            )

    # Simple count check
    if n_available < 2 * picks_remaining:
        return "at_risk", (
            f"tight: {n_available} available for {picks_remaining} picks remaining"
        )

    return "safe", f"{n_available} available, {picks_remaining} picks remaining"


def _seed_win_prob(seed: int) -> float:
    """Rough win probability estimate from seed (for when model isn't available)."""
    # Approximate first-round win rates by seed
    seed_probs = {
        1: 0.95, 2: 0.88, 3: 0.83, 4: 0.78, 5: 0.65, 6: 0.63,
        7: 0.60, 8: 0.50, 9: 0.50, 10: 0.40, 11: 0.37, 12: 0.35,
        13: 0.22, 14: 0.17, 15: 0.12, 16: 0.05,
    }
    return seed_probs.get(seed, 0.5)
