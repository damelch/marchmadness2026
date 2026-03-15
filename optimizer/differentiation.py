"""Approach A: Greedy leverage-based pick optimization with differentiation."""

from optimizer.survival import leverage_score
from simulation.engine import TournamentBracket


def rank_picks_by_leverage(
    available_teams: dict[int, int],
    win_probs: dict[int, float],
    ownership: dict[int, float],
) -> list[dict]:
    """Rank all available teams by leverage score.

    Returns list of dicts sorted by leverage (descending):
        [{team_id, seed, win_prob, ownership, leverage}, ...]
    """
    # Field survival rate: expected fraction of opponents surviving this round
    field_survival = sum(ownership.get(t, 0) * win_probs.get(t, 0.5) for t in available_teams)

    ranked = []
    for team_id, seed in available_teams.items():
        wp = win_probs.get(team_id, 0.5)
        own = ownership.get(team_id, 1 / len(available_teams))
        lev = leverage_score(wp, own, field_survival)

        ranked.append({
            "team_id": team_id,
            "seed": seed,
            "win_prob": wp,
            "ownership": own,
            "leverage": lev,
        })

    ranked.sort(key=lambda x: x["leverage"], reverse=True)
    return ranked


def optimize_single_entry(
    available_teams: dict[int, int],
    win_probs: dict[int, float],
    ownership: dict[int, float],
    min_win_prob: float = 0.3,
) -> int:
    """Select the optimal pick for a single entry.

    Filters out teams below min_win_prob threshold, then picks highest leverage.
    """
    filtered = {t: s for t, s in available_teams.items() if win_probs.get(t, 0) >= min_win_prob}
    if not filtered:
        filtered = available_teams  # fallback: don't filter

    ranked = rank_picks_by_leverage(filtered, win_probs, ownership)
    return ranked[0]["team_id"]


def optimize_multi_entry(
    n_entries: int,
    available_teams_per_entry: list[dict[int, int]],
    win_probs: dict[int, float],
    ownership: dict[int, float],
    bracket: TournamentBracket | None = None,
    min_win_prob: float = 0.3,
) -> list[int]:
    """Assign diversified picks across multiple entries.

    Algorithm:
    1. Pick highest leverage team for entry 1
    2. For each subsequent entry, prefer teams from different regions
       or with negatively correlated outcomes
    3. Never assign the same team to multiple entries in the same round
       (that would waste differentiation)

    Args:
        n_entries: Number of entries to assign picks for
        available_teams_per_entry: Per-entry available teams (respecting reuse constraints)
        win_probs: team_id -> P(win)
        ownership: team_id -> ownership fraction
        bracket: Optional bracket for region-based diversification
        min_win_prob: Minimum acceptable win probability

    Returns:
        List of team_ids, one per entry
    """
    picks = []
    used_teams = set()
    used_regions = set()

    for entry_idx in range(n_entries):
        available = available_teams_per_entry[entry_idx]
        filtered = {
            t: s for t, s in available.items()
            if win_probs.get(t, 0) >= min_win_prob and t not in used_teams
        }

        if not filtered:
            # Relax constraints: allow same team as another entry
            filtered = {
                t: s for t, s in available.items()
                if win_probs.get(t, 0) >= min_win_prob
            }
        if not filtered:
            filtered = available

        ranked = rank_picks_by_leverage(filtered, win_probs, ownership)

        # Apply diversification bonus: prefer teams from unused regions
        if bracket is not None and used_regions:
            for pick in ranked:
                team_info = bracket.teams.get(pick["team_id"], {})
                region = team_info.get("region", "")
                if region and region not in used_regions:
                    # Boost leverage by 20% for unused regions
                    pick["leverage"] *= 1.2

            ranked.sort(key=lambda x: x["leverage"], reverse=True)

        best = ranked[0]
        picks.append(best["team_id"])
        used_teams.add(best["team_id"])

        if bracket is not None:
            region = bracket.teams.get(best["team_id"], {}).get("region", "")
            if region:
                used_regions.add(region)

    return picks


def generate_differentiation_report(
    picks: list[int],
    win_probs: dict[int, float],
    ownership: dict[int, float],
    bracket: TournamentBracket,
) -> str:
    """Generate a human-readable report of pick diversification."""
    lines = ["\nDifferentiation Report", "=" * 40]

    for i, team_id in enumerate(picks):
        info = bracket.teams.get(team_id, {})
        wp = win_probs.get(team_id, 0)
        own = ownership.get(team_id, 0)

        lines.append(
            f"Entry {i+1}: ({info.get('seed', '?')}) {info.get('name', team_id)} "
            f"| WinProb={wp:.1%} | Own={own:.1%} | Region={info.get('region', '?')}"
        )

    # Correlation analysis
    regions = [bracket.teams.get(t, {}).get("region", "") for t in picks]
    unique_regions = len(set(r for r in regions if r))
    lines.append(f"\nRegion coverage: {unique_regions}/{len(picks)} entries in unique regions")

    # Expected at least one survivor
    survival_probs = [win_probs.get(t, 0.5) for t in picks]
    p_all_die = 1.0
    for p in survival_probs:
        p_all_die *= (1 - p)
    p_at_least_one = 1 - p_all_die
    lines.append(f"P(at least one survives this round): {p_at_least_one:.1%}")

    return "\n".join(lines)
