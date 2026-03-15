"""Fetch live tournament bracket state from ESPN and resolve into our bracket model."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from contest.schedule import ContestSchedule
from simulation.engine import TournamentBracket

# ESPN names that differ from our bracket.json names.
# Format: espn_display_name_lower -> bracket.json name
_NAME_ALIASES: dict[str, str] = {
    "uconn": "Connecticut",
    "iowa state": "Iowa St.",
    "iowa state cyclones": "Iowa St.",
    "michigan state": "Michigan St.",
    "michigan state spartans": "Michigan St.",
    "ohio state": "Ohio St.",
    "ohio state buckeyes": "Ohio St.",
    "utah state": "Utah St.",
    "portland state": "Portland St.",
    "tennessee state": "Tennessee St.",
    "wright state": "Wright St.",
    "north dakota state": "North Dakota St.",
    "saint john's": "St. John's",
    "st. john's red storm": "St. John's",
    "miami hurricanes": "Miami",
    "miami (fl)": "Miami",
    "miami fl": "Miami",
    "miami redhawks": "Miami (OH)",
    "miami ohio": "Miami (OH)",
    "uc irvine anteaters": "UC Irvine",
    "south florida bulls": "South Florida",
    "usf": "South Florida",
    "sf austin": "Stephen F. Austin",
    "sfa": "Stephen F. Austin",
    "smu mustangs": "SMU",
    "byu cougars": "BYU",
    "tcu horned frogs": "TCU",
    "umbc retrievers": "UMBC",
    "st. louis billikens": "St. Louis",
    "saint louis": "St. Louis",
    "queens royals": "Queens",
    "queens (nc)": "Queens",
}

# NCAA region names used by ESPN → our abstract region codes.
# This mapping depends on the actual bracket assignment each year.
# Update this dict when the bracket is released.
NCAA_REGION_MAP: dict[str, str] = {
    "West": "W",
    "East": "X",
    "South": "Y",
    "Midwest": "Z",
}


@dataclass
class LiveBracketState:
    """Current state of the real tournament bracket."""

    bracket: TournamentBracket
    alive_team_ids: set[int] = field(default_factory=set)
    eliminated_team_ids: set[int] = field(default_factory=set)
    current_round: int = 1
    current_day: int = 1
    games_completed: int = 0
    games_total: int = 63
    unmatched_teams: list[str] = field(default_factory=list)


def _normalize_name(name: str) -> str:
    """Normalize a team name for matching."""
    s = name.strip().lower()
    # Remove common suffixes like "Wildcats", "Tigers", etc.
    # but keep core identifiers
    s = re.sub(r"\s+(university|univ\.?)$", "", s)
    return s


def _build_name_index(bracket: TournamentBracket) -> dict[str, int]:
    """Build multiple name → team_id lookup keys from bracket.teams."""
    index: dict[str, int] = {}
    for team_id, info in bracket.teams.items():
        name = info["name"]
        # Exact lowercase
        index[name.lower()] = team_id
        # Expand "St." → "State"
        if "St." in name:
            expanded = name.replace("St.", "State")
            index[expanded.lower()] = team_id
        # Also store without periods
        no_dots = name.replace(".", "")
        index[no_dots.lower()] = team_id
    return index


def match_team_name(
    espn_name: str,
    name_index: dict[str, int],
    bracket: TournamentBracket,
) -> int | None:
    """Match an ESPN team name to a bracket team ID.

    Uses three tiers: exact match, alias table, fuzzy match.
    """
    normalized = _normalize_name(espn_name)

    # Tier 1: exact match against name index
    if normalized in name_index:
        return name_index[normalized]

    # Tier 2: alias table
    if normalized in _NAME_ALIASES:
        alias_name = _NAME_ALIASES[normalized]
        if alias_name.lower() in name_index:
            return name_index[alias_name.lower()]

    # Tier 3: fuzzy match with difflib
    all_bracket_names = [info["name"] for info in bracket.teams.values()]
    matches = difflib.get_close_matches(espn_name, all_bracket_names, n=1, cutoff=0.75)
    if matches:
        matched_name = matches[0]
        for team_id, info in bracket.teams.items():
            if info["name"] == matched_name:
                return team_id

    return None


def fetch_live_state(
    bracket: TournamentBracket,
    schedule: ContestSchedule | None = None,
) -> LiveBracketState | None:
    """Fetch live bracket state from ESPN and resolve results.

    Returns None if the ESPN API is unavailable.
    """
    try:
        from data.scrapers.espn_api import get_current_bracket
    except ImportError:
        return None

    try:
        games_df = get_current_bracket()
    except Exception:
        return None

    if games_df is None or games_df.empty:
        return None

    name_index = _build_name_index(bracket)
    all_team_ids = set(bracket.teams.keys())
    eliminated: set[int] = set()
    unmatched: list[str] = []
    games_completed = 0

    # Process each completed game
    for _, game in games_df.iterrows():
        if not game.get("completed", False):
            continue

        games_completed += 1

        home_name = game.get("home_name", "")
        away_name = game.get("away_name", "")
        home_winner = game.get("home_winner", False)

        winner_name = home_name if home_winner else away_name
        loser_name = away_name if home_winner else home_name

        winner_id = match_team_name(winner_name, name_index, bracket)
        loser_id = match_team_name(loser_name, name_index, bracket)

        if loser_id is not None:
            eliminated.add(loser_id)

        if winner_id is None and winner_name:
            unmatched.append(winner_name)
        if loser_id is None and loser_name:
            unmatched.append(loser_name)

        # Resolve game in bracket to advance winner
        if winner_id is not None and loser_id is not None:
            _resolve_in_bracket(bracket, winner_id, loser_id)

    alive = all_team_ids - eliminated

    # Determine current round from completed games
    current_round = _detect_current_round(games_completed)
    current_day = _detect_current_day(games_completed, schedule)

    return LiveBracketState(
        bracket=bracket,
        alive_team_ids=alive,
        eliminated_team_ids=eliminated,
        current_round=current_round,
        current_day=current_day,
        games_completed=games_completed,
        games_total=63,
        unmatched_teams=list(set(unmatched)),
    )


def _resolve_in_bracket(
    bracket: TournamentBracket, winner_id: int, loser_id: int,
) -> None:
    """Find the slot where these two teams play and resolve it."""
    for i, slot in enumerate(bracket.slots):
        team_set = {slot.team_a, slot.team_b}
        if winner_id in team_set and loser_id in team_set:
            if slot.winner is None:
                bracket.resolve_game(i, winner_id)
            return


def _detect_current_round(games_completed: int) -> int:
    """Estimate current round from number of completed games."""
    if games_completed <= 0:
        return 1
    if games_completed <= 32:
        return 1   # R64
    if games_completed <= 48:
        return 2   # R32
    if games_completed <= 56:
        return 3   # S16
    if games_completed <= 60:
        return 4   # E8
    if games_completed <= 62:
        return 5   # F4
    return 6        # Championship


def _detect_current_day(
    games_completed: int, schedule: ContestSchedule | None,
) -> int:
    """Estimate current contest day from completed games."""
    if schedule is None:
        schedule = ContestSchedule.default()

    # Cumulative games per day (standard bracket):
    # Day 1: 16 games (R64 W+X), Day 2: 16 (R64 Y+Z) = 32 total
    # Day 3: 4 (R32 W+X), Day 4: 4 (R32 Y+Z) = 40 total
    # Day 5: 2 (S16 W+X), Day 6: 2 (S16 Y+Z) = 44 total  -- Wait, that's wrong
    # Actually: S16 has 2 games per region, 2 regions per day = 4 per day
    # Let me compute from the schedule
    cumulative = 0
    games_per_day = []
    for day in schedule.days:
        if day.round_num <= 4:
            games_in_round_per_region = {1: 8, 2: 4, 3: 2, 4: 1}
            n_games = games_in_round_per_region[day.round_num] * len(day.regions)
        elif day.round_num == 5:
            n_games = 2
        else:
            n_games = 1
        cumulative += n_games
        games_per_day.append((day.day_num, cumulative))

    # Find the day we're currently on
    for day_num, cum_games in games_per_day:
        if games_completed < cum_games:
            return day_num

    return games_per_day[-1][0]  # last day


def resolve_team_name(
    name: str, bracket: TournamentBracket,
) -> int | None:
    """Resolve a user-provided team name to a bracket team ID.

    Used for parsing --used flag input.
    """
    name_index = _build_name_index(bracket)
    return match_team_name(name, name_index, bracket)
