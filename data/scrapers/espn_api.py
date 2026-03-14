"""Fetch live bracket and game data from ESPN's public API."""

import requests
import pandas as pd
from datetime import datetime


ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
)
ESPN_TEAMS_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams"
)


def fetch_scoreboard(date: str | None = None, groups: int = 100) -> dict:
    """Fetch scoreboard data from ESPN.

    Args:
        date: Date string in YYYYMMDD format, or None for today
        groups: 100 for tournament games
    """
    params = {"groups": groups, "limit": 50}
    if date:
        params["dates"] = date

    resp = requests.get(ESPN_SCOREBOARD_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def parse_games(scoreboard_data: dict) -> list[dict]:
    """Parse ESPN scoreboard JSON into game dicts."""
    games = []
    for event in scoreboard_data.get("events", []):
        game = {
            "game_id": event["id"],
            "name": event.get("name", ""),
            "date": event.get("date", ""),
            "status": event["status"]["type"]["name"],
            "completed": event["status"]["type"]["completed"],
        }

        competitors = event.get("competitions", [{}])[0].get("competitors", [])
        for comp in competitors:
            prefix = "home" if comp["homeAway"] == "home" else "away"
            game[f"{prefix}_id"] = int(comp["team"]["id"])
            game[f"{prefix}_name"] = comp["team"].get("displayName", "")
            game[f"{prefix}_seed"] = int(comp.get("curatedRank", {}).get("current", 0))
            game[f"{prefix}_score"] = int(comp.get("score", 0))
            game[f"{prefix}_winner"] = comp.get("winner", False)

        games.append(game)

    return games


def fetch_tournament_games(dates: list[str]) -> pd.DataFrame:
    """Fetch all tournament games across given dates.

    Args:
        dates: List of date strings in YYYYMMDD format
    """
    all_games = []
    for date in dates:
        try:
            data = fetch_scoreboard(date=date)
            all_games.extend(parse_games(data))
        except Exception as e:
            print(f"Error fetching games for {date}: {e}")
            continue

    return pd.DataFrame(all_games)


def get_current_bracket(year: int = 2026) -> pd.DataFrame:
    """Fetch all tournament games for the given year.

    Returns DataFrame with game details including matchups, seeds, and results.
    """
    # Tournament dates for 2026
    tournament_dates = [
        "20260319", "20260320",  # Round of 64
        "20260321", "20260322",  # Round of 32
        "20260326", "20260327",  # Sweet 16
        "20260330",              # Elite 8
        "20260404",              # Final Four
        "20260406",              # Championship
    ]

    return fetch_tournament_games(tournament_dates)
