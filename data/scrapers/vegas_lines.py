"""Fetch Vegas betting lines for NCAA tournament games.

Uses The Odds API to retrieve current spreads, totals, and moneylines
for NCAAB games.  Historical lines can be loaded from a local CSV for
training features.

Requires an API key from https://the-odds-api.com (free tier available).
Set the ODDS_API_KEY environment variable or pass it directly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

ODDS_API_BASE_URL = (
    "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _american_to_implied_prob(odds: int | float) -> float:
    """Convert American odds to implied probability.

    Args:
        odds: American-format odds (e.g., -150, +130)

    Returns:
        Implied probability as a float between 0 and 1.
        Returns 0.0 for invalid input.
    """
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return 0.0

    if odds == 0:
        return 0.0
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def _safe_float(val, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Current odds (live API)
# ---------------------------------------------------------------------------

def fetch_current_odds(
    api_key: str | None = None,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch current NCAAB odds from The Odds API.

    Args:
        api_key: The Odds API key.  Falls back to the ``ODDS_API_KEY``
                 environment variable when *None*.
        save_path: If provided, save the fetched data as CSV.

    Returns:
        DataFrame with columns: HomeTeam, AwayTeam, Spread, Total,
        HomeML, AwayML, ImpliedProbHome, ImpliedProbAway, Bookmaker.
        The last row per game is a ``consensus`` entry that averages
        all bookmakers.  Returns an empty DataFrame on failure.
    """
    key = api_key or os.environ.get("ODDS_API_KEY")
    if not key:
        print("Warning: No API key provided. Set ODDS_API_KEY or pass api_key.")
        return pd.DataFrame()

    params = {
        "apiKey": key,
        "regions": "us",
        "markets": "spreads,totals,h2h",
        "oddsFormat": "american",
    }

    try:
        resp = requests.get(ODDS_API_BASE_URL, params=params, timeout=20)
        if resp.status_code != 200:
            print(
                f"Warning: Odds API returned status {resp.status_code}: "
                f"{resp.text[:200]}"
            )
            return pd.DataFrame()

        events = resp.json()
        if not events:
            print("Warning: Odds API returned no events.")
            return pd.DataFrame()

    except Exception as e:
        print(f"Warning: Failed to fetch odds — {e}")
        return pd.DataFrame()

    rows = _parse_odds_events(events)
    if not rows:
        print("Warning: Could not parse any odds from API response.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print(f"Fetched odds for {df['HomeTeam'].nunique()} games "
          f"from {df['Bookmaker'].nunique()} bookmakers")

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Saved odds data to {path}")

    return df


def _parse_odds_events(events: list[dict]) -> list[dict]:
    """Parse The Odds API events list into flat row dicts."""
    rows: list[dict] = []

    for event in events:
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        bookmakers = event.get("bookmakers", [])

        game_rows: list[dict] = []

        for bm in bookmakers:
            bookmaker_name = bm.get("title", "unknown")
            spread = 0.0
            total = 0.0
            home_ml = 0.0
            away_ml = 0.0

            for market in bm.get("markets", []):
                market_key = market.get("key", "")
                outcomes = market.get("outcomes", [])

                if market_key == "spreads":
                    for oc in outcomes:
                        if oc.get("name") == home_team:
                            spread = _safe_float(oc.get("point"))
                        # Away spread is just the inverse; we only store home

                elif market_key == "totals":
                    for oc in outcomes:
                        if oc.get("name") == "Over":
                            total = _safe_float(oc.get("point"))

                elif market_key == "h2h":
                    for oc in outcomes:
                        if oc.get("name") == home_team:
                            home_ml = _safe_float(oc.get("price"))
                        elif oc.get("name") == away_team:
                            away_ml = _safe_float(oc.get("price"))

            row = {
                "HomeTeam": home_team,
                "AwayTeam": away_team,
                "Spread": spread,
                "Total": total,
                "HomeML": home_ml,
                "AwayML": away_ml,
                "ImpliedProbHome": _american_to_implied_prob(home_ml),
                "ImpliedProbAway": _american_to_implied_prob(away_ml),
                "Bookmaker": bookmaker_name,
            }
            game_rows.append(row)
            rows.append(row)

        # Consensus row: average across all bookmakers for this game
        if game_rows:
            consensus = {
                "HomeTeam": home_team,
                "AwayTeam": away_team,
                "Spread": sum(r["Spread"] for r in game_rows) / len(game_rows),
                "Total": sum(r["Total"] for r in game_rows) / len(game_rows),
                "HomeML": sum(r["HomeML"] for r in game_rows) / len(game_rows),
                "AwayML": sum(r["AwayML"] for r in game_rows) / len(game_rows),
                "ImpliedProbHome": sum(r["ImpliedProbHome"] for r in game_rows) / len(game_rows),
                "ImpliedProbAway": sum(r["ImpliedProbAway"] for r in game_rows) / len(game_rows),
                "Bookmaker": "consensus",
            }
            rows.append(consensus)

    return rows


# ---------------------------------------------------------------------------
# Historical Vegas lines
# ---------------------------------------------------------------------------

def load_historical_vegas(
    path: str | Path = "data/vegas_historical.csv",
) -> pd.DataFrame:
    """Load historical Vegas lines from a local CSV.

    Expected columns: Season, WTeamID, LTeamID, Spread, OverUnder.
    Spread is from the winner's perspective (negative means the winner
    was favored).

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with historical lines, or an empty DataFrame if the
        file is missing.
    """
    p = Path(path)
    if not p.exists():
        print(f"Warning: Historical Vegas file not found at {p}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(p)
        for col in ("Spread", "OverUnder"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        for col in ("Season", "WTeamID", "LTeamID"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        print(f"Loaded {len(df)} historical Vegas lines from {p}")
        return df
    except Exception as e:
        print(f"Warning: Failed to load historical Vegas data — {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Merge with matchup features
# ---------------------------------------------------------------------------

def merge_vegas_with_matchups(
    matchup_df: pd.DataFrame,
    vegas_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge Vegas lines into a matchup feature DataFrame.

    Joins on Season + team IDs, trying both orderings of TeamA/TeamB
    since the assignment is arbitrary.  Adds ``VegasSpread`` and
    ``VegasOU`` columns (defaulting to 0.0 when no match is found).

    Args:
        matchup_df: Feature-engineered matchup DataFrame.  Must contain
                    Season, TeamA (or WTeamID), and TeamB (or LTeamID).
        vegas_df:   Historical Vegas DataFrame from
                    :func:`load_historical_vegas`.

    Returns:
        A copy of *matchup_df* with VegasSpread and VegasOU appended.
    """
    if vegas_df.empty:
        merged = matchup_df.copy()
        merged["VegasSpread"] = 0.0
        merged["VegasOU"] = 0.0
        return merged

    # Determine team ID column names in the matchup DataFrame
    if "TeamA" in matchup_df.columns and "TeamB" in matchup_df.columns:
        id_a, id_b = "TeamA", "TeamB"
    elif "WTeamID" in matchup_df.columns and "LTeamID" in matchup_df.columns:
        id_a, id_b = "WTeamID", "LTeamID"
    else:
        print("Warning: matchup_df must contain TeamA/TeamB or WTeamID/LTeamID columns.")
        merged = matchup_df.copy()
        merged["VegasSpread"] = 0.0
        merged["VegasOU"] = 0.0
        return merged

    # Try merge in both orderings
    merged = matchup_df.copy()

    vegas_renamed_1 = vegas_df.rename(columns={
        "WTeamID": id_a,
        "LTeamID": id_b,
        "Spread": "VegasSpread",
        "OverUnder": "VegasOU",
    })[["Season", id_a, id_b, "VegasSpread", "VegasOU"]]

    vegas_renamed_2 = vegas_df.rename(columns={
        "WTeamID": id_b,
        "LTeamID": id_a,
        "Spread": "VegasSpread",
        "OverUnder": "VegasOU",
    })[["Season", id_a, id_b, "VegasSpread", "VegasOU"]]
    # Flip spread sign for reversed ordering
    vegas_renamed_2["VegasSpread"] = -vegas_renamed_2["VegasSpread"]

    merged = merged.merge(
        vegas_renamed_1, on=["Season", id_a, id_b], how="left",
    )

    # Fill missing with the reversed ordering
    mask = merged["VegasSpread"].isna()
    if mask.any():
        alt = merged.loc[mask].drop(columns=["VegasSpread", "VegasOU"]).merge(
            vegas_renamed_2, on=["Season", id_a, id_b], how="left",
        )
        merged.loc[mask, "VegasSpread"] = alt["VegasSpread"].values
        merged.loc[mask, "VegasOU"] = alt["VegasOU"].values

    merged["VegasSpread"] = merged["VegasSpread"].fillna(0.0)
    merged["VegasOU"] = merged["VegasOU"].fillna(0.0)

    return merged
