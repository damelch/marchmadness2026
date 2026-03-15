"""Fetch ESPN Basketball Power Index (BPI) ratings.

Uses ESPN's public core API to retrieve BPI, offensive/defensive ratings,
strength of record, and quality wins for all D-I teams.

No authentication required — this is the same API that powers espn.com/bpi.
"""

from pathlib import Path

import pandas as pd
import requests

BPI_BASE_URL = (
    "https://sports.core.api.espn.com/v2/sports/basketball"
    "/leagues/mens-college-basketball/seasons/{year}/powerindex"
)

TEAM_BASE_URL = (
    "https://sports.core.api.espn.com/v2/sports/basketball"
    "/leagues/mens-college-basketball/seasons/{year}/teams/{team_id}"
)

# ESPN team ID -> common name mapping for tournament teams
# This supplements the API team names to handle edge cases
ESPN_NAME_FIXES: dict[str, str] = {
    "UConn Huskies": "Connecticut",
    "UConn": "Connecticut",
}


def fetch_espn_bpi(
    year: int = 2026,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch BPI ratings for all D-I teams from ESPN API.

    Args:
        year: Season year (e.g., 2026 for 2025-26 season)
        save_path: If provided, save the fetched data as CSV

    Returns:
        DataFrame with columns: Team, ESPN_ID, BPI, BPIOff, BPIDef,
                                BPIRank, SOR, QualityWins, QualityLosses
    """
    url = BPI_BASE_URL.format(year=year)
    all_teams = []

    # ESPN API paginates — fetch all pages
    page = 1
    while True:
        try:
            resp = requests.get(
                url, params={"limit": 50, "page": page}, timeout=20,
            )
            if resp.status_code != 200:
                break

            data = resp.json()
            items = data.get("items", [])
            if not items:
                break

            for item in items:
                team_data = _parse_bpi_item(item, year)
                if team_data:
                    all_teams.append(team_data)

            # Check if more pages
            page_count = data.get("pageCount", 1)
            if page >= page_count:
                break
            page += 1

        except Exception as e:
            print(f"Error fetching BPI page {page}: {e}")
            break

    if not all_teams:
        # Try loading cached file
        default_path = Path(f"data/espn_bpi_{year}.csv")
        if default_path.exists():
            print(f"Loading cached ESPN BPI data from {default_path}")
            return load_espn_bpi(default_path)
        print("Could not fetch ESPN BPI data.")
        return pd.DataFrame()

    df = pd.DataFrame(all_teams)
    print(f"Fetched BPI data for {len(df)} teams")

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Saved ESPN BPI data to {path}")

    return df


def _parse_bpi_item(item: dict, year: int) -> dict | None:
    """Parse a single team's BPI data from the API response."""
    stats = item.get("stats", [])
    if not stats:
        return None

    stat_dict = {}
    for stat in stats:
        name = stat.get("name", "")
        value = stat.get("value")
        if name and value is not None:
            stat_dict[name] = value

    # Resolve team name from the team reference URL
    team_ref = item.get("team", {})
    team_url = team_ref if isinstance(team_ref, str) else team_ref.get("$ref", "")
    team_name = _resolve_team_name(team_url)
    espn_id = _extract_team_id(team_url)

    return {
        "Team": team_name,
        "ESPN_ID": espn_id,
        "BPI": stat_dict.get("bpi", 0.0),
        "BPIOff": stat_dict.get("bpioffense", 0.0),
        "BPIDef": stat_dict.get("bpidefense", 0.0),
        "BPIRank": int(stat_dict.get("bpirank", 0)),
        "BPIOffRank": int(stat_dict.get("bpioffenserank", 0)),
        "BPIDefRank": int(stat_dict.get("bpidefenserank", 0)),
        "SOR": stat_dict.get("sor", 0.0),
        "SORRank": int(stat_dict.get("sorrank", 0)),
        "SOSPast": stat_dict.get("sospast", 0.0),
        "QualityWins": int(stat_dict.get("top50bpiwins", 0)),
        "QualityLosses": int(stat_dict.get("top50bpilosses", 0)),
        "Wins": int(stat_dict.get("wins", 0)),
        "Losses": int(stat_dict.get("losses", 0)),
    }


def _extract_team_id(url: str) -> int:
    """Extract team ID from ESPN API URL."""
    try:
        # URL format: .../teams/52?lang=en&region=us
        parts = url.split("/teams/")
        if len(parts) > 1:
            id_str = parts[1].split("?")[0].split("/")[0]
            return int(id_str)
    except (ValueError, IndexError):
        pass
    return 0


# Cache for team name lookups
_team_name_cache: dict[str, str] = {}


def _resolve_team_name(team_url: str) -> str:
    """Resolve team name from ESPN API team URL."""
    if team_url in _team_name_cache:
        return _team_name_cache[team_url]

    try:
        resp = requests.get(team_url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            name = data.get("displayName", data.get("name", "Unknown"))
            # Strip mascot name for matching (e.g., "Duke Blue Devils" -> "Duke")
            location = data.get("location", name)
            _team_name_cache[team_url] = location
            return location
    except Exception:
        pass

    # Fallback: extract ID
    team_id = _extract_team_id(team_url)
    fallback = f"ESPN_{team_id}"
    _team_name_cache[team_url] = fallback
    return fallback


def load_espn_bpi(path: str | Path = "data/espn_bpi_2026.csv") -> pd.DataFrame:
    """Load cached ESPN BPI data from CSV."""
    df = pd.read_csv(path)
    for col in ["BPI", "BPIOff", "BPIDef", "SOR", "SOSPast"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def bpi_to_team_stats(
    bpi_df: pd.DataFrame,
    team_id_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Convert ESPN BPI DataFrame to team_stats rows for merging with Predictor.

    Returns DataFrame with columns: TeamID, TeamName, BPI, BPIOff, BPIDef, SOR
    """
    from data.kenpom import KENPOM_TO_KAGGLE_NAME

    rows = []
    for _, row in bpi_df.iterrows():
        team_name = str(row.get("Team", "")).strip()
        # Apply name fixes
        team_name = ESPN_NAME_FIXES.get(team_name, team_name)

        if team_id_map is not None:
            team_id = team_id_map.get(team_name)
            if team_id is None:
                kaggle_name = KENPOM_TO_KAGGLE_NAME.get(team_name)
                if kaggle_name:
                    team_id = team_id_map.get(kaggle_name)
            if team_id is None:
                # Try matching with "St." variations
                for variant in [team_name.replace("State", "St."), team_name.replace("St.", "State")]:
                    team_id = team_id_map.get(variant)
                    if team_id:
                        break
            if team_id is None:
                continue
        else:
            team_id = int(row.get("ESPN_ID", 0)) or hash(team_name) % 100000

        rows.append({
            "TeamID": team_id,
            "TeamName": team_name,
            "BPI": float(row.get("BPI", 0.0)),
            "BPIOff": float(row.get("BPIOff", 0.0)),
            "BPIDef": float(row.get("BPIDef", 0.0)),
            "SOR": float(row.get("SOR", 0.0)),
        })

    return pd.DataFrame(rows)


def fetch_and_save_bpi(
    year: int = 2026,
    output_dir: str | Path = "data",
) -> pd.DataFrame:
    """One-shot: fetch BPI and save to standard location."""
    path = Path(output_dir) / f"espn_bpi_{year}.csv"
    return fetch_espn_bpi(year=year, save_path=path)
