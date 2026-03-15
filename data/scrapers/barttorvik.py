"""Fetch Barttorvik T-Rank team ratings.

Barttorvik provides free tempo-free college basketball analytics.
Data is scraped from the public website and cached as CSV.

Key metrics:
- Barthag: predicted win probability vs average D-I team
- WAB: Wins Above Bubble (quality metric)
- AdjOE/AdjDE: adjusted offensive/defensive efficiency
"""

from pathlib import Path

import pandas as pd
import requests

# Barttorvik API endpoint for team stats (returns CSV when csv=1)
BARTTORVIK_STATS_URL = "https://barttorvik.com/team-tables.php"

# Column names from Barttorvik's team table export
BARTTORVIK_COLUMNS = [
    "Rk", "Team", "Conf", "Record", "AdjOE", "AdjOE_Rk",
    "AdjDE", "AdjDE_Rk", "Barthag", "Barthag_Rk",
    "EFG_O", "EFG_D", "TOR", "TORD", "ORB", "DRB",
    "FTRate", "FTRateD", "TwoP_O", "TwoP_D", "ThreeP_O", "ThreeP_D",
    "ThreePRate", "ThreePRateD", "AdjT", "AdjT_Rk", "WAB", "WAB_Rk",
]


def fetch_barttorvik(
    year: int = 2026,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch team ratings from Barttorvik.

    Tries the JSON/CSV API first, falls back to loading a local CSV.

    Args:
        year: Season year (e.g., 2026 for 2025-26 season)
        save_path: If provided, save the fetched data as CSV

    Returns:
        DataFrame with team ratings including Barthag, WAB, AdjOE, AdjDE
    """
    df = _try_fetch_api(year)
    if df is not None:
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            print(f"Saved Barttorvik data to {path}")
        return df

    # API blocked — try local CSV
    default_path = Path(f"data/barttorvik_{year}.csv")
    if default_path.exists():
        print(f"Loading cached Barttorvik data from {default_path}")
        return load_barttorvik(default_path)

    print(
        f"Could not fetch Barttorvik data. Please manually save T-Rank table "
        f"as data/barttorvik_{year}.csv from https://barttorvik.com/trank.php"
    )
    return pd.DataFrame()


def _try_fetch_api(year: int) -> pd.DataFrame | None:
    """Try to fetch data from Barttorvik's API endpoints."""
    # Method 1: Direct JSON endpoint
    try:
        url = f"https://barttorvik.com/trank.php?year={year}&conyes=1&json=1"
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (compatible; research bot)",
        })
        if resp.status_code == 200 and resp.text.strip().startswith("["):
            data = resp.json()
            if data and isinstance(data, list):
                return _parse_json_response(data)
    except Exception:
        pass

    # Method 2: CSV export endpoint
    try:
        params = {
            "year": year,
            "top": 363,
            "conlimit": "All",
            "venue": "All",
            "type": "pointed",
            "mingames": 0,
            "csv": 1,
        }
        resp = requests.get(
            BARTTORVIK_STATS_URL, params=params, timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; research bot)"},
        )
        if resp.status_code == 200 and "," in resp.text[:200]:
            from io import StringIO
            return pd.read_csv(StringIO(resp.text))
    except Exception:
        pass

    return None


def _parse_json_response(data: list) -> pd.DataFrame:
    """Parse Barttorvik JSON array response into DataFrame."""
    rows = []
    for entry in data:
        if isinstance(entry, list) and len(entry) >= 24:
            rows.append({
                "Team": str(entry[0]),
                "Conf": str(entry[1]),
                "Record": str(entry[2]),
                "AdjOE": _safe_float(entry[3]),
                "AdjDE": _safe_float(entry[5]),
                "Barthag": _safe_float(entry[7]),
                "EFG_O": _safe_float(entry[9]),
                "EFG_D": _safe_float(entry[10]),
                "TOR": _safe_float(entry[11]),
                "TORD": _safe_float(entry[12]),
                "ORB": _safe_float(entry[13]),
                "DRB": _safe_float(entry[14]),
                "FTRate": _safe_float(entry[15]),
                "FTRateD": _safe_float(entry[16]),
                "AdjT": _safe_float(entry[21]),
                "WAB": _safe_float(entry[23]),
            })
        elif isinstance(entry, dict):
            rows.append({
                "Team": entry.get("team", ""),
                "Conf": entry.get("conf", ""),
                "Record": entry.get("record", ""),
                "AdjOE": _safe_float(entry.get("adjoe")),
                "AdjDE": _safe_float(entry.get("adjde")),
                "Barthag": _safe_float(entry.get("barthag")),
                "AdjT": _safe_float(entry.get("adjt")),
                "WAB": _safe_float(entry.get("wab")),
            })
    return pd.DataFrame(rows)


def _safe_float(val) -> float:
    """Safely convert a value to float."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def load_barttorvik(path: str | Path = "data/barttorvik_2026.csv") -> pd.DataFrame:
    """Load cached Barttorvik data from CSV.

    Returns DataFrame with columns: Team, Conf, AdjOE, AdjDE, Barthag, AdjT, WAB
    """
    df = pd.read_csv(path)

    # Normalize column names (handle various export formats)
    col_map = {}
    for col in df.columns:
        lower = col.strip().lower().replace(" ", "")
        if lower in ("team", "teamname"):
            col_map[col] = "Team"
        elif lower in ("adjoe", "adjoffeff", "offeff"):
            col_map[col] = "AdjOE"
        elif lower in ("adjde", "adjdefeff", "defeff"):
            col_map[col] = "AdjDE"
        elif lower == "barthag":
            col_map[col] = "Barthag"
        elif lower in ("adjt", "adjtempo"):
            col_map[col] = "AdjT"
        elif lower == "wab":
            col_map[col] = "WAB"
        elif lower in ("conf", "conference"):
            col_map[col] = "Conf"

    if col_map:
        df = df.rename(columns=col_map)

    # Ensure numeric columns
    for col in ["AdjOE", "AdjDE", "Barthag", "AdjT", "WAB"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def barttorvik_to_team_stats(
    bt_df: pd.DataFrame,
    team_id_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Convert Barttorvik DataFrame to team_stats rows for merging with Predictor.

    Returns DataFrame with columns: TeamID, TeamName, Barthag, WAB, BT_AdjOE, BT_AdjDE
    """
    from data.kenpom import KENPOM_TO_KAGGLE_NAME

    rows = []
    for _, row in bt_df.iterrows():
        team_name = str(row.get("Team", "")).strip()

        if team_id_map is not None:
            team_id = team_id_map.get(team_name)
            if team_id is None:
                kaggle_name = KENPOM_TO_KAGGLE_NAME.get(team_name)
                if kaggle_name:
                    team_id = team_id_map.get(kaggle_name)
            if team_id is None:
                continue
        else:
            team_id = hash(team_name) % 100000

        rows.append({
            "TeamID": team_id,
            "TeamName": team_name,
            "Barthag": float(row.get("Barthag", 0.0)),
            "WAB": float(row.get("WAB", 0.0)),
            "BT_AdjOE": float(row.get("AdjOE", 0.0)),
            "BT_AdjDE": float(row.get("AdjDE", 0.0)),
        })

    return pd.DataFrame(rows)
