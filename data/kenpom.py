"""Load and integrate KenPom ratings data.

Reads kenpom_2026.csv and produces team_stats DataFrames compatible
with the existing Predictor and feature_engineering pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Known name mismatches between KenPom and Kaggle MTeams.csv
KENPOM_TO_KAGGLE_NAME = {
    "Connecticut": "UConn",
    "Saint Mary's": "St Mary's CA",
    "St. John's": "St John's",
    "Saint Joseph's": "St Joseph's PA",
    "Miami FL": "Miami FL",
    "Miami OH": "Miami OH",
    "UNC Wilmington": "NC Wilmington",
    "UNC Asheville": "NC A&T",  # may need manual check
    "UNC Greensboro": "NC Greensboro",
    "ETSU": "East Tennessee St.",
    "Iowa St.": "Iowa St",
    "Michigan St.": "Michigan St",
    "Ohio St.": "Ohio St",
    "Mississippi St.": "Mississippi St",
    "Penn St.": "Penn St",
    "Kansas St.": "Kansas St",
    "Colorado St.": "Colorado St",
    "Boise St.": "Boise St",
    "Utah St.": "Utah St",
    "Arizona St.": "Arizona St",
    "Oregon St.": "Oregon St",
    "Oklahoma St.": "Oklahoma St",
    "Washington St.": "Washington St",
    "Wichita St.": "Wichita St",
    "Fresno St.": "Fresno St",
    "Illinois St.": "Illinois St",
    "Indiana St.": "Indiana St",
    "Missouri St.": "Missouri St",
    "Portland St.": "Portland St",
    "Sacramento St.": "Sacramento St",
    "Montana St.": "Montana St",
    "Idaho St.": "Idaho St",
    "Weber St.": "Weber St",
    "Murray St.": "Murray St",
    "Cleveland St.": "Cleveland St",
    "Youngstown St.": "Youngstown St",
    "Georgia St.": "Georgia St",
    "Jacksonville St.": "Jacksonville St",
    "Sam Houston St.": "Sam Houston St",
    "Stephen F. Austin": "SF Austin",
    "Long Beach St.": "Long Beach St",
    "Cal St. Fullerton": "CS Fullerton",
    "Cal St. Bakersfield": "CS Bakersfield",
    "Cal St. Northridge": "CS Northridge",
    "South Carolina St.": "S Carolina St",
    "South Dakota St.": "South Dakota St",
    "North Dakota St.": "North Dakota St",
    "North Carolina Central": "NC Central",
    "North Carolina A&T": "NC A&T",
    "Texas A&M Corpus Christi": "TX A&M C. Christi",
    "Arkansas Pine Bluff": "Ark Pine Bluff",
    "Maryland Eastern Shore": "Md E Shore",
    "Mississippi Valley St.": "MS Valley St",
    "Alabama A&M": "Alabama A&M",
    "Loyola Chicago": "Loyola-Chicago",
    "Loyola Marymount": "Loyola Marymount",
    "Loyola MD": "Loyola MD",
    "Southeast Missouri": "SE Missouri St",
    "Central Connecticut": "Cent Connecticut",
    "Southeastern Louisiana": "SE Louisiana",
    "Middle Tennessee": "Middle TN",
    "East Tennessee St.": "ETSU",
    "Western Kentucky": "W Kentucky",
    "Eastern Kentucky": "E Kentucky",
    "Northern Kentucky": "N Kentucky",
    "Eastern Washington": "E Washington",
    "Western Carolina": "W Carolina",
    "Charleston Southern": "Charleston So",
    "Little Rock": "Ark Little Rock",
    "Florida A&M": "Florida A&M",
    "Alabama St.": "Alabama St",
    "Jackson St.": "Jackson St",
    "Texas Southern": "Texas Southern",
    "Grambling St.": "Grambling",
    "Alcorn St.": "Alcorn St",
    "Prairie View A&M": "Prairie View",
    "Southern": "Southern Univ",
    "The Citadel": "Citadel",
    "UMass Lowell": "MA Lowell",
    "Fairleigh Dickinson": "F Dickinson",
    "USC Upstate": "SC Upstate",
}


def load_kenpom(path: str | Path = "data/kenpom_2026.csv") -> pd.DataFrame:
    """Load KenPom ratings CSV.

    Returns DataFrame with columns:
        Rk, Team, Conf, Record, NetRtg, ORtg, DRtg, AdjT, Luck, SOS, NCSOS
    """
    df = pd.read_csv(path)

    # Clean numeric columns (remove + signs)
    for col in ["NetRtg", "Luck", "SOS", "NCSOS"]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace("+", "", regex=False), errors="coerce")

    # Parse win-loss record into WinPct
    def parse_record(rec):
        try:
            parts = str(rec).split("-")
            w, l = int(parts[0]), int(parts[1])
            return w / (w + l) if (w + l) > 0 else 0.5
        except (ValueError, IndexError):
            return 0.5

    df["WinPct"] = df["Record"].apply(parse_record)

    return df


def kenpom_to_team_stats(
    kenpom_df: pd.DataFrame,
    team_id_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Convert KenPom DataFrame to team_stats format expected by Predictor.

    Args:
        kenpom_df: Output of load_kenpom()
        team_id_map: Mapping of team name -> Kaggle TeamID.
            If None, uses KenPom rank as a synthetic ID.

    Returns:
        DataFrame with columns: TeamID, AdjO, AdjD, AdjEM, AdjT, SOS, WinPct
    """
    rows = []
    for _, row in kenpom_df.iterrows():
        team_name = row["Team"]

        if team_id_map is not None:
            # Try exact match first, then KenPom→Kaggle alias
            team_id = team_id_map.get(team_name)
            if team_id is None:
                kaggle_name = KENPOM_TO_KAGGLE_NAME.get(team_name)
                if kaggle_name:
                    team_id = team_id_map.get(kaggle_name)
            if team_id is None:
                continue  # Can't map this team
        else:
            team_id = int(row["Rk"])  # Synthetic ID from rank

        rows.append({
            "TeamID": team_id,
            "TeamName": team_name,
            "AdjO": float(row["ORtg"]),
            "AdjD": float(row["DRtg"]),
            "AdjEM": float(row["NetRtg"]),
            "AdjT": float(row["AdjT"]),
            "SOS": float(row["SOS"]),
            "WinPct": float(row["WinPct"]),
            "KenpomRank": int(row["Rk"]),
        })

    return pd.DataFrame(rows)


def build_team_id_map(teams_csv: str | Path = "data/raw/MTeams.csv") -> dict[str, int]:
    """Build team name -> TeamID mapping from Kaggle MTeams.csv."""
    path = Path(teams_csv)
    if not path.exists():
        return {}
    teams = pd.read_csv(path)
    return dict(zip(teams["TeamName"], teams["TeamID"]))


def load_kenpom_as_team_stats(
    kenpom_path: str | Path = "data/kenpom_2026.csv",
    teams_csv: str | Path = "data/raw/MTeams.csv",
) -> pd.DataFrame:
    """One-shot: load KenPom CSV and return Predictor-compatible team_stats.

    If Kaggle MTeams.csv is available, maps to real TeamIDs.
    Otherwise uses KenPom rank as synthetic IDs.
    """
    kenpom_df = load_kenpom(kenpom_path)
    team_id_map = build_team_id_map(teams_csv) or None
    return kenpom_to_team_stats(kenpom_df, team_id_map)


def kenpom_predict_matchup(
    team_a_name: str,
    team_b_name: str,
    kenpom_df: pd.DataFrame,
) -> float:
    """Predict P(team_a wins) using KenPom efficiency margins directly.

    Uses the log5 formula with KenPom ratings:
        P(A wins) = 1 / (1 + 10^((NetRtg_B - NetRtg_A) * pace_factor / point_spread_factor))

    This is a simple but effective model for when the full ML pipeline isn't available.
    """
    a = kenpom_df[kenpom_df["Team"] == team_a_name]
    b = kenpom_df[kenpom_df["Team"] == team_b_name]

    if a.empty or b.empty:
        return 0.5

    em_a = float(a.iloc[0]["NetRtg"])
    em_b = float(b.iloc[0]["NetRtg"])

    # Efficiency margin difference → win probability
    # Empirically, ~3.5 points of AdjEM ≈ 1 point of spread
    # And each point of spread ≈ 3% win probability shift from 50%
    em_diff = em_a - em_b
    spread = em_diff * 0.29  # Convert AdjEM diff to approximate point spread

    # Logistic conversion (spread to probability)
    return 1.0 / (1.0 + 10 ** (-spread / 8.0))
