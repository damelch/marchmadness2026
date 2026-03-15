"""Historical seed-vs-seed win rates for the NCAA tournament."""


import numpy as np
import pandas as pd


def parse_seed(seed_str: str) -> int:
    """Parse seed string like 'W01', 'Z16a' to numeric seed."""
    # Remove region letter(s) and play-in suffix
    numeric = ""
    for ch in seed_str[1:]:
        if ch.isdigit():
            numeric += ch
    return int(numeric)


def build_seed_win_rates(tourney_results: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    """Build a matrix of historical seed-vs-seed win rates.

    Args:
        tourney_results: MNCAATourneyCompactResults with columns
            Season, WTeamID, LTeamID
        seeds: MNCAATourneySeeds with columns Season, Seed, TeamID

    Returns:
        DataFrame indexed by (higher_seed, lower_seed) with columns:
        win_rate (of higher seed), games, wins
    """
    # Map team IDs to numeric seeds
    seed_map = seeds.copy()
    seed_map["NumSeed"] = seed_map["Seed"].apply(parse_seed)

    # Merge seeds onto results
    df = tourney_results.merge(
        seed_map[["Season", "TeamID", "NumSeed"]],
        left_on=["Season", "WTeamID"],
        right_on=["Season", "TeamID"],
    ).rename(columns={"NumSeed": "WSeed"})

    df = df.merge(
        seed_map[["Season", "TeamID", "NumSeed"]],
        left_on=["Season", "LTeamID"],
        right_on=["Season", "TeamID"],
        suffixes=("_w", "_l"),
    ).rename(columns={"NumSeed": "LSeed"})

    # For each game, identify the higher seed (lower number) and lower seed
    records = []
    for _, row in df.iterrows():
        higher = min(row["WSeed"], row["LSeed"])
        lower = max(row["WSeed"], row["LSeed"])
        higher_won = row["WSeed"] == higher
        records.append({"higher_seed": higher, "lower_seed": lower, "higher_won": higher_won})

    matchups = pd.DataFrame(records)

    # Aggregate
    grouped = matchups.groupby(["higher_seed", "lower_seed"]).agg(
        games=("higher_won", "count"),
        wins=("higher_won", "sum"),
    )
    grouped["win_rate"] = grouped["wins"] / grouped["games"]

    return grouped.reset_index()


# Fallback historical rates (approximate, from 1985-2025 data)
DEFAULT_SEED_WIN_RATES = {
    (1, 16): 0.99,
    (2, 15): 0.94,
    (3, 14): 0.85,
    (4, 13): 0.79,
    (5, 12): 0.65,
    (6, 11): 0.63,
    (7, 10): 0.61,
    (8, 9): 0.51,
    # Round of 32 (approximate)
    (1, 8): 0.80,
    (1, 9): 0.83,
    (2, 7): 0.71,
    (2, 10): 0.74,
    (3, 6): 0.59,
    (3, 11): 0.68,
    (4, 5): 0.55,
    (4, 12): 0.70,
}


def get_seed_win_prob(
    seed_a: int,
    seed_b: int,
    seed_win_rates: dict[tuple[int, int], float] | None = None,
) -> float:
    """Get historical win probability for seed_a over seed_b.

    Uses actual data if available, otherwise falls back to a logistic model
    based on seed difference.
    """
    if seed_win_rates is None:
        seed_win_rates = DEFAULT_SEED_WIN_RATES

    higher = min(seed_a, seed_b)
    lower = max(seed_a, seed_b)
    key = (higher, lower)

    if key in seed_win_rates:
        rate = seed_win_rates[key]
        return rate if seed_a == higher else 1.0 - rate

    # Fallback: logistic model based on seed difference
    # Fitted to historical data: P(lower_seed_wins) = sigmoid(0.15 * seed_diff)
    seed_diff = seed_b - seed_a  # positive means A is favored
    prob_a_wins = 1.0 / (1.0 + np.exp(-0.15 * seed_diff))
    return float(prob_a_wins)
