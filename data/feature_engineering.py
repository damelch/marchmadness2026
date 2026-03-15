"""Build matchup features for win probability modeling."""

from pathlib import Path

import numpy as np
import pandas as pd

from data.seed_history import parse_seed


def compute_possessions(row: pd.Series, prefix: str) -> float:
    """Estimate possessions from box score stats."""
    fga = row[f"{prefix}FGA"]
    ora = row[f"{prefix}OR"]
    to = row[f"{prefix}TO"]
    fta = row[f"{prefix}FTA"]
    return fga - ora + to + 0.475 * fta


def compute_team_stats(detailed_results: pd.DataFrame, season: int) -> pd.DataFrame:
    """Compute per-team season statistics from detailed results.

    Returns DataFrame with columns:
        TeamID, Season, WinPct, AdjO, AdjD, AdjEM, AdjT,
        AvgPoss, PointsPerPoss, PointsAllowedPerPoss
    """
    df = detailed_results[detailed_results["Season"] == season].copy()

    if df.empty:
        return pd.DataFrame()

    # Calculate possessions for each game
    df["WPoss"] = df.apply(lambda r: compute_possessions(r, "W"), axis=1)
    df["LPoss"] = df.apply(lambda r: compute_possessions(r, "L"), axis=1)
    df["AvgPoss"] = (df["WPoss"] + df["LPoss"]) / 2

    # Build per-team stats
    team_stats = {}

    # Process winners
    for _, row in df.iterrows():
        wid, lid = row["WTeamID"], row["LTeamID"]
        poss = row["AvgPoss"]

        if poss == 0:
            continue

        for team_id, scored, allowed, won in [
            (wid, row["WScore"], row["LScore"], True),
            (lid, row["LScore"], row["WScore"], False),
        ]:
            if team_id not in team_stats:
                team_stats[team_id] = {
                    "games": 0,
                    "wins": 0,
                    "total_off_eff": 0,
                    "total_def_eff": 0,
                    "total_poss": 0,
                    "total_scored": 0,
                    "total_allowed": 0,
                    "opp_ids": [],
                    "game_margins": [],  # per-game efficiency margins for consistency
                    "close_games": 0,    # games decided by <= 5 points
                    "close_wins": 0,     # wins in close games
                }
            stats = team_stats[team_id]
            stats["games"] += 1
            stats["wins"] += int(won)
            off_eff = (scored / poss) * 100
            def_eff = (allowed / poss) * 100
            stats["total_off_eff"] += off_eff
            stats["total_def_eff"] += def_eff
            stats["total_poss"] += poss
            stats["total_scored"] += scored
            stats["total_allowed"] += allowed
            stats["opp_ids"].append(lid if team_id == wid else wid)
            stats["game_margins"].append(off_eff - def_eff)
            # Track close games (decided by <= 5 points)
            margin = abs(scored - allowed)
            if margin <= 5:
                stats["close_games"] += 1
                stats["close_wins"] += int(won)

    # Convert to DataFrame
    rows = []
    for team_id, stats in team_stats.items():
        g = stats["games"]
        if g == 0:
            continue

        win_pct = stats["wins"] / g
        avg_scored = stats.get("total_scored", 0) / g if g > 0 else 0
        avg_allowed = stats.get("total_allowed", 0) / g if g > 0 else 0

        # Pythagorean expected win%: scored^11.5 / (scored^11.5 + allowed^11.5)
        # Exponent 11.5 is standard for college basketball (KenPom uses ~10-12)
        exp = 11.5
        if avg_scored > 0 and avg_allowed > 0:
            pyth_win_pct = avg_scored**exp / (avg_scored**exp + avg_allowed**exp)
        else:
            pyth_win_pct = 0.5

        # Luck = actual win% - expected win% (positive = overperforming)
        luck = win_pct - pyth_win_pct

        rows.append(
            {
                "TeamID": team_id,
                "Season": season,
                "WinPct": win_pct,
                "RawOE": stats["total_off_eff"] / g,
                "RawDE": stats["total_def_eff"] / g,
                "AvgPoss": stats["total_poss"] / g,
                "Games": g,
                "Luck": luck,
            }
        )

    team_df = pd.DataFrame(rows)

    if team_df.empty:
        return team_df

    # Iterative SOS adjustment (simplified KenPom-style)
    # Start with raw efficiencies, adjust for opponent strength over 10 iterations
    team_df = team_df.set_index("TeamID")
    team_df["AdjO"] = team_df["RawOE"]
    team_df["AdjD"] = team_df["RawDE"]

    league_avg_oe = team_df["RawOE"].mean()
    league_avg_de = team_df["RawDE"].mean()

    for _ in range(10):
        # For each team, compute average opponent AdjD and AdjO
        new_adj_o = []
        new_adj_d = []

        for team_id in team_df.index:
            opps = team_stats[team_id]["opp_ids"]
            valid_opps = [o for o in opps if o in team_df.index]
            if not valid_opps:
                new_adj_o.append(team_df.loc[team_id, "RawOE"])
                new_adj_d.append(team_df.loc[team_id, "RawDE"])
                continue

            # Adjust offense: raw_oe * (league_avg_de / avg_opp_adj_d)
            avg_opp_adj_d = team_df.loc[valid_opps, "AdjD"].mean()
            adj_o = team_df.loc[team_id, "RawOE"] * (league_avg_de / max(avg_opp_adj_d, 1))

            # Adjust defense: raw_de * (league_avg_oe / avg_opp_adj_o)
            avg_opp_adj_o = team_df.loc[valid_opps, "AdjO"].mean()
            adj_d = team_df.loc[team_id, "RawDE"] * (league_avg_oe / max(avg_opp_adj_o, 1))

            new_adj_o.append(adj_o)
            new_adj_d.append(adj_d)

        team_df["AdjO"] = new_adj_o
        team_df["AdjD"] = new_adj_d

    team_df["AdjEM"] = team_df["AdjO"] - team_df["AdjD"]
    team_df["AdjT"] = team_df["AvgPoss"]

    # Compute scoring margin consistency (std dev of per-game margins)
    em_std_values = []
    for team_id in team_df.index:
        margins = team_stats[team_id]["game_margins"]
        em_std_values.append(float(np.std(margins)) if len(margins) > 1 else 0.0)
    team_df["AdjEMStd"] = em_std_values

    # Compute SOS as average opponent AdjEM
    sos_values = []
    for team_id in team_df.index:
        opps = team_stats[team_id]["opp_ids"]
        valid_opps = [o for o in opps if o in team_df.index]
        if valid_opps:
            sos_values.append(team_df.loc[valid_opps, "AdjEM"].mean())
        else:
            sos_values.append(0.0)
    team_df["SOS"] = sos_values

    # Close game win% (games decided by <= 5 points) — proxy for tournament readiness
    close_game_values = []
    for team_id in team_df.index:
        cg = team_stats[team_id]["close_games"]
        cw = team_stats[team_id]["close_wins"]
        close_game_values.append(cw / cg if cg > 0 else 0.5)
    team_df["CloseGameWinPct"] = close_game_values

    return team_df.reset_index()[
        ["TeamID", "Season", "WinPct", "AdjO", "AdjD", "AdjEM", "AdjT",
         "SOS", "AdjEMStd", "Luck", "CloseGameWinPct", "Games"]
    ]


def compute_massey_composite(massey_df: pd.DataFrame, season: int, day_num: int = 133) -> pd.DataFrame:
    """Compute composite Massey ordinal ranking (average across all systems).

    day_num=133 is roughly Selection Sunday.
    """
    df = massey_df[(massey_df["Season"] == season) & (massey_df["RankingDayNum"] == day_num)]
    if df.empty:
        # Try the closest available day
        avail = massey_df[massey_df["Season"] == season]["RankingDayNum"]
        if avail.empty:
            return pd.DataFrame(columns=["TeamID", "Season", "MasseyRank"])
        closest_day = avail.iloc[(avail - day_num).abs().argsort().iloc[0]]
        df = massey_df[
            (massey_df["Season"] == season) & (massey_df["RankingDayNum"] == closest_day)
        ]

    composite = df.groupby("TeamID")["OrdinalRank"].mean().reset_index()
    composite.columns = ["TeamID", "MasseyRank"]
    composite["Season"] = season
    return composite


def compute_tourney_experience(seeds_df: pd.DataFrame, season: int, lookback: int = 5) -> pd.DataFrame:
    """Count number of tournament appearances in the last N years for each team."""
    past = seeds_df[
        (seeds_df["Season"] >= season - lookback) & (seeds_df["Season"] < season)
    ]
    exp = past.groupby("TeamID").size().reset_index(name="TourneyExp")
    exp["Season"] = season
    return exp


def build_matchup_features(
    tourney_results: pd.DataFrame,
    seeds_df: pd.DataFrame,
    regular_detailed: pd.DataFrame,
    massey_df: pd.DataFrame | None = None,
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Build feature matrix for all historical tournament matchups.

    Each row represents a matchup (TeamA vs TeamB) with the label being
    whether TeamA won. We create two rows per game (A vs B and B vs A)
    for symmetry augmentation.

    Returns DataFrame with feature columns and 'Result' (1 = TeamA won).
    """
    if seasons is None:
        seasons = sorted(tourney_results["Season"].unique())

    all_rows = []

    for season in seasons:
        # Compute team stats for this season
        team_stats = compute_team_stats(regular_detailed, season)
        if team_stats.empty:
            continue

        # Get Massey composite if available
        massey = None
        if massey_df is not None and not massey_df.empty:
            massey = compute_massey_composite(massey_df, season)

        # Tournament experience
        tourney_exp = compute_tourney_experience(seeds_df, season)

        # Get seed mapping for this season
        season_seeds = seeds_df[seeds_df["Season"] == season].copy()
        season_seeds["NumSeed"] = season_seeds["Seed"].apply(parse_seed)
        seed_map = dict(zip(season_seeds["TeamID"], season_seeds["NumSeed"]))

        # Build features for each tournament game
        season_games = tourney_results[tourney_results["Season"] == season]

        for _, game in season_games.iterrows():
            winner_id = game["WTeamID"]
            loser_id = game["LTeamID"]

            # Derive round number from DayNum (Kaggle convention)
            day_num = game.get("DayNum", 136)
            round_num = _daynum_to_round(day_num)

            for team_a, team_b, result in [
                (winner_id, loser_id, 1),
                (loser_id, winner_id, 0),
            ]:
                features = _compute_pair_features(
                    team_a, team_b, season, seed_map, team_stats, massey, tourney_exp,
                    round_num=round_num,
                )
                if features is not None:
                    features["Result"] = result
                    features["Season"] = season
                    all_rows.append(features)

    df = pd.DataFrame(all_rows)

    # Merge Vegas features if historical data is available
    try:
        from data.scrapers.vegas_lines import load_historical_vegas, merge_vegas_with_matchups

        vegas_df = load_historical_vegas()
        if not vegas_df.empty:
            df = merge_vegas_with_matchups(df, vegas_df)
    except Exception:
        pass

    # Ensure Vegas columns exist even without data
    for col in ("VegasSpread", "VegasOU"):
        if col not in df.columns:
            df[col] = 0.0

    return df


def _daynum_to_round(day_num: int) -> int:
    """Map Kaggle DayNum to tournament round (1-6).

    Kaggle convention: R64 starts ~DayNum 136-137, R32 ~138-139, etc.
    """
    if day_num <= 137:
        return 1  # Round of 64
    elif day_num <= 139:
        return 2  # Round of 32
    elif day_num <= 144:
        return 3  # Sweet 16
    elif day_num <= 146:
        return 4  # Elite 8
    elif day_num <= 152:
        return 5  # Final Four
    else:
        return 6  # Championship


def _compute_pair_features(
    team_a: int,
    team_b: int,
    season: int,
    seed_map: dict[int, int],
    team_stats: pd.DataFrame,
    massey: pd.DataFrame | None,
    tourney_exp: pd.DataFrame,
    round_num: int = 1,
) -> dict | None:
    """Compute feature dict for a single A-vs-B matchup."""
    seed_a = seed_map.get(team_a)
    seed_b = seed_map.get(team_b)
    if seed_a is None or seed_b is None:
        return None

    stats_a = team_stats[team_stats["TeamID"] == team_a]
    stats_b = team_stats[team_stats["TeamID"] == team_b]
    if stats_a.empty or stats_b.empty:
        return None

    sa = stats_a.iloc[0]
    sb = stats_b.iloc[0]

    seed_diff = seed_b - seed_a  # positive = A is favored

    features = {
        "TeamA": team_a,
        "TeamB": team_b,
        "SeedA": seed_a,
        "SeedB": seed_b,
        "SeedDiff": seed_diff,
        "AdjODiff": sa["AdjO"] - sb["AdjO"],
        "AdjDDiff": sa["AdjD"] - sb["AdjD"],  # lower is better for defense
        "AdjEMDiff": sa["AdjEM"] - sb["AdjEM"],
        "AdjTAvg": (sa["AdjT"] + sb["AdjT"]) / 2,
        "SOSDiff": sa["SOS"] - sb["SOS"],
        "WinPctDiff": sa["WinPct"] - sb["WinPct"],
    }

    # Massey rank
    if massey is not None and not massey.empty:
        rank_a = massey[massey["TeamID"] == team_a]
        rank_b = massey[massey["TeamID"] == team_b]
        if not rank_a.empty and not rank_b.empty:
            features["MasseyRankDiff"] = rank_b.iloc[0]["MasseyRank"] - rank_a.iloc[0]["MasseyRank"]
        else:
            features["MasseyRankDiff"] = 0.0
    else:
        features["MasseyRankDiff"] = 0.0

    # Tournament experience
    exp_a = tourney_exp[tourney_exp["TeamID"] == team_a]
    exp_b = tourney_exp[tourney_exp["TeamID"] == team_b]
    features["TourneyExpDiff"] = (
        (exp_a.iloc[0]["TourneyExp"] if not exp_a.empty else 0)
        - (exp_b.iloc[0]["TourneyExp"] if not exp_b.empty else 0)
    )

    # --- New features ---

    # Luck: actual win% minus Pythagorean expected win% (positive = overperforming)
    luck_a = sa.get("Luck", 0.0) if "Luck" in sa.index else 0.0
    luck_b = sb.get("Luck", 0.0) if "Luck" in sb.index else 0.0
    features["LuckDiff"] = float(luck_a) - float(luck_b)

    # Close game win% — proxy for tournament readiness (clutch performance)
    cg_a = sa.get("CloseGameWinPct", 0.5) if "CloseGameWinPct" in sa.index else 0.5
    cg_b = sb.get("CloseGameWinPct", 0.5) if "CloseGameWinPct" in sb.index else 0.5
    features["CloseGameDiff"] = float(cg_a) - float(cg_b)

    # Seed-round interaction (being a 1-seed in R64 vs E8 is different)
    features["SeedRoundInteraction"] = seed_diff * round_num

    # Scoring margin consistency (lower std = more consistent team)
    em_std_a = sa.get("AdjEMStd", 0.0) if "AdjEMStd" in sa.index else 0.0
    em_std_b = sb.get("AdjEMStd", 0.0) if "AdjEMStd" in sb.index else 0.0
    features["AdjEMStdDiff"] = float(em_std_a) - float(em_std_b)

    # --- Barttorvik features (if available) ---

    # Barthag: predicted win% vs average D-I team (0 to 1 scale)
    barthag_a = sa.get("Barthag", 0.0) if "Barthag" in sa.index else 0.0
    barthag_b = sb.get("Barthag", 0.0) if "Barthag" in sb.index else 0.0
    features["BarthagDiff"] = float(barthag_a) - float(barthag_b)

    # WAB: Wins Above Bubble (quality metric, can be negative)
    wab_a = sa.get("WAB", 0.0) if "WAB" in sa.index else 0.0
    wab_b = sb.get("WAB", 0.0) if "WAB" in sb.index else 0.0
    features["WABDiff"] = float(wab_a) - float(wab_b)

    # --- ESPN BPI features (if available) ---

    # BPI: overall power index (points above/below average)
    bpi_a = sa.get("BPI", 0.0) if "BPI" in sa.index else 0.0
    bpi_b = sb.get("BPI", 0.0) if "BPI" in sb.index else 0.0
    features["BPIDiff"] = float(bpi_a) - float(bpi_b)

    # BPI Offense and Defense
    bpi_off_a = sa.get("BPIOff", 0.0) if "BPIOff" in sa.index else 0.0
    bpi_off_b = sb.get("BPIOff", 0.0) if "BPIOff" in sb.index else 0.0
    features["BPIOffDiff"] = float(bpi_off_a) - float(bpi_off_b)

    bpi_def_a = sa.get("BPIDef", 0.0) if "BPIDef" in sa.index else 0.0
    bpi_def_b = sb.get("BPIDef", 0.0) if "BPIDef" in sb.index else 0.0
    features["BPIDefDiff"] = float(bpi_def_a) - float(bpi_def_b)

    return features


FEATURE_COLUMNS = [
    "SeedDiff",
    "AdjODiff",
    "AdjDDiff",
    "AdjEMDiff",
    "AdjTAvg",
    "SOSDiff",
    "WinPctDiff",
    "MasseyRankDiff",
    "TourneyExpDiff",
    "LuckDiff",
    "CloseGameDiff",
    "SeedRoundInteraction",
    "AdjEMStdDiff",
    # Barttorvik features
    "BarthagDiff",
    "WABDiff",
    # ESPN BPI features
    "BPIDiff",
    "BPIOffDiff",
    "BPIDefDiff",
    # Vegas betting line features (zero-filled when not available)
    "VegasSpread",
    "VegasOU",
]


def save_features(df: pd.DataFrame, output_dir: str | Path = "data/processed") -> Path:
    """Save feature matrix to parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "matchup_features.parquet"
    df.to_parquet(path, index=False)
    return path


def load_features(data_dir: str | Path = "data/processed") -> pd.DataFrame:
    """Load saved feature matrix."""
    path = Path(data_dir) / "matchup_features.parquet"
    return pd.read_parquet(path)
