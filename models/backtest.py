"""Backtesting framework for NCAA tournament survivor pool strategies.

Replays past NCAA tournaments using Kaggle historical data to measure
how different pick strategies (optimizer, top seeds, random, contrarian)
perform across seasons. Uses leave-one-season-out (LOSO) model training
to avoid data leakage.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from data.feature_engineering import _daynum_to_round
from data.seed_history import parse_seed
from models.train import get_model
from optimizer.analytical import optimal_day_picks
from optimizer.ownership import estimate_ownership_from_bracket
from simulation.engine import TournamentBracket

if TYPE_CHECKING:
    from models.train import WinProbabilityModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backtest schedule: maps contest days to Kaggle round numbers and regions.
# This mirrors ContestSchedule.default() but is hardcoded here so backtests
# are reproducible regardless of any future schedule changes.
# ---------------------------------------------------------------------------

_BACKTEST_SCHEDULE: list[dict] = [
    {"day": 1, "round": 1, "regions": ["W", "X"], "num_picks": 2},
    {"day": 2, "round": 1, "regions": ["Y", "Z"], "num_picks": 2},
    {"day": 3, "round": 2, "regions": ["W", "X"], "num_picks": 1},
    {"day": 4, "round": 2, "regions": ["Y", "Z"], "num_picks": 1},
    {"day": 5, "round": 3, "regions": ["W", "X"], "num_picks": 1},
    {"day": 6, "round": 3, "regions": ["Y", "Z"], "num_picks": 1},
    {"day": 7, "round": 4, "regions": ["W", "X", "Y", "Z"], "num_picks": 1},
    {"day": 8, "round": 5, "regions": ["FF"], "num_picks": 1},
    {"day": 9, "round": 6, "regions": ["FF"], "num_picks": 1},
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Result of backtesting a single season with a given strategy."""

    season: int
    strategy: str  # "optimizer", "top_seeds", "random", "contrarian"
    n_entries: int
    days_survived: list[int]  # per entry: how many days survived (0-9)
    final_alive: int  # entries alive at end
    picks_per_day: dict[int, list] = field(default_factory=dict)

    @property
    def avg_days_survived(self) -> float:
        """Average number of days an entry survived."""
        if not self.days_survived:
            return 0.0
        return sum(self.days_survived) / len(self.days_survived)

    @property
    def max_days_survived(self) -> int:
        """Best single-entry result."""
        return max(self.days_survived) if self.days_survived else 0

    @property
    def survival_rate(self) -> float:
        """Fraction of entries that survived all days."""
        if not self.days_survived:
            return 0.0
        total_days = len(_BACKTEST_SCHEDULE)
        return sum(1 for d in self.days_survived if d >= total_days) / len(self.days_survived)


# ---------------------------------------------------------------------------
# Bracket & result reconstruction from Kaggle data
# ---------------------------------------------------------------------------


def reconstruct_bracket(
    season: int,
    seeds_df: pd.DataFrame,
    teams_df: pd.DataFrame,
) -> TournamentBracket:
    """Build a TournamentBracket from Kaggle seed data for a given season.

    Args:
        season: Tournament year (e.g. 2023).
        seeds_df: MNCAATourneySeeds with columns Season, Seed, TeamID.
        teams_df: MTeams with columns TeamID, TeamName.

    Returns:
        Populated TournamentBracket with all 64 (or 68) teams seeded.
    """
    bracket = TournamentBracket()

    season_seeds = seeds_df[seeds_df["Season"] == season]
    if season_seeds.empty:
        logger.warning("No seeds found for season %d", season)
        return bracket

    # Build team name lookup
    name_map: dict[int, str] = {}
    if teams_df is not None and not teams_df.empty:
        name_map = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))

    for _, row in season_seeds.iterrows():
        team_id = int(row["TeamID"])
        seed_str = str(row["Seed"])

        # Parse region (first character) and seed number
        region = seed_str[0]  # W, X, Y, or Z
        seed_num = parse_seed(seed_str)

        team_name = name_map.get(team_id, f"Team{team_id}")

        # Play-in teams (seeds like "W16a", "W16b") share the same seed slot.
        # set_seed silently ignores seeds not in its mapping, so play-in
        # duplicates are handled gracefully.
        bracket.set_seed(team_id, seed_num, region, name=team_name)

    return bracket


def get_actual_winners(
    season: int,
    results_df: pd.DataFrame,
) -> dict[int, set[int]]:
    """Get actual game winners grouped by bracket round.

    Args:
        season: Tournament year.
        results_df: MNCAATourneyCompactResults with columns
            Season, DayNum, WTeamID, WScore, LTeamID, LScore.

    Returns:
        Mapping of round_num (1-6) to set of winning TeamIDs.
    """
    season_results = results_df[results_df["Season"] == season]
    winners: dict[int, set[int]] = defaultdict(set)

    for _, row in season_results.iterrows():
        day_num = int(row["DayNum"])
        round_num = _daynum_to_round(day_num)
        winners[round_num].add(int(row["WTeamID"]))

    return dict(winners)


def _resolve_bracket_through_round(
    bracket: TournamentBracket,
    results_df: pd.DataFrame,
    season: int,
    up_to_round: int,
) -> None:
    """Resolve bracket games through a given round using actual results.

    This propagates winners forward so that later-round matchups are populated.

    Args:
        bracket: Bracket to modify in place.
        results_df: Kaggle compact results.
        season: Tournament year.
        up_to_round: Resolve games in rounds 1..up_to_round (inclusive).
    """
    season_results = results_df[results_df["Season"] == season]

    for rnd in range(1, up_to_round + 1):
        round_results = season_results[
            season_results["DayNum"].apply(_daynum_to_round) == rnd
        ]
        for _, row in round_results.iterrows():
            winner_id = int(row["WTeamID"])
            loser_id = int(row["LTeamID"])

            # Find the slot for this game
            for i, slot in enumerate(bracket.slots):
                if slot.round_num != rnd:
                    continue
                teams = {slot.team_a, slot.team_b}
                if winner_id in teams and loser_id in teams:
                    bracket.resolve_game(i, winner_id)
                    break


# ---------------------------------------------------------------------------
# Pick strategies
# ---------------------------------------------------------------------------


def _strategy_top_seeds(
    available_teams: dict[int, int],
    n_picks: int,
    **kwargs,
) -> list[int]:
    """Pick the highest-seeded (lowest seed number) available teams.

    Args:
        available_teams: team_id -> seed number.
        n_picks: How many teams to pick.

    Returns:
        List of team IDs for the top-seeded teams.
    """
    sorted_teams = sorted(available_teams.items(), key=lambda x: x[1])
    return [t for t, _ in sorted_teams[:n_picks]]


def _strategy_random(
    available_teams: dict[int, int],
    n_picks: int,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> list[int]:
    """Pick random teams from the available pool.

    Args:
        available_teams: team_id -> seed number.
        n_picks: How many teams to pick.
        rng: NumPy random generator for reproducibility.

    Returns:
        List of randomly selected team IDs.
    """
    if rng is None:
        rng = np.random.default_rng()
    team_ids = list(available_teams.keys())
    n_picks = min(n_picks, len(team_ids))
    chosen = rng.choice(team_ids, size=n_picks, replace=False)
    return list(chosen)


def _strategy_contrarian(
    available_teams: dict[int, int],
    n_picks: int,
    ownership: dict[int, float] | None = None,
    **kwargs,
) -> list[int]:
    """Pick lowest-ownership teams that still have a reasonable win probability.

    Filters to teams with win_prob >= 0.3 before selecting lowest ownership.
    Falls back to all available teams if filter is too restrictive.

    Args:
        available_teams: team_id -> seed number.
        n_picks: How many teams to pick.
        ownership: team_id -> ownership fraction.

    Returns:
        List of team IDs with lowest public ownership.
    """
    if ownership is None:
        # Without ownership data, fall back to picking worst seeds (most contrarian)
        sorted_teams = sorted(available_teams.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in sorted_teams[:n_picks]]

    win_probs = kwargs.get("win_probs", {})

    # Filter to teams with reasonable win probability
    viable = {
        t: ownership.get(t, 0.0)
        for t in available_teams
        if win_probs.get(t, 0.5) >= 0.3
    }
    if len(viable) < n_picks:
        viable = {t: ownership.get(t, 0.0) for t in available_teams}

    # Sort by ownership ascending (lowest first)
    sorted_teams = sorted(viable.items(), key=lambda x: x[1])
    return [t for t, _ in sorted_teams[:n_picks]]


def _strategy_optimizer(
    available_teams: dict[int, int],
    n_picks: int,
    n_entries: int,
    win_probs: dict[int, float],
    ownership: dict[int, float],
    pool_size: int,
    prize_pool: float,
    used_teams_per_entry: list[set[int]],
    matchup_pairs: list[tuple[int, int]] | None = None,
    **kwargs,
) -> list[list[int]]:
    """Use the analytical optimizer to pick teams.

    Returns a list of pick-sets, one per entry. Each pick-set contains
    ``n_picks`` team IDs.

    Args:
        available_teams: team_id -> seed.
        n_picks: Picks required this day (1 or 2).
        n_entries: Total live entries.
        win_probs: team_id -> P(win).
        ownership: team_id -> public ownership fraction.
        pool_size: Simulated pool size.
        prize_pool: Prize money.
        used_teams_per_entry: Already-used teams per entry.
        matchup_pairs: (team_a, team_b) pairs for this day's games.

    Returns:
        List of pick-sets: [[teamA, teamB], ...] or [[teamA], ...].
    """
    return optimal_day_picks(
        n_entries=n_entries,
        available_teams=available_teams,
        win_probs=win_probs,
        ownership=ownership,
        pool_size=pool_size,
        prize_pool=prize_pool,
        num_picks=n_picks,
        used_teams_per_entry=used_teams_per_entry,
        matchup_pairs=matchup_pairs,
    )


# ---------------------------------------------------------------------------
# Win-probability helpers
# ---------------------------------------------------------------------------


def _build_win_probs(
    bracket: TournamentBracket,
    round_num: int,
    regions: list[str],
    model: WinProbabilityModel,
    features_df: pd.DataFrame,
    season: int,
) -> dict[int, float]:
    """Compute win probabilities for all matchups in a day's games.

    Uses the trained model to predict P(team_a wins) for each game,
    then maps each team to its win probability.

    Args:
        bracket: Current bracket state with matchups populated.
        round_num: Bracket round (1-6).
        regions: Regions playing on this contest day.
        model: Trained win probability model.
        features_df: Full feature DataFrame (used for column structure).
        season: Current season (excluded from training).

    Returns:
        team_id -> P(win this game).
    """
    matchups = bracket.get_day_matchups(round_num, regions)
    win_probs: dict[int, float] = {}

    for team_a, team_b, slot_idx in matchups:
        if team_a is None or team_b is None:
            continue

        # Build a minimal feature row for this matchup
        prob = _predict_matchup(model, team_a, team_b, bracket, features_df, season)
        win_probs[team_a] = prob
        win_probs[team_b] = 1.0 - prob

    return win_probs


def _predict_matchup(
    model: WinProbabilityModel,
    team_a: int,
    team_b: int,
    bracket: TournamentBracket,
    features_df: pd.DataFrame,
    season: int,
) -> float:
    """Predict P(team_a wins) using the model.

    Falls back to seed-based heuristic if feature construction fails.

    Args:
        model: Trained model with predict_proba().
        team_a: First team ID.
        team_b: Second team ID.
        bracket: Bracket for seed/team lookups.
        features_df: Feature data for column schema reference.
        season: Season being backtested.

    Returns:
        Probability that team_a wins (0.0 to 1.0).
    """
    info_a = bracket.teams.get(team_a, {})
    info_b = bracket.teams.get(team_b, {})
    seed_a = info_a.get("seed", 8)
    seed_b = info_b.get("seed", 8)

    # Try to find a pre-computed feature row for this exact matchup
    mask = (
        (features_df["Season"] == season)
        & (features_df["TeamA"] == team_a)
        & (features_df["TeamB"] == team_b)
    )
    rows = features_df[mask]
    if not rows.empty:
        try:
            prob = model.predict_proba(rows.iloc[[0]])
            if hasattr(prob, "__len__") and len(prob) > 0:
                return float(prob[0]) if np.isfinite(prob[0]) else 0.5
            return float(prob) if np.isfinite(prob) else 0.5
        except Exception:
            pass

    # Try the reverse matchup
    mask_rev = (
        (features_df["Season"] == season)
        & (features_df["TeamA"] == team_b)
        & (features_df["TeamB"] == team_a)
    )
    rows_rev = features_df[mask_rev]
    if not rows_rev.empty:
        try:
            prob_rev = model.predict_proba(rows_rev.iloc[[0]])
            val = float(prob_rev[0]) if hasattr(prob_rev, "__len__") else float(prob_rev)
            return 1.0 - val if np.isfinite(val) else 0.5
        except Exception:
            pass

    # Fallback: seed-based logistic approximation
    # P(better seed wins) ~ logistic(0.15 * seed_diff)
    seed_diff = seed_b - seed_a  # positive means A is favored
    return 1.0 / (1.0 + np.exp(-0.15 * seed_diff))


# ---------------------------------------------------------------------------
# Core backtest logic
# ---------------------------------------------------------------------------


def backtest_season(
    season: int,
    features_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    results_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    strategy: str = "optimizer",
    n_entries: int = 5,
    pool_size: int = 10000,
    prize_pool: float = 250000,
    model_type: str = "xgboost",
    calibrate: bool = True,
) -> BacktestResult:
    """Run a full backtest for one season.

    Workflow:
        1. Train model on all seasons EXCEPT this one (LOSO).
        2. Reconstruct this season's bracket from Kaggle seeds.
        3. For each contest day:
           a. Use the chosen strategy to select picks for alive entries.
           b. Check picks against actual tournament winners.
           c. Eliminate entries whose picks lost.
        4. Return survival results.

    Args:
        season: Year to backtest (held out of training).
        features_df: Pre-built matchup features for all seasons.
        seeds_df: MNCAATourneySeeds.
        results_df: MNCAATourneyCompactResults.
        teams_df: MTeams.
        strategy: One of "optimizer", "top_seeds", "random", "contrarian".
        n_entries: Number of simulated entries.
        pool_size: Simulated pool size for EV calculations.
        prize_pool: Simulated prize pool for EV calculations.
        model_type: Model architecture (passed to ``get_model``).
        calibrate: Whether to calibrate the model.

    Returns:
        BacktestResult with per-entry survival data.
    """
    logger.info("Backtesting season %d with strategy '%s'", season, strategy)

    # 1. Train LOSO model (train on everything except this season)
    train_df = features_df[features_df["Season"] != season]
    if train_df.empty or len(train_df) < 50:
        logger.warning("Insufficient training data for season %d", season)
        return BacktestResult(
            season=season,
            strategy=strategy,
            n_entries=n_entries,
            days_survived=[0] * n_entries,
            final_alive=0,
        )

    model = get_model(model_type, calibrate)
    model.fit(train_df, train_df["Result"])

    # 2. Reconstruct bracket
    bracket = reconstruct_bracket(season, seeds_df, teams_df)
    if not bracket.teams:
        logger.warning("No bracket data for season %d", season)
        return BacktestResult(
            season=season,
            strategy=strategy,
            n_entries=n_entries,
            days_survived=[0] * n_entries,
            final_alive=0,
        )

    # 3. Get actual winners per round
    actual_winners = get_actual_winners(season, results_df)

    # 4. Simulate contest days
    rng = np.random.default_rng(seed=season)
    alive = [True] * n_entries
    days_survived = [0] * n_entries
    used_teams: list[set[int]] = [set() for _ in range(n_entries)]
    picks_per_day: dict[int, list] = {}

    for day_info in _BACKTEST_SCHEDULE:
        day_num = day_info["day"]
        round_num = day_info["round"]
        regions = day_info["regions"]
        num_picks = day_info["num_picks"]

        # Resolve bracket through prior rounds so matchups are populated
        if round_num > 1:
            _resolve_bracket_through_round(bracket, results_df, season, round_num - 1)

        # Get matchups for this day
        matchups = bracket.get_day_matchups(round_num, regions)
        if not matchups:
            logger.debug(
                "No matchups for season %d day %d (round %d, regions %s)",
                season, day_num, round_num, regions,
            )
            continue

        # Build available teams and win probs
        available_teams: dict[int, int] = {}
        matchup_pairs: list[tuple[int, int]] = []
        for team_a, team_b, _ in matchups:
            if team_a is not None and team_a in bracket.teams:
                available_teams[team_a] = bracket.teams[team_a]["seed"]
            if team_b is not None and team_b in bracket.teams:
                available_teams[team_b] = bracket.teams[team_b]["seed"]
            if team_a is not None and team_b is not None:
                matchup_pairs.append((team_a, team_b))

        if not available_teams:
            continue

        # Compute win probabilities
        win_probs = _build_win_probs(
            bracket, round_num, regions, model, features_df, season,
        )

        # Compute ownership estimates
        ownership = estimate_ownership_from_bracket(
            bracket, round_num, win_probs,
            pool_sophistication=0.5,
            method="heuristic",
            pool_size=pool_size,
            prize_pool=prize_pool,
            regions=regions if regions != ["FF"] else None,
        )

        # Count alive entries
        alive_indices = [i for i in range(n_entries) if alive[i]]
        n_alive = len(alive_indices)
        if n_alive == 0:
            break

        # Get picks based on strategy
        day_picks: list[list[int]] = []

        if strategy == "optimizer":
            alive_used = [used_teams[i] for i in alive_indices]
            try:
                raw_picks = _strategy_optimizer(
                    available_teams=available_teams,
                    n_picks=num_picks,
                    n_entries=n_alive,
                    win_probs=win_probs,
                    ownership=ownership,
                    pool_size=pool_size,
                    prize_pool=prize_pool,
                    used_teams_per_entry=alive_used,
                    matchup_pairs=matchup_pairs,
                )
                day_picks = raw_picks
            except Exception as e:
                logger.warning(
                    "Optimizer failed for season %d day %d: %s. Falling back to top_seeds.",
                    season, day_num, e,
                )
                # Fallback: give each entry the top seeds
                top_picks = _strategy_top_seeds(available_teams, num_picks)
                day_picks = [top_picks[:] for _ in range(n_alive)]

        elif strategy == "top_seeds":
            top_picks = _strategy_top_seeds(available_teams, num_picks)
            day_picks = [top_picks[:] for _ in range(n_alive)]

        elif strategy == "random":
            day_picks = [
                _strategy_random(available_teams, num_picks, rng=rng)
                for _ in range(n_alive)
            ]

        elif strategy == "contrarian":
            contrarian_picks = _strategy_contrarian(
                available_teams, num_picks,
                ownership=ownership,
                win_probs=win_probs,
            )
            day_picks = [contrarian_picks[:] for _ in range(n_alive)]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Pad if fewer picks returned than alive entries
        while len(day_picks) < n_alive:
            fallback = _strategy_top_seeds(available_teams, num_picks)
            day_picks.append(fallback)

        # Record picks
        picks_per_day[day_num] = list(zip(alive_indices, day_picks))

        # Get actual round winners
        round_winners = actual_winners.get(round_num, set())

        # Check each alive entry's picks against actual results
        for idx_in_alive, entry_idx in enumerate(alive_indices):
            pick_set = day_picks[idx_in_alive]

            # All picks must have won for the entry to survive
            all_won = all(team_id in round_winners for team_id in pick_set)

            if all_won:
                days_survived[entry_idx] = day_num
                # Track used teams
                for t in pick_set:
                    used_teams[entry_idx].add(t)
            else:
                alive[entry_idx] = False

    final_alive = sum(alive)

    return BacktestResult(
        season=season,
        strategy=strategy,
        n_entries=n_entries,
        days_survived=days_survived,
        final_alive=final_alive,
        picks_per_day=picks_per_day,
    )


# ---------------------------------------------------------------------------
# Multi-season / multi-strategy runner
# ---------------------------------------------------------------------------


def backtest_all(
    features_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    results_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    seasons: list[int] | None = None,
    strategies: list[str] | None = None,
    n_entries: int = 5,
    pool_size: int = 10000,
    prize_pool: float = 250000,
    model_type: str = "xgboost",
    calibrate: bool = True,
) -> pd.DataFrame:
    """Run backtests across multiple seasons and strategies.

    Args:
        features_df: Pre-built matchup features for all seasons.
        seeds_df: MNCAATourneySeeds.
        results_df: MNCAATourneyCompactResults.
        teams_df: MTeams.
        seasons: Seasons to test. Defaults to all available seasons
            that have both seeds and results.
        strategies: Strategies to test. Defaults to all four.
        n_entries: Number of entries per strategy per season.
        pool_size: Simulated pool size.
        prize_pool: Simulated prize pool.
        model_type: Model architecture.
        calibrate: Whether to calibrate models.

    Returns:
        Summary DataFrame with columns: Season, Strategy, NEntries,
        AvgDaysSurvived, MaxDaysSurvived, FinalAlive, SurvivalRate.
    """
    if strategies is None:
        strategies = ["optimizer", "top_seeds", "random", "contrarian"]

    if seasons is None:
        # Use seasons that exist in both seeds and results
        seed_seasons = set(seeds_df["Season"].unique())
        result_seasons = set(results_df["Season"].unique())
        feature_seasons = set(features_df["Season"].unique())
        seasons = sorted(seed_seasons & result_seasons & feature_seasons)

    if not seasons:
        logger.warning("No valid seasons found for backtesting.")
        return pd.DataFrame()

    rows = []
    for season in seasons:
        for strategy in strategies:
            logger.info("Running backtest: season=%d strategy=%s", season, strategy)
            try:
                result = backtest_season(
                    season=season,
                    features_df=features_df,
                    seeds_df=seeds_df,
                    results_df=results_df,
                    teams_df=teams_df,
                    strategy=strategy,
                    n_entries=n_entries,
                    pool_size=pool_size,
                    prize_pool=prize_pool,
                    model_type=model_type,
                    calibrate=calibrate,
                )
                rows.append({
                    "Season": season,
                    "Strategy": strategy,
                    "NEntries": result.n_entries,
                    "AvgDaysSurvived": result.avg_days_survived,
                    "MaxDaysSurvived": result.max_days_survived,
                    "FinalAlive": result.final_alive,
                    "SurvivalRate": result.survival_rate,
                })
            except Exception as e:
                logger.error(
                    "Backtest failed for season %d strategy %s: %s",
                    season, strategy, e,
                )
                rows.append({
                    "Season": season,
                    "Strategy": strategy,
                    "NEntries": n_entries,
                    "AvgDaysSurvived": 0.0,
                    "MaxDaysSurvived": 0,
                    "FinalAlive": 0,
                    "SurvivalRate": 0.0,
                })

    summary = pd.DataFrame(rows)

    if not summary.empty:
        # Log aggregate stats
        for strat in strategies:
            strat_df = summary[summary["Strategy"] == strat]
            if not strat_df.empty:
                logger.info(
                    "Strategy '%s' across %d seasons: avg_days=%.2f, "
                    "avg_survival_rate=%.3f, total_final_alive=%d",
                    strat,
                    len(strat_df),
                    strat_df["AvgDaysSurvived"].mean(),
                    strat_df["SurvivalRate"].mean(),
                    strat_df["FinalAlive"].sum(),
                )

    return summary
