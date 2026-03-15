"""Portfolio distribution analysis for multi-entry survivor pools.

Analyzes team concentration, simulates survival distributions,
and computes entry correlation (how many entries die together on upsets).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from contest.schedule import ContestSchedule
from entries.manager import EntryManager
from simulation.engine import TournamentBracket, simulate_tournament


@dataclass
class TeamConcentration:
    """How entries are distributed across teams on a given day."""

    day_num: int
    label: str
    team_counts: dict[int, int]         # team_id -> number of entries picking it
    team_fractions: dict[int, float]     # team_id -> fraction of alive entries
    max_concentration: float             # highest fraction on any one team
    max_concentration_team: int
    hhi: float                           # Herfindahl-Hirschman Index (0-1, lower = more diversified)
    n_unique_teams: int
    n_entries: int


@dataclass
class SurvivalDistribution:
    """Monte Carlo survival analysis across bracket outcomes."""

    day_num: int
    label: str
    mean_alive: float
    median_alive: float
    std_alive: float
    min_alive: int
    max_alive: int
    p_zero: float            # P(all entries eliminated)
    p_at_least_one: float    # P(at least one entry survives)
    percentiles: dict[int, int]  # {5: x, 25: x, 50: x, 75: x, 95: x}
    alive_counts: list[int]  # raw counts per sim for histogramming


@dataclass
class CorrelationAnalysis:
    """How correlated entry eliminations are."""

    # Per-team: how many entries die if this team loses
    team_exposure: dict[int, int]    # team_id -> entries eliminated if team loses
    max_exposure_team: int
    max_exposure_count: int
    # Pairwise: fraction of sims where entries i and j both die on the same day
    # (only computed for small portfolios)
    pairwise_correlation: np.ndarray | None  # (n_entries x n_entries) matrix
    mean_pairwise: float             # average pairwise death correlation


@dataclass
class DistributionReport:
    """Full distribution analysis report."""

    n_entries: int
    n_alive: int
    concentration_by_day: list[TeamConcentration]
    survival: list[SurvivalDistribution]
    correlation: CorrelationAnalysis
    bracket: TournamentBracket


def analyze_distribution(
    bracket: TournamentBracket,
    entry_manager: EntryManager,
    schedule: ContestSchedule,
    predict_fn,
    n_sims: int = 10000,
    rng_seed: int = 42,
) -> DistributionReport:
    """Run full distribution analysis on the entry portfolio.

    Args:
        bracket: Tournament bracket with teams seeded.
        entry_manager: Entry manager with picks recorded.
        schedule: Contest schedule.
        predict_fn: Function (team_a, team_b) -> P(team_a wins).
        n_sims: Number of Monte Carlo simulations.
        rng_seed: Random seed for reproducibility.

    Returns:
        DistributionReport with concentration, survival, and correlation data.
    """
    alive_entries = entry_manager.get_alive_entries()
    n_alive = len(alive_entries)

    # 1. Team concentration per day
    concentration = _compute_concentration(alive_entries, schedule, bracket)

    # 2. Monte Carlo survival distribution
    sim_results = simulate_tournament(bracket, predict_fn, n_sims, rng_seed)
    survival = _compute_survival(
        alive_entries, sim_results, bracket, schedule,
    )

    # 3. Correlation analysis
    correlation = _compute_correlation(
        alive_entries, sim_results, bracket, schedule,
    )

    return DistributionReport(
        n_entries=len(entry_manager.entries),
        n_alive=n_alive,
        concentration_by_day=concentration,
        survival=survival,
        correlation=correlation,
        bracket=bracket,
    )


def _compute_concentration(
    entries: list,
    schedule: ContestSchedule,
    bracket: TournamentBracket,
) -> list[TeamConcentration]:
    """Compute team concentration for each day with picks."""
    results = []
    n_entries = len(entries)
    if n_entries == 0:
        return results

    for day in schedule.days:
        counts: Counter = Counter()
        entries_with_picks = 0
        for entry in entries:
            day_picks = entry.picks.get(day.day_num)
            if day_picks:
                entries_with_picks += 1
                for team_id in day_picks:
                    counts[team_id] += 1

        if entries_with_picks == 0:
            continue

        fractions = {t: c / entries_with_picks for t, c in counts.items()}
        hhi = sum(f * f for f in fractions.values())

        max_team = max(counts, key=counts.get) if counts else 0
        max_frac = fractions.get(max_team, 0)

        results.append(TeamConcentration(
            day_num=day.day_num,
            label=day.label,
            team_counts=dict(counts),
            team_fractions=fractions,
            max_concentration=max_frac,
            max_concentration_team=max_team,
            hhi=hhi,
            n_unique_teams=len(counts),
            n_entries=entries_with_picks,
        ))

    return results


def _compute_survival(
    entries: list,
    sim_results: np.ndarray,
    bracket: TournamentBracket,
    schedule: ContestSchedule,
) -> list[SurvivalDistribution]:
    """Simulate survival across bracket outcomes for each day."""
    n_sims = sim_results.shape[0]
    results = []

    for day in schedule.days:
        # Find which entries have picks for this day
        entries_with_picks = [e for e in entries if day.day_num in e.picks]
        if not entries_with_picks:
            continue

        # For each sim, count how many entries survive this day
        alive_counts = np.zeros(n_sims, dtype=np.int32)

        # Get the slots for this day's round + regions
        day_slots = []
        for i, slot in enumerate(bracket.slots):
            if slot.round_num == day.round_num and slot.region in day.regions:
                day_slots.append(i)

        for sim_idx in range(n_sims):
            # Winners in this sim for this day's games
            winners = set()
            for slot_idx in day_slots:
                w = sim_results[sim_idx, slot_idx]
                if w > 0:
                    winners.add(int(w))

            # Count surviving entries
            survived = 0
            for entry in entries_with_picks:
                day_picks = entry.picks[day.day_num]
                if all(t in winners for t in day_picks):
                    survived += 1
            alive_counts[sim_idx] = survived

        mean_a = float(np.mean(alive_counts))
        results.append(SurvivalDistribution(
            day_num=day.day_num,
            label=day.label,
            mean_alive=mean_a,
            median_alive=float(np.median(alive_counts)),
            std_alive=float(np.std(alive_counts)),
            min_alive=int(np.min(alive_counts)),
            max_alive=int(np.max(alive_counts)),
            p_zero=float(np.mean(alive_counts == 0)),
            p_at_least_one=float(np.mean(alive_counts > 0)),
            percentiles={
                5: int(np.percentile(alive_counts, 5)),
                25: int(np.percentile(alive_counts, 25)),
                50: int(np.percentile(alive_counts, 50)),
                75: int(np.percentile(alive_counts, 75)),
                95: int(np.percentile(alive_counts, 95)),
            },
            alive_counts=alive_counts.tolist(),
        ))

    return results


def _compute_correlation(
    entries: list,
    sim_results: np.ndarray,
    bracket: TournamentBracket,
    schedule: ContestSchedule,
) -> CorrelationAnalysis:
    """Compute entry elimination correlation.

    Team exposure: if team X loses, how many entries die?
    Pairwise: for each pair of entries, how often do they die on the same day?
    """
    n_entries = len(entries)
    n_sims = sim_results.shape[0]

    # Team exposure: count entries per team across all days
    team_to_entries: dict[int, set[int]] = {}
    for entry in entries:
        for _day_num, picks in entry.picks.items():
            for team_id in picks:
                team_to_entries.setdefault(team_id, set()).add(entry.entry_id)

    team_exposure = {t: len(eids) for t, eids in team_to_entries.items()}
    max_team = max(team_exposure, key=team_exposure.get) if team_exposure else 0
    max_count = team_exposure.get(max_team, 0)

    # Pairwise correlation (only for manageable portfolio sizes)
    pairwise = None
    mean_pairwise = 0.0

    if n_entries <= 200:
        # For each sim, determine which entries survive ALL their picked days
        # An entry "dies" if ANY of its picked days has a losing pick
        entry_died = np.zeros((n_sims, n_entries), dtype=bool)

        for ei, entry in enumerate(entries):
            for day in schedule.days:
                day_picks = entry.picks.get(day.day_num)
                if not day_picks:
                    continue

                # Get slots for this day
                day_slots = []
                for si, slot in enumerate(bracket.slots):
                    if slot.round_num == day.round_num and slot.region in day.regions:
                        day_slots.append(si)

                for sim_idx in range(n_sims):
                    if entry_died[sim_idx, ei]:
                        continue  # already dead from prior day
                    winners = {int(sim_results[sim_idx, s]) for s in day_slots
                               if sim_results[sim_idx, s] > 0}
                    if not all(t in winners for t in day_picks):
                        entry_died[sim_idx, ei] = True

        # Pairwise: fraction of sims where both entries die
        pairwise = np.zeros((n_entries, n_entries), dtype=np.float32)
        for i in range(n_entries):
            for j in range(i, n_entries):
                both_died = np.mean(entry_died[:, i] & entry_died[:, j])
                pairwise[i, j] = both_died
                pairwise[j, i] = both_died

        # Mean off-diagonal pairwise correlation
        if n_entries > 1:
            mask = ~np.eye(n_entries, dtype=bool)
            mean_pairwise = float(np.mean(pairwise[mask]))

    return CorrelationAnalysis(
        team_exposure=team_exposure,
        max_exposure_team=max_team,
        max_exposure_count=max_count,
        pairwise_correlation=pairwise,
        mean_pairwise=mean_pairwise,
    )
