"""Tests for distribution analysis module."""

import numpy as np

from contest.schedule import ContestDay, ContestSchedule
from entries.manager import EntryManager
from optimizer.distribution import (
    _compute_concentration,
    _compute_correlation,
    _compute_survival,
    analyze_distribution,
)
from simulation.engine import TournamentBracket


def _make_full_bracket() -> TournamentBracket:
    """Create a 64-team bracket."""
    bracket = TournamentBracket()
    team_id = 1000
    for region in ["W", "X", "Y", "Z"]:
        for seed in range(1, 17):
            bracket.set_seed(team_id, seed, region, name=f"{region}{seed}")
            team_id += 1
    return bracket


def _make_entries(n: int, bracket: TournamentBracket) -> EntryManager:
    """Create entries with Day 1 picks (double pick W+X)."""
    mgr = EntryManager()
    mgr.create_entries(n)
    teams = list(bracket.teams.keys())
    for i, entry in enumerate(mgr.entries):
        # Assign picks: entry i picks teams at index 2*i, 2*i+1
        idx = (2 * i) % len(teams)
        t1 = teams[idx]
        t2 = teams[(idx + 1) % len(teams)]
        entry.add_picks(1, [t1, t2])
    return mgr


class TestConcentration:
    def test_counts_correct(self):
        bracket = _make_full_bracket()
        mgr = _make_entries(4, bracket)
        sched = ContestSchedule([
            ContestDay(1, "R64 Thursday", "2026-03-19", 1, 2, ["W", "X"]),
        ])

        results = _compute_concentration(mgr.get_alive_entries(), sched, bracket)
        assert len(results) == 1
        conc = results[0]
        assert conc.n_entries == 4
        assert conc.hhi > 0  # some concentration
        assert conc.max_concentration > 0

    def test_hhi_one_team(self):
        """All entries on same team → HHI = 1."""
        bracket = _make_full_bracket()
        mgr = EntryManager()
        mgr.create_entries(5)
        teams = list(bracket.teams.keys())
        for entry in mgr.entries:
            entry.add_picks(1, [teams[0]])

        sched = ContestSchedule([
            ContestDay(1, "R64 Thursday", "2026-03-19", 1, 1, ["W"]),
        ])
        results = _compute_concentration(mgr.entries, sched, bracket)
        assert results[0].hhi == 1.0
        assert results[0].max_concentration == 1.0

    def test_no_picks(self):
        bracket = _make_full_bracket()
        mgr = EntryManager()
        mgr.create_entries(3)
        sched = ContestSchedule([
            ContestDay(1, "R64 Thursday", "2026-03-19", 1, 2, ["W", "X"]),
        ])
        results = _compute_concentration(mgr.entries, sched, bracket)
        assert len(results) == 0


class TestSurvival:
    def test_returns_distributions(self):
        bracket = _make_full_bracket()
        mgr = _make_entries(5, bracket)
        sched = ContestSchedule([
            ContestDay(1, "R64 Thursday", "2026-03-19", 1, 2, ["W", "X"]),
        ])

        def predict_fn(a, b):
            return 0.5

        sim_results = np.zeros((100, 63), dtype=np.int32)
        # Fill in round 1 W+X slots (0-15) with random winners
        rng = np.random.default_rng(42)
        for slot_idx in range(16):
            slot = bracket.slots[slot_idx]
            if slot.team_a and slot.team_b:
                winners = rng.choice([slot.team_a, slot.team_b], size=100)
                sim_results[:, slot_idx] = winners

        results = _compute_survival(mgr.entries, sim_results, bracket, sched)
        assert len(results) == 1
        surv = results[0]
        assert surv.mean_alive >= 0
        assert surv.p_at_least_one >= 0
        assert len(surv.alive_counts) == 100


class TestCorrelation:
    def test_team_exposure(self):
        bracket = _make_full_bracket()
        mgr = EntryManager()
        mgr.create_entries(3)
        teams = list(bracket.teams.keys())
        # All 3 entries pick the same team
        for entry in mgr.entries:
            entry.add_picks(1, [teams[0]])

        sched = ContestSchedule([
            ContestDay(1, "R64 Thursday", "2026-03-19", 1, 1, ["W"]),
        ])

        sim_results = np.zeros((10, 63), dtype=np.int32)
        corr = _compute_correlation(mgr.entries, sim_results, bracket, sched)
        # Team 0 should have exposure = 3
        assert corr.team_exposure[teams[0]] == 3
        assert corr.max_exposure_count == 3

    def test_pairwise_computed(self):
        bracket = _make_full_bracket()
        mgr = _make_entries(3, bracket)
        sched = ContestSchedule([
            ContestDay(1, "R64 Thursday", "2026-03-19", 1, 2, ["W", "X"]),
        ])
        sim_results = np.zeros((10, 63), dtype=np.int32)
        corr = _compute_correlation(mgr.entries, sim_results, bracket, sched)
        assert corr.pairwise_correlation is not None
        assert corr.pairwise_correlation.shape == (3, 3)


class TestAnalyzeDistribution:
    def test_full_report(self):
        bracket = _make_full_bracket()
        mgr = _make_entries(5, bracket)
        sched = ContestSchedule([
            ContestDay(1, "R64 Thursday", "2026-03-19", 1, 2, ["W", "X"]),
        ])

        def predict_fn(a, b):
            return 0.5

        report = analyze_distribution(bracket, mgr, sched, predict_fn, n_sims=50)
        assert report.n_alive == 5
        assert len(report.concentration_by_day) == 1
        assert len(report.survival) == 1
        assert report.correlation is not None
