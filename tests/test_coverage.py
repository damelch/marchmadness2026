"""Tests for coverage guarantee algorithm."""

from contest.schedule import ContestDay, ContestSchedule
from optimizer.coverage import compute_coverage
from simulation.engine import TournamentBracket


def _make_bracket_with_matchups() -> TournamentBracket:
    """Create a bracket with Round 1 matchups in region W."""
    bracket = TournamentBracket()
    teams = [
        # Region W (8 matchups in R1)
        (1001, "Duke", 1, "W"),
        (1002, "Howard", 16, "W"),
        (1003, "Iowa", 8, "W"),
        (1004, "Texas A&M", 9, "W"),
        (1005, "Arkansas", 5, "W"),
        (1006, "Stephen F. Austin", 12, "W"),
        (1007, "Texas Tech", 4, "W"),
        (1008, "Utah Valley", 13, "W"),
        (1009, "Wisconsin", 6, "W"),
        (1010, "Miami (OH)", 11, "W"),
        (1011, "Iowa St.", 3, "W"),
        (1012, "Portland St.", 14, "W"),
        (1013, "Saint Mary's", 7, "W"),
        (1014, "NC State", 10, "W"),
        (1015, "Illinois", 2, "W"),
        (1016, "Furman", 15, "W"),
        # Region X
        (1017, "Florida", 1, "X"),
        (1018, "Queens", 16, "X"),
        (1019, "UCLA", 8, "X"),
        (1020, "St. Louis", 9, "X"),
        (1021, "North Carolina", 5, "X"),
        (1022, "South Florida", 12, "X"),
        (1023, "Kansas", 4, "X"),
        (1024, "North Dakota St.", 13, "X"),
        (1025, "Louisville", 6, "X"),
        (1026, "Santa Clara", 11, "X"),
        (1027, "Purdue", 3, "X"),
        (1028, "Hofstra", 14, "X"),
        (1029, "Georgia", 7, "X"),
        (1030, "Ohio St.", 10, "X"),
        (1031, "Connecticut", 2, "X"),
        (1032, "Merrimack", 15, "X"),
    ]
    for tid, name, seed, region in teams:
        bracket.set_seed(tid, seed, region, name=name)
    return bracket


class TestCoverage:
    def test_all_available(self):
        bracket = _make_bracket_with_matchups()
        alive = set(bracket.teams.keys())
        used: set[int] = set()
        # Use a single-day schedule covering only W+X (the regions in our test bracket)
        sched = ContestSchedule([
            ContestDay(1, "R64 Thursday", "2026-03-19", 1, 2, ["W", "X"]),
        ])

        result = compute_coverage(bracket, alive, used, 1, sched)
        assert result.risk_level == "safe"
        assert len(result.available_teams) == len(alive)
        assert len(result.uncovered_matchups) == 0

    def test_used_teams_excluded(self):
        bracket = _make_bracket_with_matchups()
        alive = set(bracket.teams.keys())
        used = {1001, 1003}  # Duke and Iowa
        sched = ContestSchedule.default()

        result = compute_coverage(bracket, alive, used, 1, sched)
        available_ids = {t["id"] for t in result.available_teams}
        assert 1001 not in available_ids
        assert 1003 not in available_ids

    def test_eliminated_teams_excluded(self):
        bracket = _make_bracket_with_matchups()
        alive = set(bracket.teams.keys()) - {1002, 1004}  # Howard and Texas A&M eliminated
        used: set[int] = set()
        sched = ContestSchedule.default()

        result = compute_coverage(bracket, alive, used, 1, sched)
        available_ids = {t["id"] for t in result.available_teams}
        assert 1002 not in available_ids
        assert 1004 not in available_ids

    def test_uncovered_matchup(self):
        """Both sides of a matchup are used → uncovered."""
        bracket = _make_bracket_with_matchups()
        alive = set(bracket.teams.keys())
        # Use both Duke (1001) and Howard (1002) who play each other in R1
        used = {1001, 1002}
        sched = ContestSchedule.default()

        result = compute_coverage(bracket, alive, used, 1, sched)
        assert len(result.uncovered_matchups) == 1
        assert set(result.uncovered_matchups[0]) == {1001, 1002}
        assert result.risk_level == "critical"

    def test_safety_set_covers_matchups(self):
        bracket = _make_bracket_with_matchups()
        alive = set(bracket.teams.keys())
        used: set[int] = set()
        sched = ContestSchedule.default()
        # Day 1 covers regions W and X
        result = compute_coverage(bracket, alive, used, 1, sched)
        # Safety set should have one team from each matchup
        assert len(result.safety_set) > 0

    def test_critical_when_no_teams(self):
        bracket = _make_bracket_with_matchups()
        alive = set(bracket.teams.keys())
        # Use ALL teams
        used = set(alive)
        sched = ContestSchedule.default()

        result = compute_coverage(bracket, alive, used, 1, sched)
        assert result.risk_level == "critical"
        assert len(result.available_teams) == 0

    def test_win_probs_affect_safety_set_order(self):
        bracket = _make_bracket_with_matchups()
        alive = set(bracket.teams.keys())
        used: set[int] = set()
        sched = ContestSchedule.default()
        # Give Duke a very high win prob
        win_probs = {1001: 0.99, 1002: 0.01}

        result = compute_coverage(bracket, alive, used, 1, sched, win_probs)
        # Duke should be in the safety set (higher win prob)
        assert 1001 in result.safety_set

    def test_available_sorted_by_win_prob(self):
        bracket = _make_bracket_with_matchups()
        alive = set(bracket.teams.keys())
        used: set[int] = set()
        sched = ContestSchedule.default()
        win_probs = {1001: 0.95, 1015: 0.90, 1003: 0.50}

        result = compute_coverage(bracket, alive, used, 1, sched, win_probs)
        # First available should be highest win prob
        probs = [t["win_prob"] for t in result.available_teams if t["id"] in win_probs]
        assert probs == sorted(probs, reverse=True)

    def test_picks_remaining_calculation(self):
        bracket = _make_bracket_with_matchups()
        alive = set(bracket.teams.keys())
        sched = ContestSchedule.default()

        result_day1 = compute_coverage(bracket, alive, set(), 1, sched)
        result_day5 = compute_coverage(bracket, alive, set(), 5, sched)
        assert result_day1.picks_remaining > result_day5.picks_remaining
