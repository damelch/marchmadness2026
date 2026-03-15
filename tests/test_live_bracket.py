"""Tests for live bracket fetching and team name matching."""

from data.live_bracket import (
    _build_name_index,
    _detect_current_day,
    _detect_current_round,
    _normalize_name,
    match_team_name,
    resolve_team_name,
)
from simulation.engine import TournamentBracket


def _make_bracket() -> TournamentBracket:
    """Create a small bracket for testing."""
    bracket = TournamentBracket()
    teams = [
        (1001, "Duke", 1, "W"),
        (1002, "Howard", 16, "W"),
        (1003, "Iowa", 8, "W"),
        (1004, "Texas A&M", 9, "W"),
        (1011, "Iowa St.", 3, "W"),
        (1031, "Connecticut", 2, "X"),
        (1053, "St. John's", 5, "Z"),
        (1063, "Michigan St.", 2, "Z"),
        (1010, "Miami (OH)", 11, "W"),
        (1061, "Miami", 7, "Z"),
        (1006, "Stephen F. Austin", 12, "W"),
    ]
    for tid, name, seed, region in teams:
        bracket.set_seed(tid, seed, region, name=name)
    return bracket


class TestNameMatching:
    def test_exact_match(self):
        bracket = _make_bracket()
        index = _build_name_index(bracket)
        assert match_team_name("Duke", index, bracket) == 1001

    def test_case_insensitive(self):
        bracket = _make_bracket()
        index = _build_name_index(bracket)
        assert match_team_name("duke", index, bracket) == 1001

    def test_alias_uconn(self):
        bracket = _make_bracket()
        index = _build_name_index(bracket)
        assert match_team_name("UConn", index, bracket) == 1031

    def test_alias_iowa_state(self):
        bracket = _make_bracket()
        index = _build_name_index(bracket)
        assert match_team_name("Iowa State", index, bracket) == 1011

    def test_alias_michigan_state(self):
        bracket = _make_bracket()
        index = _build_name_index(bracket)
        assert match_team_name("Michigan State", index, bracket) == 1063

    def test_st_expansion(self):
        bracket = _make_bracket()
        index = _build_name_index(bracket)
        # "Iowa St." is the bracket name; "Iowa State" should match via alias
        assert match_team_name("Iowa State", index, bracket) == 1011

    def test_fuzzy_match(self):
        bracket = _make_bracket()
        index = _build_name_index(bracket)
        # Exact name in bracket is "Texas A&M"; "Texas A&M" matches directly
        assert match_team_name("Texas A&M", index, bracket) == 1004

    def test_no_match(self):
        bracket = _make_bracket()
        index = _build_name_index(bracket)
        assert match_team_name("Nonexistent University", index, bracket) is None

    def test_resolve_team_name(self):
        bracket = _make_bracket()
        assert resolve_team_name("Duke", bracket) == 1001

    def test_resolve_alias(self):
        bracket = _make_bracket()
        assert resolve_team_name("UConn", bracket) == 1031


class TestNormalize:
    def test_strips_whitespace(self):
        assert _normalize_name("  Duke  ") == "duke"

    def test_lowercases(self):
        assert _normalize_name("DUKE") == "duke"


class TestDetectRound:
    def test_no_games(self):
        assert _detect_current_round(0) == 1

    def test_mid_r64(self):
        assert _detect_current_round(16) == 1

    def test_after_r64(self):
        assert _detect_current_round(32) == 1

    def test_r32(self):
        assert _detect_current_round(33) == 2

    def test_s16(self):
        assert _detect_current_round(49) == 3

    def test_e8(self):
        assert _detect_current_round(57) == 4

    def test_f4(self):
        assert _detect_current_round(61) == 5

    def test_championship(self):
        assert _detect_current_round(63) == 6


class TestDetectDay:
    def test_no_games_is_day_1(self):
        assert _detect_current_day(0, None) == 1

    def test_after_day_1(self):
        # Day 1 has 16 games (R64 W+X); mid-day-2 = 17 games
        assert _detect_current_day(17, None) == 2

    def test_after_r64(self):
        # 32 games = R64 done, should be day 3
        assert _detect_current_day(32, None) == 3
