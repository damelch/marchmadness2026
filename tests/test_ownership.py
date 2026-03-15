"""Tests for ownership estimation."""

from optimizer.ownership import (
    OwnershipConfig,
    estimate_ownership,
)


def _make_teams(n=8):
    """Create simple team dict: id -> seed."""
    return {i: i for i in range(1, n + 1)}


def _make_win_probs(teams):
    """Higher seed = higher win prob."""
    return {t: max(0.5, 1.0 - seed * 0.05) for t, seed in teams.items()}


class TestOwnershipBasics:
    def test_sums_to_one(self):
        teams = _make_teams()
        wp = _make_win_probs(teams)
        own = estimate_ownership(teams, wp, round_num=1)
        assert abs(sum(own.values()) - 1.0) < 1e-10

    def test_sums_to_one_with_config(self):
        teams = _make_teams(16)
        wp = _make_win_probs(teams)
        config = OwnershipConfig(brand_bias={"Duke": 2.0})
        own = estimate_ownership(teams, wp, round_num=1, config=config)
        assert abs(sum(own.values()) - 1.0) < 1e-10

    def test_higher_seed_gets_more_ownership(self):
        teams = {1: 1, 2: 16}
        wp = {1: 0.97, 2: 0.03}
        own = estimate_ownership(teams, wp, round_num=1)
        assert own[1] > own[2]

    def test_empty_teams(self):
        own = estimate_ownership({}, {}, round_num=1)
        assert own == {}


class TestBrandBias:
    def test_brand_bias_increases_favorites(self):
        teams = {1: 3, 2: 3}  # Same seed
        wp = {1: 0.85, 2: 0.85}  # Same win prob
        names = {1: "Duke", 2: "Random Team"}

        own_with_brand = estimate_ownership(
            teams, wp, round_num=1, team_names=names,
            config=OwnershipConfig(),
        )
        own_without_brand = estimate_ownership(
            teams, wp, round_num=1, team_names=names,
            config=OwnershipConfig(brand_bias={}),
        )
        # Duke should have higher ownership with brand bias
        assert own_with_brand[1] > own_without_brand[1]

    def test_brand_bias_no_effect_on_unknown(self):
        teams = {1: 5, 2: 5}
        wp = {1: 0.8, 2: 0.8}
        names = {1: "Team A", 2: "Team B"}
        own = estimate_ownership(
            teams, wp, round_num=1, team_names=names,
            config=OwnershipConfig(),
        )
        # Unknown teams should be equal
        assert abs(own[1] - own[2]) < 1e-10


class TestSophistication:
    def test_casual_pool_favors_top_seeds(self):
        teams = {1: 1, 2: 8}
        wp = {1: 0.97, 2: 0.52}
        own_casual = estimate_ownership(teams, wp, round_num=1, pool_sophistication=0.0)
        own_sharp = estimate_ownership(teams, wp, round_num=1, pool_sophistication=1.0)
        # Casual pool should concentrate more on the 1-seed
        assert own_casual[1] > own_sharp[1]

    def test_sharp_pool_more_uniform(self):
        teams = _make_teams(4)
        wp = _make_win_probs(teams)
        own_casual = estimate_ownership(teams, wp, round_num=1, pool_sophistication=0.0)
        own_sharp = estimate_ownership(teams, wp, round_num=1, pool_sophistication=1.0)
        # Sharp pool should have lower max ownership
        assert max(own_sharp.values()) < max(own_casual.values())


class TestRecencyBias:
    def test_recency_boosts_champion(self):
        teams = {1: 2, 2: 2}  # Same seed
        wp = {1: 0.94, 2: 0.94}
        names = {1: "Connecticut", 2: "Other Team"}
        own = estimate_ownership(
            teams, wp, round_num=1, team_names=names,
            config=OwnershipConfig(),
        )
        # UConn should be higher due to recency bias
        assert own[1] > own[2]
