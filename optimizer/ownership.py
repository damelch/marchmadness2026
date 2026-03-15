"""Estimate public ownership percentages for survivor pool picks.

Supports three methods:
- "heuristic": Seed-based popularity model (fast, good for casual pools)
- "nash": Game-theoretic Nash equilibrium (optimal for sharp pools)
- "blend": Weighted combination based on pool sophistication
"""

from dataclasses import dataclass, field

from simulation.engine import TournamentBracket

# Seed-based popularity bias (casual pools overweight favorites)
SEED_POPULARITY_BIAS = {
    1: 3.0,
    2: 2.0,
    3: 1.5,
    4: 1.1,
    5: 1.0,
    6: 1.0,
    7: 0.95,
    8: 0.9,
    9: 0.85,
    10: 0.85,
    11: 0.8,
    12: 0.8,
    13: 0.7,
    14: 0.6,
    15: 0.2,
    16: 0.05,
}

# Brand recognition bias — blue bloods get picked more regardless of seed
BRAND_BIAS: dict[str, float] = {
    "Duke": 1.4,
    "North Carolina": 1.3,
    "Kentucky": 1.3,
    "Kansas": 1.25,
    "Connecticut": 1.2,
    "UConn": 1.2,
    "UCLA": 1.15,
    "Michigan St.": 1.1,
    "Gonzaga": 1.1,
    "Villanova": 1.1,
}

# Recency bias — recent deep runs boost public perception
# team_name -> multiplier (applied on top of seed + brand bias)
DEFAULT_RECENCY: dict[str, float] = {
    "Connecticut": 1.15,  # Back-to-back champ 2023-2024
    "UConn": 1.15,
}


@dataclass
class OwnershipConfig:
    """Configuration for ownership estimation."""
    seed_popularity_bias: dict[int, float] = field(default_factory=lambda: dict(SEED_POPULARITY_BIAS))
    brand_bias: dict[str, float] = field(default_factory=lambda: dict(BRAND_BIAS))
    recency_bias: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RECENCY))
    base_alpha: float = 2.5
    alpha_sophistication_factor: float = 1.5
    round_alpha_increment: float = 0.3


def estimate_ownership(
    available_teams: dict[int, int],
    win_probs: dict[int, float],
    round_num: int,
    pool_sophistication: float = 0.5,
    team_names: dict[int, str] | None = None,
    config: OwnershipConfig | None = None,
) -> dict[int, float]:
    """Estimate what fraction of the pool will pick each team (heuristic model).

    Args:
        available_teams: team_id -> seed for teams playing this round
        win_probs: team_id -> P(win this round)
        round_num: Current round number (1-6)
        pool_sophistication: 0 = very casual (heavy herding), 1 = sharp (closer to optimal)
        team_names: team_id -> name for brand/recency bias lookup
        config: OwnershipConfig with tunable parameters

    Returns:
        Dict of team_id -> ownership fraction (sums to 1.0)
    """
    if config is None:
        config = OwnershipConfig()
    if team_names is None:
        team_names = {}

    # Alpha controls concentration on favorites
    # Higher alpha = more herding on chalk
    # Casual pools: alpha=2.5, Sharp pools: alpha=1.0
    base_alpha = config.base_alpha - config.alpha_sophistication_factor * pool_sophistication

    # Later rounds concentrate more (fewer teams, less creativity)
    alpha = base_alpha + config.round_alpha_increment * (round_num - 1)

    raw_ownership = {}
    for team_id, seed in available_teams.items():
        wp = win_probs.get(team_id, 0.5)
        seed_bias = config.seed_popularity_bias.get(seed, 1.0)

        # Blend seed bias based on sophistication
        # Sophisticated pools weight win_prob more, casual pools weight seed more
        effective_bias = 1.0 + (seed_bias - 1.0) * (1.0 - pool_sophistication)

        # Brand recognition bias (blue bloods get picked more)
        name = team_names.get(team_id, "")
        brand_mult = config.brand_bias.get(name, 1.0)

        # Recency bias (recent champions/F4 teams)
        recency_mult = config.recency_bias.get(name, 1.0)

        raw_ownership[team_id] = (wp ** alpha) * effective_bias * brand_mult * recency_mult

    # Normalize to sum to 1
    total = sum(raw_ownership.values())
    if total == 0:
        n = len(raw_ownership)
        return {t: 1 / n for t in raw_ownership}

    return {t: v / total for t, v in raw_ownership.items()}


def get_ownership(
    available_teams: dict[int, int],
    win_probs: dict[int, float],
    round_num: int,
    pool_size: int = 100,
    prize_pool: float = 5000,
    pool_sophistication: float = 0.5,
    method: str = "blend",
    team_names: dict[int, str] | None = None,
    config: OwnershipConfig | None = None,
) -> dict[int, float]:
    """Get ownership estimates using the specified method.

    Args:
        method: "heuristic", "nash", or "blend"
        pool_sophistication: For blend, controls mix (0=all heuristic, 1=all Nash)
        team_names: team_id -> name for brand/recency bias
        config: OwnershipConfig with tunable parameters
    """
    if method == "heuristic":
        return estimate_ownership(
            available_teams, win_probs, round_num, pool_sophistication,
            team_names=team_names, config=config,
        )

    if method == "nash":
        from optimizer.nash import nash_equilibrium
        return nash_equilibrium(win_probs, pool_size, prize_pool)

    # Blend: weighted combination
    from optimizer.nash import blended_ownership
    heuristic = estimate_ownership(
        available_teams, win_probs, round_num, pool_sophistication,
        team_names=team_names, config=config,
    )
    return blended_ownership(
        win_probs, pool_size, prize_pool, heuristic,
        field_efficiency=pool_sophistication,
    )


def estimate_ownership_from_bracket(
    bracket: TournamentBracket,
    round_num: int,
    win_probs: dict[int, float],
    pool_sophistication: float = 0.5,
    method: str = "blend",
    pool_size: int = 100,
    prize_pool: float = 5000,
    regions: list[str] | None = None,
    config: OwnershipConfig | None = None,
) -> dict[int, float]:
    """Convenience function: estimate ownership from bracket structure."""
    if regions:
        matchups = bracket.get_day_matchups(round_num, regions)
    else:
        matchups = bracket.get_round_matchups(round_num)
    available = {}
    team_names = {}
    for team_a, team_b, _ in matchups:
        if team_a and team_a in bracket.teams:
            available[team_a] = bracket.teams[team_a]["seed"]
            team_names[team_a] = bracket.teams[team_a].get("name", "")
        if team_b and team_b in bracket.teams:
            available[team_b] = bracket.teams[team_b]["seed"]
            team_names[team_b] = bracket.teams[team_b].get("name", "")

    return get_ownership(
        available, win_probs, round_num, pool_size, prize_pool,
        pool_sophistication, method,
        team_names=team_names, config=config,
    )
