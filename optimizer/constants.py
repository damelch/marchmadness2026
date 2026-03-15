"""Centralized constants for optimizer tuning parameters.

All magic numbers from the optimizer modules are collected here for easy
tuning and A/B testing across different pool configurations.
"""

# --- Future Value (analytical.py) ---
FV_WEIGHT = 0.3  # Base penalty weight for using high-FV teams now (0-1)

# Top seeds are scarce resources — extra FV multiplier by seed
SEED_FV_MULTIPLIER: dict[int, float] = {
    1: 4.0,   # 1-seeds are irreplaceable
    2: 3.5,   # 2-seeds nearly as valuable
    3: 1.5,   # 3-seeds have some premium
}

# --- Dynamic Programming (dp.py) ---
DISCOUNT_FACTOR = 0.75           # Per-day discount for future value (0-1)
DOUBLE_PICK_SCARCITY = 1.8       # Scarcity multiplier for double-pick days
TOP_SEED_SCARCITY_BONUS = 0.3    # Bonus scarcity for seeds <= 2
MIN_ADVANCEMENT_PROB = 0.01      # Skip teams with < 1% advancement chance
MIN_ROUND_WIN_PROB = 0.1         # Skip teams with < 10% round win prob

# --- Local Search (analytical.py) ---
MAX_SWAP_ITERATIONS = 20         # Max iterations for single-pick local search
SWAP_IMPROVEMENT_THRESHOLD = 1.001  # Min improvement ratio to accept swap (0.1%)
TOP_TEAMS_FOR_SWAP = 30          # Number of top teams to try in single-pick search
DOUBLE_PICK_CANDIDATE_LIMIT = 60  # Top pairs for double-pick greedy
DOUBLE_PICK_SEARCH_LIMIT = 50    # Top pairs for double-pick local search

# --- Diversification (analytical.py) ---
CONCENTRATION_PENALTY = 0.4      # Penalty scale for portfolio concentration
OVERLAP_PENALTY = 0.25           # Overlap penalty coefficient in greedy assignment
DIVERSITY_BONUS_SCALE = 0.25     # HHI-based diversity bonus as fraction of best EV

# --- Nash Solver (nash.py) ---
NASH_MAX_ITER = 2000
NASH_TOLERANCE = 1e-8
NASH_LEARNING_RATE = 0.5
NASH_MIN_OWNERSHIP = 1e-6
EQUILIBRIUM_THRESHOLD = 0.05    # Within 5% EV spread = equilibrium

# --- Ant Colony Optimization (aco.py) ---
ACO_N_ANTS = 30                  # Ants per generation
ACO_N_ITERATIONS = 80            # Number of generations
ACO_ALPHA = 1.0                  # Pheromone importance exponent
ACO_BETA = 2.0                   # Heuristic importance exponent
ACO_RHO = 0.1                    # Evaporation rate (0-1)
ACO_ELITE_WEIGHT = 2.0           # Extra pheromone for best-ever solution
ACO_TOP_K = 5                    # Top ants that deposit pheromone per generation
ACO_MIN_PHEROMONE = 0.01         # Floor to prevent starvation
ACO_MAX_PHEROMONE = 10.0         # Ceiling to prevent premature convergence
