# March Madness Survivor Pool Optimizer

Maximize expected value in NCAA tournament survivor pools. Pick one team per round to win — if they lose, you're eliminated. Last survivor(s) split the pot.

This isn't a bracket contest optimizer. It's built specifically for **survivor pools** where the key decisions are: which team to pick each round, when to burn a top team vs. save it, and how to differentiate across multiple entries.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Before bracket is announced — train on historical data
marchmadness download        # Kaggle NCAA data (2013-2025)
marchmadness features        # Build KenPom-style feature matrix
marchmadness train           # Train XGBoost model
marchmadness evaluate        # Verify calibration

# After Selection Sunday — optimize picks
marchmadness simulate        # Monte Carlo advancement probabilities
marchmadness optimize --round 1 --method both

# During tournament
marchmadness results --round 1 <winning_team_ids>
marchmadness optimize --round 2
marchmadness status
```

Requires Python >= 3.11 and a [Kaggle API key](https://www.kaggle.com/docs/api) for data download.

## How It Works

### Win Probability Model

Three-tier prediction with automatic fallback:

**1. XGBoost with isotonic calibration** (primary)

Trained on 12 seasons of NCAA tournament results (2013-2025, excluding 2020). Features are differences between teams:

| Feature | Source | What it captures |
|---------|--------|-----------------|
| SeedDiff | Tournament seeding | Committee ranking |
| AdjEMDiff | KenPom / box scores | Overall team strength |
| AdjODiff | KenPom / box scores | Offensive efficiency |
| AdjDDiff | KenPom / box scores | Defensive efficiency |
| AdjTAvg | KenPom / box scores | Game tempo |
| SOSDiff | KenPom / box scores | Strength of schedule |
| WinPctDiff | Season record | Win percentage |
| MasseyRankDiff | Massey Ordinals | Composite ranking |
| TourneyExpDiff | Historical seeds | Tournament experience |

Isotonic calibration ensures a predicted 70% probability actually wins ~70% of the time. Evaluated with leave-one-season-out cross-validation.

**2. KenPom direct prediction** (fallback)

If no trained model exists, converts KenPom adjusted efficiency margin difference to win probability via logistic function. The 2026 KenPom ratings for all 365 D-I teams are included (`data/kenpom_2026.csv`).

**3. Seed-based historical rates** (last resort)

Uses empirical seed-vs-seed win rates from tournament history, smoothed with a logistic model.

### Optimization Engine

The optimizer uses three complementary techniques:

**Analytical EV** — Exact closed-form expected value for each pick:

```
EV = P(win) * prize_pool / E[survivors | our pick wins]
```

No Monte Carlo noise. Accounts for opponent ownership (picking a team everyone else picked means you share the prize with more survivors). Uses greedy assignment + local search for multi-entry optimization.

**Nash Equilibrium** — Game-theoretic optimal ownership distribution. At equilibrium, every picked team has equal EV (no profitable deviation). Computed via replicator dynamics:

1. Start with uniform ownership
2. Compute EV per team given current ownership
3. Shift ownership toward higher-EV teams
4. Repeat until convergence

Your edge comes from the gap between Nash (what the field should do) and heuristic (what the field actually does). Against casual pools that overweight chalk, Nash identifies which contrarian picks carry the most value.

**Dynamic Programming** — Multi-round planning via backward induction. Should you use a 1-seed now (99% safe) or save it for a later round where alternatives are scarce?

For each team, computes:
- `future_value` = how valuable this team is in future rounds
- `scarcity` = how many viable alternatives exist in each future round
- Adjusted EV = current_round_EV - future_value_penalty

The result: picks that balance safety now against optionality later.

**Monte Carlo simulation** (50k runs) is used only for computing `P(team reaches round R)` — it captures tournament structure correlations (e.g., two 1-seeds in the same region can't both make the Final Four). All other math is exact.

### Ownership Model

Three modes:

- **Heuristic** — Seed-based popularity bias for casual pools (1-seeds get picked ~50% of the time, 16-seeds ~1%)
- **Nash** — Mathematically optimal ownership for sharp pools
- **Blend** — Weighted mix controlled by `pool_sophistication` parameter (recommended)

## Configuration

Edit `config.yaml`:

```yaml
pool:
  num_entries: 5          # How many entries you're buying
  pool_size: 100          # Total entries in the pool
  prize_pool: 5000        # Total prize money
  entry_cost: 50          # Cost per entry
  risk_tolerance: 0.5     # 0=conservative, 1=aggressive
  rules:
    reuse_allowed: false   # Can't pick same team twice across rounds
```

## Bracket Setup

After Selection Sunday, create `data/bracket.json`:

```json
{
  "teams": [
    {"id": 1234, "name": "Duke", "seed": 1, "region": "W"},
    {"id": 1235, "name": "Kansas", "seed": 2, "region": "W"}
  ]
}
```

Team IDs should match Kaggle's TeamID from `MTeams.csv` if you've trained a model. If using KenPom-only mode, any consistent integer IDs work.

## Project Structure

```
marchmadness2026/
├── cli.py                     # CLI commands
├── config.yaml                # Pool and model settings
├── data/
│   ├── scrapers/
│   │   ├── kaggle_data.py     # Historical NCAA data
│   │   └── espn_api.py        # Live bracket & scores
│   ├── feature_engineering.py # KenPom-style features from box scores
│   ├── seed_history.py        # Historical seed-vs-seed win rates
│   ├── kenpom.py              # KenPom ratings integration
│   └── kenpom_2026.csv        # 2026 KenPom ratings (365 teams)
├── models/
│   ├── train.py               # Logistic, XGBoost, Ensemble
│   ├── predict.py             # Win probability predictor
│   └── evaluate.py            # Calibration & accuracy
├── simulation/
│   ├── engine.py              # Monte Carlo tournament simulator
│   └── analysis.py            # Survivor pool outcome analysis
├── optimizer/
│   ├── analytical.py          # Exact closed-form EV
│   ├── nash.py                # Nash equilibrium solver
│   ├── dp.py                  # Dynamic programming planner
│   ├── ownership.py           # Ownership estimation (heuristic/Nash/blend)
│   ├── portfolio.py           # Portfolio optimizer
│   ├── survival.py            # Survival probability math
│   ├── differentiation.py     # Leverage-based pick ranking
│   └── kelly.py               # Kelly Criterion for entry count
├── entries/
│   ├── manager.py             # Track picks and eliminations
│   └── generator.py           # Full optimization pipeline
└── tests/                     # 66 tests
```

## Tests

```bash
pytest tests/ -v
```

66 tests covering model training, bracket simulation, analytical EV math, Nash equilibrium convergence, DP future values, and KenPom integration.
