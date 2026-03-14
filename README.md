# March Madness Survivor Pool Optimizer

Maximize expected value in NCAA tournament survivor pools. Pick teams each day to win straight-up — if any pick loses, you're eliminated. Last survivor(s) split the pot.

This isn't a bracket contest optimizer. It's built specifically for **survivor pools** where the key decisions are: which team(s) to pick each day, when to burn a top team vs. save it, and how to differentiate across multiple entries. Supports **double-pick days** where both picks must win to survive.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Before bracket is announced — train on historical data
marchmadness download        # Kaggle NCAA data (2013-2025)
marchmadness features        # Build KenPom-style feature matrix
marchmadness train           # Train model (default: xgboost, or set in config.yaml)
marchmadness evaluate        # Verify calibration

# After Selection Sunday — optimize picks
marchmadness simulate        # Monte Carlo advancement probabilities
marchmadness schedule        # View the 9-day contest schedule
marchmadness optimize --day 1 --method both

# During tournament
marchmadness results --day 1 <winning_team_ids>
marchmadness optimize --day 2
marchmadness status
```

Requires Python >= 3.11 and a [Kaggle API key](https://www.kaggle.com/docs/api) for data download.

### Docker

```bash
# Build the image
docker build -t marchmadness .

# Run any command
docker run --rm marchmadness schedule
docker run --rm marchmadness optimize --day 1 --method both

# Mount volumes for persistent data and config
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/entries:/app/entries \
  marchmadness optimize --day 1

# Interactive shell
docker run --rm -it --entrypoint bash marchmadness
```

The Docker image uses Python 3.12 and includes all dependencies. Mount your `data/`, `entries/`, and `config.yaml` to persist state between runs.

## How It Works

### Win Probability Model

Three-tier prediction with automatic fallback:

**1. Stacked ensemble with 6 base models** (primary)

A meta-learner (logistic regression) trained on out-of-fold predictions from 6 base models: Logistic Regression, XGBoost, LightGBM, CatBoost, Random Forest, and Gaussian Naive Bayes. The meta-learner uses leave-one-season-out cross-validation to learn optimal model weights without overfitting. Individual models are also available (`xgboost`, `lightgbm`, `catboost`, `randomforest`, `naivebayes`, `logistic`).

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

**Dynamic Programming** — Multi-day planning via backward induction across all 9 contest days. Should you use a 1-seed now (99% safe) or save it for a later day where alternatives are scarce?

For each team, computes:
- `future_value` = how valuable this team is in future days
- `scarcity` = how many viable alternatives exist in each future day (double-pick days get extra weight)
- Adjusted EV = current_day_EV - future_value_penalty

The result: picks that balance safety now against optionality later.

**Monte Carlo simulation** (50k runs) is used only for computing `P(team reaches round R)` — it captures tournament structure correlations (e.g., two 1-seeds in the same region can't both make the Final Four). All other math is exact.

### Ownership Model

Three modes:

- **Heuristic** — Seed-based popularity bias for casual pools (1-seeds get picked ~50% of the time, 16-seeds ~1%)
- **Nash** — Mathematically optimal ownership for sharp pools
- **Blend** — Weighted mix controlled by `pool_sophistication` parameter (recommended)

## Example Output

Sample optimizer output for one region (8 games), using KenPom 2026 ratings with a 100-entry pool and $5,000 prize:

```
OPTIMIZER OUTPUT - Round 1 (sample region)
===========================================================================
Team                 Seed   Win%  Heur Own  Nash Own       EV
---------------------------------------------------------------------------
Duke                    1 90.6%    20.5%    80.5%  $  59.88  <--
Florida                 2 86.1%    16.6%    19.5%  $  56.65  <--
Iowa St.                3 82.4%    13.9%     0.0%  $  54.16  <--
Gonzaga                 4 74.4%    10.9%     0.0%  $  48.77  <--
Alabama                 5 68.3%     8.5%     0.0%  $  44.96
Kansas                  6 65.5%     7.8%     0.0%  $  43.20
Iowa                    7 57.3%     5.7%     0.0%  $  38.22
Auburn                  8 52.3%     4.6%     0.0%  $  35.24
Kentucky                9 47.7%     3.7%     0.0%  $  32.60
Missouri               10 42.7%     2.9%     0.0%  $  29.64
Northwestern           11 34.5%     1.8%     0.0%  $  24.73
Pittsburgh             12 31.7%     1.5%     0.0%  $  23.01
Drake                  13 25.6%     0.9%     0.0%  $  19.31
Colorado               14 17.6%     0.4%     0.0%  $  14.04
Boise St.              15 13.9%     0.2%     0.0%  $  11.66
Akron                  16  9.4%     0.1%     0.0%  $   8.52
```

Reading this output:

- **Win%** — KenPom-derived probability of winning the round
- **Heur Own** — Estimated % of the field picking this team (casual pool model)
- **Nash Own** — Game-theoretically optimal ownership (what a sharp pool would do)
- **EV** — Expected value of picking this team given pool size, prize, and field behavior
- **`<--`** — Recommended picks (highest EV)

Notice the gap between Heuristic and Nash ownership — Duke gets 20.5% in a casual pool but Nash says 80.5%. This gap is where your edge lives. The optimizer finds picks where Win% is high relative to how many opponents are picking that team.

### Multi-Entry Pick Distribution

With multiple entries, the optimizer diversifies picks across independent games so your entries don't all live or die together:

```
MULTI-ENTRY PICKS (5 entries, 100-person pool, $5,000 prize)
===========================================================================
  Entry    Pick                      Seed   Win%   Own%       EV
---------------------------------------------------------------------------
  1        ( 1) Duke                    1  90.6%  5.3%   $65.70
  2        ( 1) Michigan                1  90.2%  5.3%   $65.45
  3        ( 1) Arizona                 1  89.4%  5.2%   $64.83
  4        ( 2) Florida                 2  86.1%  4.3%   $62.36
  5        ( 1) Houston                 1  86.0%  4.8%   $62.24

  Total EV:            $320.57
  P(at least 1 alive): 100.0%
  Unique teams:        5/5
```

Each entry gets a different team from a different game. This way:
- If Duke loses (9.4% chance), only Entry 1 is eliminated — the other 4 survive
- P(at least 1 survives) is effectively 100% vs. 90.6% if all 5 picked Duke
- The DP planner may swap some entries to lower-seed picks to save top teams for later rounds

The optimizer balances **total EV** against **diversification** — concentrating all entries on the highest-EV team maximizes raw EV but creates catastrophic correlation risk.

## Contest Schedule

The optimizer uses a **day-based** schedule matching real survivor pool contest rules. The tournament spans 9 decision days with 12 total picks:

| Day | Round | Regions | Picks | Note |
|-----|-------|---------|-------|------|
| 1 | R64 Thu | W, X | **2** | Both must win |
| 2 | R64 Fri | Y, Z | **2** | Both must win |
| 3 | R32 Sat | W, X | 1 | |
| 4 | R32 Sun | Y, Z | 1 | |
| 5 | S16 Thu | W, X | 1 | |
| 6 | S16 Fri | Y, Z | 1 | |
| 7 | E8 | All | **2** | Both must win |
| 8 | Final Four | — | 1 | |
| 9 | Championship | — | 1 | |

On double-pick days (1, 2, 7), you select two teams and **both must win** for your entry to survive. Teams can only be used once across the entire tournament.

Run `marchmadness schedule` to view the full schedule with dates.

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
    reuse_allowed: false   # Can't pick same team twice across days

model:
  type: "stacked"         # logistic, xgboost, lightgbm, catboost,
                          # randomforest, naivebayes, ensemble, stacked
  calibrate: true         # Isotonic calibration for base models
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
├── cli.py                     # CLI commands (day-based)
├── config.yaml                # Pool, model, and contest schedule settings
├── Dockerfile                 # Docker containerization
├── contest/
│   └── schedule.py            # Day-based contest schedule (9 days, 12 picks)
├── data/
│   ├── scrapers/
│   │   ├── kaggle_data.py     # Historical NCAA data
│   │   └── espn_api.py        # Live bracket & scores
│   ├── feature_engineering.py # KenPom-style features from box scores
│   ├── seed_history.py        # Historical seed-vs-seed win rates
│   ├── kenpom.py              # KenPom ratings integration
│   └── kenpom_2026.csv        # 2026 KenPom ratings (365 teams)
├── models/
│   ├── train.py               # Logistic, XGBoost, LightGBM, CatBoost, RF, NB, Stacked Ensemble
│   ├── predict.py             # Win probability predictor
│   └── evaluate.py            # Calibration & accuracy
├── simulation/
│   ├── engine.py              # Monte Carlo tournament simulator
│   └── analysis.py            # Survivor pool outcome analysis
├── optimizer/
│   ├── analytical.py          # Exact closed-form EV (single + double-pick)
│   ├── nash.py                # Nash equilibrium solver
│   ├── dp.py                  # Dynamic programming planner (9-day)
│   ├── ownership.py           # Ownership estimation (heuristic/Nash/blend)
│   ├── portfolio.py           # Portfolio optimizer
│   ├── survival.py            # Survival probability math
│   ├── differentiation.py     # Leverage-based pick ranking
│   └── kelly.py               # Kelly Criterion for entry count
├── entries/
│   ├── manager.py             # Track picks and eliminations (day-based)
│   └── generator.py           # Full optimization pipeline
└── tests/                     # Test suite
```

## Tests

```bash
pytest tests/ -v
```

111 tests covering model training (all 8 model types including stacked ensemble pickle round-trip), bracket simulation, analytical EV math, Nash equilibrium convergence, DP future values, and KenPom integration.
