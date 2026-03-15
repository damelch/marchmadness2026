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
marchmadness optimize --day 1 --method hybrid

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
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/entries:/app/entries \
  marchmadness optimize --day 1 --method hybrid

# Override pool settings on the fly (defaults: 10k pool, 150 max/user)
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/entries:/app/entries \
  marchmadness optimize --day 1 --pool-size 22000 --num-entries 5 --max-entries 150

# Run tests
docker run --rm --entrypoint pytest marchmadness -v
```

Or use the Makefile:

```bash
make build                  # Build Docker image
make test                   # Run 120 tests
make lint                   # Ruff lint check (zero violations)
make format                 # Ruff format check
make simulate               # 50k Monte Carlo sims
make optimize-day1          # Day 1 picks (uses config.yaml)
make optimize-day2          # Day 2 picks
make optimize-all           # Both R64 days

# Override pool settings
make optimize-day1 POOL_SIZE=22000 NUM_ENTRIES=5 MAX_ENTRIES=150
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
| LuckDiff | KenPom / derived | Regression-to-mean indicator |
| NCSOSDiff | KenPom | Non-conference strength of schedule |
| SeedRoundInteraction | Derived | Seed advantage amplified by round |
| AdjEMStdDiff | Box scores | Scoring margin consistency |

Multiple calibration methods ensure a predicted 70% probability actually wins ~70% of the time:
- **Isotonic** (default) — non-parametric calibration via `CalibratedClassifierCV`
- **Platt scaling** (`calibrate: "sigmoid"`) — parametric logistic calibration
- **Temperature scaling** (`calibrate: "temperature"`) — learns a single parameter T on LOSO out-of-fold predictions to minimize log-loss

Evaluation includes per-season LOSO cross-validation, seed-tier calibration diagnostics (Blowout/Competitive/Close matchups), and Expected Calibration Error (ECE).

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

Top seeds (1 and 2) get extra preservation multipliers — they're irreplaceable in later rounds when the field thins out. The optimizer strongly prefers burning 3-7 seeds in the Round of 64 and banking 1-seeds for Elite 8 and beyond.

**Monte Carlo simulation** (50k runs) is used only for computing `P(team reaches round R)` — it captures tournament structure correlations (e.g., two 1-seeds in the same region can't both make the Final Four). All other math is exact.

### Ownership Model

Three modes:

- **Heuristic** — Seed-based popularity bias for casual pools (1-seeds get picked ~30-40% of the time, 16-seeds <1%)
- **Nash** — Mathematically optimal ownership for sharp pools
- **Blend** — Weighted mix controlled by `pool_sophistication` parameter (recommended)

Field sophistication is auto-estimated from contest structure: large paid multi-entry contests (22k entries, 150 max/user) get higher sophistication than a 100-person office pool.

The ownership model also accounts for:
- **Brand recognition bias** — blue blood programs (Duke 1.4x, UNC 1.3x, Kentucky 1.3x, Kansas 1.25x, UConn 1.2x) attract disproportionate public picks regardless of seed
- **Recency bias** — recent champions get a pickup boost (e.g., UConn 1.15x after back-to-back titles)

All bias parameters are configurable in `config.yaml` under the `ownership:` section.

### Portfolio Diversification

With multiple entries, the optimizer spreads picks across independent games with a **concentration penalty** — if one team appears in too many entries, a single upset wipes out the entire portfolio. The penalty scales quadratically with exposure, so the optimizer naturally produces varied picks across entries.

## Example Output

```
======================================================================
PICK RECOMMENDATIONS - Day 1 (R64 Thursday)
  Pool: 22,000 entries | Prize: $3,000,000 | Max/user: 150
  ** Double-pick day: both picks must win to survive **
======================================================================
  Entry 0: (3) Iowa St. Win=93.6% Own=2.0% + (3) Purdue Win=85.2% Own=1.8%
  Entry 1: (4) Kansas Win=89.0% Own=1.7% + (2) Connecticut Win=96.0% Own=3.0%
  Entry 2: (6) Louisville Win=85.0% Own=1.5% + (2) Connecticut Win=96.0% Own=3.0%
  Entry 3: (2) Illinois Win=88.2% Own=2.1% + (10) Ohio St. Win=86.4% Own=1.5%
  Entry 4: (3) Iowa St. Win=93.6% Own=2.0% + (2) Connecticut Win=96.0% Own=3.0%
  Entry 1: Switched from (1) Duke + (1) Florida to (3) Iowa St. + (3) Purdue
           (FV=2202724.42) — saving higher-FV teams for later

Portfolio Analysis:
  Total EV: $663.51
  Joint survival: 100.0%
```

Reading this output:
- **Win** — KenPom-derived probability of winning the game
- **Own** — Estimated % of the field picking this team
- **FV** — Future value of the teams that were swapped out (why the DP planner saved them)
- **Switched from** — Shows which "obvious" picks (1-seeds) the DP planner replaced with mid-seeds to preserve top seeds for later rounds

Key strategy: all four 1-seeds (Duke, Florida, Michigan, Arizona) are banked for later rounds. The optimizer uses 3-7 seeds with 85-93% win probability and only 1-3% ownership — high leverage against a field that's piling on the chalk.

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
  pool_size: 22000        # Total entries in the pool (CLI: --pool-size)
  prize_pool: 3000000     # Total prize money
  entry_cost: 150         # Cost per entry
  risk_tolerance: 0.7     # 0=conservative, 1=aggressive
  max_entries_per_user: 150  # Max entries allowed per user (CLI: --max-entries)
  payout_structure: "winner_take_all"
  rules:
    reuse_allowed: false   # Can't pick same team twice across days

model:
  type: "stacked"         # logistic, xgboost, lightgbm, catboost,
                          # randomforest, naivebayes, ensemble, stacked
  calibrate: true         # true/"isotonic", "sigmoid" (Platt), "temperature"

ownership:
  method: blend           # heuristic, nash, or blend
  brand_bias:
    Duke: 1.4
    North Carolina: 1.3
    Kentucky: 1.3
    Kansas: 1.25
    Connecticut: 1.2
    UConn: 1.2
  recency_bias:
    Connecticut: 1.15
    UConn: 1.15
```

CLI flags override config values: `--pool-size 22000 --num-entries 5 --max-entries 150`

Defaults when nothing is specified: pool size = 10,000, max entries = 150.

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
├── Makefile                   # Docker build/test/optimize shortcuts
├── contest/
│   └── schedule.py            # Day-based contest schedule (9 days, 12 picks)
├── data/
│   ├── scrapers/
│   │   ├── kaggle_data.py     # Historical NCAA data
│   │   └── espn_api.py        # Live bracket & scores
│   ├── feature_engineering.py # KenPom-style features from box scores
│   ├── seed_history.py        # Historical seed-vs-seed win rates
│   ├── kenpom.py              # KenPom ratings integration
│   ├── kenpom_2026.csv        # 2026 KenPom ratings (365 teams)
│   └── bracket.json           # Tournament bracket (64 teams)
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
│   ├── ownership.py           # Ownership estimation (heuristic/Nash/blend + brand/recency bias)
│   ├── constants.py           # Centralized magic numbers and tuning parameters
│   ├── portfolio.py           # Portfolio optimizer
│   ├── survival.py            # Survival probability math
│   ├── differentiation.py     # Leverage-based pick ranking
│   └── kelly.py               # Kelly Criterion for entry count
├── entries/
│   ├── manager.py             # Track picks and eliminations (day-based)
│   └── generator.py           # Full optimization pipeline
└── tests/                     # Test suite (120 tests)
```

## Tests

```bash
pytest tests/ -v
# or
make test
```

120 tests covering model training (all 8 model types including stacked ensemble pickle round-trip), bracket simulation, analytical EV math, Nash equilibrium convergence, DP future values, KenPom integration, and ownership model behavior (brand bias, recency bias, sophistication scaling).

Linting is enforced via [ruff](https://docs.astral.sh/ruff/) with rules for errors, warnings, import sorting, modern Python idioms, and common bugs.

## License

Apache 2.0 — see [LICENSE](LICENSE).
