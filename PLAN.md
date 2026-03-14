# March Madness Survivor Pool Optimizer

## Contest Rules
- Pick ONE team per round to win their game
- If your team loses, you're eliminated
- Last survivor(s) split the prize pool equally
- Multiple entries allowed
- Cannot reuse a team across rounds (configurable)

## Tournament Schedule (2026)
| Round | Games | Date |
|-------|-------|------|
| Round of 64 (Day 1) | 16 | 3/19 @ 11:15 AM |
| Round of 64 (Day 2) | 16 | 3/20 @ 11:15 AM |
| Round of 32 (Day 3) | 8 | 3/21 @ 11:45 AM |
| Round of 32 (Day 4) | 8 | 3/22 @ 11:10 AM |
| Round of 16 (Day 5) | 4 | 3/26 @ 6:09 PM |
| Round of 16 (Day 6) | 4 | 3/27 @ 6:09 PM |
| Elite 8 (Day 7+8) | 4 | 3/30 @ 5:09 PM |
| Final Four | 2 | 4/4 @ 5:09 PM |
| Championship | 1 | 4/6 @ 7:30 PM |

## Architecture

```
marchmadness2026/
├── cli.py                     # CLI: download, features, train, evaluate, simulate, optimize
├── config.yaml                # Pool size, prize pool, entries, model settings
├── pyproject.toml
├── data/
│   ├── scrapers/
│   │   ├── kaggle_data.py     # Download NCAA historical data
│   │   └── espn_api.py        # Live bracket & scores from ESPN
│   ├── feature_engineering.py # KenPom-style features (AdjO, AdjD, AdjEM, SOS)
│   └── seed_history.py        # Historical seed-vs-seed win rates
├── models/
│   ├── train.py               # Logistic, XGBoost, Ensemble with calibration
│   ├── predict.py             # Win probability for any matchup
│   └── evaluate.py            # Log-loss, Brier score, calibration curves
├── simulation/
│   ├── engine.py              # Monte Carlo tournament simulator (50k sims)
│   └── analysis.py            # Survivor pool outcome simulation
├── optimizer/
│   ├── analytical.py          # ★ Exact closed-form EV (replaces MC for single-round)
│   ├── nash.py                # ★ Nash equilibrium solver (game-theoretic ownership)
│   ├── dp.py                  # ★ Dynamic programming multi-round planner
│   ├── ownership.py           # Ownership: heuristic, Nash, or blended
│   ├── portfolio.py           # Portfolio optimizer (now uses analytical EV)
│   ├── survival.py            # Survival probability math & leverage scores
│   ├── differentiation.py     # Fast greedy leverage-based picks
│   └── kelly.py               # Kelly Criterion for optimal number of entries
├── entries/
│   ├── manager.py             # Track picks, used teams, eliminations
│   └── generator.py           # Full hybrid optimization pipeline
└── tests/                     # 56 tests across all modules
```

## Hybrid Optimization Pipeline

The optimizer uses three complementary techniques — each for what it does best:

### 1. Analytical EV (`optimizer/analytical.py`)
**What**: Exact closed-form expected value for each pick
**Why**: Instant and exact. No simulation variance.
**How**: `EV = P(win) * prize / E[survivors | our pick wins]`
- Accounts for opponent behavior via ownership model
- Handles correlation between picks in the same game
- Greedy + local search for multi-entry optimization

### 2. Nash Equilibrium (`optimizer/nash.py`)
**What**: Game-theoretic optimal ownership distribution
**Why**: The heuristic ownership model guesses. Nash computes the mathematically optimal strategy.
**How**: Replicator dynamics (multiplicative weights):
1. Start with uniform ownership
2. Compute EV per team → shift ownership toward higher-EV teams
3. Repeat until all picked teams have equal EV (= Nash equilibrium)

Key insight: **Your edge comes from the gap between Nash and how the field actually plays.** Against a casual pool (heavy chalk), Nash tells you which contrarian picks have the most value.

### 3. Dynamic Programming (`optimizer/dp.py`)
**What**: Multi-round "save team for later" planning
**Why**: Using a 1-seed in Round 1 (99% win) wastes a valuable pick you'll need in later rounds.
**How**: Backward induction:
- Compute `future_value[team]` = how valuable is this team in future rounds
- `team_scarcity` = how many alternatives exist in future rounds
- Adjust current-round EV: `adjusted_ev = round_ev - future_value`

### Where Monte Carlo Fits
MC simulation (50k tournament sims) is used ONLY for:
- Computing `P(team reaches round R)` — tournament outcome correlations
- Feeding advancement probabilities to the DP planner
- Validation: comparing analytical EV to MC estimates

### Ownership Methods
Three modes via `method` parameter:
- `"heuristic"`: Seed-based popularity model (fast, for casual pools)
- `"nash"`: Full Nash equilibrium (for sharp pools)
- `"blend"`: Weighted mix based on `pool_sophistication` (recommended default)

## How It Works (Full Pipeline)

```
1. ML Model → Win probabilities for all matchups
2. MC Simulation → P(team reaches each round) [correlations]
3. Nash Equilibrium → Optimal ownership distribution
4. Blend with heuristic → Estimated field behavior
5. DP Future Values → Which teams to save for later
6. Analytical Optimizer → Pick assignments with future-value adjustment
7. Output → Pick recommendations with EV breakdown
```

## Workflow

### Before Bracket (now)
1. `marchmadness download` - get Kaggle data
2. `marchmadness features` - build feature matrix
3. `marchmadness train` - train model
4. `marchmadness evaluate` - verify calibration

### After Selection Sunday
1. Create `data/bracket.json` with real teams:
```json
{
  "teams": [
    {"id": 1234, "name": "Duke", "seed": 1, "region": "W"},
    {"id": 1235, "name": "Kansas", "seed": 2, "region": "W"}
  ]
}
```
2. `marchmadness simulate` - run MC simulation with real matchups
3. `marchmadness optimize --round 1 --method hybrid` - get picks

### During Tournament
1. `marchmadness results --round 1 <winning_team_ids>` - update results
2. `marchmadness optimize --round 2` - get next round picks
3. `marchmadness status` - view all entries

## Config (config.yaml)
- `pool.num_entries` - how many entries you're buying
- `pool.pool_size` - total entries in the pool
- `pool.prize_pool` - total prize money
- `pool.entry_cost` - cost per entry
- `pool.risk_tolerance` - 0=conservative, 1=aggressive

## TODO / Iteration Ideas
- [ ] Add real bracket data once announced
- [ ] Plug in actual pool size, prize pool, max entries
- [ ] Tune `pool_sophistication` for your specific pool
- [ ] Add web scraper for public ownership data
- [ ] Add day-by-day optimization (currently round-by-round)
- [ ] Compare model predictions vs Vegas lines as sanity check
- [ ] Add visualization: bracket with color-coded pick recommendations
- [ ] Track actual results vs predictions for model improvement
