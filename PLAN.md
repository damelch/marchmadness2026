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
├── cli.py                  # CLI: download, features, train, evaluate, simulate, optimize, results, status
├── config.yaml             # Pool size, prize pool, entries, model settings, sim params
├── pyproject.toml
├── data/
│   ├── scrapers/
│   │   ├── kaggle_data.py  # Download NCAA historical data
│   │   └── espn_api.py     # Live bracket & scores from ESPN
│   ├── feature_engineering.py  # KenPom-style features (AdjO, AdjD, AdjEM, SOS, etc.)
│   └── seed_history.py     # Historical seed-vs-seed win rates
├── models/
│   ├── train.py            # Logistic, XGBoost, Ensemble with calibration
│   ├── predict.py          # Win probability for any matchup
│   └── evaluate.py         # Log-loss, Brier score, calibration curves
├── simulation/
│   ├── engine.py           # Monte Carlo tournament simulator (50k sims)
│   └── analysis.py         # Survivor pool outcome simulation & pick EV analysis
├── optimizer/
│   ├── ownership.py        # Estimate public pick distribution (seed bias + sophistication)
│   ├── survival.py         # Survival probability math & leverage scores
│   ├── differentiation.py  # Approach A: greedy leverage-based multi-entry diversification
│   ├── portfolio.py        # Approach B: correlation-aware portfolio optimization via MC
│   └── kelly.py            # Kelly Criterion for optimal number of entries
├── entries/
│   ├── manager.py          # Track picks, used teams, eliminations per entry
│   └── generator.py        # Orchestrate full optimization pipeline
└── tests/                  # 30 tests covering models, optimizer, simulation
```

## How It Works

### 1. ML Win Probability Model
- Trains on 12 years of NCAA tournament data (2013-2025, skip 2020)
- Features: seed diff, adjusted offensive/defensive efficiency, tempo, SOS, Massey rank, tournament experience
- XGBoost with isotonic calibration (critical: optimizer needs calibrated probabilities)
- Leave-one-season-out cross-validation

### 2. Monte Carlo Simulation
- Simulates entire tournament 50,000 times
- Pre-caches all pairwise win probabilities
- Produces team advancement probabilities (P(reach R32), P(reach S16), etc.)

### 3. Optimizer (Two Approaches)

**Approach A: Leverage-Based Differentiation**
- Ranks teams by `leverage = win_prob * (1 - ownership) / field_survival_rate`
- Assigns picks greedily across entries, preferring different regions
- Fast, intuitive, good for quick decisions

**Approach B: Correlation-Aware Portfolio**
- Uses MC simulation results to evaluate expected payout of each pick combination
- Greedy assignment + local search (pairwise swaps)
- Accounts for opponent behavior via ownership model
- More accurate but slower

### 4. Ownership Estimation
- Models public pick behavior: `ownership ~ win_prob^alpha * seed_popularity_bias`
- `pool_sophistication` parameter: 0 = casual (heavy herding on favorites), 1 = sharp
- Later rounds increase alpha (more concentration)
- **This is where the edge comes from**: picking contrarian winners

### 5. Kelly Criterion
- Determines optimal number of entries given bankroll and edge
- Half-Kelly default for safety
- Accounts for diminishing returns (correlated entries)

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
    {"id": 1235, "name": "Kansas", "seed": 2, "region": "W"},
    ...
  ]
}
```
2. `marchmadness simulate` - run MC simulation with real matchups
3. `marchmadness optimize --round 1` - get Round 1 picks

### During Tournament
1. `marchmadness results --round 1 <winning_team_ids>` - update results
2. `marchmadness optimize --round 2` - get next round picks
3. `marchmadness status` - view all entries

## Config (config.yaml)
Key settings to update:
- `pool.num_entries` - how many entries you're buying
- `pool.pool_size` - total entries in the pool
- `pool.prize_pool` - total prize money
- `pool.entry_cost` - cost per entry
- `pool.risk_tolerance` - 0=conservative, 1=aggressive (affects ownership model)

## TODO / Iteration Ideas
- [ ] Add real bracket data once announced
- [ ] Plug in actual pool size, prize pool, max entries
- [ ] Tune `pool_sophistication` parameter for your specific pool
- [ ] Add web scraper for public ownership data (if available)
- [ ] Implement "save team for later" heuristic (future_value_estimate in portfolio.py)
- [ ] Add day-by-day optimization (currently round-by-round)
- [ ] Consider regular season pre-training to augment small tournament dataset
- [ ] Add visualization: bracket with color-coded pick recommendations
- [ ] Compare model predictions vs Vegas lines as sanity check
- [ ] Track actual results vs predictions for model improvement
