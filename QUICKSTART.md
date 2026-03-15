# Quickstart Guide

A step-by-step guide to running the March Madness Survivor Pool Optimizer.
No programming experience required — just follow along.

---

## What This Tool Does

You're in a survivor pool: each game day you pick one or two teams to win.
If any pick loses, you're out. This tool crunches 12 years of NCAA data
and runs 50,000 simulated tournaments to tell you **which teams to pick
each day** to maximize your expected payout.

---

## What You'll Need

| Item | Why | Time |
|------|-----|------|
| **A computer** (Windows, Mac, or Linux) | Runs everything locally | — |
| **Docker Desktop** | Runs the tool without installing Python | 5 min install |
| **A free Kaggle account** | Downloads historical NCAA game data | 2 min signup |

### Install Docker Desktop

1. Go to [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
2. Download the installer for your OS
3. Run the installer, accept defaults
4. Open Docker Desktop and wait until it says **"Docker Desktop is running"**

### Create a Kaggle Account & API Key

1. Go to [kaggle.com](https://www.kaggle.com/) and sign up (free)
2. Click your profile icon (top-right) → **Settings**
3. Scroll to **API** → click **Create New Token**
4. This downloads a file called `kaggle.json`
5. Move that file to the right place:
   - **Windows**: Create a folder `C:\Users\<YourName>\.kaggle\` and put `kaggle.json` inside
   - **Mac/Linux**: Run `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/`

---

## Step-by-Step Setup

Open a terminal:
- **Windows**: Press `Win + R`, type `cmd`, press Enter
- **Mac**: Open **Terminal** (search for it in Spotlight with `Cmd + Space`)
- **Linux**: Open your terminal app

### 1. Download this project

#### Option A — With Git (recommended)

If you already have Git installed:

```bash
git clone https://github.com/damelch/marchmadness2026.git
cd marchmadness2026
```

#### Option B — Install Git first

**Windows:**
1. Download Git from [gitforwindows.org](https://gitforwindows.org/)
2. Run the installer — accept all defaults
3. Close and reopen your terminal, then run the `git clone` command above

**Mac:**
1. Open Terminal and type `git`. If it's not installed, macOS will prompt
   you to install the Command Line Developer Tools — click **Install**
2. Once installed, run the `git clone` command above

**Linux:**
```bash
sudo apt install git    # Ubuntu/Debian
sudo dnf install git    # Fedora
```
Then run the `git clone` command above.

#### Option C — No Git at all

1. Go to [github.com/damelch/marchmadness2026](https://github.com/damelch/marchmadness2026)
2. Click the green **Code** button → **Download ZIP**
3. Unzip the file
4. Open a terminal and navigate into the unzipped folder:
   - **Windows**: `cd C:\Users\<YourName>\Downloads\marchmadness2026-main`
   - **Mac/Linux**: `cd ~/Downloads/marchmadness2026-main`

### 2. Build the tool

```bash
docker build -t marchmadness .
```

This takes 1-2 minutes the first time. You'll see a bunch of output — that's
normal. When it finishes you should see `Successfully tagged marchmadness:latest`.

### 3. Verify it works

```bash
docker run --rm marchmadness --help
```

You should see a list of commands like `download`, `train`, `optimize`, etc.

---

## Phase 1: Build the Model (Before the Tournament)

Do these steps **once**, before Selection Sunday. They download historical
data and train the prediction model.

### Step 1 — Download historical NCAA data

```bash
docker run --rm \
  -v ~/.kaggle:/root/.kaggle \
  -v ./data:/app/data \
  marchmadness download
```

> **Windows Command Prompt** — replace `~/.kaggle` with `%USERPROFILE%\.kaggle`
> and `./data` with the full path like `C:\Users\YourName\marchmadness2026\data`

This downloads 12 seasons of tournament results from Kaggle (~50 MB).
Takes about 30 seconds.

### Step 2 — Fetch extra ratings (optional but recommended)

```bash
docker run --rm -v ./data:/app/data marchmadness fetch-bpi
docker run --rm -v ./data:/app/data marchmadness fetch-barttorvik
```

These grab ESPN BPI and Barttorvik power ratings — free, no account needed.
They add extra features that improve predictions.

### Step 3 — Build features

```bash
docker run --rm -v ./data:/app/data marchmadness features
```

This creates the matchup feature matrix from the raw data (~6,000 game rows,
18 features each). Takes a few seconds.

### Step 4 — Train the model

```bash
docker run --rm \
  -v ./data:/app/data \
  -v ./models:/app/models \
  marchmadness train
```

Trains a stacked ensemble on 12 seasons of data. Takes 1-3 minutes depending
on your computer. When done it saves `models/saved/model.pkl`.

### Step 5 — Check how good it is (optional)

```bash
docker run --rm \
  -v ./data:/app/data \
  -v ./models:/app/models \
  marchmadness evaluate
```

Prints accuracy, log-loss, and calibration stats. A well-calibrated model
means when it says "80% chance to win" that team really does win about 80%
of the time.

---

## Phase 2: Get Your Picks (After Selection Sunday)

Once the bracket is announced, you'll set up your pool info and ask the
optimizer for picks.

### Step 1 — Edit the config file

Open `config.yaml` in any text editor (Notepad, TextEdit, VS Code — anything
works). Update these fields to match your pool:

```yaml
pool:
  num_entries: 5         # How many entries you bought
  pool_size: 22000       # Total entries in the pool
  prize_pool: 3000000    # Total prize money ($)
  entry_cost: 150        # Cost per entry ($)
  max_entries_per_user: 150
```

Save the file.

### Step 2 — Set up the bracket

After Selection Sunday, update `data/bracket.json` with the real 64-team
bracket. The file already has a template — just update the team names, seeds,
and regions to match the actual bracket.

### Step 3 — See the contest schedule

```bash
docker run --rm -v ./config.yaml:/app/config.yaml marchmadness schedule
```

This shows the 9 game days, dates, and how many picks you need each day.

### Step 4 — Get Day 1 picks

```bash
docker run --rm \
  -v ./data:/app/data \
  -v ./models:/app/models \
  -v ./entries:/app/entries \
  -v ./config.yaml:/app/config.yaml \
  marchmadness optimize --day 1
```

**This is the main event.** It runs 50,000 tournament simulations, evaluates
every team, and recommends the best picks for each of your entries. Output
looks something like:

```
=== Day 1 Recommendations (R64 Thursday) ===

Entry 1: (3) Iowa St.  + (3) Purdue    Win=93%  Own=2.0%   EV=$145.20
Entry 2: (4) Kansas     + (2) UConn     Win=89%  Own=1.7%   EV=$138.50
Entry 3: (1) Duke       + (5) Michigan  Win=95%  Own=8.2%   EV=$112.30
...

Total portfolio EV: $663.51
Joint survival probability: 100%
```

**How to read this:**
- **Win%** = probability both picks survive that day
- **Own%** = what fraction of the pool you expect to also pick that team
  (lower = better — you want to be different)
- **EV** = expected dollar value of that entry

### Step 5 — Enter your picks

Go to your pool's website and submit the recommended picks for each entry.

---

## During the Tournament

After each game day, record which teams won and get updated picks for the
next day.

### Record results

```bash
docker run --rm \
  -v ./entries:/app/entries \
  -v ./config.yaml:/app/config.yaml \
  marchmadness results --day 1 1234 5678
```

Replace `1234 5678` with the team IDs of the winners from that day's games.
The tool marks any eliminated entries and remembers which teams you've
already used (no repeats allowed).

### Check your status

```bash
docker run --rm \
  -v ./entries:/app/entries \
  -v ./config.yaml:/app/config.yaml \
  marchmadness status
```

Shows which entries are alive, which are eliminated, and what teams have
been used.

### Get next day's picks

```bash
docker run --rm \
  -v ./data:/app/data \
  -v ./models:/app/models \
  -v ./entries:/app/entries \
  -v ./config.yaml:/app/config.yaml \
  marchmadness optimize --day 2
```

Change `--day 2` to whatever day you're on. The optimizer accounts for
teams you've already used and entries that were eliminated.

**Repeat this cycle for all 9 game days.**

### Get mid-tournament advice (live bracket)

Once the tournament is underway, `advise` pulls live results from ESPN,
shows which teams are still available for each entry, and guarantees you
won't run out of picks in future rounds:

```bash
docker run --rm \
  -v ./data:/app/data \
  -v ./models:/app/models \
  -v ./entries:/app/entries \
  -v ./config.yaml:/app/config.yaml \
  marchmadness advise
```

It auto-detects the current day and round. You can also specify used teams
manually:

```bash
docker run --rm \
  -v ./data:/app/data \
  -v ./models:/app/models \
  -v ./config.yaml:/app/config.yaml \
  marchmadness advise --used "0:Duke,Florida;1:Duke,UConn"
```

The format is `entry:team1,team2;entry:team1,team2`. The tool will:
- Show available (alive + unused) teams for each entry
- Flag entries that are **at risk** or **critical** (running low on options)
- Recommend a pick that maximizes EV while keeping future coverage safe

If ESPN is unavailable, add `--no-live` to skip the live fetch and use
your local bracket file instead.

### Analyze your portfolio (large entry pools)

If you have many entries, `analyze` runs a full distribution analysis to
show how diversified your portfolio is:

```bash
docker run --rm \
  -v ./data:/app/data \
  -v ./models:/app/models \
  -v ./entries:/app/entries \
  -v ./output:/app/output \
  -v ./config.yaml:/app/config.yaml \
  marchmadness analyze --sims 10000
```

This generates:
- **Team concentration** — are too many entries on the same team? (HHI index)
- **Survival distribution** — how many entries survive each round on average?
- **Entry correlation** — if one team loses, how many entries die together?
- **Charts** saved to `output/` (heatmap, histogram, funnel, correlation matrix, exposure bar)

See `samples/analyze_output.txt` for example output.

---

## Troubleshooting

### "docker: command not found"
Docker Desktop isn't running or isn't installed. Open Docker Desktop and
wait for it to start.

### "kaggle.json not found" or download fails
Make sure `kaggle.json` is in the right folder:
- Windows: `C:\Users\<YourName>\.kaggle\kaggle.json`
- Mac/Linux: `~/.kaggle/kaggle.json`

### "No module named ..." or import errors
You might be running Python directly instead of using Docker. Always use
the `docker run` commands shown above.

### Optimize is slow
The `hybrid` method (default) takes about 60 seconds. If you want faster
results while testing, add `--method analytical` (takes ~1 second but
slightly less optimal).

### "bracket.json not found"
You need to create the bracket file after Selection Sunday. See the
template in `data/bracket.json`.

---

## Quick Reference

| What | Command |
|------|---------|
| Build the tool | `docker build -t marchmadness .` |
| Download data | `docker run --rm -v ~/.kaggle:/root/.kaggle -v ./data:/app/data marchmadness download` |
| Train model | `docker run --rm -v ./data:/app/data -v ./models:/app/models marchmadness train` |
| Get picks | `docker run --rm -v ./data:/app/data -v ./models:/app/models -v ./entries:/app/entries -v ./config.yaml:/app/config.yaml marchmadness optimize --day 1` |
| Mid-tournament advice | `docker run --rm -v ./data:/app/data -v ./models:/app/models -v ./entries:/app/entries -v ./config.yaml:/app/config.yaml marchmadness advise` |
| Analyze portfolio | `docker run --rm -v ./data:/app/data -v ./models:/app/models -v ./entries:/app/entries -v ./output:/app/output -v ./config.yaml:/app/config.yaml marchmadness analyze` |
| Record results | `docker run --rm -v ./entries:/app/entries -v ./config.yaml:/app/config.yaml marchmadness results --day 1 <team_ids>` |
| Check status | `docker run --rm -v ./entries:/app/entries -v ./config.yaml:/app/config.yaml marchmadness status` |

---

## Using Make (Shortcut)

If you have `make` installed (Mac/Linux usually do, Windows needs
[Git Bash](https://gitforwindows.org/) or similar), there are shortcuts:

```bash
make build              # Build Docker image
make test               # Run all tests
make optimize-day1      # Day 1 picks with defaults from config.yaml
make optimize-day2      # Day 2 picks
make optimize-all       # Both Round of 64 days
```

Override pool settings on the fly:

```bash
make optimize-day1 POOL_SIZE=10000 NUM_ENTRIES=3
```
