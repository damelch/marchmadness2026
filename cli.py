"""Command-line interface for the March Madness Survivor Pool Optimizer."""

from pathlib import Path

import click
import pandas as pd
import yaml

from contest.schedule import ContestSchedule


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


@click.group()
@click.option("--config", default="config.yaml", help="Path to config file")
@click.pass_context
def main(ctx, config):
    """March Madness Survivor Pool Optimizer"""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["config"] = load_config(config)
    ctx.obj["schedule"] = ContestSchedule.from_config(ctx.obj["config"])


@main.command()
@click.pass_context
def download(ctx):
    """Download historical NCAA tournament data from Kaggle."""
    from data.scrapers.kaggle_data import download_kaggle_data

    config = ctx.obj["config"]
    raw_dir = config["data"]["raw_dir"]

    click.echo("Downloading Kaggle NCAA tournament data...")
    download_kaggle_data("march-machine-learning-mania-2025", raw_dir)
    click.echo("Done!")


@main.command("fetch-bpi")
@click.option("--year", default=2026, help="Season year")
@click.pass_context
def fetch_bpi(ctx, year):
    """Fetch ESPN BPI ratings (no auth required)."""
    from data.scrapers.espn_bpi import fetch_espn_bpi

    path = f"data/espn_bpi_{year}.csv"
    click.echo(f"Fetching ESPN BPI data for {year}...")
    df = fetch_espn_bpi(year=year, save_path=path)
    if not df.empty:
        click.echo(f"Saved {len(df)} teams to {path}")
    else:
        click.echo("Failed to fetch BPI data. Check network connection.")


@main.command("fetch-barttorvik")
@click.option("--year", default=2026, help="Season year")
@click.pass_context
def fetch_barttorvik(ctx, year):
    """Fetch Barttorvik T-Rank ratings."""
    from data.scrapers.barttorvik import fetch_barttorvik

    path = f"data/barttorvik_{year}.csv"
    click.echo(f"Fetching Barttorvik data for {year}...")
    df = fetch_barttorvik(year=year, save_path=path)
    if not df.empty:
        click.echo(f"Saved {len(df)} teams to {path}")
    else:
        click.echo(
            "Could not fetch Barttorvik data (may require browser).\n"
            f"Manually save T-Rank table as {path} from https://barttorvik.com/trank.php"
        )


@main.command()
@click.pass_context
def features(ctx):
    """Build feature matrix from raw data."""
    from data.feature_engineering import build_matchup_features, save_features
    from data.scrapers.kaggle_data import load_dataset

    config = ctx.obj["config"]
    raw_dir = config["data"]["raw_dir"]
    proc_dir = config["data"]["processed_dir"]

    click.echo("Loading datasets...")
    tourney = load_dataset("tourney_compact", raw_dir)
    seeds = load_dataset("seeds", raw_dir)
    regular = load_dataset("regular_detailed", raw_dir)

    massey = None
    try:
        massey = load_dataset("massey", raw_dir)
    except FileNotFoundError:
        click.echo("Massey ordinals not found, proceeding without them")

    seasons = config["model"]["historical_seasons"]
    click.echo(f"Building features for {len(seasons)} seasons...")

    features_df = build_matchup_features(tourney, seeds, regular, massey, seasons)
    path = save_features(features_df, proc_dir)

    click.echo(f"Saved {len(features_df)} feature rows to {path}")
    click.echo(f"Seasons: {features_df['Season'].nunique()}")
    click.echo(f"Features: {features_df.columns.tolist()}")


@main.command()
@click.option("--model-type", default=None, help="Model type: logistic, xgboost, ensemble")
@click.pass_context
def train(ctx):
    """Train win probability model."""
    from data.feature_engineering import load_features
    from models.train import save_model, train_model

    config = ctx.obj["config"]
    model_type = config["model"]["type"]
    calibrate = config["model"]["calibrate"]

    click.echo(f"Training {model_type} model (calibrate={calibrate})...")

    features_df = load_features(config["data"]["processed_dir"])
    model = train_model(features_df, model_type, calibrate)
    save_model(model)

    click.echo("Model saved to models/saved/model.pkl")


@main.command()
@click.option("--model-type", default=None, help="Override model type from config")
@click.option("--output", default="models/saved", help="Directory for calibration charts")
@click.pass_context
def evaluate(ctx, model_type, output):
    """Evaluate model calibration and accuracy."""
    from data.feature_engineering import load_features
    from models.evaluate import evaluate_model

    config = ctx.obj["config"]
    mt = model_type or config["model"]["type"]
    calibrate = config["model"]["calibrate"]

    features_df = load_features(config["data"]["processed_dir"])
    evaluate_model(features_df, mt, calibrate, output_dir=output)


@main.command()
@click.pass_context
def simulate(ctx):
    """Run Monte Carlo tournament simulation (requires bracket)."""
    from models.predict import Predictor
    from models.train import load_model
    from simulation.engine import simulate_tournament, team_advancement_probs

    config = ctx.obj["config"]
    n_sims = config["simulation"]["num_sims"]

    click.echo("Loading model and bracket...")

    bracket_path = Path("data/bracket.json")
    if not bracket_path.exists():
        click.echo("No bracket file found at data/bracket.json")
        click.echo("Create one with `marchmadness bracket` or wait for Selection Sunday")
        click.echo("\nRunning demo simulation with seed-based probabilities...")

        from data.seed_history import get_seed_win_prob
        bracket = _create_demo_bracket()
        def predict_fn(a, b):
            return get_seed_win_prob(
                    bracket.teams[a]["seed"], bracket.teams[b]["seed"]
                )
    else:
        import json
        with open(bracket_path) as f:
            bracket_data = json.load(f)
        bracket = _load_bracket(bracket_data)

        seed_map = {t: info["seed"] for t, info in bracket.teams.items()}
        model_path = Path("models/saved/model.pkl")
        kenpom_path = Path("data/kenpom_2026.csv")

        if model_path.exists():
            model = load_model()
            predictor = Predictor(model=model, seed_map=seed_map)
            predict_fn = predictor.predict_matchup
            click.echo("Using trained ML model")
        elif kenpom_path.exists():
            from data.kenpom import load_kenpom
            kenpom_df = load_kenpom(kenpom_path)
            # Map bracket team IDs to names for KenPom lookup
            id_to_name = {t: info["name"] for t, info in bracket.teams.items()}
            predictor = Predictor(
                kenpom_df=kenpom_df,
                kenpom_id_to_name=id_to_name,
                seed_map=seed_map,
            )
            predict_fn = predictor.predict_matchup
            click.echo("No trained model — using KenPom ratings")
        else:
            from data.seed_history import get_seed_win_prob
            def predict_fn(a, b):
                return get_seed_win_prob(
                            bracket.teams[a]["seed"], bracket.teams[b]["seed"]
                        )
            click.echo("No trained model or KenPom data — using seed-based probabilities")

    click.echo(f"Running {n_sims:,} simulations...")
    sim_results = simulate_tournament(
        bracket, predict_fn, n_sims=n_sims,
        rng_seed=config["simulation"]["seed"],
    )

    probs = team_advancement_probs(sim_results, bracket)
    click.echo("\nTeam Advancement Probabilities:")
    click.echo(probs.to_string(index=False, float_format="{:.1%}".format))


@main.command()
@click.pass_context
def schedule(ctx):
    """Show the contest day schedule."""
    sched = ctx.obj["schedule"]

    click.echo(f"\n{'='*60}")
    click.echo("CONTEST SCHEDULE")
    click.echo(f"{'='*60}")
    click.echo(f"{'Day':>4}  {'Label':<20}  {'Date':<12}  {'Round':>5}  {'Picks':>5}  {'Regions'}")
    click.echo(f"{'-'*60}")

    total_picks = 0
    for day in sched.days:
        regions_str = ", ".join(day.regions)
        pick_str = str(day.num_picks)
        if day.is_double_pick:
            pick_str += " *"
        click.echo(
            f"{day.day_num:>4}  {day.label:<20}  {day.date:<12}  {day.round_num:>5}  "
            f"{pick_str:>5}  {regions_str}"
        )
        total_picks += day.num_picks

    click.echo(f"{'-'*60}")
    click.echo(f"Total: {sched.total_days()} days, {total_picks} picks")
    click.echo("* = double-pick day (both picks must win)")


@main.command()
@click.option("--day", "day_num", type=int, required=True, help="Contest day number (1-9)")
@click.option("--method", default="hybrid", type=click.Choice(["differentiation", "analytical", "hybrid", "aco"]))
@click.option("--pool-size", type=int, default=None, help="Total entries in the contest (default: 10000)")
@click.option("--num-entries", type=int, default=None, help="Number of your entries (default: from config)")
@click.option("--max-entries", type=int, default=None, help="Max entries per user allowed (default: 150)")
@click.pass_context
def optimize(ctx, day_num, method, pool_size, num_entries, max_entries):
    """Generate optimal picks for a contest day."""
    from entries.generator import generate_picks
    from entries.manager import EntryManager
    from models.predict import Predictor
    from models.train import load_model

    config = ctx.obj["config"]
    sched = ctx.obj["schedule"]

    # CLI overrides > config > defaults
    if pool_size is not None:
        config["pool"]["pool_size"] = pool_size
    elif config["pool"].get("pool_size") is None:
        config["pool"]["pool_size"] = 10000
    if num_entries is not None:
        config["pool"]["num_entries"] = num_entries
    if max_entries is not None:
        config["pool"]["max_entries_per_user"] = max_entries
    elif config["pool"].get("max_entries_per_user") is None:
        config["pool"]["max_entries_per_user"] = 150

    day = sched.get_day(day_num)
    click.echo(f"Optimizing picks for Day {day_num} ({day.label}) — {day.num_picks} pick(s)...")

    bracket_path = Path("data/bracket.json")
    if not bracket_path.exists():
        click.echo("No bracket found. Using demo bracket with seed-based probs.")
        bracket = _create_demo_bracket()
        predictor = Predictor(seed_map={
            t: info["seed"] for t, info in bracket.teams.items()
        })
    else:
        import json
        with open(bracket_path) as f:
            bracket = _load_bracket(json.load(f))

        seed_map = {t: info["seed"] for t, info in bracket.teams.items()}
        model_path = Path("models/saved/model.pkl")
        kenpom_path = Path("data/kenpom_2026.csv")

        if model_path.exists():
            model = load_model()
            predictor = Predictor(model=model, seed_map=seed_map)
        elif kenpom_path.exists():
            from data.kenpom import load_kenpom
            kenpom_df = load_kenpom(kenpom_path)
            id_to_name = {t: info["name"] for t, info in bracket.teams.items()}
            predictor = Predictor(
                kenpom_df=kenpom_df,
                kenpom_id_to_name=id_to_name,
                seed_map=seed_map,
            )
        else:
            predictor = Predictor(seed_map=seed_map)

    entry_path = Path("entries/state.json")
    if entry_path.exists():
        mgr = EntryManager.load(entry_path)
    else:
        n = config["pool"]["num_entries"]
        mgr = EntryManager(reuse_allowed=config["pool"]["rules"]["reuse_allowed"])
        mgr.create_entries(n)
        click.echo(f"Created {n} new entries")

    results = generate_picks(bracket, predictor, mgr, day_num, sched, config, method)

    pool_size_actual = config["pool"]["pool_size"]
    prize_pool_actual = config["pool"]["prize_pool"]
    max_ent = config["pool"].get("max_entries_per_user", 150)

    click.echo(f"\n{'='*70}")
    click.echo(f"PICK RECOMMENDATIONS - Day {day_num} ({day.label})")
    click.echo(f"  Pool: {pool_size_actual:,} entries | Prize: ${prize_pool_actual:,.0f} | Max/user: {max_ent}")
    if day.is_double_pick:
        click.echo("  ** Double-pick day: both picks must win to survive **")
    click.echo(f"{'='*70}")

    for entry_id, team_ids in results.get("recommendations", {}).items():
        parts = []
        for team_id in team_ids:
            info = bracket.teams.get(team_id, {})
            wp = results["win_probs"].get(team_id, 0)
            own = results["ownership"].get(team_id, 0)
            parts.append(
                f"({info.get('seed', '?')}) {info.get('name', team_id)} "
                f"Win={wp:.1%} Own={own:.1%}"
            )
        pick_str = " + ".join(parts)
        click.echo(f"  Entry {entry_id}: {pick_str}")

    if "dp_analysis" in results:
        for line in results["dp_analysis"].get("reasoning", []):
            click.echo(f"  {line}")

    if "evaluation" in results:
        ev = results["evaluation"]
        click.echo("\nPortfolio Analysis:")
        click.echo(f"  Total EV: ${ev.get('total_ev', 0):.2f}")
        if "joint_survival" in ev:
            click.echo(f"  Joint survival: {ev['joint_survival']:.1%}")


@main.command()
@click.option("--day", "day_num", type=int, default=None,
              help="Contest day number (auto-detected if omitted)")
@click.option("--used", "used_teams_str", type=str, default=None,
              help='Used teams per entry: "0:Duke,Florida;1:Duke,UConn"')
@click.option("--entries-alive", type=int, default=None,
              help="Number of alive entries (default: from state or config)")
@click.option("--no-live", is_flag=True, default=False,
              help="Skip ESPN live fetch; use bracket.json as-is (all teams alive)")
@click.option("--method", default="analytical",
              type=click.Choice(["analytical", "hybrid"]))
@click.pass_context
def advise(ctx, day_num, used_teams_str, entries_alive, no_live, method):
    """Get mid-tournament pick advice with coverage guarantees."""
    import json as _json

    from data.live_bracket import fetch_live_state
    from entries.manager import EntryManager
    from models.predict import Predictor
    from optimizer.coverage import compute_coverage

    config = ctx.obj["config"]
    sched = ctx.obj["schedule"]

    # --- Load bracket ---
    bracket_path = Path("data/bracket.json")
    if not bracket_path.exists():
        click.echo("Error: data/bracket.json not found.")
        return
    with open(bracket_path) as f:
        bracket = _load_bracket(_json.load(f))

    # --- Build predictor ---
    seed_map = {t: info["seed"] for t, info in bracket.teams.items()}
    model_path = Path("models/saved/model.pkl")
    if model_path.exists():
        from models.train import load_model
        model = load_model()
        predictor = Predictor(model=model, seed_map=seed_map)
    else:
        predictor = Predictor(seed_map=seed_map)

    # --- Fetch live bracket state ---
    all_team_ids = set(bracket.teams.keys())
    live_state = None

    if not no_live:
        click.echo("Fetching live bracket from ESPN...")
        live_state = fetch_live_state(bracket, sched)
        if live_state is None:
            click.echo("  Warning: ESPN API unavailable. Using all teams as alive.")

    if live_state is not None:
        alive_teams = live_state.alive_team_ids
        detected_day = live_state.current_day
        games_completed = live_state.games_completed
        if live_state.unmatched_teams:
            click.echo(f"  Warning: Could not match ESPN names: {live_state.unmatched_teams}")
    else:
        alive_teams = all_team_ids
        detected_day = 1
        games_completed = 0

    if day_num is None:
        day_num = detected_day
        click.echo(f"  Auto-detected contest day: {day_num}")

    day = sched.get_day(day_num)

    # --- Parse used teams ---
    entries_data: dict[int, set[int]] = {}

    # First try loading from state.json
    entry_path = Path("entries/state.json")
    if entry_path.exists():
        mgr = EntryManager.load(entry_path)
        for entry in mgr.get_alive_entries():
            entries_data[entry.entry_id] = set(entry.used_teams)

    # Override/supplement with --used flag
    if used_teams_str:
        entries_data = _parse_used_teams(used_teams_str, bracket)

    # If no entries at all, create default entries
    if not entries_data:
        n = entries_alive or config["pool"].get("num_entries", 1)
        for i in range(n):
            entries_data[i] = set()

    # If --entries-alive limits the count
    if entries_alive is not None and entries_alive < len(entries_data):
        entries_data = dict(list(entries_data.items())[:entries_alive])

    # --- Compute win probabilities for today's matchups ---
    matchups = bracket.get_day_matchups(day.round_num, day.regions)
    teams_playing = set()
    win_probs: dict[int, float] = {}
    for team_a, team_b, _slot in matchups:
        if team_a is not None and team_b is not None:
            teams_playing.add(team_a)
            teams_playing.add(team_b)
            try:
                p = predictor.predict_matchup(team_a, team_b)
                win_probs[team_a] = p
                win_probs[team_b] = 1.0 - p
            except Exception:
                # Seed-based fallback
                sa = bracket.teams.get(team_a, {}).get("seed", 8)
                sb = bracket.teams.get(team_b, {}).get("seed", 8)
                pa = 0.5 + 0.03 * (sb - sa)
                pa = max(0.05, min(0.95, pa))
                win_probs[team_a] = pa
                win_probs[team_b] = 1.0 - pa

    # --- Header ---
    source = "ESPN Live API" if live_state else "bracket.json (offline)"
    n_alive = len(alive_teams)
    n_total = len(all_team_ids)
    click.echo(f"\n{'='*70}")
    click.echo(f"LIVE BRACKET ADVISORY — Day {day_num} ({day.label})")
    click.echo(f"  Source: {source} | Round: {day.round_num} | Regions: {', '.join(day.regions)}")
    click.echo(f"  Alive: {n_alive}/{n_total} teams | Games completed: {games_completed}/63")
    click.echo(f"{'='*70}")

    # --- Per-entry analysis ---
    for entry_id, used_teams in sorted(entries_data.items()):
        coverage = compute_coverage(
            bracket, alive_teams, used_teams,
            day_num, sched, win_probs,
        )
        coverage.entry_id = entry_id

        # Used teams display
        used_names = []
        for t in sorted(used_teams):
            info = bracket.teams.get(t, {})
            used_names.append(f"({info.get('seed', '?')}) {info.get('name', t)}")
        used_str = ", ".join(used_names) if used_names else "(none)"

        click.echo(f"\nENTRY {entry_id}:")
        click.echo(f"  Used teams: {used_str}")
        click.echo(f"  Available (alive & unused): {len(coverage.available_teams)} teams")

        # Show available teams
        for t in coverage.available_teams[:12]:
            marker = "*" if t["id"] in coverage.safety_set else " "
            click.echo(
                f"   {marker}({t['seed']}) {t['name']:20s} [{t['region']}]  "
                f"Win={t['win_prob']:.1%}"
            )
        if len(coverage.available_teams) > 12:
            click.echo(f"    ... and {len(coverage.available_teams) - 12} more")

        # Coverage assessment
        risk_icon = {"safe": "SAFE", "at_risk": "AT RISK", "critical": "CRITICAL"}
        click.echo(
            f"  Coverage: {risk_icon[coverage.risk_level]} — {coverage.risk_reason}"
        )

        # Safety set
        if coverage.safety_set and coverage.risk_level != "safe":
            safety_names = []
            for t_id in coverage.safety_set:
                info = bracket.teams.get(t_id, {})
                safety_names.append(info.get("name", str(t_id)))
            click.echo(f"  Safety set: {{{', '.join(safety_names)}}}")

        # Uncovered matchups warning
        if coverage.uncovered_matchups:
            click.echo("  WARNING: Uncovered matchups (both sides already used):")
            for a, b in coverage.uncovered_matchups:
                na = bracket.teams.get(a, {}).get("name", str(a))
                nb = bracket.teams.get(b, {}).get("name", str(b))
                click.echo(f"    {na} vs {nb}")

        # Future day risks
        if coverage.future_risks:
            risky_days = [
                r for r in coverage.future_risks
                if r.worst_case_available < 2 * r.picks_needed
            ]
            if risky_days:
                click.echo("  Future day outlook:")
                for r in risky_days:
                    status = "OK"
                    if r.worst_case_available < r.picks_needed:
                        status = "BLOCKED"
                    elif r.worst_case_available < 2 * r.picks_needed:
                        status = "TIGHT"
                    scenarios = ""
                    if r.blocked_scenarios > 0:
                        scenarios = (
                            f" ({r.blocked_scenarios}/{r.total_scenarios} "
                            f"scenarios blocked)"
                        )
                    click.echo(
                        f"    Day {r.day_num} ({r.label}): {status} — "
                        f"worst case {r.worst_case_available} avail, "
                        f"need {r.picks_needed}{scenarios}"
                    )

        # Recommendation: best available by EV
        if coverage.available_teams:
            best = coverage.available_teams[0]
            click.echo(
                f"  > Recommended: ({best['seed']}) {best['name']} — "
                f"Win={best['win_prob']:.1%}"
            )

    click.echo(f"\n{'='*70}")


def _parse_used_teams(
    used_str: str, bracket,
) -> dict[int, set[int]]:
    """Parse --used flag: '0:Duke,Florida;1:Duke,UConn' -> {0: {id, id}, ...}."""
    from data.live_bracket import resolve_team_name

    result: dict[int, set[int]] = {}
    for entry_part in used_str.split(";"):
        entry_part = entry_part.strip()
        if not entry_part:
            continue
        if ":" in entry_part:
            eid_str, teams_str = entry_part.split(":", 1)
            eid = int(eid_str.strip())
        else:
            # Single entry mode — entry 0
            eid = 0
            teams_str = entry_part

        team_ids: set[int] = set()
        for name in teams_str.split(","):
            name = name.strip()
            if not name:
                continue
            tid = resolve_team_name(name, bracket)
            if tid is not None:
                team_ids.add(tid)
            else:
                click.echo(f"  Warning: Could not resolve team name '{name}'")
        result[eid] = team_ids

    return result


@main.command("results")
@click.option("--day", "day_num", type=int, required=True)
@click.argument("winners", nargs=-1, type=int)
@click.pass_context
def results_cmd(ctx, day_num, winners):
    """Update entries with actual results. Pass winning team IDs as arguments."""
    from entries.manager import EntryManager

    mgr = EntryManager.load("entries/state.json")
    stats = mgr.update_results(day_num, set(winners))
    mgr.save()

    day = ctx.obj["schedule"].get_day(day_num)
    click.echo(f"\nDay {day_num} ({day.label}) Results:")
    click.echo(f"  Survived: {stats['survived']}")
    click.echo(f"  Eliminated: {stats['eliminated']}")
    click.echo(f"  Total alive: {stats['total_alive']}")


@main.command()
@click.pass_context
def status(ctx):
    """Show current entry status and picks."""
    from entries.manager import EntryManager

    entry_path = Path("entries/state.json")
    if not entry_path.exists():
        click.echo("No entries found. Run `marchmadness optimize` first.")
        return

    mgr = EntryManager.load(entry_path)
    sheets = mgr.export_pick_sheets()

    click.echo(f"\n{'='*60}")
    click.echo("ENTRY STATUS")
    click.echo(f"{'='*60}")

    alive = len([e for e in mgr.entries if e.alive])
    click.echo(f"Total entries: {len(mgr.entries)} | Alive: {alive}")
    click.echo()

    df = pd.DataFrame(sheets)
    click.echo(df.to_string(index=False))


@main.command()
@click.option("--output", "output_dir", default="output",
              help="Directory for chart PNGs (default: output/)")
@click.option("--sims", "n_sims", type=int, default=10000,
              help="Monte Carlo simulations (default: 10000)")
@click.pass_context
def analyze(ctx, output_dir, n_sims):
    """Distribution analysis: concentration, survival, and correlation across entries."""
    import json as _json

    from entries.manager import EntryManager
    from models.predict import Predictor
    from optimizer.distribution import analyze_distribution
    from visualization.charts import generate_all_charts

    sched = ctx.obj["schedule"]

    # Load bracket
    bracket_path = Path("data/bracket.json")
    if not bracket_path.exists():
        click.echo("Error: data/bracket.json not found.")
        return
    with open(bracket_path) as f:
        bracket = _load_bracket(_json.load(f))

    # Load entries
    entry_path = Path("entries/state.json")
    if not entry_path.exists():
        click.echo("No entries found. Run `marchmadness optimize` first.")
        return
    mgr = EntryManager.load(entry_path)
    alive = mgr.get_alive_entries()
    if not alive:
        click.echo("All entries eliminated. Nothing to analyze.")
        return

    # Build predictor
    seed_map = {t: info["seed"] for t, info in bracket.teams.items()}
    model_path = Path("models/saved/model.pkl")
    if model_path.exists():
        from models.train import load_model
        model = load_model()
        predictor = Predictor(model=model, seed_map=seed_map)
    else:
        predictor = Predictor(seed_map=seed_map)

    click.echo(f"Analyzing {len(alive)} alive entries ({n_sims:,} simulations)...")
    report = analyze_distribution(
        bracket, mgr, sched, predictor.predict_matchup, n_sims=n_sims,
    )

    # --- Team Concentration ---
    click.echo(f"\n{'='*70}")
    click.echo("TEAM CONCENTRATION")
    click.echo(f"{'='*70}")
    for conc in report.concentration_by_day:
        max_info = bracket.teams.get(conc.max_concentration_team, {})
        max_name = max_info.get("name", "?")
        click.echo(
            f"  Day {conc.day_num} ({conc.label}): "
            f"{conc.n_unique_teams} unique teams | "
            f"HHI={conc.hhi:.3f} | "
            f"Max: {conc.max_concentration:.0%} on ({max_info.get('seed', '?')}) {max_name}"
        )

    # --- Survival Distribution ---
    click.echo(f"\n{'='*70}")
    click.echo("SURVIVAL DISTRIBUTION (Monte Carlo)")
    click.echo(f"{'='*70}")
    click.echo(f"  {'Day':<25s} {'Mean':>6s} {'Med':>5s} {'Std':>5s} "
               f"{'Min':>4s} {'Max':>4s} {'P(0)':>6s} {'P(1+)':>6s}")
    click.echo(f"  {'-'*62}")
    for surv in report.survival:
        click.echo(
            f"  {surv.label:<25s} {surv.mean_alive:>6.1f} {surv.median_alive:>5.0f} "
            f"{surv.std_alive:>5.1f} {surv.min_alive:>4d} {surv.max_alive:>4d} "
            f"{surv.p_zero:>6.1%} {surv.p_at_least_one:>6.1%}"
        )

    # --- Correlation ---
    click.echo(f"\n{'='*70}")
    click.echo("ENTRY CORRELATION")
    click.echo(f"{'='*70}")
    click.echo(f"  Mean pairwise elimination correlation: {report.correlation.mean_pairwise:.3f}")

    # Top exposure
    sorted_exp = sorted(
        report.correlation.team_exposure.items(), key=lambda x: x[1], reverse=True,
    )[:10]
    click.echo("\n  Top team exposure (entries eliminated if team loses):")
    for team_id, count in sorted_exp:
        info = bracket.teams.get(team_id, {})
        frac = count / report.n_alive if report.n_alive else 0
        click.echo(
            f"    ({info.get('seed', '?')}) {info.get('name', team_id):20s}: "
            f"{count} entries ({frac:.0%})"
        )

    # --- Generate charts ---
    click.echo(f"\nGenerating charts to {output_dir}/...")
    saved = generate_all_charts(report, output_dir)
    for p in saved:
        click.echo(f"  Saved: {p}")

    click.echo(f"\n{'='*70}")


@main.command("fetch-odds")
@click.option("--historical", is_flag=True, help="Validate historical CSV instead of fetching live")
@click.option("--save", default=None, help="Save live odds to this CSV path")
@click.pass_context
def fetch_odds(ctx, historical, save):
    """Fetch Vegas betting lines (live or validate historical)."""
    from data.scrapers.vegas_lines import fetch_current_odds, load_historical_vegas

    if historical:
        click.echo("Validating historical Vegas lines CSV...")
        df = load_historical_vegas()
        if df.empty:
            click.echo(
                "No historical Vegas data found.\n"
                "Place a CSV at data/vegas_historical.csv with columns:\n"
                "  Season, WTeamID, LTeamID, Spread, OverUnder"
            )
        else:
            click.echo(f"Found {len(df)} rows covering seasons: "
                        f"{sorted(df['Season'].unique())}")
    else:
        click.echo("Fetching live NCAA tournament odds...")
        save_path = save or "data/vegas_live.csv"
        df = fetch_current_odds(save_path=save_path)
        if df.empty:
            click.echo(
                "No odds returned. Set ODDS_API_KEY env var or pass --historical.\n"
                "Get a free key at https://the-odds-api.com"
            )
        else:
            click.echo(f"Fetched odds for {df['HomeTeam'].nunique()} games")


@main.command()
@click.option("--season", default=None, type=int, help="Single season to backtest")
@click.option("--entries", default=5, help="Number of simulated entries")
@click.option("--pool", default=10000, help="Simulated pool size")
@click.option("--prize", default=250000.0, help="Simulated prize pool ($)")
@click.option("--strategy", default=None, help="Single strategy (optimizer/top_seeds/random/contrarian)")
@click.pass_context
def backtest(ctx, season, entries, pool, prize, strategy):
    """Backtest optimizer against historical tournaments (2015-2024)."""
    import logging

    from data.feature_engineering import load_features
    from data.scrapers.kaggle_data import load_dataset
    from models.backtest import backtest_all, backtest_season

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = ctx.obj["config"]
    raw_dir = config["data"]["raw_dir"]
    model_type = config["model"]["type"]
    calibrate = config["model"]["calibrate"]

    click.echo("Loading data...")
    features_df = load_features(config["data"]["processed_dir"])
    seeds_df = load_dataset("seeds", raw_dir)
    results_df = load_dataset("tourney_compact", raw_dir)
    teams_df = load_dataset("teams", raw_dir)

    strategies = [strategy] if strategy else None
    seasons = [season] if season else None

    if season:
        click.echo(f"Backtesting season {season}...")
        strats = strategies or ["optimizer", "top_seeds", "random", "contrarian"]
        for strat in strats:
            result = backtest_season(
                season=season,
                features_df=features_df,
                seeds_df=seeds_df,
                results_df=results_df,
                teams_df=teams_df,
                strategy=strat,
                n_entries=entries,
                pool_size=pool,
                prize_pool=prize,
                model_type=model_type,
                calibrate=calibrate,
            )
            click.echo(
                f"  {strat:12s}  survived={result.avg_days_survived:.1f}/{len(result.days_survived)} "
                f"final_alive={result.final_alive}/{result.n_entries}"
            )
    else:
        click.echo("Running full backtest across all seasons...")
        summary = backtest_all(
            features_df=features_df,
            seeds_df=seeds_df,
            results_df=results_df,
            teams_df=teams_df,
            seasons=seasons,
            strategies=strategies,
            n_entries=entries,
            pool_size=pool,
            prize_pool=prize,
            model_type=model_type,
            calibrate=calibrate,
        )

        if summary.empty:
            click.echo("No results — check that data exists for the requested seasons.")
        else:
            click.echo(f"\n{'='*70}")
            click.echo(f"BACKTEST RESULTS ({entries} entries, pool={pool:,})")
            click.echo(f"{'='*70}")
            click.echo(f"{'Strategy':<14} {'AvgSurvived':>12} {'BestSeason':>12} {'WorstSeason':>13} {'SurvivalRate':>13}")
            click.echo("-" * 70)

            for strat in summary["Strategy"].unique():
                sdf = summary[summary["Strategy"] == strat]
                avg_surv = sdf["AvgDaysSurvived"].mean()
                best_idx = sdf["AvgDaysSurvived"].idxmax()
                worst_idx = sdf["AvgDaysSurvived"].idxmin()
                best_s = f"{int(sdf.loc[best_idx, 'Season'])} ({sdf.loc[best_idx, 'FinalAlive']}/{entries})"
                worst_s = f"{int(sdf.loc[worst_idx, 'Season'])} ({sdf.loc[worst_idx, 'FinalAlive']}/{entries})"
                sr = sdf["SurvivalRate"].mean()
                click.echo(f"{strat:<14} {avg_surv:>12.1f} {best_s:>12} {worst_s:>13} {sr:>12.1%}")

            click.echo(f"{'='*70}")


def _create_demo_bracket():
    """Create a demo bracket with fake teams for testing."""
    from simulation.engine import TournamentBracket

    bracket = TournamentBracket()
    team_id = 1000
    for region in ["W", "X", "Y", "Z"]:
        for seed in range(1, 17):
            bracket.set_seed(team_id, seed, region, name=f"{region}{seed}-Team")
            team_id += 1
    return bracket


def _load_bracket(data: dict):
    """Load bracket from JSON data."""
    from simulation.engine import TournamentBracket

    bracket = TournamentBracket()
    for team in data.get("teams", []):
        bracket.set_seed(
            team["id"], team["seed"], team["region"],
            name=team.get("name", ""),
        )
    return bracket


if __name__ == "__main__":
    main()
