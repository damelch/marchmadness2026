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
@click.pass_context
def evaluate(ctx, model_type):
    """Evaluate model calibration and accuracy."""
    from data.feature_engineering import load_features
    from models.evaluate import evaluate_model

    config = ctx.obj["config"]
    mt = model_type or config["model"]["type"]
    calibrate = config["model"]["calibrate"]

    features_df = load_features(config["data"]["processed_dir"])
    evaluate_model(features_df, mt, calibrate)


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
