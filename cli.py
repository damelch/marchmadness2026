"""Command-line interface for the March Madness Survivor Pool Optimizer."""

import click
import yaml
import pandas as pd
from pathlib import Path


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


@main.command()
@click.pass_context
def features(ctx):
    """Build feature matrix from raw data."""
    from data.scrapers.kaggle_data import load_dataset
    from data.feature_engineering import build_matchup_features, save_features

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
    from models.train import train_model, save_model

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
    from simulation.engine import simulate_tournament, team_advancement_probs, TournamentBracket
    from models.train import load_model
    from models.predict import Predictor

    config = ctx.obj["config"]
    n_sims = config["simulation"]["num_sims"]

    click.echo("Loading model and bracket...")

    # Check if bracket exists
    bracket_path = Path("data/bracket.json")
    if not bracket_path.exists():
        click.echo("No bracket file found at data/bracket.json")
        click.echo("Create one with `marchmadness bracket` or wait for Selection Sunday")
        click.echo("\nRunning demo simulation with seed-based probabilities...")

        from data.seed_history import get_seed_win_prob
        bracket = _create_demo_bracket()
        predict_fn = lambda a, b: get_seed_win_prob(
            bracket.teams[a]["seed"], bracket.teams[b]["seed"]
        )
    else:
        import json
        with open(bracket_path) as f:
            bracket_data = json.load(f)
        bracket = _load_bracket(bracket_data)
        model = load_model()
        predictor = Predictor(model=model, seed_map={
            t: info["seed"] for t, info in bracket.teams.items()
        })
        predict_fn = predictor.predict_matchup

    click.echo(f"Running {n_sims:,} simulations...")
    sim_results = simulate_tournament(
        bracket, predict_fn, n_sims=n_sims,
        rng_seed=config["simulation"]["seed"],
    )

    probs = team_advancement_probs(sim_results, bracket)
    click.echo("\nTeam Advancement Probabilities:")
    click.echo(probs.to_string(index=False, float_format="{:.1%}".format))


@main.command()
@click.option("--round", "round_num", type=int, required=True, help="Round number (1-6)")
@click.option("--method", default="both", type=click.Choice(["differentiation", "portfolio", "both"]))
@click.pass_context
def optimize(ctx, round_num, method):
    """Generate optimal picks for the current round."""
    from entries.generator import generate_picks, load_config
    from entries.manager import EntryManager
    from simulation.engine import TournamentBracket
    from models.predict import Predictor
    from models.train import load_model

    config = ctx.obj["config"]

    click.echo(f"Optimizing picks for Round {round_num} using {method} method...")

    # Load bracket and model
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
        model = load_model()
        predictor = Predictor(model=model, seed_map={
            t: info["seed"] for t, info in bracket.teams.items()
        })

    # Load or create entry manager
    entry_path = Path("entries/state.json")
    if entry_path.exists():
        mgr = EntryManager.load(entry_path)
    else:
        n = config["pool"]["num_entries"]
        mgr = EntryManager(reuse_allowed=config["pool"]["rules"]["reuse_allowed"])
        mgr.create_entries(n)
        click.echo(f"Created {n} new entries")

    results = generate_picks(bracket, predictor, mgr, round_num, config, method)

    # Display results
    click.echo(f"\n{'='*60}")
    click.echo(f"PICK RECOMMENDATIONS - Round {round_num}")
    click.echo(f"{'='*60}")

    for entry_id, team_id in results.get("recommendations", {}).items():
        info = bracket.teams.get(team_id, {})
        wp = results["win_probs"].get(team_id, 0)
        own = results["ownership"].get(team_id, 0)
        click.echo(
            f"  Entry {entry_id}: ({info.get('seed','?')}) {info.get('name', team_id)} "
            f"| Win={wp:.1%} | Own={own:.1%}"
        )

    if "differentiation" in results:
        click.echo(results["differentiation"]["report"])

    if "portfolio" in results:
        ev = results["portfolio"]["evaluation"]
        click.echo(f"\nPortfolio Analysis:")
        click.echo(f"  Total EV: ${ev['total_ev']:.2f}")
        click.echo(f"  Joint survival: {ev['joint_survival']:.1%}")


@main.command()
@click.option("--round", "round_num", type=int, required=True)
@click.argument("winners", nargs=-1, type=int)
@click.pass_context
def results(ctx, round_num, winners):
    """Update entries with actual results. Pass winning team IDs as arguments."""
    from entries.manager import EntryManager

    mgr = EntryManager.load("entries/state.json")
    stats = mgr.update_results(round_num, set(winners))
    mgr.save()

    click.echo(f"\nRound {round_num} Results:")
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
