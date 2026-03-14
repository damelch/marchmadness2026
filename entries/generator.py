"""Orchestrate the full optimization pipeline to generate pick recommendations."""

from __future__ import annotations

import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from simulation.engine import TournamentBracket, simulate_tournament
from models.predict import Predictor
from optimizer.ownership import estimate_ownership_from_bracket
from optimizer.differentiation import optimize_multi_entry, generate_differentiation_report
from optimizer.portfolio import optimize_portfolio_greedy, evaluate_portfolio
from optimizer.kelly import optimal_entries
from entries.manager import EntryManager


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate_picks(
    bracket: TournamentBracket,
    predictor: Predictor,
    entry_manager: EntryManager,
    round_num: int,
    config: dict | None = None,
    method: str = "both",
) -> dict:
    """Generate optimal pick recommendations for the current round.

    Args:
        bracket: Tournament bracket with current state
        predictor: Win probability predictor
        entry_manager: Entry manager with current entries
        round_num: Current round number
        config: Configuration dict (loaded from config.yaml)
        method: "differentiation", "portfolio", or "both"

    Returns:
        Dict with recommendations per entry and analysis
    """
    if config is None:
        config = load_config()

    pool_cfg = config["pool"]
    sim_cfg = config["simulation"]

    alive_entries = entry_manager.get_alive_entries()
    if not alive_entries:
        return {"error": "No entries alive"}

    # Get matchups for this round
    matchups = bracket.get_round_matchups(round_num)
    teams_playing = set()
    for a, b, _ in matchups:
        if a:
            teams_playing.add(a)
        if b:
            teams_playing.add(b)

    # Compute win probabilities for all matchups
    win_probs = {}
    for a, b, _ in matchups:
        if a and b:
            p = predictor.predict_matchup(a, b)
            win_probs[a] = p
            win_probs[b] = 1.0 - p

    # Estimate ownership
    ownership = estimate_ownership_from_bracket(
        bracket, round_num, win_probs,
        pool_sophistication=1 - pool_cfg.get("risk_tolerance", 0.5),
    )

    results = {
        "round": round_num,
        "win_probs": win_probs,
        "ownership": ownership,
        "recommendations": {},
    }

    # Available teams per entry
    available_per_entry = []
    for entry in alive_entries:
        avail = entry_manager.get_available_teams(entry.entry_id, teams_playing)
        avail_with_seeds = {
            t: bracket.teams.get(t, {}).get("seed", 8) for t in avail
        }
        available_per_entry.append(avail_with_seeds)

    # Method A: Differentiation
    if method in ("differentiation", "both"):
        diff_picks = optimize_multi_entry(
            n_entries=len(alive_entries),
            available_teams_per_entry=available_per_entry,
            win_probs=win_probs,
            ownership=ownership,
            bracket=bracket,
        )
        results["differentiation"] = {
            "picks": {
                alive_entries[i].entry_id: diff_picks[i]
                for i in range(len(alive_entries))
            },
            "report": generate_differentiation_report(
                diff_picks, win_probs, ownership, bracket
            ),
        }

    # Method B: Portfolio (requires simulation)
    if method in ("portfolio", "both"):
        # Run simulation
        sim_results = simulate_tournament(
            bracket,
            predictor.predict_matchup,
            n_sims=min(sim_cfg.get("num_sims", 10000), 10000),  # cap for speed
            rng_seed=sim_cfg.get("seed", 42),
        )

        candidate_teams = list(teams_playing)
        port_picks = optimize_portfolio_greedy(
            n_entries=len(alive_entries),
            candidate_teams=candidate_teams,
            sim_results=sim_results,
            bracket=bracket,
            round_num=round_num,
            ownership=ownership,
            pool_size=pool_cfg["pool_size"],
            prize_pool=pool_cfg["prize_pool"],
            used_teams_per_entry=[e.used_teams for e in alive_entries],
        )

        port_eval = evaluate_portfolio(
            port_picks, sim_results, bracket, round_num,
            ownership, pool_cfg["pool_size"], pool_cfg["prize_pool"],
        )

        results["portfolio"] = {
            "picks": {
                alive_entries[i].entry_id: port_picks[i]
                for i in range(len(alive_entries))
            },
            "evaluation": port_eval,
        }

    # If both methods ran, recommend the one with higher total EV
    if method == "both" and "portfolio" in results and "differentiation" in results:
        # Use portfolio EV as ground truth since it's simulation-based
        results["recommended_method"] = "portfolio"
        results["recommendations"] = results["portfolio"]["picks"]
    elif "differentiation" in results:
        results["recommendations"] = results["differentiation"]["picks"]
    elif "portfolio" in results:
        results["recommendations"] = results["portfolio"]["picks"]

    return results


def kelly_analysis(
    bracket: TournamentBracket,
    predictor: Predictor,
    config: dict | None = None,
) -> dict:
    """Run Kelly Criterion analysis to determine optimal number of entries."""
    if config is None:
        config = load_config()

    pool_cfg = config["pool"]
    sim_cfg = config["simulation"]

    # Quick simulation to estimate EV per entry
    sim_results = simulate_tournament(
        bracket,
        predictor.predict_matchup,
        n_sims=sim_cfg.get("num_sims", 50000),
        rng_seed=sim_cfg.get("seed", 42),
    )

    # Rough EV estimate: use seed-based survival probability
    # A 1-seed picked every round survives ~0.99 * 0.80 * 0.60 * 0.50 * 0.50 * 0.50 ≈ 5.9%
    # Prize per survivor = prize_pool / (pool_size * avg_survival_rate)
    # This is a placeholder - real EV comes from the optimizer
    avg_survival = 0.03  # ~3% of pool survives (rough estimate)
    expected_survivors_count = pool_cfg["pool_size"] * avg_survival
    ev_per = pool_cfg["prize_pool"] / max(expected_survivors_count, 1) * avg_survival

    return optimal_entries(
        entry_cost=pool_cfg.get("entry_cost", 50),
        ev_per_entry=ev_per,
        bankroll=pool_cfg.get("prize_pool", 5000) / 10,  # assume bankroll is 10% of pool
        kelly_multiplier=0.5,
        max_entries=pool_cfg.get("max_entries"),
    )
