"""Orchestrate the full optimization pipeline to generate pick recommendations.

Pipeline: win probs → MC for advancement → Nash ownership → DP future values → analytical optimizer
"""

from __future__ import annotations

import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from simulation.engine import TournamentBracket, simulate_tournament
from models.predict import Predictor
from optimizer.ownership import estimate_ownership_from_bracket
from optimizer.analytical import exact_round_ev, optimal_multi_entry
from optimizer.nash import nash_equilibrium, best_response, verify_equilibrium
from optimizer.dp import compute_round_win_probs, compute_advancement_probs, dp_optimal_picks
from optimizer.differentiation import optimize_multi_entry, generate_differentiation_report
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
    method: str = "hybrid",
) -> dict:
    """Generate optimal pick recommendations for the current round.

    Methods:
        "hybrid": Full pipeline — Nash ownership + DP future values + analytical EV (recommended)
        "analytical": Analytical EV with heuristic ownership (fast, no simulation needed)
        "differentiation": Legacy greedy leverage-based picks (fastest)

    Args:
        bracket: Tournament bracket with current state
        predictor: Win probability predictor
        entry_manager: Entry manager with current entries
        round_num: Current round number
        config: Configuration dict (loaded from config.yaml)
        method: "hybrid", "analytical", or "differentiation"
    """
    if config is None:
        config = load_config()

    pool_cfg = config["pool"]
    sim_cfg = config["simulation"]
    pool_size = pool_cfg["pool_size"]
    prize_pool = pool_cfg["prize_pool"]

    alive_entries = entry_manager.get_alive_entries()
    if not alive_entries:
        return {"error": "No entries alive"}

    # Step 1: Get matchups and compute win probabilities (all methods need this)
    matchups = bracket.get_round_matchups(round_num)
    teams_playing = set()
    matchup_pairs = []
    for a, b, _ in matchups:
        if a:
            teams_playing.add(a)
        if b:
            teams_playing.add(b)
        if a and b:
            matchup_pairs.append((a, b))

    win_probs = {}
    for a, b, _ in matchups:
        if a and b:
            p = predictor.predict_matchup(a, b)
            win_probs[a] = p
            win_probs[b] = 1.0 - p

    results = {
        "round": round_num,
        "method": method,
        "win_probs": win_probs,
        "recommendations": {},
    }

    # Step 2: Estimate ownership (method-dependent)
    if method == "hybrid":
        # Nash equilibrium blended with heuristic
        ownership = estimate_ownership_from_bracket(
            bracket, round_num, win_probs,
            pool_sophistication=pool_cfg.get("risk_tolerance", 0.5),
            method="blend",
            pool_size=pool_size,
            prize_pool=prize_pool,
        )

        # Verify Nash quality
        nash_own = nash_equilibrium(win_probs, pool_size, prize_pool)
        nash_check = verify_equilibrium(nash_own, win_probs, pool_size, prize_pool)
        results["nash_equilibrium"] = {
            "ownership": nash_own,
            "is_valid": nash_check["is_equilibrium"],
            "max_ev_diff": nash_check["max_ev_diff"],
        }
    elif method == "analytical":
        ownership = estimate_ownership_from_bracket(
            bracket, round_num, win_probs,
            pool_sophistication=1 - pool_cfg.get("risk_tolerance", 0.5),
            method="heuristic",
        )
    else:  # differentiation
        ownership = estimate_ownership_from_bracket(
            bracket, round_num, win_probs,
            pool_sophistication=1 - pool_cfg.get("risk_tolerance", 0.5),
            method="heuristic",
        )

    results["ownership"] = ownership

    # Available teams per entry
    available_per_entry_sets = []
    available_per_entry_dicts = []
    for entry in alive_entries:
        avail = entry_manager.get_available_teams(entry.entry_id, teams_playing)
        available_per_entry_sets.append(avail)
        available_per_entry_dicts.append({
            t: bracket.teams.get(t, {}).get("seed", 8) for t in avail
        })

    # Step 3: Optimize picks (method-dependent)
    if method == "hybrid":
        # Run MC simulation for advancement probs (needed for DP)
        sim_results = simulate_tournament(
            bracket, predictor.predict_matchup,
            n_sims=sim_cfg.get("num_sims", 50000),
            rng_seed=sim_cfg.get("seed", 42),
        )

        # Compute round win probs and advancement probs
        round_win_probs = compute_round_win_probs(bracket, predictor.predict_matchup, sim_results)
        adv_probs = compute_advancement_probs(sim_results, bracket)

        # DP optimal picks with future value consideration
        dp_result = dp_optimal_picks(
            n_entries=len(alive_entries),
            bracket=bracket,
            round_win_probs=round_win_probs,
            adv_probs=adv_probs,
            ownership=ownership,
            pool_size=pool_size,
            prize_pool=prize_pool,
            current_round=round_num,
            used_teams_per_entry=[e.used_teams for e in alive_entries],
        )

        picks = dp_result["picks"]
        results["dp_analysis"] = {
            "future_values": dp_result["future_values"],
            "picks_without_fv": dp_result["picks_without_fv"],
            "reasoning": dp_result["reasoning"],
        }

        # Also compute best response against estimated field
        br = best_response(win_probs, ownership, pool_size, prize_pool, len(alive_entries))
        results["best_response"] = br[:10]  # Top 10 teams by EV

        # Evaluate the picks analytically
        ev_result = exact_round_ev(picks, win_probs, ownership, pool_size, prize_pool, matchup_pairs)
        results["evaluation"] = ev_result

    elif method == "analytical":
        # Analytical optimizer (no simulation needed)
        all_available = {}
        for d in available_per_entry_dicts:
            all_available.update(d)

        picks = optimal_multi_entry(
            n_entries=len(alive_entries),
            available_teams=all_available,
            win_probs=win_probs,
            ownership=ownership,
            pool_size=pool_size,
            prize_pool=prize_pool,
            used_teams_per_entry=[e.used_teams for e in alive_entries],
        )

        ev_result = exact_round_ev(picks, win_probs, ownership, pool_size, prize_pool, matchup_pairs)
        results["evaluation"] = ev_result

    else:  # differentiation
        picks = optimize_multi_entry(
            n_entries=len(alive_entries),
            available_teams_per_entry=available_per_entry_dicts,
            win_probs=win_probs,
            ownership=ownership,
            bracket=bracket,
        )
        results["differentiation_report"] = generate_differentiation_report(
            picks, win_probs, ownership, bracket,
        )

    # Map picks to entry IDs
    results["recommendations"] = {
        alive_entries[i].entry_id: picks[i]
        for i in range(len(alive_entries))
    }

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
    pool_size = pool_cfg["pool_size"]
    prize_pool = pool_cfg["prize_pool"]

    # Use Nash equilibrium to estimate EV per entry
    # Get round 1 win probs
    matchups = bracket.get_round_matchups(1)
    win_probs = {}
    for a, b, _ in matchups:
        if a and b:
            from data.seed_history import get_seed_win_prob
            sa = bracket.teams.get(a, {}).get("seed", 8)
            sb = bracket.teams.get(b, {}).get("seed", 8)
            p = get_seed_win_prob(sa, sb)
            win_probs[a] = p
            win_probs[b] = 1.0 - p

    # Nash ownership for round 1
    from optimizer.analytical import exact_pick_ev
    nash_own = nash_equilibrium(win_probs, pool_size, prize_pool)

    # Best response EV
    br = best_response(win_probs, nash_own, pool_size, prize_pool)
    ev_per = br[0]["ev"] if br else 0

    return optimal_entries(
        entry_cost=pool_cfg.get("entry_cost", 50),
        ev_per_entry=ev_per,
        bankroll=pool_cfg.get("prize_pool", 5000) / 10,
        kelly_multiplier=0.5,
        max_entries=pool_cfg.get("max_entries"),
    )
