"""Orchestrate the full optimization pipeline to generate pick recommendations.

Pipeline: win probs → MC for advancement → Nash ownership → DP future values → analytical optimizer

Updated for day-based contest schedule with double-pick support.
"""

from __future__ import annotations

import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from simulation.engine import TournamentBracket, simulate_tournament
from models.predict import Predictor
from optimizer.ownership import estimate_ownership_from_bracket
from optimizer.analytical import exact_round_ev, exact_day_ev, optimal_day_picks
from optimizer.nash import nash_equilibrium, best_response, verify_equilibrium
from optimizer.dp import compute_round_win_probs, compute_advancement_probs, dp_optimal_picks
from optimizer.differentiation import optimize_multi_entry, generate_differentiation_report
from optimizer.kelly import optimal_entries
from entries.manager import EntryManager
from contest.schedule import ContestSchedule


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate_picks(
    bracket: TournamentBracket,
    predictor: Predictor,
    entry_manager: EntryManager,
    day_num: int,
    schedule: ContestSchedule,
    config: dict | None = None,
    method: str = "hybrid",
) -> dict:
    """Generate optimal pick recommendations for a contest day.

    Methods:
        "hybrid": Full pipeline — Nash ownership + DP future values + analytical EV (recommended)
        "analytical": Analytical EV with heuristic ownership (fast, no simulation needed)
        "differentiation": Legacy greedy leverage-based picks (fastest, single-pick only)

    Args:
        bracket: Tournament bracket with current state
        predictor: Win probability predictor
        entry_manager: Entry manager with current entries
        day_num: Contest day number (1-9)
        schedule: Contest schedule mapping days to games
        config: Configuration dict (loaded from config.yaml)
        method: "hybrid", "analytical", or "differentiation"
    """
    if config is None:
        config = load_config()

    pool_cfg = config["pool"]
    sim_cfg = config["simulation"]
    pool_size = pool_cfg["pool_size"]
    prize_pool = pool_cfg["prize_pool"]
    max_entries_per_user = pool_cfg.get("max_entries_per_user", 150)

    # Estimate field sophistication from contest structure:
    # - Larger pools with multi-entry and higher buy-ins attract sharper players
    # - Scale from 0.3 (casual office pool) to 0.8 (large paid multi-entry)
    base_sophistication = pool_cfg.get("risk_tolerance", 0.5)
    if pool_size >= 10000 and max_entries_per_user >= 50:
        pool_sophistication = max(base_sophistication, 0.65)
    elif pool_size >= 1000:
        pool_sophistication = max(base_sophistication, 0.5)
    else:
        pool_sophistication = base_sophistication

    day = schedule.get_day(day_num)
    num_picks = day.num_picks
    round_num = day.round_num

    alive_entries = entry_manager.get_alive_entries()
    if not alive_entries:
        return {"error": "No entries alive"}

    # Step 1: Get matchups for this day (filtered by region)
    matchups = schedule.get_games_for_day(day_num, bracket)
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
        "day": day_num,
        "day_label": day.label,
        "round": round_num,
        "num_picks": num_picks,
        "method": method,
        "win_probs": win_probs,
        "recommendations": {},
    }

    # Step 2: Estimate ownership (method-dependent, filtered to today's regions)
    if method == "hybrid":
        ownership = estimate_ownership_from_bracket(
            bracket, round_num, win_probs,
            pool_sophistication=pool_sophistication,
            method="blend",
            pool_size=pool_size,
            prize_pool=prize_pool,
            regions=day.regions,
        )

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
            pool_sophistication=pool_sophistication,
            method="heuristic",
            regions=day.regions,
        )
    else:  # differentiation
        ownership = estimate_ownership_from_bracket(
            bracket, round_num, win_probs,
            pool_sophistication=pool_sophistication,
            method="heuristic",
            regions=day.regions,
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
            current_day=day_num,
            schedule=schedule,
            used_teams_per_entry=[e.used_teams for e in alive_entries],
            num_picks=num_picks,
            matchup_pairs=matchup_pairs,
        )

        picks = dp_result["picks"]
        results["dp_analysis"] = {
            "future_values": dp_result["future_values"],
            "picks_without_fv": dp_result["picks_without_fv"],
            "reasoning": dp_result["reasoning"],
        }

        # Best response against estimated field
        br = best_response(win_probs, ownership, pool_size, prize_pool, len(alive_entries))
        results["best_response"] = br[:10]

        # Evaluate the picks
        if num_picks == 1:
            flat_picks = [ps[0] for ps in picks]
            ev_result = exact_round_ev(
                flat_picks, win_probs, ownership, pool_size, prize_pool, matchup_pairs,
            )
        else:
            ev_per = []
            for ps in picks:
                ev = exact_day_ev(
                    ps, win_probs, ownership, pool_size, prize_pool,
                    num_picks_per_opponent=num_picks, n_our_entries=len(alive_entries),
                    matchup_pairs=matchup_pairs,
                )
                ev_per.append(ev)
            joint_surv = 1.0
            for ps in picks:
                p = 1.0
                for t in ps:
                    p *= win_probs.get(t, 0.5)
                joint_surv *= (1.0 - p)
            ev_result = {
                "ev_per_entry": ev_per,
                "total_ev": sum(ev_per),
                "joint_survival": 1.0 - joint_surv,
                "per_entry_survival": [
                    math.prod(win_probs.get(t, 0.5) for t in ps) for ps in picks
                ],
            }
        results["evaluation"] = ev_result

    elif method == "analytical":
        all_available = {}
        for d in available_per_entry_dicts:
            all_available.update(d)

        picks = optimal_day_picks(
            n_entries=len(alive_entries),
            available_teams=all_available,
            win_probs=win_probs,
            ownership=ownership,
            pool_size=pool_size,
            prize_pool=prize_pool,
            num_picks=num_picks,
            used_teams_per_entry=[e.used_teams for e in alive_entries],
            matchup_pairs=matchup_pairs,
        )

        if num_picks == 1:
            flat_picks = [ps[0] for ps in picks]
            ev_result = exact_round_ev(
                flat_picks, win_probs, ownership, pool_size, prize_pool, matchup_pairs,
            )
        else:
            ev_per = []
            for ps in picks:
                ev = exact_day_ev(
                    ps, win_probs, ownership, pool_size, prize_pool,
                    num_picks_per_opponent=num_picks, n_our_entries=len(alive_entries),
                    matchup_pairs=matchup_pairs,
                )
                ev_per.append(ev)
            ev_result = {
                "ev_per_entry": ev_per,
                "total_ev": sum(ev_per),
            }
        results["evaluation"] = ev_result

    else:  # differentiation (single-pick only)
        if num_picks > 1:
            results["error"] = "Differentiation method only supports single-pick days"
            # Fall back to analytical for double-pick
            all_available = {}
            for d in available_per_entry_dicts:
                all_available.update(d)
            picks = optimal_day_picks(
                n_entries=len(alive_entries),
                available_teams=all_available,
                win_probs=win_probs,
                ownership=ownership,
                pool_size=pool_size,
                prize_pool=prize_pool,
                num_picks=num_picks,
                used_teams_per_entry=[e.used_teams for e in alive_entries],
                matchup_pairs=matchup_pairs,
            )
        else:
            diff_picks = optimize_multi_entry(
                n_entries=len(alive_entries),
                available_teams_per_entry=available_per_entry_dicts,
                win_probs=win_probs,
                ownership=ownership,
                bracket=bracket,
            )
            picks = [[t] for t in diff_picks]
            results["differentiation_report"] = generate_differentiation_report(
                diff_picks, win_probs, ownership, bracket,
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
    pool_size = pool_cfg["pool_size"]
    prize_pool = pool_cfg["prize_pool"]

    # Use Nash equilibrium to estimate EV per entry
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

    nash_own = nash_equilibrium(win_probs, pool_size, prize_pool)

    br = best_response(win_probs, nash_own, pool_size, prize_pool)
    ev_per = br[0]["ev"] if br else 0

    return optimal_entries(
        entry_cost=pool_cfg.get("entry_cost", 50),
        ev_per_entry=ev_per,
        bankroll=pool_cfg.get("prize_pool", 5000) / 10,
        kelly_multiplier=0.5,
        max_entries=pool_cfg.get("max_entries"),
    )


# Need math.prod for Python 3.11
import math
