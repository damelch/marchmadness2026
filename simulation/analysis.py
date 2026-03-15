"""Post-simulation analysis for survivor pool strategy."""

import numpy as np
import pandas as pd

from simulation.engine import TournamentBracket


def survivor_pool_sim(
    sim_results: np.ndarray,
    bracket: TournamentBracket,
    entries: list[list[int]],
    ownership_by_round: list[dict[int, float]],
    pool_size: int,
    prize_pool: float,
    rng_seed: int = 42,
) -> dict:
    """Simulate survivor pool outcomes across all tournament simulations.

    Args:
        sim_results: (n_sims, 63) array of game winners
        bracket: Tournament bracket structure
        entries: List of entries, each is [team_id_round1, team_id_round2, ...]
        ownership_by_round: For each round, dict of team_id -> fraction of pool picking them
        pool_size: Total number of entries in the pool (including ours)
        prize_pool: Total prize money
        rng_seed: Random seed for opponent simulation

    Returns:
        Dict with entry_survival_probs, expected_payouts, joint_survival_prob
    """
    rng = np.random.default_rng(rng_seed)
    n_sims = sim_results.shape[0]
    n_entries = len(entries)
    n_rounds = len(entries[0]) if entries else 0
    n_opponents = pool_size - n_entries

    # Precompute: for each round, which teams won in each sim
    round_winners_by_sim = []
    for round_num in range(1, n_rounds + 1):
        round_slots = [
            i for i, s in enumerate(bracket.slots) if s.round_num == round_num
        ]
        # Set of winners per sim
        winners_per_sim = []
        for sim in range(n_sims):
            winners = set(sim_results[sim, round_slots])
            winners.discard(0)
            winners_per_sim.append(winners)
        round_winners_by_sim.append(winners_per_sim)

    # Track entry survival
    entry_alive = np.ones((n_sims, n_entries), dtype=bool)
    entry_survival_by_round = np.ones((n_rounds, n_entries), dtype=float)

    for r in range(n_rounds):
        for e_idx, entry in enumerate(entries):
            if r >= len(entry):
                continue
            team_pick = entry[r]
            for sim in range(n_sims):
                if not entry_alive[sim, e_idx]:
                    continue
                if team_pick not in round_winners_by_sim[r][sim]:
                    entry_alive[sim, e_idx] = False

        # Record survival rate after this round
        for e_idx in range(n_entries):
            entry_survival_by_round[r, e_idx] = entry_alive[:, e_idx].mean()

    # Simulate opponents
    # For each sim, for each opponent, simulate their picks and survival
    opponent_alive = np.ones((n_sims, n_opponents), dtype=bool)

    for r in range(n_rounds):
        ownership = ownership_by_round[r]
        teams = list(ownership.keys())
        probs = np.array([ownership[t] for t in teams])
        probs = probs / probs.sum()  # normalize

        # For each opponent, sample which team they picked
        opponent_picks = rng.choice(teams, size=(n_sims, n_opponents), p=probs)

        for sim in range(n_sims):
            winners = round_winners_by_sim[r][sim]
            for opp in range(n_opponents):
                if not opponent_alive[sim, opp]:
                    continue
                if opponent_picks[sim, opp] not in winners:
                    opponent_alive[sim, opp] = False

    # Calculate payouts
    expected_payouts = np.zeros(n_entries)

    for sim in range(n_sims):
        total_survivors = entry_alive[sim].sum() + opponent_alive[sim].sum()
        if total_survivors == 0:
            continue
        payout_per = prize_pool / total_survivors
        for e_idx in range(n_entries):
            if entry_alive[sim, e_idx]:
                expected_payouts[e_idx] += payout_per

    expected_payouts /= n_sims

    # Joint survival: P(at least one of our entries survives)
    any_alive = entry_alive.any(axis=1)
    joint_survival = any_alive.mean()

    return {
        "entry_survival_by_round": entry_survival_by_round.tolist(),
        "expected_payouts": expected_payouts.tolist(),
        "joint_survival_prob": float(joint_survival),
        "avg_opponents_surviving": float(opponent_alive.sum(axis=1).mean()),
    }


def analyze_pick_ev(
    sim_results: np.ndarray,
    bracket: TournamentBracket,
    round_num: int,
    ownership: dict[int, float],
    pool_size: int,
    prize_pool: float,
) -> pd.DataFrame:
    """Analyze EV of each possible pick for a single round.

    Returns DataFrame with columns: TeamID, WinProb, Ownership, Leverage, EV
    """
    n_sims = sim_results.shape[0]

    round_slots = [
        i for i, s in enumerate(bracket.slots) if s.round_num == round_num
    ]

    # Get teams playing in this round
    teams_playing = set()
    for slot_idx in round_slots:
        slot = bracket.slots[slot_idx]
        if slot.team_a:
            teams_playing.add(slot.team_a)
        if slot.team_b:
            teams_playing.add(slot.team_b)

    rows = []
    for team_id in teams_playing:
        # Win probability from simulation
        win_count = 0
        for sim in range(n_sims):
            for slot_idx in round_slots:
                if sim_results[sim, slot_idx] == team_id:
                    win_count += 1
                    break

        win_prob = win_count / n_sims
        own_pct = ownership.get(team_id, 1 / len(teams_playing))

        # Leverage: picking a winner others didn't
        # When this team wins, (1 - own_pct) of opponents must have picked someone else
        # Some of those opponents might also survive (their pick also won)
        other_teams_win_probs = {
            t: sum(
                1 for sim in range(n_sims)
                for si in round_slots
                if sim_results[sim, si] == t
            ) / n_sims
            for t in teams_playing if t != team_id
        }

        # Expected fraction of opponents eliminated when this team wins
        # Opponents who picked this team survive; others survive if their pick won
        opp_survival_if_team_wins = own_pct  # those who picked same team
        for other_team, other_own in ownership.items():
            if other_team == team_id:
                continue
            other_wp = other_teams_win_probs.get(other_team, 0.5)
            opp_survival_if_team_wins += other_own * other_wp

        leverage = win_prob * (1 - opp_survival_if_team_wins)

        # Simplified EV
        ev = win_prob * prize_pool / max(1, pool_size * opp_survival_if_team_wins)

        info = bracket.teams.get(team_id, {})
        rows.append({
            "TeamID": team_id,
            "TeamName": info.get("name", ""),
            "Seed": info.get("seed", 0),
            "WinProb": win_prob,
            "Ownership": own_pct,
            "Leverage": leverage,
            "EV": ev,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("EV", ascending=False)
    return df
