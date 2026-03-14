"""Monte Carlo tournament simulation engine."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BracketSlot:
    """A single game slot in the tournament bracket."""
    region: str
    round_num: int       # 1=R64, 2=R32, 3=S16, 4=E8, 5=F4, 6=Championship
    game_index: int
    team_a: int | None = None
    team_b: int | None = None
    winner: int | None = None
    next_slot: int | None = None  # index of the slot the winner advances to


class TournamentBracket:
    """Full 64-team tournament bracket structure."""

    REGIONS = ["W", "X", "Y", "Z"]

    def __init__(self):
        self.slots: list[BracketSlot] = []
        self.teams: dict[int, dict] = {}  # team_id -> {name, seed, region}
        self._build_bracket_structure()

    def _build_bracket_structure(self) -> None:
        """Build the 63-game bracket tree."""
        slot_idx = 0

        # Rounds 1-4: regional games (per region)
        for region_idx, region in enumerate(self.REGIONS):
            region_offset = region_idx * 15  # 15 games per region

            # Round 1: 8 games per region
            for g in range(8):
                next_slot = region_offset + 8 + g // 2
                self.slots.append(BracketSlot(
                    region=region, round_num=1, game_index=g,
                    next_slot=next_slot,
                ))

            # Round 2: 4 games per region
            for g in range(4):
                next_slot = region_offset + 12 + g // 2
                self.slots.append(BracketSlot(
                    region=region, round_num=2, game_index=g,
                    next_slot=next_slot,
                ))

            # Round 3 (Sweet 16): 2 games per region
            for g in range(2):
                next_slot = region_offset + 14
                self.slots.append(BracketSlot(
                    region=region, round_num=3, game_index=g,
                    next_slot=next_slot,
                ))

            # Round 4 (Elite 8): 1 game per region
            # Next slot is Final Four
            self.slots.append(BracketSlot(
                region=region, round_num=4, game_index=0,
                next_slot=60 + region_idx // 2,
            ))

        # Final Four: 2 games (slots 60-61)
        self.slots.append(BracketSlot(
            region="FF", round_num=5, game_index=0, next_slot=62,
        ))
        self.slots.append(BracketSlot(
            region="FF", round_num=5, game_index=1, next_slot=62,
        ))

        # Championship: 1 game (slot 62)
        self.slots.append(BracketSlot(
            region="FF", round_num=6, game_index=0, next_slot=None,
        ))

    def set_seed(self, team_id: int, seed: int, region: str, name: str = "") -> None:
        """Place a team in the bracket by seed and region."""
        self.teams[team_id] = {"name": name, "seed": seed, "region": region}

        # Standard bracket seeding order for Round 1 matchups:
        # Game 0: 1v16, Game 1: 8v9, Game 2: 5v12, Game 3: 4v13
        # Game 4: 6v11, Game 5: 3v14, Game 6: 7v10, Game 7: 2v15
        SEED_TO_GAME = {1: (0, "a"), 16: (0, "b"), 8: (1, "a"), 9: (1, "b"),
                        5: (2, "a"), 12: (2, "b"), 4: (3, "a"), 13: (3, "b"),
                        6: (4, "a"), 11: (4, "b"), 3: (5, "a"), 14: (5, "b"),
                        7: (6, "a"), 10: (6, "b"), 2: (7, "a"), 15: (7, "b")}

        if seed not in SEED_TO_GAME:
            return

        game_idx, slot_pos = SEED_TO_GAME[seed]
        region_idx = self.REGIONS.index(region)
        abs_slot = region_idx * 15 + game_idx

        if slot_pos == "a":
            self.slots[abs_slot].team_a = team_id
        else:
            self.slots[abs_slot].team_b = team_id

    def get_round_matchups(self, round_num: int) -> list[tuple[int | None, int | None, int]]:
        """Get matchups for a given round.

        Returns list of (team_a, team_b, slot_index).
        """
        matchups = []
        for i, slot in enumerate(self.slots):
            if slot.round_num == round_num:
                matchups.append((slot.team_a, slot.team_b, i))
        return matchups

    def get_day_matchups(
        self, round_num: int, regions: list[str],
    ) -> list[tuple[int | None, int | None, int]]:
        """Get matchups for a specific contest day, filtered by region.

        Returns list of (team_a, team_b, slot_index).
        """
        matchups = []
        for i, slot in enumerate(self.slots):
            if slot.round_num == round_num and slot.region in regions:
                matchups.append((slot.team_a, slot.team_b, i))
        return matchups

    def resolve_game(self, slot_index: int, winner: int) -> None:
        """Set the winner of a game and advance them to the next slot."""
        slot = self.slots[slot_index]
        slot.winner = winner

        if slot.next_slot is not None:
            next_s = self.slots[slot.next_slot]
            # Determine which position (a or b) the winner fills
            # Lower game_index fills position a, higher fills b
            if next_s.team_a is None:
                next_s.team_a = winner
            else:
                next_s.team_b = winner

    def get_team_game(self, team_id: int, round_num: int) -> int | None:
        """Find the slot index where a team plays in a given round."""
        for i, slot in enumerate(self.slots):
            if slot.round_num == round_num and (slot.team_a == team_id or slot.team_b == team_id):
                return i
        return None

    def get_opponent(self, team_id: int, round_num: int) -> int | None:
        """Find a team's opponent in a given round."""
        for slot in self.slots:
            if slot.round_num != round_num:
                continue
            if slot.team_a == team_id:
                return slot.team_b
            if slot.team_b == team_id:
                return slot.team_a
        return None

    def copy(self) -> TournamentBracket:
        """Deep copy the bracket."""
        import copy
        return copy.deepcopy(self)


def simulate_tournament(
    bracket: TournamentBracket,
    predict_fn: Callable[[int, int], float],
    n_sims: int = 50000,
    rng_seed: int = 42,
) -> np.ndarray:
    """Run Monte Carlo simulation of the entire tournament.

    Args:
        bracket: Tournament bracket with Round 1 matchups set
        predict_fn: Function (team_a, team_b) -> P(team_a wins)
        n_sims: Number of simulations
        rng_seed: Random seed for reproducibility

    Returns:
        Array of shape (n_sims, 63) with winner TeamID for each slot
    """
    rng = np.random.default_rng(rng_seed)

    # Pre-compute probability cache for all possible matchups
    all_teams = list(bracket.teams.keys())
    prob_cache: dict[tuple[int, int], float] = {}
    for i, a in enumerate(all_teams):
        for b in all_teams[i + 1:]:
            p = predict_fn(a, b)
            prob_cache[(a, b)] = p
            prob_cache[(b, a)] = 1.0 - p

    n_slots = len(bracket.slots)
    results = np.zeros((n_sims, n_slots), dtype=np.int32)

    # Pre-generate all random draws
    rand_draws = rng.random((n_sims, n_slots))

    for sim in range(n_sims):
        # Copy initial bracket state (only team assignments)
        teams_a = [s.team_a for s in bracket.slots]
        teams_b = [s.team_b for s in bracket.slots]
        next_slots = [s.next_slot for s in bracket.slots]

        for slot_idx in range(n_slots):
            a = teams_a[slot_idx]
            b = teams_b[slot_idx]

            if a is None or b is None:
                continue

            p_a = prob_cache.get((a, b), 0.5)
            winner = a if rand_draws[sim, slot_idx] < p_a else b
            results[sim, slot_idx] = winner

            # Propagate winner to next slot
            if next_slots[slot_idx] is not None:
                ns = next_slots[slot_idx]
                if teams_a[ns] is None or teams_a[ns] == 0:
                    teams_a[ns] = winner
                else:
                    teams_b[ns] = winner

    return results


def team_advancement_probs(
    sim_results: np.ndarray,
    bracket: TournamentBracket,
) -> pd.DataFrame:
    """Compute probability each team reaches each round.

    Returns DataFrame: TeamID, TeamName, Seed, Region, R64, R32, S16, E8, F4, Champ, Winner
    """
    import pandas as pd

    all_teams = list(bracket.teams.keys())
    round_names = ["R64", "R32", "S16", "E8", "F4", "Champ", "Winner"]
    n_sims = sim_results.shape[0]

    rows = []
    for team_id in all_teams:
        info = bracket.teams[team_id]
        row = {
            "TeamID": team_id,
            "TeamName": info.get("name", ""),
            "Seed": info.get("seed", 0),
            "Region": info.get("region", ""),
        }

        # R64: team always plays (probability 1.0 for tournament teams)
        row["R64"] = 1.0

        # For each round, check if team won a game in the previous round
        for round_num in range(1, 7):
            round_label = round_names[round_num]
            # Count sims where team won a game in this round
            round_slots = [
                i for i, s in enumerate(bracket.slots) if s.round_num == round_num
            ]
            wins_in_round = sum(
                np.sum(sim_results[:, slot_idx] == team_id) for slot_idx in round_slots
            )
            row[round_label] = wins_in_round / n_sims

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("Seed")
    return df
