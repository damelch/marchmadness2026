"""Track entries, picks, and eliminations across the tournament."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Entry:
    entry_id: int
    picks: dict[int, int] = field(default_factory=dict)  # round_num -> team_id
    used_teams: set[int] = field(default_factory=set)
    alive: bool = True
    eliminated_round: int | None = None

    def add_pick(self, round_num: int, team_id: int, reuse_allowed: bool = False) -> None:
        if not reuse_allowed and team_id in self.used_teams:
            raise ValueError(
                f"Entry {self.entry_id}: Team {team_id} already used in a prior round"
            )
        self.picks[round_num] = team_id
        self.used_teams.add(team_id)

    def check_result(self, round_num: int, winners: set[int]) -> bool:
        """Check if this entry survived a round. Returns True if survived."""
        if not self.alive:
            return False
        pick = self.picks.get(round_num)
        if pick is None:
            return True  # No pick yet for this round
        if pick not in winners:
            self.alive = False
            self.eliminated_round = round_num
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "picks": {str(k): v for k, v in self.picks.items()},
            "used_teams": list(self.used_teams),
            "alive": self.alive,
            "eliminated_round": self.eliminated_round,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Entry:
        return cls(
            entry_id=d["entry_id"],
            picks={int(k): v for k, v in d["picks"].items()},
            used_teams=set(d["used_teams"]),
            alive=d["alive"],
            eliminated_round=d.get("eliminated_round"),
        )


class EntryManager:
    """Manage multiple entries across a survivor pool."""

    def __init__(self, reuse_allowed: bool = False):
        self.entries: list[Entry] = []
        self.reuse_allowed = reuse_allowed

    def create_entries(self, n: int) -> list[Entry]:
        """Create N new entries."""
        start_id = len(self.entries)
        new_entries = [Entry(entry_id=start_id + i) for i in range(n)]
        self.entries.extend(new_entries)
        return new_entries

    def add_pick(self, entry_id: int, round_num: int, team_id: int) -> None:
        """Record a pick for an entry."""
        entry = self._get_entry(entry_id)
        if not entry.alive:
            raise ValueError(f"Entry {entry_id} is eliminated")
        entry.add_pick(round_num, team_id, self.reuse_allowed)

    def update_results(self, round_num: int, winners: set[int]) -> dict:
        """Update all entries with round results.

        Returns dict with survival stats.
        """
        survived = 0
        eliminated = 0
        already_dead = 0

        for entry in self.entries:
            if not entry.alive:
                already_dead += 1
                continue
            if entry.check_result(round_num, winners):
                survived += 1
            else:
                eliminated += 1

        return {
            "round": round_num,
            "survived": survived,
            "eliminated": eliminated,
            "already_dead": already_dead,
            "total_alive": survived,
        }

    def get_alive_entries(self) -> list[Entry]:
        return [e for e in self.entries if e.alive]

    def get_available_teams(
        self, entry_id: int, teams_playing: set[int]
    ) -> set[int]:
        """Get teams this entry can still pick (playing and not yet used)."""
        entry = self._get_entry(entry_id)
        if self.reuse_allowed:
            return teams_playing
        return teams_playing - entry.used_teams

    def export_pick_sheets(self) -> list[dict]:
        """Export all entries as a list of dicts for display."""
        rows = []
        for entry in self.entries:
            row = {
                "Entry": entry.entry_id,
                "Status": "ALIVE" if entry.alive else f"OUT (R{entry.eliminated_round})",
            }
            for r, team in sorted(entry.picks.items()):
                row[f"Round {r}"] = team
            rows.append(row)
        return rows

    def save(self, path: str | Path = "entries/state.json") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "reuse_allowed": self.reuse_allowed,
            "entries": [e.to_dict() for e in self.entries],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path = "entries/state.json") -> EntryManager:
        with open(path) as f:
            data = json.load(f)
        mgr = cls(reuse_allowed=data["reuse_allowed"])
        mgr.entries = [Entry.from_dict(d) for d in data["entries"]]
        return mgr

    def _get_entry(self, entry_id: int) -> Entry:
        for e in self.entries:
            if e.entry_id == entry_id:
                return e
        raise ValueError(f"Entry {entry_id} not found")
