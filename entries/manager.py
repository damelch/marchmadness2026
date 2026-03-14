"""Track entries, picks, and eliminations across the tournament."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Entry:
    entry_id: int
    picks: dict[int, list[int]] = field(default_factory=dict)  # day_num -> [team_ids]
    used_teams: set[int] = field(default_factory=set)
    alive: bool = True
    eliminated_day: int | None = None

    def add_picks(
        self, day_num: int, team_ids: list[int], reuse_allowed: bool = False,
    ) -> None:
        """Record picks for a contest day (1 or 2 teams)."""
        if not reuse_allowed:
            for t in team_ids:
                if t in self.used_teams:
                    raise ValueError(
                        f"Entry {self.entry_id}: Team {t} already used on a prior day"
                    )
        self.picks[day_num] = list(team_ids)
        self.used_teams.update(team_ids)

    def check_day_result(self, day_num: int, winners: set[int]) -> bool:
        """Check if this entry survived a day. ALL picks must win."""
        if not self.alive:
            return False
        day_picks = self.picks.get(day_num)
        if day_picks is None:
            return True  # No picks yet for this day
        if all(t in winners for t in day_picks):
            return True
        self.alive = False
        self.eliminated_day = day_num
        return False

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "picks": {str(k): v for k, v in self.picks.items()},
            "used_teams": list(self.used_teams),
            "alive": self.alive,
            "eliminated_day": self.eliminated_day,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Entry:
        raw_picks = d.get("picks", {})
        picks = {}
        for k, v in raw_picks.items():
            day = int(k)
            # Migration: old format stored {round: team_id} as int values
            if isinstance(v, int):
                picks[day] = [v]
            else:
                picks[day] = list(v)

        used_teams = set(d.get("used_teams", []))

        return cls(
            entry_id=d["entry_id"],
            picks=picks,
            used_teams=used_teams,
            alive=d.get("alive", True),
            eliminated_day=d.get("eliminated_day", d.get("eliminated_round")),
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

    def add_picks(
        self, entry_id: int, day_num: int, team_ids: list[int],
    ) -> None:
        """Record picks for an entry on a contest day."""
        entry = self._get_entry(entry_id)
        if not entry.alive:
            raise ValueError(f"Entry {entry_id} is eliminated")
        entry.add_picks(day_num, team_ids, self.reuse_allowed)

    def update_results(self, day_num: int, winners: set[int]) -> dict:
        """Update all entries with day results.

        Returns dict with survival stats.
        """
        survived = 0
        eliminated = 0
        already_dead = 0

        for entry in self.entries:
            if not entry.alive:
                already_dead += 1
                continue
            if entry.check_day_result(day_num, winners):
                survived += 1
            else:
                eliminated += 1

        return {
            "day": day_num,
            "survived": survived,
            "eliminated": eliminated,
            "already_dead": already_dead,
            "total_alive": survived,
        }

    def get_alive_entries(self) -> list[Entry]:
        return [e for e in self.entries if e.alive]

    def get_available_teams(
        self, entry_id: int, teams_playing: set[int],
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
                "Status": "ALIVE" if entry.alive else f"OUT (Day {entry.eliminated_day})",
            }
            for day, teams in sorted(entry.picks.items()):
                if len(teams) == 1:
                    row[f"Day {day}"] = teams[0]
                else:
                    row[f"Day {day}"] = ", ".join(str(t) for t in teams)
            rows.append(row)
        return rows

    def save(self, path: str | Path = "entries/state.json") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 2,
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
