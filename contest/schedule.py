"""Contest schedule mapping tournament rounds to pick days.

The NCAA tournament is round-based (R64, R32, S16, E8, F4, Championship),
but survivor pool contests operate on a day-based schedule where some days
require multiple picks. This module bridges the two.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContestDay:
    """A single decision point in the contest."""

    day_num: int
    label: str
    date: str
    round_num: int
    num_picks: int
    regions: list[str]

    @property
    def is_double_pick(self) -> bool:
        return self.num_picks >= 2


class ContestSchedule:
    """Maps contest days to tournament bracket games."""

    def __init__(self, days: list[ContestDay]):
        self.days = sorted(days, key=lambda d: d.day_num)
        self._by_num = {d.day_num: d for d in self.days}

    @classmethod
    def from_config(cls, config: dict) -> ContestSchedule:
        """Build schedule from config.yaml contest section."""
        contest_cfg = config.get("contest", {})
        days_cfg = contest_cfg.get("days", [])

        if not days_cfg:
            return cls.default()

        days = []
        for d in days_cfg:
            days.append(ContestDay(
                day_num=d["day"],
                label=d.get("label", f"Day {d['day']}"),
                date=d.get("date", ""),
                round_num=d["round"],
                num_picks=d.get("picks", 1),
                regions=d.get("regions", []),
            ))

        return cls(days)

    @classmethod
    def default(cls) -> ContestSchedule:
        """Default 9-day schedule matching standard NCAA survivor pool."""
        return cls([
            ContestDay(1, "R64 Thursday", "2026-03-19", 1, 2, ["W", "X"]),
            ContestDay(2, "R64 Friday", "2026-03-20", 1, 2, ["Y", "Z"]),
            ContestDay(3, "R32 Saturday", "2026-03-21", 2, 1, ["W", "X"]),
            ContestDay(4, "R32 Sunday", "2026-03-22", 2, 1, ["Y", "Z"]),
            ContestDay(5, "S16 Thursday", "2026-03-26", 3, 1, ["W", "X"]),
            ContestDay(6, "S16 Friday", "2026-03-27", 3, 1, ["Y", "Z"]),
            ContestDay(7, "Elite 8", "2026-03-30", 4, 2, ["W", "X", "Y", "Z"]),
            ContestDay(8, "Final Four", "2026-04-04", 5, 1, ["FF"]),
            ContestDay(9, "Championship", "2026-04-06", 6, 1, ["FF"]),
        ])

    def get_day(self, day_num: int) -> ContestDay:
        if day_num not in self._by_num:
            raise ValueError(f"Day {day_num} not in schedule (valid: {list(self._by_num.keys())})")
        return self._by_num[day_num]

    def get_games_for_day(self, day_num: int, bracket) -> list[tuple]:
        """Get matchups available on this day, filtered by region.

        Returns list of (team_a, team_b, slot_index) tuples.
        """
        day = self.get_day(day_num)
        return bracket.get_day_matchups(day.round_num, day.regions)

    def get_remaining_days(self, current_day: int) -> list[ContestDay]:
        """Days after current_day (exclusive)."""
        return [d for d in self.days if d.day_num > current_day]

    def total_picks_remaining(self, current_day: int) -> int:
        """Total picks needed from current_day onward (inclusive)."""
        return sum(d.num_picks for d in self.days if d.day_num >= current_day)

    def total_days(self) -> int:
        return len(self.days)

    def __repr__(self) -> str:
        return f"ContestSchedule({len(self.days)} days, {sum(d.num_picks for d in self.days)} total picks)"
