"""Tests for day-based entry manager."""

import pytest

from entries.manager import Entry, EntryManager


class TestEntryDayPicks:
    def test_single_pick(self):
        e = Entry(entry_id=0)
        e.add_picks(3, [101])
        assert e.picks[3] == [101]
        assert 101 in e.used_teams

    def test_double_pick(self):
        e = Entry(entry_id=0)
        e.add_picks(1, [101, 102])
        assert e.picks[1] == [101, 102]
        assert 101 in e.used_teams
        assert 102 in e.used_teams

    def test_no_reuse(self):
        e = Entry(entry_id=0)
        e.add_picks(1, [101, 102])
        with pytest.raises(ValueError, match="already used"):
            e.add_picks(3, [101])

    def test_reuse_allowed(self):
        e = Entry(entry_id=0)
        e.add_picks(1, [101], reuse_allowed=True)
        e.add_picks(3, [101], reuse_allowed=True)
        assert e.picks[1] == [101]
        assert e.picks[3] == [101]


class TestDaySurvival:
    def test_single_pick_survives(self):
        e = Entry(entry_id=0)
        e.add_picks(3, [101])
        assert e.check_day_result(3, {101, 200, 300}) is True
        assert e.alive is True

    def test_single_pick_eliminated(self):
        e = Entry(entry_id=0)
        e.add_picks(3, [101])
        assert e.check_day_result(3, {200, 300}) is False
        assert e.alive is False
        assert e.eliminated_day == 3

    def test_double_pick_both_win(self):
        e = Entry(entry_id=0)
        e.add_picks(1, [101, 102])
        assert e.check_day_result(1, {101, 102, 200}) is True
        assert e.alive is True

    def test_double_pick_one_loses(self):
        """If either pick on a double-pick day loses, entry is eliminated."""
        e = Entry(entry_id=0)
        e.add_picks(1, [101, 102])
        assert e.check_day_result(1, {101, 200}) is False  # 102 lost
        assert e.alive is False
        assert e.eliminated_day == 1

    def test_double_pick_both_lose(self):
        e = Entry(entry_id=0)
        e.add_picks(1, [101, 102])
        assert e.check_day_result(1, {200, 300}) is False
        assert e.alive is False

    def test_no_pick_yet_survives(self):
        e = Entry(entry_id=0)
        assert e.check_day_result(5, {200}) is True

    def test_already_dead(self):
        e = Entry(entry_id=0)
        e.alive = False
        assert e.check_day_result(3, {101}) is False


class TestSerialization:
    def test_round_trip(self):
        e = Entry(entry_id=0)
        e.add_picks(1, [101, 102])
        e.add_picks(3, [200])

        d = e.to_dict()
        e2 = Entry.from_dict(d)

        assert e2.entry_id == 0
        assert e2.picks[1] == [101, 102]
        assert e2.picks[3] == [200]
        assert e2.used_teams == {101, 102, 200}

    def test_migrate_old_format(self):
        """Old format stored {round_num: team_id} as int values."""
        old_data = {
            "entry_id": 0,
            "picks": {"1": 101, "2": 200},
            "used_teams": [101, 200],
            "alive": True,
            "eliminated_round": None,
        }
        e = Entry.from_dict(old_data)
        assert e.picks[1] == [101]
        assert e.picks[2] == [200]
        assert e.alive is True


class TestEntryManager:
    def test_add_picks(self):
        mgr = EntryManager()
        mgr.create_entries(2)
        mgr.add_picks(0, 1, [101, 102])
        mgr.add_picks(1, 1, [103, 104])
        assert mgr.entries[0].picks[1] == [101, 102]

    def test_update_results_day(self):
        mgr = EntryManager()
        mgr.create_entries(3)
        mgr.add_picks(0, 1, [101, 102])  # both win
        mgr.add_picks(1, 1, [101, 200])  # 200 loses
        mgr.add_picks(2, 1, [103, 104])  # both win

        stats = mgr.update_results(1, {101, 102, 103, 104})
        assert stats["survived"] == 2
        assert stats["eliminated"] == 1

    def test_save_load(self, tmp_path):
        mgr = EntryManager()
        mgr.create_entries(2)
        mgr.add_picks(0, 1, [101, 102])
        mgr.add_picks(1, 1, [103])

        path = tmp_path / "state.json"
        mgr.save(path)

        mgr2 = EntryManager.load(path)
        assert len(mgr2.entries) == 2
        assert mgr2.entries[0].picks[1] == [101, 102]
        assert mgr2.entries[1].picks[1] == [103]

    def test_export_pick_sheets(self):
        mgr = EntryManager()
        mgr.create_entries(1)
        mgr.add_picks(0, 1, [101, 102])

        sheets = mgr.export_pick_sheets()
        assert len(sheets) == 1
        assert "Day 1" in sheets[0]
