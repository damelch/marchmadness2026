"""Generate win probability predictions for matchups."""

from pathlib import Path

import pandas as pd

from data.seed_history import get_seed_win_prob
from models.train import WinProbabilityModel, load_model


class Predictor:
    """Predict win probabilities for tournament matchups."""

    def __init__(
        self,
        model: WinProbabilityModel | None = None,
        team_stats: pd.DataFrame | None = None,
        massey: pd.DataFrame | None = None,
        seed_map: dict[int, int] | None = None,
        kenpom_df: pd.DataFrame | None = None,
        kenpom_id_to_name: dict[int, str] | None = None,
    ):
        self.model = model
        self.team_stats = team_stats
        self.massey = massey
        self.seed_map = seed_map or {}
        self.kenpom_df = kenpom_df
        self.kenpom_id_to_name = kenpom_id_to_name or {}

    @classmethod
    def from_saved(
        cls,
        model_path: str | Path = "models/saved/model.pkl",
        team_stats: pd.DataFrame | None = None,
        massey: pd.DataFrame | None = None,
        seed_map: dict[int, int] | None = None,
    ) -> "Predictor":
        model = load_model(model_path)
        return cls(model=model, team_stats=team_stats, massey=massey, seed_map=seed_map)

    @classmethod
    def from_kenpom(
        cls,
        kenpom_path: str | Path = "data/kenpom_2026.csv",
        teams_csv: str | Path = "data/raw/MTeams.csv",
        model_path: str | Path | None = "models/saved/model.pkl",
        seed_map: dict[int, int] | None = None,
        barttorvik_path: str | Path | None = None,
        espn_bpi_path: str | Path | None = None,
    ) -> "Predictor":
        """Create a Predictor using KenPom ratings as team stats.

        If a trained model exists, uses KenPom stats as features for the ML model.
        Otherwise, falls back to KenPom-based direct prediction.

        Optionally merges Barttorvik and ESPN BPI data for additional features.
        """
        from data.kenpom import load_kenpom, load_kenpom_as_team_stats

        kenpom_df = load_kenpom(kenpom_path)
        team_stats = load_kenpom_as_team_stats(kenpom_path, teams_csv)

        # Build ID -> name mapping for KenPom direct prediction fallback
        id_to_name = dict(zip(team_stats["TeamID"], team_stats["TeamName"]))

        # Merge Barttorvik data if available
        team_stats = _merge_barttorvik(team_stats, barttorvik_path, teams_csv)

        # Merge ESPN BPI data if available
        team_stats = _merge_espn_bpi(team_stats, espn_bpi_path, teams_csv)

        model = None
        if model_path and Path(model_path).exists():
            model = load_model(model_path)

        return cls(
            model=model,
            team_stats=team_stats,
            seed_map=seed_map or {},
            kenpom_df=kenpom_df,
            kenpom_id_to_name=id_to_name,
        )

    def predict_matchup(self, team_a: int, team_b: int, round_num: int = 1) -> float:
        """Predict P(team_a wins) for a single matchup.

        Priority:
        1. ML model with team stats (best if trained model exists)
        2. KenPom efficiency margin direct prediction
        3. Seed-based historical probability (fallback)
        """
        # 1. ML model
        if self.model is not None and self.team_stats is not None:
            features = self._build_features(team_a, team_b, round_num=round_num)
            if features is not None:
                df = pd.DataFrame([features])
                return float(self.model.predict_proba(df)[0])

        # 2. KenPom direct prediction
        if self.kenpom_df is not None:
            name_a = self.kenpom_id_to_name.get(team_a)
            name_b = self.kenpom_id_to_name.get(team_b)
            if name_a and name_b:
                # Only use KenPom if both teams are actually in the dataset
                has_a = (self.kenpom_df["Team"] == name_a).any()
                has_b = (self.kenpom_df["Team"] == name_b).any()
                if has_a and has_b:
                    from data.kenpom import kenpom_predict_matchup
                    return kenpom_predict_matchup(name_a, name_b, self.kenpom_df)

        # 3. Seed-based fallback
        seed_a = self.seed_map.get(team_a, 8)
        seed_b = self.seed_map.get(team_b, 8)
        return get_seed_win_prob(seed_a, seed_b)

    def predict_round(self, matchups: list[tuple[int, int]]) -> dict[tuple[int, int], float]:
        """Predict win probabilities for all matchups in a round.

        Returns dict mapping (team_a, team_b) -> P(team_a wins).
        """
        results = {}
        for team_a, team_b in matchups:
            prob = self.predict_matchup(team_a, team_b)
            results[(team_a, team_b)] = prob
        return results

    def predict_all_matchups(self, team_ids: list[int]) -> dict[tuple[int, int], float]:
        """Predict win probabilities for all possible pairings.

        Used by the simulation engine to cache all probabilities.
        """
        cache = {}
        for i, a in enumerate(team_ids):
            for b in team_ids[i + 1 :]:
                p = self.predict_matchup(a, b)
                cache[(a, b)] = p
                cache[(b, a)] = 1.0 - p
        return cache

    def _build_features(self, team_a: int, team_b: int, round_num: int = 1) -> dict | None:
        """Build feature dict for a matchup."""
        seed_a = self.seed_map.get(team_a)
        seed_b = self.seed_map.get(team_b)
        if seed_a is None or seed_b is None:
            return None

        sa = self.team_stats[self.team_stats["TeamID"] == team_a]
        sb = self.team_stats[self.team_stats["TeamID"] == team_b]
        if sa.empty or sb.empty:
            return None

        sa, sb = sa.iloc[0], sb.iloc[0]

        seed_diff = seed_b - seed_a

        features = {
            "SeedDiff": seed_diff,
            "AdjODiff": sa["AdjO"] - sb["AdjO"],
            "AdjDDiff": sa["AdjD"] - sb["AdjD"],
            "AdjEMDiff": sa["AdjEM"] - sb["AdjEM"],
            "AdjTAvg": (sa["AdjT"] + sb["AdjT"]) / 2,
            "SOSDiff": sa["SOS"] - sb["SOS"],
            "WinPctDiff": sa["WinPct"] - sb["WinPct"],
            "MasseyRankDiff": 0.0,
            "TourneyExpDiff": 0.0,
            # New features
            "LuckDiff": float(sa.get("Luck", 0.0)) - float(sb.get("Luck", 0.0)),
            "CloseGameDiff": 0.0,  # Not available from KenPom; model uses other features
            "SeedRoundInteraction": seed_diff * round_num,
            "AdjEMStdDiff": float(sa.get("AdjEMStd", 0.0)) - float(sb.get("AdjEMStd", 0.0)),
            # Barttorvik features
            "BarthagDiff": float(sa.get("Barthag", 0.0)) - float(sb.get("Barthag", 0.0)),
            "WABDiff": float(sa.get("WAB", 0.0)) - float(sb.get("WAB", 0.0)),
            # ESPN BPI features
            "BPIDiff": float(sa.get("BPI", 0.0)) - float(sb.get("BPI", 0.0)),
            "BPIOffDiff": float(sa.get("BPIOff", 0.0)) - float(sb.get("BPIOff", 0.0)),
            "BPIDefDiff": float(sa.get("BPIDef", 0.0)) - float(sb.get("BPIDef", 0.0)),
        }

        if self.massey is not None:
            rank_a = self.massey[self.massey["TeamID"] == team_a]
            rank_b = self.massey[self.massey["TeamID"] == team_b]
            if not rank_a.empty and not rank_b.empty:
                features["MasseyRankDiff"] = (
                    rank_b.iloc[0]["MasseyRank"] - rank_a.iloc[0]["MasseyRank"]
                )

        return features


def _merge_barttorvik(
    team_stats: pd.DataFrame,
    barttorvik_path: str | Path | None,
    teams_csv: str | Path = "data/raw/MTeams.csv",
) -> pd.DataFrame:
    """Merge Barttorvik stats into team_stats if data file exists."""
    if barttorvik_path is None:
        # Auto-detect common paths
        for candidate in ["data/barttorvik_2026.csv", "data/barttorvik.csv"]:
            if Path(candidate).exists():
                barttorvik_path = candidate
                break

    if barttorvik_path is None or not Path(barttorvik_path).exists():
        return team_stats

    try:
        from data.kenpom import build_team_id_map
        from data.scrapers.barttorvik import barttorvik_to_team_stats, load_barttorvik

        bt_df = load_barttorvik(barttorvik_path)
        team_id_map = build_team_id_map(teams_csv) or None
        bt_stats = barttorvik_to_team_stats(bt_df, team_id_map)

        if not bt_stats.empty:
            merge_cols = ["TeamID", "Barthag", "WAB"]
            available = [c for c in merge_cols if c in bt_stats.columns]
            team_stats = team_stats.merge(
                bt_stats[available], on="TeamID", how="left",
            )
            for col in ["Barthag", "WAB"]:
                if col in team_stats.columns:
                    team_stats[col] = team_stats[col].fillna(0.0)
            print(f"Merged Barttorvik data for {bt_stats['TeamID'].nunique()} teams")
    except Exception as e:
        print(f"Warning: Could not merge Barttorvik data: {e}")

    return team_stats


def _merge_espn_bpi(
    team_stats: pd.DataFrame,
    espn_bpi_path: str | Path | None,
    teams_csv: str | Path = "data/raw/MTeams.csv",
) -> pd.DataFrame:
    """Merge ESPN BPI stats into team_stats if data file exists."""
    if espn_bpi_path is None:
        for candidate in ["data/espn_bpi_2026.csv", "data/espn_bpi.csv"]:
            if Path(candidate).exists():
                espn_bpi_path = candidate
                break

    if espn_bpi_path is None or not Path(espn_bpi_path).exists():
        return team_stats

    try:
        from data.kenpom import build_team_id_map
        from data.scrapers.espn_bpi import bpi_to_team_stats, load_espn_bpi

        bpi_df = load_espn_bpi(espn_bpi_path)
        team_id_map = build_team_id_map(teams_csv) or None
        bpi_stats = bpi_to_team_stats(bpi_df, team_id_map)

        if not bpi_stats.empty:
            merge_cols = ["TeamID", "BPI", "BPIOff", "BPIDef"]
            available = [c for c in merge_cols if c in bpi_stats.columns]
            team_stats = team_stats.merge(
                bpi_stats[available], on="TeamID", how="left",
            )
            for col in ["BPI", "BPIOff", "BPIDef"]:
                if col in team_stats.columns:
                    team_stats[col] = team_stats[col].fillna(0.0)
            print(f"Merged ESPN BPI data for {bpi_stats['TeamID'].nunique()} teams")
    except Exception as e:
        print(f"Warning: Could not merge ESPN BPI data: {e}")

    return team_stats
