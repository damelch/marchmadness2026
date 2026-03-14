"""Generate win probability predictions for matchups."""

import numpy as np
import pandas as pd
from pathlib import Path

from data.feature_engineering import FEATURE_COLUMNS, compute_team_stats, compute_massey_composite
from data.seed_history import get_seed_win_prob
from models.train import load_model, WinProbabilityModel


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
    ) -> "Predictor":
        """Create a Predictor using KenPom ratings as team stats.

        If a trained model exists, uses KenPom stats as features for the ML model.
        Otherwise, falls back to KenPom-based direct prediction.
        """
        from data.kenpom import load_kenpom, load_kenpom_as_team_stats

        kenpom_df = load_kenpom(kenpom_path)
        team_stats = load_kenpom_as_team_stats(kenpom_path, teams_csv)

        # Build ID -> name mapping for KenPom direct prediction fallback
        id_to_name = dict(zip(team_stats["TeamID"], team_stats["TeamName"]))

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

    def predict_matchup(self, team_a: int, team_b: int) -> float:
        """Predict P(team_a wins) for a single matchup.

        Priority:
        1. ML model with team stats (best if trained model exists)
        2. KenPom efficiency margin direct prediction
        3. Seed-based historical probability (fallback)
        """
        # 1. ML model
        if self.model is not None and self.team_stats is not None:
            features = self._build_features(team_a, team_b)
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

    def _build_features(self, team_a: int, team_b: int) -> dict | None:
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

        features = {
            "SeedDiff": seed_b - seed_a,
            "AdjODiff": sa["AdjO"] - sb["AdjO"],
            "AdjDDiff": sa["AdjD"] - sb["AdjD"],
            "AdjEMDiff": sa["AdjEM"] - sb["AdjEM"],
            "AdjTAvg": (sa["AdjT"] + sb["AdjT"]) / 2,
            "SOSDiff": sa["SOS"] - sb["SOS"],
            "WinPctDiff": sa["WinPct"] - sb["WinPct"],
            "MasseyRankDiff": 0.0,
            "TourneyExpDiff": 0.0,
        }

        if self.massey is not None:
            rank_a = self.massey[self.massey["TeamID"] == team_a]
            rank_b = self.massey[self.massey["TeamID"] == team_b]
            if not rank_a.empty and not rank_b.empty:
                features["MasseyRankDiff"] = (
                    rank_b.iloc[0]["MasseyRank"] - rank_a.iloc[0]["MasseyRank"]
                )

        return features
