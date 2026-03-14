"""Tests for win probability models."""

import numpy as np
import pandas as pd
import pytest
from data.seed_history import parse_seed, get_seed_win_prob
from data.feature_engineering import FEATURE_COLUMNS


def test_parse_seed():
    assert parse_seed("W01") == 1
    assert parse_seed("Z16a") == 16
    assert parse_seed("X08") == 8
    assert parse_seed("Y11") == 11


def test_seed_win_prob_favorites():
    """Higher seeds should generally be favored."""
    assert get_seed_win_prob(1, 16) > 0.9
    assert get_seed_win_prob(2, 15) > 0.85
    assert get_seed_win_prob(8, 9) > 0.45  # Nearly 50/50


def test_seed_win_prob_symmetry():
    """P(A beats B) + P(B beats A) should equal 1."""
    for seed_a in range(1, 17):
        for seed_b in range(seed_a + 1, 17):
            p1 = get_seed_win_prob(seed_a, seed_b)
            p2 = get_seed_win_prob(seed_b, seed_a)
            assert abs(p1 + p2 - 1.0) < 1e-10, f"Seeds {seed_a} vs {seed_b}"


def test_seed_win_prob_range():
    """All probabilities should be in [0, 1]."""
    for seed_a in range(1, 17):
        for seed_b in range(1, 17):
            if seed_a == seed_b:
                continue
            p = get_seed_win_prob(seed_a, seed_b)
            assert 0 <= p <= 1, f"Seeds {seed_a} vs {seed_b}: {p}"


def test_feature_columns_defined():
    """Ensure feature columns list is not empty."""
    assert len(FEATURE_COLUMNS) >= 5


class TestSyntheticModel:
    """Test model training with synthetic data."""

    def _make_synthetic_data(self, n=200) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        rows = []
        for i in range(n):
            seed_diff = rng.integers(-15, 16)
            adj_em_diff = seed_diff * -2.0 + rng.normal(0, 3)
            result = 1 if (seed_diff + rng.normal(0, 5)) > 0 else 0
            rows.append({
                "Season": 2020 + i % 5,
                "SeedA": max(1, 8 - seed_diff // 2),
                "SeedB": max(1, 8 + seed_diff // 2),
                "TeamA": 1000 + i,
                "TeamB": 2000 + i,
                "SeedDiff": seed_diff,
                "AdjODiff": rng.normal(0, 5),
                "AdjDDiff": rng.normal(0, 5),
                "AdjEMDiff": adj_em_diff,
                "AdjTAvg": rng.normal(67, 3),
                "SOSDiff": rng.normal(0, 2),
                "WinPctDiff": seed_diff * 0.02 + rng.normal(0, 0.1),
                "MasseyRankDiff": -seed_diff * 5 + rng.normal(0, 10),
                "TourneyExpDiff": rng.integers(-3, 4),
                "Result": result,
            })
        return pd.DataFrame(rows)

    def test_logistic_model(self):
        from models.train import LogisticBaseline
        df = self._make_synthetic_data()
        model = LogisticBaseline()
        model.fit(df, df["Result"])
        preds = model.predict_proba(df)
        assert preds.shape == (len(df),)
        assert all(0 <= p <= 1 for p in preds)

    def test_xgboost_model(self):
        from models.train import XGBoostModel
        df = self._make_synthetic_data()
        model = XGBoostModel(calibrate=False)
        model.fit(df, df["Result"])
        preds = model.predict_proba(df)
        assert preds.shape == (len(df),)
        assert all(0 <= p <= 1 for p in preds)

    def test_ensemble_model(self):
        from models.train import EnsembleModel
        df = self._make_synthetic_data()
        model = EnsembleModel(calibrate=False)
        model.fit(df, df["Result"])
        preds = model.predict_proba(df)
        assert preds.shape == (len(df),)
        assert all(0 <= p <= 1 for p in preds)
        # Ensemble should be between logistic and xgboost
        assert abs(preds.mean() - 0.5) < 0.3

    def test_lightgbm_model(self):
        from models.train import LightGBMModel
        df = self._make_synthetic_data()
        model = LightGBMModel(calibrate=False)
        model.fit(df, df["Result"])
        preds = model.predict_proba(df)
        assert preds.shape == (len(df),)
        assert all(0 <= p <= 1 for p in preds)

    def test_catboost_model(self):
        from models.train import CatBoostModel
        df = self._make_synthetic_data()
        model = CatBoostModel(calibrate=False)
        model.fit(df, df["Result"])
        preds = model.predict_proba(df)
        assert preds.shape == (len(df),)
        assert all(0 <= p <= 1 for p in preds)

    def test_randomforest_model(self):
        from models.train import RandomForestModel
        df = self._make_synthetic_data()
        model = RandomForestModel()
        model.fit(df, df["Result"])
        preds = model.predict_proba(df)
        assert preds.shape == (len(df),)
        assert all(0 <= p <= 1 for p in preds)

    def test_naivebayes_model(self):
        from models.train import NaiveBayesModel
        df = self._make_synthetic_data()
        model = NaiveBayesModel()
        model.fit(df, df["Result"])
        preds = model.predict_proba(df)
        assert preds.shape == (len(df),)
        assert all(0 <= p <= 1 for p in preds)

    def test_stacked_model(self):
        from models.train import StackedEnsemble
        df = self._make_synthetic_data(n=300)
        model = StackedEnsemble(calibrate=False)
        model.fit(df, df["Result"])
        preds = model.predict_proba(df)
        assert preds.shape == (len(df),)
        assert all(0 <= p <= 1 for p in preds)

    def test_stacked_model_pickleable(self):
        import pickle
        from models.train import StackedEnsemble
        df = self._make_synthetic_data(n=300)
        model = StackedEnsemble(calibrate=False)
        model.fit(df, df["Result"])
        data = pickle.dumps(model)
        loaded = pickle.loads(data)
        preds_orig = model.predict_proba(df)
        preds_loaded = loaded.predict_proba(df)
        np.testing.assert_array_almost_equal(preds_orig, preds_loaded)
