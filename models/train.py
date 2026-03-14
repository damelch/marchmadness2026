"""Train win probability models for NCAA tournament matchups."""

import pickle
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import log_loss, brier_score_loss

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import catboost as cb
except ImportError:
    cb = None

from data.feature_engineering import FEATURE_COLUMNS


class WinProbabilityModel(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


class LogisticBaseline:
    """Logistic regression baseline model."""

    def __init__(self):
        self.model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X[FEATURE_COLUMNS], y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[FEATURE_COLUMNS])[:, 1]


class XGBoostModel:
    """XGBoost model with isotonic calibration."""

    def __init__(self, calibrate: bool = True):
        if xgb is None:
            raise ImportError("xgboost is required. Install with: pip install xgboost")
        self.calibrate = calibrate
        self.base_model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            max_depth=4,
            n_estimators=200,
            learning_rate=0.05,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        features = X[FEATURE_COLUMNS].values
        if self.calibrate:
            self.model = CalibratedClassifierCV(
                self.base_model, method="isotonic", cv=5
            )
        else:
            self.model = self.base_model
        self.model.fit(features, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.model.predict_proba(X[FEATURE_COLUMNS].values)[:, 1]
        return probs


class EnsembleModel:
    """Simple average of Logistic and XGBoost predictions."""

    def __init__(self, calibrate: bool = True):
        self.logistic = LogisticBaseline()
        self.xgboost = XGBoostModel(calibrate=calibrate)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.logistic.fit(X, y)
        self.xgboost.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p1 = self.logistic.predict_proba(X)
        p2 = self.xgboost.predict_proba(X)
        return (p1 + p2) / 2


class LightGBMModel:
    """LightGBM gradient boosting model."""

    def __init__(self, calibrate: bool = True):
        if lgb is None:
            raise ImportError("lightgbm is required. Install with: pip install lightgbm")
        self.calibrate = calibrate
        self.base_model = lgb.LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            num_leaves=31,
            n_estimators=200,
            learning_rate=0.05,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
        )
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        features = X[FEATURE_COLUMNS].values
        if self.calibrate:
            self.model = CalibratedClassifierCV(
                self.base_model, method="isotonic", cv=5
            )
        else:
            self.model = self.base_model
        self.model.fit(features, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[FEATURE_COLUMNS].values)[:, 1]


class CatBoostModel:
    """CatBoost gradient boosting model."""

    def __init__(self, calibrate: bool = True):
        if cb is None:
            raise ImportError("catboost is required. Install with: pip install catboost")
        self.calibrate = calibrate
        self.base_model = cb.CatBoostClassifier(
            depth=4,
            iterations=200,
            learning_rate=0.05,
            random_seed=42,
            verbose=0,
        )
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        features = X[FEATURE_COLUMNS].values
        if self.calibrate:
            self.model = CalibratedClassifierCV(
                self.base_model, method="isotonic", cv=5
            )
        else:
            self.model = self.base_model
        self.model.fit(features, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[FEATURE_COLUMNS].values)[:, 1]


class RandomForestModel:
    """Random Forest classifier."""

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X[FEATURE_COLUMNS].values, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[FEATURE_COLUMNS].values)[:, 1]


class NaiveBayesModel:
    """Gaussian Naive Bayes model. Provides ensemble diversity."""

    def __init__(self):
        self.model = GaussianNB()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X[FEATURE_COLUMNS].values, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[FEATURE_COLUMNS].values)[:, 1]


class StackedEnsemble:
    """Stacked ensemble: 6 base models + logistic regression meta-learner.

    Uses leave-one-season-out CV to generate out-of-fold predictions,
    then trains a meta-learner on those predictions.
    """

    def __init__(self, calibrate: bool = True):
        self.calibrate = calibrate
        self.base_models = None
        self.meta_learner = LogisticRegression(max_iter=1000)

    def _make_base_models(self):
        return [
            ("logistic", LogisticBaseline()),
            ("xgboost", XGBoostModel(calibrate=self.calibrate)),
            ("lightgbm", LightGBMModel(calibrate=self.calibrate)),
            ("catboost", CatBoostModel(calibrate=self.calibrate)),
            ("randomforest", RandomForestModel()),
            ("naivebayes", NaiveBayesModel()),
        ]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        seasons = sorted(X["Season"].unique())
        n_models = 6
        oof_preds = np.zeros((len(X), n_models))
        oof_mask = np.zeros(len(X), dtype=bool)

        # Generate out-of-fold predictions via leave-one-season-out
        for holdout_season in seasons:
            train_idx = X["Season"] != holdout_season
            test_idx = X["Season"] == holdout_season

            if train_idx.sum() < 50 or test_idx.sum() < 10:
                continue

            fold_models = self._make_base_models()
            for i, (name, model) in enumerate(fold_models):
                model.fit(X[train_idx], y[train_idx])
                oof_preds[test_idx.values, i] = model.predict_proba(X[test_idx])

            oof_mask |= test_idx.values

        # Train meta-learner on OOF predictions
        self.meta_learner.fit(oof_preds[oof_mask], y[oof_mask])

        # Retrain all base models on full data for inference
        self.base_models = self._make_base_models()
        for name, model in self.base_models:
            model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        base_preds = np.column_stack([
            model.predict_proba(X) for name, model in self.base_models
        ])
        return self.meta_learner.predict_proba(base_preds)[:, 1]


def get_model(model_type: str = "xgboost", calibrate: bool = True) -> WinProbabilityModel:
    """Factory function for models."""
    models = {
        "logistic": lambda: LogisticBaseline(),
        "xgboost": lambda: XGBoostModel(calibrate=calibrate),
        "lightgbm": lambda: LightGBMModel(calibrate=calibrate),
        "catboost": lambda: CatBoostModel(calibrate=calibrate),
        "randomforest": lambda: RandomForestModel(),
        "naivebayes": lambda: NaiveBayesModel(),
        "ensemble": lambda: EnsembleModel(calibrate=calibrate),
        "stacked": lambda: StackedEnsemble(calibrate=calibrate),
    }
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    return models[model_type]()


def train_model(
    features_df: pd.DataFrame,
    model_type: str = "xgboost",
    calibrate: bool = True,
) -> WinProbabilityModel:
    """Train a win probability model on the full feature dataset."""
    model = get_model(model_type, calibrate)
    X = features_df[FEATURE_COLUMNS]
    y = features_df["Result"]
    model.fit(features_df, y)
    return model


def cross_validate(
    features_df: pd.DataFrame,
    model_type: str = "xgboost",
    calibrate: bool = True,
) -> pd.DataFrame:
    """Leave-one-season-out cross-validation.

    Returns DataFrame with per-season metrics.
    """
    seasons = sorted(features_df["Season"].unique())
    results = []

    for holdout_season in seasons:
        train = features_df[features_df["Season"] != holdout_season]
        test = features_df[features_df["Season"] == holdout_season]

        if len(train) < 50 or len(test) < 10:
            continue

        model = get_model(model_type, calibrate)
        model.fit(train, train["Result"])
        preds = model.predict_proba(test)

        y_true = test["Result"].values
        results.append(
            {
                "Season": holdout_season,
                "LogLoss": log_loss(y_true, preds),
                "BrierScore": brier_score_loss(y_true, preds),
                "NumGames": len(test) // 2,  # divided by 2 due to symmetry augmentation
                "MeanPred": preds.mean(),
            }
        )

    return pd.DataFrame(results)


def save_model(model: WinProbabilityModel, path: str | Path = "models/saved/model.pkl") -> None:
    """Serialize model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str | Path = "models/saved/model.pkl") -> WinProbabilityModel:
    """Load model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
