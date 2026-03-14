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

try:
    import xgboost as xgb
except ImportError:
    xgb = None

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


def get_model(model_type: str = "xgboost", calibrate: bool = True) -> WinProbabilityModel:
    """Factory function for models."""
    models = {
        "logistic": lambda: LogisticBaseline(),
        "xgboost": lambda: XGBoostModel(calibrate=calibrate),
        "ensemble": lambda: EnsembleModel(calibrate=calibrate),
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
