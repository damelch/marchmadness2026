"""Evaluate model calibration and accuracy."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss

from models.train import cross_validate, WinProbabilityModel
from data.feature_engineering import FEATURE_COLUMNS


def evaluate_model(
    features_df: pd.DataFrame,
    model_type: str = "xgboost",
    calibrate: bool = True,
    output_dir: str = "models/saved",
) -> dict:
    """Run full evaluation suite: cross-validation + calibration analysis."""
    # Cross-validation
    cv_results = cross_validate(features_df, model_type, calibrate)

    print(f"\n{'='*60}")
    print(f"Model Evaluation: {model_type} (calibrated={calibrate})")
    print(f"{'='*60}")

    if cv_results.empty:
        print("No cross-validation results (not enough data)")
        return {"cv_results": cv_results}

    print(f"\nLeave-One-Season-Out Cross-Validation:")
    print(f"  Mean Log-Loss:    {cv_results['LogLoss'].mean():.4f} (+/- {cv_results['LogLoss'].std():.4f})")
    print(f"  Mean Brier Score: {cv_results['BrierScore'].mean():.4f} (+/- {cv_results['BrierScore'].std():.4f})")
    print(f"\nPer-Season Results:")
    print(cv_results.to_string(index=False))

    # Seed-based baseline comparison
    seed_baseline = _seed_baseline_scores(features_df)
    print(f"\nSeed-Based Baseline:")
    print(f"  Log-Loss:    {seed_baseline['log_loss']:.4f}")
    print(f"  Brier Score: {seed_baseline['brier_score']:.4f}")

    improvement = (seed_baseline["log_loss"] - cv_results["LogLoss"].mean()) / seed_baseline["log_loss"]
    print(f"\nModel improvement over seed baseline: {improvement:.1%}")

    return {
        "cv_results": cv_results,
        "seed_baseline": seed_baseline,
        "improvement": improvement,
    }


def _seed_baseline_scores(features_df: pd.DataFrame) -> dict:
    """Compute baseline scores using only seed difference."""
    from data.seed_history import get_seed_win_prob

    # Use seed-based probabilities as predictions
    preds = []
    for _, row in features_df.iterrows():
        p = get_seed_win_prob(int(row["SeedA"]), int(row["SeedB"]))
        preds.append(p)

    y_true = features_df["Result"].values
    preds = np.array(preds)
    preds = np.clip(preds, 0.01, 0.99)

    return {
        "log_loss": log_loss(y_true, preds),
        "brier_score": brier_score_loss(y_true, preds),
    }


def plot_calibration(
    features_df: pd.DataFrame,
    model: WinProbabilityModel,
    save_path: str | None = None,
) -> None:
    """Plot calibration curve: predicted vs actual win rate."""
    preds = model.predict_proba(features_df)
    y_true = features_df["Result"].values

    prob_true, prob_pred = calibration_curve(y_true, preds, n_bins=10, strategy="uniform")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Calibration curve
    ax1.plot(prob_pred, prob_true, "s-", label="Model")
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.set_xlabel("Predicted probability")
    ax1.set_ylabel("Actual win rate")
    ax1.set_title("Calibration Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Prediction distribution
    ax2.hist(preds, bins=30, edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Distribution")
    ax2.axvline(x=0.5, color="r", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved calibration plot to {save_path}")
    else:
        plt.show()
