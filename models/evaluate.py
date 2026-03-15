"""Evaluate model calibration and accuracy."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

from models.train import WinProbabilityModel, cross_validate


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

    print("\nLeave-One-Season-Out Cross-Validation:")
    print(f"  Mean Log-Loss:    {cv_results['LogLoss'].mean():.4f} (+/- {cv_results['LogLoss'].std():.4f})")
    print(f"  Mean Brier Score: {cv_results['BrierScore'].mean():.4f} (+/- {cv_results['BrierScore'].std():.4f})")
    print("\nPer-Season Results:")
    print(cv_results.to_string(index=False))

    # Seed-based baseline comparison
    seed_baseline = _seed_baseline_scores(features_df)
    print("\nSeed-Based Baseline:")
    print(f"  Log-Loss:    {seed_baseline['log_loss']:.4f}")
    print(f"  Brier Score: {seed_baseline['brier_score']:.4f}")

    improvement = (seed_baseline["log_loss"] - cv_results["LogLoss"].mean()) / seed_baseline["log_loss"]
    print(f"\nModel improvement over seed baseline: {improvement:.1%}")

    # Seed-tier calibration
    tier_metrics = _seed_tier_calibration(features_df, model_type, calibrate)
    if tier_metrics is not None:
        print("\nPer-Seed-Tier Calibration:")
        print(tier_metrics.to_string(index=False))

    # Round-level calibration
    round_metrics = _round_calibration(features_df, model_type, calibrate)
    if round_metrics is not None:
        print("\nPer-Round Calibration:")
        print(round_metrics.to_string(index=False))

    # Per-round calibration chart
    import os
    chart_path = os.path.join(output_dir, "calibration_by_round.png")
    _round_calibration_chart(features_df, model_type, calibrate, save_path=chart_path)
    print(f"Saved per-round calibration chart to {chart_path}")

    return {
        "cv_results": cv_results,
        "seed_baseline": seed_baseline,
        "improvement": improvement,
        "tier_metrics": tier_metrics,
        "round_metrics": round_metrics,
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


def _seed_tier_calibration(
    features_df: pd.DataFrame,
    model_type: str = "xgboost",
    calibrate: bool = True,
) -> pd.DataFrame | None:
    """Compute calibration metrics by seed tier.

    Tiers: Blowouts (1-4 vs 13-16), Competitive (5-8 vs 9-12), Close (same tier).
    Uses LOSO predictions to avoid overfitting.
    """
    from models.train import get_model

    if "SeedA" not in features_df.columns or "SeedB" not in features_df.columns:
        return None

    # Generate OOF predictions
    oof_preds = np.zeros(len(features_df))
    oof_mask = np.zeros(len(features_df), dtype=bool)
    seasons = sorted(features_df["Season"].unique())

    for holdout in seasons:
        train_idx = features_df["Season"] != holdout
        test_idx = features_df["Season"] == holdout
        if train_idx.sum() < 50 or test_idx.sum() < 10:
            continue
        model = get_model(model_type, calibrate)
        model.fit(features_df[train_idx], features_df[train_idx]["Result"])
        oof_preds[test_idx.values] = model.predict_proba(features_df[test_idx])
        oof_mask |= test_idx.values

    if oof_mask.sum() < 20:
        return None

    df = features_df[oof_mask].copy()
    preds = oof_preds[oof_mask]
    y_true = df["Result"].values

    # Classify seed tiers
    def _tier(seed_a, seed_b):
        higher, lower = min(seed_a, seed_b), max(seed_a, seed_b)
        if higher <= 4 and lower >= 13:
            return "Blowout (1-4 vs 13-16)"
        elif higher <= 4 and lower <= 4:
            return "Close (top vs top)"
        elif higher >= 5 and higher <= 8 and lower >= 9 and lower <= 12:
            return "Competitive (5-8 vs 9-12)"
        else:
            return "Other"

    tiers = [_tier(int(r["SeedA"]), int(r["SeedB"])) for _, r in df.iterrows()]

    results = []
    for tier in sorted(set(tiers)):
        mask = np.array([t == tier for t in tiers])
        if mask.sum() < 10:
            continue
        tier_preds = np.clip(preds[mask], 0.01, 0.99)
        tier_y = y_true[mask]
        results.append({
            "Tier": tier,
            "N": int(mask.sum()),
            "LogLoss": log_loss(tier_y, tier_preds),
            "BrierScore": brier_score_loss(tier_y, tier_preds),
            "MeanPred": tier_preds.mean(),
            "ActualWinRate": tier_y.mean(),
        })

    return pd.DataFrame(results) if results else None


def _round_calibration(
    features_df: pd.DataFrame,
    model_type: str = "xgboost",
    calibrate: bool = True,
) -> pd.DataFrame | None:
    """Compute calibration metrics by tournament round.

    Uses LOSO predictions to avoid overfitting.  Requires SeedRoundInteraction
    feature (which encodes round info) or falls back to DayNum if available.
    """
    from models.train import get_model

    # We can infer round from SeedRoundInteraction / SeedDiff
    # SeedRoundInteraction = SeedDiff * round_num, so round = interaction / seed_diff
    if "SeedRoundInteraction" not in features_df.columns or "SeedDiff" not in features_df.columns:
        return None

    # Generate OOF predictions
    oof_preds = np.zeros(len(features_df))
    oof_mask = np.zeros(len(features_df), dtype=bool)
    seasons = sorted(features_df["Season"].unique())

    for holdout in seasons:
        train_idx = features_df["Season"] != holdout
        test_idx = features_df["Season"] == holdout
        if train_idx.sum() < 50 or test_idx.sum() < 10:
            continue
        model = get_model(model_type, calibrate)
        model.fit(features_df[train_idx], features_df[train_idx]["Result"])
        oof_preds[test_idx.values] = model.predict_proba(features_df[test_idx])
        oof_mask |= test_idx.values

    if oof_mask.sum() < 20:
        return None

    df = features_df[oof_mask].copy()
    preds = oof_preds[oof_mask]
    y_true = df["Result"].values

    # Recover round number: SeedRoundInteraction / SeedDiff (when SeedDiff != 0)
    seed_diff = df["SeedDiff"].values
    interaction = df["SeedRoundInteraction"].values
    rounds = np.where(seed_diff != 0, np.round(interaction / seed_diff).astype(int), 0)

    round_names = {1: "R64", 2: "R32", 3: "S16", 4: "E8", 5: "F4", 6: "Championship"}

    results = []
    for rnd in sorted(set(rounds)):
        if rnd < 1 or rnd > 6:
            continue
        mask = rounds == rnd
        if mask.sum() < 10:
            continue
        rnd_preds = np.clip(preds[mask], 0.01, 0.99)
        rnd_y = y_true[mask]
        ll = log_loss(rnd_y, rnd_preds)
        ll_ci_low, ll_ci_high = np.nan, np.nan
        if mask.sum() >= 30:
            rng = np.random.default_rng(42)
            boot_losses = []
            n = len(rnd_y)
            for _ in range(1000):
                idx = rng.integers(0, n, size=n)
                boot_losses.append(log_loss(rnd_y[idx], rnd_preds[idx]))
            ll_ci_low, ll_ci_high = np.percentile(boot_losses, [2.5, 97.5])
        results.append({
            "Round": round_names.get(rnd, f"R{rnd}"),
            "N": int(mask.sum()),
            "LogLoss": ll,
            "LogLoss_CI_Low": ll_ci_low,
            "LogLoss_CI_High": ll_ci_high,
            "BrierScore": brier_score_loss(rnd_y, rnd_preds),
            "MeanPred": rnd_preds.mean(),
            "ActualWinRate": rnd_y.mean(),
            "ECE": compute_ece(rnd_y, rnd_preds),
        })

    return pd.DataFrame(results) if results else None


def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE = sum_bins(|actual - predicted| * fraction_in_bin)
    Lower is better. 0 = perfectly calibrated.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_pred[mask].mean()
        ece += abs(bin_acc - bin_conf) * mask.sum() / len(y_true)
    return ece


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

    # ECE
    ece = compute_ece(y_true, preds)

    # Calibration curve
    ax1.plot(prob_pred, prob_true, "s-", label="Model")
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.set_xlabel("Predicted probability")
    ax1.set_ylabel("Actual win rate")
    ax1.set_title(f"Calibration Curve (ECE={ece:.4f})")
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


def _round_calibration_chart(
    features_df: pd.DataFrame,
    model_type: str = "xgboost",
    calibrate: bool = True,
    save_path: str = "output/calibration_by_round.png",
) -> None:
    """Generate 6-panel calibration plot, one per tournament round.

    Each panel shows predicted vs actual win rate in 5 bins,
    plus a diagonal reference line for perfect calibration.
    Also shows sample size N in each panel title.
    """
    import os

    from models.train import get_model

    if "SeedRoundInteraction" not in features_df.columns or "SeedDiff" not in features_df.columns:
        return

    # Generate LOSO OOF predictions
    oof_preds = np.zeros(len(features_df))
    oof_mask = np.zeros(len(features_df), dtype=bool)
    seasons = sorted(features_df["Season"].unique())

    for holdout in seasons:
        train_idx = features_df["Season"] != holdout
        test_idx = features_df["Season"] == holdout
        if train_idx.sum() < 50 or test_idx.sum() < 10:
            continue
        model = get_model(model_type, calibrate)
        model.fit(features_df[train_idx], features_df[train_idx]["Result"])
        oof_preds[test_idx.values] = model.predict_proba(features_df[test_idx])
        oof_mask |= test_idx.values

    if oof_mask.sum() < 20:
        return

    df = features_df[oof_mask].copy()
    preds = oof_preds[oof_mask]
    y_true = df["Result"].values

    # Recover round numbers
    seed_diff = df["SeedDiff"].values
    interaction = df["SeedRoundInteraction"].values
    rounds = np.where(seed_diff != 0, np.round(interaction / seed_diff).astype(int), 0)

    round_names = {1: "R64", 2: "R32", 3: "S16", 4: "E8", 5: "F4", 6: "Championship"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, rnd in enumerate(range(1, 7)):
        ax = axes[i]
        mask = rounds == rnd
        n = mask.sum()
        rnd_name = round_names[rnd]

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")

        if n >= 5:
            rnd_preds = np.clip(preds[mask], 0.01, 0.99)
            rnd_y = y_true[mask]
            try:
                prob_true, prob_pred = calibration_curve(
                    rnd_y, rnd_preds, n_bins=5, strategy="uniform"
                )
                ax.plot(prob_pred, prob_true, "s-", color="#1f77b4", label="Model")
            except ValueError:
                pass

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Actual win rate")
        ax.set_title(f"{rnd_name} (N={n})")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Per-Round Calibration", fontsize=14, y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
