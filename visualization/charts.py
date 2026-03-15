"""Generate matplotlib charts for distribution analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from optimizer.distribution import DistributionReport


def generate_all_charts(
    report: DistributionReport,
    output_dir: str | Path = "output",
) -> list[str]:
    """Generate all distribution charts and save to output_dir.

    Returns list of saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    if report.concentration_by_day:
        path = output_dir / "concentration_heatmap.png"
        _plot_concentration_heatmap(report, path)
        saved.append(str(path))

    if report.survival:
        path = output_dir / "survival_distribution.png"
        _plot_survival_distribution(report, path)
        saved.append(str(path))

        path = output_dir / "survival_funnel.png"
        _plot_survival_funnel(report, path)
        saved.append(str(path))

    if report.correlation.pairwise_correlation is not None:
        path = output_dir / "correlation_matrix.png"
        _plot_correlation_matrix(report, path)
        saved.append(str(path))

    if report.correlation.team_exposure:
        path = output_dir / "team_exposure.png"
        _plot_team_exposure(report, path)
        saved.append(str(path))

    return saved


def _plot_concentration_heatmap(report: DistributionReport, path: Path) -> None:
    """Heatmap: rows = teams, columns = days, color = fraction of entries."""
    conc_days = report.concentration_by_day
    if not conc_days:
        return

    # Collect all teams across all days
    all_teams: set[int] = set()
    for c in conc_days:
        all_teams.update(c.team_fractions.keys())
    team_list = sorted(all_teams)

    # Build matrix
    matrix = np.zeros((len(team_list), len(conc_days)))
    for j, c in enumerate(conc_days):
        for i, team_id in enumerate(team_list):
            matrix[i, j] = c.team_fractions.get(team_id, 0)

    # Team labels
    bracket = report.bracket
    team_labels = []
    for t in team_list:
        info = bracket.teams.get(t, {})
        team_labels.append(f"({info.get('seed', '?')}) {info.get('name', t)}")

    day_labels = [c.label for c in conc_days]

    fig, ax = plt.subplots(figsize=(max(8, len(conc_days) * 1.5), max(6, len(team_list) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Fraction of entries")

    ax.set_xticks(range(len(day_labels)))
    ax.set_xticklabels(day_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(team_labels)))
    ax.set_yticklabels(team_labels, fontsize=7)

    # Annotate cells with fraction > 5%
    for i in range(len(team_list)):
        for j in range(len(conc_days)):
            val = matrix[i, j]
            if val > 0.05:
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=6, color="black" if val < 0.5 else "white")

    ax.set_title(f"Team Concentration Heatmap ({report.n_alive} alive entries)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_survival_distribution(report: DistributionReport, path: Path) -> None:
    """Histogram of survival counts for each day."""
    survival_days = [s for s in report.survival if s.alive_counts]
    if not survival_days:
        return

    n_days = len(survival_days)
    cols = min(3, n_days)
    rows = (n_days + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

    axes = [axes] if n_days == 1 else (axes.flatten() if hasattr(axes, "flatten") else [axes])

    for idx, surv in enumerate(survival_days):
        if idx >= len(axes):
            break
        ax = axes[idx]
        counts = np.array(surv.alive_counts)
        max_val = int(np.max(counts))

        ax.hist(counts, bins=range(0, max_val + 2), color="#3498db",
                edgecolor="white", alpha=0.85, density=True)
        ax.axvline(surv.mean_alive, color="red", linestyle="--",
                   label=f"Mean: {surv.mean_alive:.1f}")
        ax.axvline(surv.median_alive, color="orange", linestyle=":",
                   label=f"Median: {surv.median_alive:.0f}")
        ax.set_title(f"Day {surv.day_num} ({surv.label})", fontsize=10)
        ax.set_xlabel("Entries surviving")
        ax.set_ylabel("Probability")
        ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(n_days, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Survival Distribution ({report.n_alive} entries, Monte Carlo)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_survival_funnel(report: DistributionReport, path: Path) -> None:
    """Funnel chart: mean/percentile survival across days."""
    survival_days = [s for s in report.survival if s.alive_counts]
    if not survival_days:
        return

    days = [s.day_num for s in survival_days]
    means = [s.mean_alive for s in survival_days]
    p5 = [s.percentiles[5] for s in survival_days]
    p25 = [s.percentiles[25] for s in survival_days]
    p75 = [s.percentiles[75] for s in survival_days]
    p95 = [s.percentiles[95] for s in survival_days]
    labels = [s.label for s in survival_days]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(days, p5, p95, alpha=0.15, color="blue", label="5th-95th percentile")
    ax.fill_between(days, p25, p75, alpha=0.3, color="blue", label="25th-75th percentile")
    ax.plot(days, means, "o-", color="red", linewidth=2, markersize=6, label="Mean")

    ax.set_xticks(days)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Entries alive")
    ax.set_title("Survival Funnel Across Tournament Days", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_correlation_matrix(report: DistributionReport, path: Path) -> None:
    """Heatmap of pairwise entry death correlation."""
    corr = report.correlation.pairwise_correlation
    if corr is None:
        return

    n = corr.shape[0]
    if n > 50:
        # Too many entries for a readable matrix; show top 50
        corr = corr[:50, :50]
        n = 50

    fig, ax = plt.subplots(figsize=(max(6, n * 0.2), max(5, n * 0.2)))
    im = ax.imshow(corr, cmap="Reds", vmin=0, vmax=min(1, float(np.max(corr)) * 1.2))
    plt.colorbar(im, ax=ax, label="P(both eliminated)")

    ax.set_xlabel("Entry ID")
    ax.set_ylabel("Entry ID")
    ax.set_title(
        f"Entry Elimination Correlation (mean={report.correlation.mean_pairwise:.3f})",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_team_exposure(report: DistributionReport, path: Path) -> None:
    """Bar chart: entries at risk per team."""
    exposure = report.correlation.team_exposure
    if not exposure:
        return

    bracket = report.bracket

    # Sort by exposure descending, show top 20
    sorted_teams = sorted(exposure.items(), key=lambda x: x[1], reverse=True)[:20]
    team_ids = [t for t, _ in sorted_teams]
    counts = [c for _, c in sorted_teams]

    labels = []
    for t in team_ids:
        info = bracket.teams.get(t, {})
        labels.append(f"({info.get('seed', '?')}) {info.get('name', t)}")

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.35)))
    y_pos = range(len(labels))
    colors = ["#e74c3c" if c > report.n_alive * 0.3 else
              "#f39c12" if c > report.n_alive * 0.15 else
              "#3498db" for c in counts]
    ax.barh(y_pos, counts, color=colors, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Entries exposed (eliminated if team loses)")
    ax.set_title("Team Exposure — Single Points of Failure", fontsize=12)

    # Add count labels
    for i, c in enumerate(counts):
        ax.text(c + 0.3, i, str(c), va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
