"""Generate pipeline flow diagram as PNG using matplotlib."""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def draw_pipeline():
    fig, ax = plt.subplots(1, 1, figsize=(14, 18))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 18)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Colors
    C_DATA = "#4A90D9"       # Blue - data sources
    C_PROCESS = "#50B86C"    # Green - processing steps
    C_MODEL = "#E67E22"      # Orange - model
    C_OPT = "#9B59B6"        # Purple - optimization
    C_OUTPUT = "#E74C3C"     # Red - outputs
    C_ARROW = "#555555"
    C_PHASE = "#F0F0F0"

    def box(x, y, w, h, text, color, fontsize=9, bold=False):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="#333333", linewidth=1.2, alpha=0.9,
        )
        ax.add_patch(rect)
        weight = "bold" if bold else "normal"
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            color="white", fontweight=weight, wrap=True,
            fontfamily="sans-serif",
        )

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.5),
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.15, my, label, fontsize=7, color="#666666",
                    fontfamily="sans-serif")

    def phase_bg(y, h, label):
        rect = mpatches.FancyBboxPatch(
            (0.3, y), 13.4, h, boxstyle="round,pad=0.2",
            facecolor=C_PHASE, edgecolor="#CCCCCC", linewidth=1, alpha=0.5,
        )
        ax.add_patch(rect)
        ax.text(0.7, y + h - 0.3, label, fontsize=10, fontweight="bold",
                color="#444444", fontfamily="sans-serif")

    # =========================================================================
    # Phase 1: Build
    # =========================================================================
    phase_bg(10.2, 7.5, "PHASE 1: BUILD (before Selection Sunday)")

    # Row 1: Data sources
    box(0.5, 16.5, 3, 0.8, "Kaggle NCAA Data\n(12 seasons)", C_DATA, 8)
    box(4, 16.5, 2.5, 0.8, "KenPom CSV\n(365 teams)", C_DATA, 8)
    box(7, 16.5, 2.5, 0.8, "Barttorvik\n(optional)", C_DATA, 8)
    box(10, 16.5, 2.5, 0.8, "ESPN BPI\n(optional)", C_DATA, 8)

    # Row 2: Feature Engineering
    box(3, 14.8, 7, 1, "Feature Engineering\n18 matchup features (team A - team B differences)\n~6,000 training rows across 12 seasons", C_PROCESS, 8)

    arrow(2, 16.5, 5, 15.85)
    arrow(5.25, 16.5, 6, 15.85)
    arrow(8.25, 16.5, 7.5, 15.85)
    arrow(11.25, 16.5, 9, 15.85)

    # Feature detail annotations
    ax.text(10.5, 15.3, "SeedDiff, AdjEMDiff, AdjODiff,\nAdjDDiff, SOSDiff, WinPctDiff,\nMasseyRankDiff, BarthagDiff,\nBPIDiff, WABDiff, + 8 more",
            fontsize=6.5, color="#777777", fontfamily="monospace", va="top")

    # Row 3: Training
    box(3, 13, 7, 1.1, "Model Training (LOSO Cross-Validation)\nStacked Ensemble: 6 base models → logistic meta-learner\nLeave-one-season-out prevents overfitting", C_MODEL, 8)

    arrow(6.5, 14.8, 6.5, 14.15)

    # Row 4: Calibration + Output
    box(1, 11.3, 3.5, 0.8, "Calibration\nIsotonic / Platt / Temperature", C_MODEL, 8)
    box(5.5, 11.3, 3, 0.8, "Evaluation\nLOSO CV, ECE, seed-tier", C_PROCESS, 8)
    box(9.5, 11.3, 3.5, 0.8, "models/saved/model.pkl\n(trained model)", C_OUTPUT, 8)

    arrow(6.5, 13, 6.5, 12.55)
    arrow(4.5, 11.7, 5.5, 11.7)
    arrow(6.5, 12.55, 2.75, 12.15)
    arrow(6.5, 12.55, 7, 12.15)
    arrow(4.5, 11.7, 9.5, 11.7)

    # =========================================================================
    # Phase 2: Optimize
    # =========================================================================
    phase_bg(0.3, 9.5, "PHASE 2: OPTIMIZE (after Selection Sunday)")

    # Row 5: Bracket + Predictor
    box(0.5, 8.6, 2.5, 0.8, "bracket.json\n(64 teams)", C_DATA, 8)
    box(3.5, 8.6, 3, 0.8, "Predictor (3-tier fallback)\nML → KenPom → Seed-based", C_MODEL, 8)
    box(7.5, 8.6, 3, 0.8, "kenpom_2026.csv\n(current season)", C_DATA, 8)

    arrow(3, 9, 3.5, 9)
    arrow(7.5, 9, 6.5, 9)
    # model.pkl feeds predictor
    arrow(11.25, 11.3, 5, 9.45)

    # Row 6: MC Simulation
    box(3, 7, 4, 0.9, "Monte Carlo Simulation\n50,000 full-bracket simulations\nP(team reaches round R)", C_PROCESS, 8)

    arrow(5, 8.6, 5, 7.95)

    # Row 7: Optimization components (side by side)
    box(0.5, 5.2, 2.8, 1.2, "Ownership Model\nNash equilibrium\n+ heuristic blend\n+ brand/recency bias", C_OPT, 7.5)
    box(3.8, 5.2, 2.8, 1.2, "DP Future Values\nBackward induction\nacross 9 contest days\nSave top seeds for later", C_OPT, 7.5)
    box(7.1, 5.2, 2.5, 1.2, "Analytical EV\nExact closed-form\nEV per pick", C_OPT, 7.5)
    box(10.1, 5.2, 2.8, 1.2, "Portfolio Search\nGreedy + swap (hybrid)\nor ACO (aco method)", C_OPT, 7.5)

    arrow(5, 7, 1.9, 6.45)
    arrow(5, 7, 5.2, 6.45)
    arrow(5, 7, 8.35, 6.45)

    # Connect ownership → portfolio
    arrow(3.3, 5.8, 7.1, 5.8)
    # Connect DP → portfolio
    arrow(6.6, 5.8, 7.1, 5.8)
    # Connect analytical → portfolio
    arrow(9.6, 5.8, 10.1, 5.8)

    # Row 8: Output
    box(3.5, 3.3, 6, 1.2, "Pick Recommendations\nEntry 1: (3) Iowa St. + (3) Purdue   Win=93% Own=2%\nEntry 2: (4) Kansas + (2) UConn      Win=89% Own=1.7%\nTotal EV: $663.51  |  Joint survival: 100%", C_OUTPUT, 7.5)

    arrow(11.5, 5.2, 6.5, 4.55)

    # Row 9: Tournament tracking
    box(0.5, 1.3, 3, 0.8, "Record Results\nmarchmadness results\n--day N", C_PROCESS, 8)
    box(4, 1.3, 3, 0.8, "Entry Manager\nTrack alive/eliminated\nEnforce no-reuse", C_PROCESS, 8)
    box(7.5, 1.3, 3, 0.8, "Next Day\nRe-optimize with\nupdated bracket", C_PROCESS, 8)

    arrow(6.5, 3.3, 6.5, 2.75)
    arrow(6.5, 2.75, 1.5, 2.15)
    arrow(6.5, 2.75, 5.5, 2.15)
    arrow(3.5, 1.7, 4, 1.7)
    arrow(7, 1.7, 7.5, 1.7)
    # Loop back
    ax.annotate(
        "", xy=(11.5, 8.6), xytext=(10.5, 1.7),
        arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.2,
                        connectionstyle="arc3,rad=-0.3", linestyle="dashed"),
    )
    ax.text(12, 5.0, "repeat\neach\nday", fontsize=7, color="#888888",
            fontfamily="sans-serif", ha="center", style="italic")

    # Title
    ax.text(7, 17.7, "March Madness Survivor Pool Optimizer — Pipeline",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color="#222222", fontfamily="sans-serif")

    plt.tight_layout()
    plt.savefig("docs/pipeline.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print("Saved docs/pipeline.png")


if __name__ == "__main__":
    draw_pipeline()
