"""Generate pipeline flow diagram as PNG using matplotlib."""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def draw_pipeline():
    fig, ax = plt.subplots(1, 1, figsize=(14, 22))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 22)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Colors
    C_DATA = "#4A90D9"       # Blue - data sources
    C_PROCESS = "#50B86C"    # Green - processing steps
    C_MODEL = "#E67E22"      # Orange - model
    C_OPT = "#9B59B6"        # Purple - optimization
    C_OUTPUT = "#E74C3C"     # Red - outputs
    C_ARROW = "#555555"
    C_PHASE = "#F5F5F5"

    def box(x, y, w, h, text, color, fontsize=9, bold=False):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="#333333", linewidth=1.2, alpha=0.92,
        )
        ax.add_patch(rect)
        weight = "bold" if bold else "normal"
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            color="white", fontweight=weight, wrap=True,
            fontfamily="sans-serif",
        )

    def arrow_v(x, y1, y2):
        """Vertical arrow from (x, y1) down to (x, y2)."""
        ax.annotate(
            "", xy=(x, y2), xytext=(x, y1),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.5),
        )

    def arrow_h(y, x1, x2):
        """Horizontal arrow from (x1, y) to (x2, y)."""
        ax.annotate(
            "", xy=(x2, y), xytext=(x1, y),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.5),
        )

    def arrow_diag(x1, y1, x2, y2, lw=1.3):
        """Diagonal arrow."""
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=lw),
        )

    def phase_bg(y, h, label):
        rect = mpatches.FancyBboxPatch(
            (0.3, y), 13.4, h, boxstyle="round,pad=0.25",
            facecolor=C_PHASE, edgecolor="#CCCCCC", linewidth=1, alpha=0.5,
        )
        ax.add_patch(rect)
        ax.text(0.7, y + h - 0.35, label, fontsize=11, fontweight="bold",
                color="#444444", fontfamily="sans-serif")

    # =========================================================================
    # Phase 1: BUILD
    # =========================================================================
    phase_bg(13.0, 8.5, "PHASE 1: BUILD (before Selection Sunday)")

    # Row 1: Data sources — evenly spaced
    dsw, dsh = 2.8, 0.85
    dsy = 20.2
    box(0.7, dsy, dsw, dsh, "Kaggle NCAA Data\n(12 seasons)", C_DATA, 8.5)
    box(3.9, dsy, dsw, dsh, "KenPom CSV\n(365 teams)", C_DATA, 8.5)
    box(7.1, dsy, dsw, dsh, "Barttorvik\n(optional)", C_DATA, 8.5)
    box(10.3, dsy, dsw, dsh, "ESPN BPI\n(optional)", C_DATA, 8.5)

    # Row 2: Feature Engineering
    fey = 18.4
    box(2.5, fey, 9, 1.1,
        "Feature Engineering\n"
        "18 matchup features (team A \u2212 team B differences)\n"
        "~6,000 training rows across 12 seasons", C_PROCESS, 8.5)

    # Arrows: data sources → feature engineering
    for bx in [0.7, 3.9, 7.1, 10.3]:
        arrow_v(bx + dsw / 2, dsy, fey + 1.1)

    # Row 3: Model Training
    mty = 16.5
    box(2.5, mty, 9, 1.2,
        "Model Training (LOSO Cross-Validation)\n"
        "Stacked Ensemble: 6 base models \u2192 logistic meta-learner\n"
        "Leave-one-season-out prevents overfitting", C_MODEL, 8.5)

    arrow_v(7, fey, mty + 1.2)

    # Row 4: Calibration / Evaluation / Saved model
    r4y = 14.4
    box(0.7, r4y, 3.5, 0.9, "Calibration\nIsotonic / Platt / Temperature",
        C_MODEL, 8.5)
    box(5.2, r4y, 3.5, 0.9, "Evaluation\nLOSO CV, ECE, seed-tier",
        C_PROCESS, 8.5)
    box(9.7, r4y, 3.5, 0.9, "models/saved/model.pkl\n(trained model)",
        C_OUTPUT, 8.5)

    # Training → three outputs (gentle diagonals)
    arrow_diag(5.5, mty, 2.45, r4y + 0.9)
    arrow_v(7, mty, r4y + 0.9)
    arrow_diag(8.5, mty, 11.45, r4y + 0.9)

    # Calibration → Evaluation
    arrow_h(r4y + 0.45, 4.2, 5.2)

    # =========================================================================
    # Legend (between phases)
    # =========================================================================
    legend_y = 13.2
    legend_items = [
        (C_DATA, "Data Sources"),
        (C_PROCESS, "Processing"),
        (C_MODEL, "Model / Training"),
        (C_OPT, "Optimization"),
        (C_OUTPUT, "Outputs"),
    ]
    for i, (color, label) in enumerate(legend_items):
        lx = 0.7 + i * 2.6
        rect = mpatches.FancyBboxPatch(
            (lx, legend_y), 0.3, 0.3, boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="#333333", linewidth=0.8, alpha=0.9,
        )
        ax.add_patch(rect)
        ax.text(lx + 0.45, legend_y + 0.15, label,
                fontsize=7.5, color="#555555", va="center",
                fontfamily="sans-serif")

    # =========================================================================
    # Phase 2: OPTIMIZE
    # =========================================================================
    phase_bg(0.3, 12.5, "PHASE 2: OPTIMIZE (after Selection Sunday)")

    # Row 5: Bracket + Predictor + KenPom
    r5y = 11.5
    box(0.7, r5y, 3, 0.85, "bracket.json\n(64 teams)", C_DATA, 8.5)
    box(4.2, r5y, 5, 0.85,
        "Predictor (3-tier fallback)\n"
        "ML model \u2192 KenPom \u2192 Seed-based", C_MODEL, 8.5)
    box(9.7, r5y, 3.5, 0.85, "kenpom_2026.csv\n(current season)",
        C_DATA, 8.5)

    # Arrows into predictor
    arrow_h(r5y + 0.42, 3.7, 4.2)   # bracket → predictor
    arrow_h(r5y + 0.42, 9.7, 9.2)   # kenpom → predictor

    # model.pkl → predictor (vertical)
    arrow_v(11.45, r4y, r5y + 0.85)

    # Row 6: Monte Carlo Simulation
    r6y = 9.6
    box(3.5, r6y, 7, 1.0,
        "Monte Carlo Simulation\n"
        "50,000 full-bracket simulations\n"
        "P(team reaches round R)", C_PROCESS, 8.5)

    arrow_v(7, r5y, r6y + 1.0)

    # Row 7: Three optimizer inputs (Ownership, DP, Analytical EV)
    r7y = 7.5
    ow = 3.6
    oh = 1.3
    gap = 0.3
    ox1 = 0.7
    ox2 = ox1 + ow + gap
    ox3 = ox2 + ow + gap
    box(ox1, r7y, ow, oh,
        "Ownership Model\nNash equilibrium\n+ heuristic blend\n+ brand/recency bias",
        C_OPT, 7.5)
    box(ox2, r7y, ow, oh,
        "DP Future Values\nBackward induction\n9 contest days\nSave top seeds for later",
        C_OPT, 7.5)
    box(ox3, r7y, ow, oh,
        "Analytical EV\nExact closed-form\nEV per pick",
        C_OPT, 7.5)

    # MC Simulation → three optimizer boxes (clean fan-out)
    mc_cx = 7.0
    arrow_v(mc_cx, r6y, r7y + oh)                             # center → DP
    arrow_diag(mc_cx - 1.5, r6y, ox1 + ow / 2, r7y + oh)     # left → Ownership
    arrow_diag(mc_cx + 1.5, r6y, ox3 + ow / 2, r7y + oh)     # right → Analytical

    # Row 8: Portfolio Search (wide, centered below the three boxes)
    r8y = 5.7
    box(3.5, r8y, 7, 1.1,
        "Portfolio Search\n"
        "Greedy assignment + local swap search (hybrid)\n"
        "or Ant Colony Optimization (aco)", C_OPT, 8.5)

    # Three optimizer boxes → Portfolio Search (vertical arrows)
    arrow_v(ox1 + ow / 2, r7y, r8y + 1.1)   # Ownership → Portfolio
    arrow_v(ox2 + ow / 2, r7y, r8y + 1.1)   # DP → Portfolio
    arrow_v(ox3 + ow / 2, r7y, r8y + 1.1)   # Analytical → Portfolio

    # Row 9: Pick Recommendations
    r9y = 3.8
    box(2.5, r9y, 9, 1.2,
        "Pick Recommendations\n"
        "Entry 1: (3) Iowa St. + (3) Purdue    Win 93%  Own 2%\n"
        "Entry 2: (4) Kansas + (2) UConn        Win 89%  Own 1.7%\n"
        "Total EV: $663  |  Joint survival: 100%", C_OUTPUT, 8)

    arrow_v(7, r8y, r9y + 1.2)

    # Row 10: Tournament tracking
    r10y = 1.5
    tw = 3.0
    box(0.7, r10y, tw, 0.9,
        "Record Results\nmarchmadness results\n--day N", C_PROCESS, 8.5)
    box(4.2, r10y, tw, 0.9,
        "Entry Manager\nTrack alive/eliminated\nEnforce no-reuse", C_PROCESS, 8.5)
    box(7.7, r10y, tw, 0.9,
        "Next Day\nRe-optimize with\nupdated bracket", C_PROCESS, 8.5)

    # Recommendations → tracking (fan out)
    arrow_diag(5.5, r9y, ox1 + tw / 2, r10y + 0.9)
    arrow_v(7, r9y, r10y + 0.9)

    # Horizontal: Record → Entry Manager → Next Day
    arrow_h(r10y + 0.45, 0.7 + tw, 4.2)
    arrow_h(r10y + 0.45, 4.2 + tw, 7.7)

    # Loop-back: Next Day → back to MC Simulation
    # Clean right-angle path up the right margin
    loop_start_x = 7.7 + tw          # right edge of "Next Day"
    loop_start_y = r10y + 0.45       # vertical center
    loop_rail_x = 13.0               # right margin rail
    loop_end_y = r6y + 0.5           # middle of MC Simulation
    loop_end_x = 3.5 + 7             # right edge of MC Sim

    # Horizontal: Next Day right edge → right margin
    ax.annotate(
        "", xy=(loop_rail_x, loop_start_y), xytext=(loop_start_x, loop_start_y),
        arrowprops=dict(arrowstyle="-", color=C_ARROW, lw=1.2,
                        linestyle="dashed"),
    )
    # Vertical: up the right margin
    ax.annotate(
        "", xy=(loop_rail_x, loop_end_y), xytext=(loop_rail_x, loop_start_y),
        arrowprops=dict(arrowstyle="-", color=C_ARROW, lw=1.2,
                        linestyle="dashed"),
    )
    # Horizontal: right margin → MC Simulation (with arrowhead)
    ax.annotate(
        "", xy=(loop_end_x, loop_end_y), xytext=(loop_rail_x, loop_end_y),
        arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.2,
                        linestyle="dashed"),
    )
    # Label
    ax.text(loop_rail_x + 0.15, (loop_start_y + loop_end_y) / 2,
            "repeat each\ngame day",
            fontsize=7.5, color="#888888", fontfamily="sans-serif",
            ha="left", va="center", style="italic")

    # =========================================================================
    # Title
    # =========================================================================
    ax.text(7, 21.6, "March Madness Survivor Pool Optimizer \u2014 Pipeline",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color="#222222", fontfamily="sans-serif")

    plt.tight_layout()
    plt.savefig("docs/pipeline.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print("Saved docs/pipeline.png")


if __name__ == "__main__":
    draw_pipeline()
