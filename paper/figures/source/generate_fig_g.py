"""
Generate Figure G: Matched Forced-Choice (2AFC) vs. Abstention-Aware (3-Option) Comparison

Directly answers Reviewer 2, Point 6:
Empirically demonstrates how forced choice manufactures false preferences at parity
and attenuates conditional payoff sensitivity through positional bias.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = "/Users/karan/Desktop/APART/paper/figures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
})

def make_fig():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8))

    # -------------------------------------------------------------
    # Panel A: Parity (1:1) Comparison: 3-Option vs Forced Choice
    # -------------------------------------------------------------
    categories = ["3-Option\n(Abstention Allowed)", "2-Option\n(Forced Choice)"]
    # 3-Option: Indiff=97.5% (156/160), Option A=1.25% (2/160), Option B=1.25% (2/160)
    # Forced:   Indiff=0.0%, Option A=20.6% (33/160), Option B=79.4% (127/160)
    
    indiff_vals = [97.5, 0.0]
    opt_a_vals  = [1.25, 20.6]
    opt_b_vals  = [1.25, 79.4]

    x = np.arange(len(categories))
    w = 0.45

    ax1.bar(x, indiff_vals, w, label="Indifferent (Abstention)", color="#A2C8EC", zorder=3)
    ax1.bar(x, opt_a_vals,  w, bottom=indiff_vals, label="Option A", color="#4878CF", zorder=3)
    ax1.bar(x, opt_b_vals,  w, bottom=[i+a for i,a in zip(indiff_vals, opt_a_vals)], label="Option B", color="#EE854A", zorder=3)

    # Add text labels
    ax1.text(0, 48, "97.5%\n(Indifferent)", ha="center", va="center", color="#222", fontweight="bold", fontsize=8.5)
    ax1.text(1, 10, "20.6% (A)", ha="center", va="center", color="white", fontweight="bold", fontsize=8)
    ax1.text(1, 60, "79.4% (Option B)\n[Directional Imbalance]", ha="center", va="center", color="white", fontweight="bold", fontsize=8.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=9.5)
    ax1.set_ylabel("Response Percentage at 1:1 Parity (%)", fontsize=10)
    ax1.set_ylim(0, 108)
    ax1.set_title("(a) Choice at 1:1 Payoff Parity (N=160)", fontsize=10.5, pad=6)
    ax1.legend(loc="upper left", fontsize=8.0, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel B: Dominant Choice Accuracy across Disparity Ratios
    # -------------------------------------------------------------
    # Ratios: 1:9, 1:5, 1:2, 2:1, 5:1, 9:1
    ratio_labels = ["1:9\n(9:1 Sub)", "1:5\n(5:1 Sub)", "1:2\n(2:1 Sub)", "2:1\n(2:1 Add)", "5:1\n(5:1 Add)", "9:1\n(9:1 Add)"]
    x_r = np.arange(len(ratio_labels))

    # 3-Option P(Dom | Commit):
    p_3opt = [100.0, 85.7, 75.0, 55.6, 78.9, 85.3]
    # 2-Option Forced P(Dom):
    p_fc   = [96.9, 97.5, 86.9, 73.8, 78.8, 83.8]

    ax2.plot(x_r, p_3opt, marker="o", color="#1f77b4", label="3-Option (Conditional on Commit)", linewidth=2.0)
    ax2.plot(x_r, p_fc,   marker="s", color="#ff7f0e", linestyle="--", label="2-Option Forced Choice (All Trials)", linewidth=2.0)

    ax2.axhline(50, color="gray", linestyle=":", linewidth=1, alpha=0.6, label="50% Chance")
    ax2.set_xticks(x_r)
    ax2.set_xticklabels(ratio_labels, fontsize=8.0)
    ax2.set_xlabel("Payoff Disparity Ratio", fontsize=10)
    ax2.set_ylabel("Dominant Option Selection (%)", fontsize=10)
    ax2.set_ylim(45, 105)
    ax2.set_title("(b) Dominant Choice by Payoff Disparity", fontsize=10.5, pad=6)
    ax2.legend(loc="lower right", fontsize=8.0, framealpha=0.9)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"FigG_ForcedChoiceComparison.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved FigG_ForcedChoiceComparison.png and .pdf successfully.")

if __name__ == "__main__":
    make_fig()
