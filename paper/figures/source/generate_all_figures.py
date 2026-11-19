"""
Generate all 6 publication-ready figures for:
"Stress-Testing LLM Preference Elicitation: Separating Commitment,
 Conditional Direction, and Payoff Sensitivity Across Models"

Figures generated (saved to paper/figures/):
  FigA_ResponseDist     - Cross-model stacked response distribution
  FigB_Commitment       - Commitment sensitivity by payoff magnitude × model
  FigC_Conditional      - Conditional dominant choice sensitivity × model
  FigD_PositionBias     - Stage-2 payoff sensitivity and position moderation
  FigE_NumericalControl - C4 numerical comprehension accuracy
  FigF_BehavioralProfile- 2-D behavioral profile (Commitment rate vs Dominant accuracy)

All data sourced from frozen CSVs: results/{model}/primary.csv and c4.csv
No model inference required.
"""

import csv
import math
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import defaultdict

warnings.filterwarnings('ignore')

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), '..') 
os.makedirs(OUT_DIR, exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "legend.framealpha": 0.85,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
})

# ── Model metadata ────────────────────────────────────────────────────────────
MODELS = [
    {"key": "qwen25_3b",    "label": "Qwen 2.5 3B",     "color": "#1f77b4", "marker": "o"},
    {"key": "llama32_3b",   "label": "Llama 3.2 3B",    "color": "#ff7f0e", "marker": "s"},
    {"key": "phi4mini_38b", "label": "Phi-4 Mini 3.8B", "color": "#2ca02c", "marker": "^"},
    {"key": "gemma3_4b",    "label": "Gemma 3 4B",      "color": "#d62728", "marker": "D"},
]

RATIO_ORDER  = ["1:9", "1:5", "1:2", "1:1", "2:1", "5:1", "9:1"]
LOG_Q_VALUES = [-2.197, -1.609, -0.693, 0.0, 0.693, 1.609, 2.197]
ABS_LOG_ORDER = [0.0, 0.693, 1.609, 2.197]
ABS_LABELS    = ["0.0\n(1:1)", "0.69\n(2:1)", "1.61\n(5:1)", "2.20\n(9:1)"]

BASE = "/Users/karan/Desktop/APART/results"

VALID_CATS = {"add", "subtract", "indifferent"}

# ── Data loader ───────────────────────────────────────────────────────────────
def load_primary(model_key):
    path = os.path.join(BASE, model_key, "primary.csv")
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            ch = r.get("underlying_choice", "").strip().lower()
            if ch not in VALID_CATS:
                continue
            lq = float(r.get("log_Q_ratio", r.get("log_Q", 0)))
            rows.append({
                "choice":    ch,
                "log_q":     lq,
                "abs_log_q": abs(lq),
                "ratio":     r.get("Q_ratio_label", ""),
                "add_first": r.get("add_is_option_a", "True").strip().lower() in ("true", "1"),
                "phrasing":  r.get("phrasing_id", ""),
                "dilemma":   r.get("dilemma_id", ""),
            })
    return rows

def load_c4(model_key):
    path = os.path.join(BASE, model_key, "c4.csv")
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            ch = r.get("underlying_choice", "").strip().lower()
            if ch not in VALID_CATS:
                continue
            lq = float(r.get("log_Q_ratio", r.get("log_Q", 0)))
            correct = False
            if abs(lq) < 1e-6:
                correct = (ch == "indifferent")
            elif lq > 0:
                correct = (ch == "add")
            else:
                correct = (ch == "subtract")
            rows.append({"correct": correct, "valid": True})
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# FIG A — Cross-model stacked response distribution
# ══════════════════════════════════════════════════════════════════════════════
def fig_a():
    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    model_labels  = [m["label"] for m in MODELS]
    add_pcts      = []
    sub_pcts      = []
    indiff_pcts   = []

    for m in MODELS:
        rows = load_primary(m["key"])
        total = len(rows)
        add_pcts.append(100 * sum(1 for r in rows if r["choice"] == "add")       / total)
        sub_pcts.append(100 * sum(1 for r in rows if r["choice"] == "subtract")  / total)
        indiff_pcts.append(100 * sum(1 for r in rows if r["choice"] == "indifferent") / total)

    x    = np.arange(len(model_labels))
    w    = 0.52
    bars_add   = ax.bar(x, add_pcts,   w, label="Add (Directional)",      color="#4878CF", zorder=3)
    bars_sub   = ax.bar(x, sub_pcts,   w, bottom=add_pcts,                label="Subtract (Directional)", color="#EE854A", zorder=3)
    bars_ind   = ax.bar(x, indiff_pcts, w,
                        bottom=[a+s for a,s in zip(add_pcts, sub_pcts)],
                        label="Indifferent (Non-directional)", color="#A2C8EC", zorder=3)

    for i, (a, s, ind) in enumerate(zip(add_pcts, sub_pcts, indiff_pcts)):
        if a > 2:   ax.text(i, a/2,         f"{a:.1f}%",   ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
        if s > 2:   ax.text(i, a + s/2,     f"{s:.1f}%",   ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
        if ind > 2: ax.text(i, a+s+ind/2,   f"{ind:.1f}%", ha="center", va="center", fontsize=7.5, color="#333", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.set_ylabel("Percentage of Valid Trials (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, "FigA_ResponseDist")


# ══════════════════════════════════════════════════════════════════════════════
# FIG B — Commitment sensitivity: P(Commit) vs |log r| per model
# ══════════════════════════════════════════════════════════════════════════════
def fig_b():
    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    abs_vals = ABS_LOG_ORDER   # 0, 0.693, 1.609, 2.197

    for m in MODELS:
        rows = load_primary(m["key"])
        commits_by_abs = defaultdict(list)
        for r in rows:
            commits_by_abs[round(r["abs_log_q"], 3)].append(1 if r["choice"] in ("add","subtract") else 0)

        y_vals = []
        for av in abs_vals:
            bucket = commits_by_abs.get(round(av, 3), [])
            y_vals.append(np.mean(bucket) if bucket else np.nan)

        ax.plot(range(len(abs_vals)), y_vals,
                color=m["color"], marker=m["marker"], label=m["label"], zorder=3)

    ax.set_xticks(range(len(abs_vals)))
    ax.set_xticklabels(ABS_LABELS, fontsize=8.5)
    ax.set_xlabel(r"$|\log r|$ — Payoff Disparity Magnitude")
    ax.set_ylabel(r"$P(\mathrm{Commitment})$")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8.5, loc="center right")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6)

    fig.tight_layout()
    _save(fig, "FigB_Commitment")


# ══════════════════════════════════════════════════════════════════════════════
# FIG C — Conditional dominant-choice sensitivity per model
# ══════════════════════════════════════════════════════════════════════════════
def fig_c():
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharey=True)

    abs_vals = [0.693, 1.609, 2.197]  # exclude parity (0) — Stage 2 only
    abs_tick_labels = ["0.69\n(2:1)", "1.61\n(5:1)", "2.20\n(9:1)"]

    for ax, m in zip(axes, MODELS):
        rows = load_primary(m["key"])
        # only directional, non-parity
        stage2 = [r for r in rows if r["choice"] in ("add","subtract") and abs(r["log_q"]) > 1e-6]

        # split by position
        add_first_dom = defaultdict(list)
        sub_first_dom = defaultdict(list)

        for r in stage2:
            dominant = (r["log_q"] > 0 and r["choice"] == "add") or \
                       (r["log_q"] < 0 and r["choice"] == "subtract")
            key = round(r["abs_log_q"], 3)
            if r["add_first"]:
                add_first_dom[key].append(1 if dominant else 0)
            else:
                sub_first_dom[key].append(1 if dominant else 0)

        y_af, y_sf = [], []
        for av in abs_vals:
            k = round(av, 3)
            y_af.append(np.mean(add_first_dom[k]) if add_first_dom[k] else np.nan)
            y_sf.append(np.mean(sub_first_dom[k]) if sub_first_dom[k] else np.nan)

        ax.plot(range(len(abs_vals)), y_af, color=m["color"], marker="o",
                linestyle="-",  label="Add-first",  zorder=3)
        ax.plot(range(len(abs_vals)), y_sf, color=m["color"], marker="^",
                linestyle="--", label="Sub-first",  zorder=3, alpha=0.75)

        ax.set_xticks(range(len(abs_vals)))
        ax.set_xticklabels(abs_tick_labels, fontsize=8)
        ax.set_xlabel(r"$|\log r|$", fontsize=9)
        ax.set_title(m["label"], fontsize=9, pad=4)
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.9, alpha=0.5)
        n = len(stage2)
        ax.text(0.03, 0.05, f"N={n}", transform=ax.transAxes, fontsize=7.5, color="gray")

    axes[0].set_ylabel(r"$P(\mathrm{Dominant} \mid \mathrm{Committed})$")
    # shared legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], color="gray", marker="o", linestyle="-",  label="Add-first"),
               Line2D([0],[0], color="gray", marker="^", linestyle="--", label="Subtract-first")]
    axes[-1].legend(handles=handles, fontsize=8, loc="lower right")

    fig.tight_layout()
    _save(fig, "FigC_Conditional")


# ══════════════════════════════════════════════════════════════════════════════
# FIG D — Stage-2 payoff sensitivity and position moderation
# ══════════════════════════════════════════════════════════════════════════════
def fig_d():
    # Firth estimates from results/firth_stage2_robustness.json
    firth_payoff = {
        "Qwen 2.5 3B":     1.231,
        "Llama 3.2 3B":    0.282,
        "Phi-4 Mini 3.8B": 0.225,
        "Gemma 3 4B":      1.531,
    }
    firth_ci = {
        "Qwen 2.5 3B":     [0.499, 1.962],
        "Llama 3.2 3B":    [0.037, 0.526],
        "Phi-4 Mini 3.8B": [-0.015, 0.464],
        "Gemma 3 4B":      [0.510, 2.552],
    }
    # Interactive MLE position interaction coefficients from Table I
    pos_interaction = {
        "Qwen 2.5 3B":     -2.311,
        "Llama 3.2 3B":     0.244,
        "Phi-4 Mini 3.8B":  0.379,
        "Gemma 3 4B":      -0.811,
    }

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))

    model_labels = [m["label"] for m in MODELS]
    colors = [m["color"] for m in MODELS]
    x = np.arange(len(model_labels))
    w = 0.5

    # Left panel: Firth payoff β with 95% CI
    ax = axes[0]
    betas   = [firth_payoff[l] for l in model_labels]
    ci_lo   = [firth_payoff[l] - firth_ci[l][0] for l in model_labels]
    ci_hi   = [firth_ci[l][1] - firth_payoff[l] for l in model_labels]
    ax.bar(x, betas, w, color=colors, zorder=3, alpha=0.85)
    ax.errorbar(x, betas, yerr=[ci_lo, ci_hi],
                fmt="none", ecolor="black", elinewidth=1.5, capsize=4, zorder=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=8.5, rotation=12, ha="right")
    ax.set_ylabel(r"Firth $\hat{\beta}_{\mathrm{payoff}}$ [95% CI]")
    ax.set_title("Stage-2 Payoff Sensitivity", fontsize=10)
    for i, (b, lo, hi) in enumerate(zip(betas, ci_lo, ci_hi)):
        ax.text(i, b + hi + 0.08, f"{b:.3f}", ha="center", va="bottom", fontsize=8)

    # Right panel: Signed Interaction (|log r| × Pos)
    ax2 = axes[1]
    int_betas = [pos_interaction[l] for l in model_labels]
    bar_colors = [("tomato" if pb < 0 else "steelblue") for pb in int_betas]
    ax2.bar(x, int_betas, w, color=bar_colors, zorder=3, alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, fontsize=8.5, rotation=12, ha="right")
    ax2.set_ylabel(r"MLE Interaction $\hat{\beta}_{|\log r| \times \mathrm{Pos}}$")
    ax2.set_title("Stage-2 Payoff $\\times$ Position Interaction", fontsize=10)
    ax2.set_ylim(-3.0, 1.0)
    for i, pb in enumerate(int_betas):
        offset = 0.08 if pb >= 0 else -0.15
        va = "bottom" if pb >= 0 else "top"
        ax2.text(i, pb + offset, f"{pb:.3f}", ha="center", va=va, fontsize=8)

    fig.tight_layout()
    _save(fig, "FigD_PositionBias")


# ══════════════════════════════════════════════════════════════════════════════
# FIG E — C4 Numerical Comprehension Control
# ══════════════════════════════════════════════════════════════════════════════
def fig_e():
    c4_data = {
        "Qwen 2.5 3B":     {"acc": 100.0, "valid": 1120, "correct": 1120},
        "Llama 3.2 3B":    {"acc":  53.8, "valid": 1052, "correct":  566},
        "Phi-4 Mini 3.8B": {"acc":  71.0, "valid": 1115, "correct":  792},
        "Gemma 3 4B":      {"acc":  98.3, "valid": 1120, "correct": 1101},
    }

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    model_labels = [m["label"] for m in MODELS]
    colors = [m["color"] for m in MODELS]
    accs   = [c4_data[l]["acc"] for l in model_labels]
    valids = [c4_data[l]["valid"] for l in model_labels]
    corrects=[c4_data[l]["correct"] for l in model_labels]

    x = np.arange(len(model_labels))
    w = 0.5
    bars = ax.bar(x, accs, w, color=colors, zorder=3, alpha=0.88)

    for i, (a, c, v) in enumerate(zip(accs, corrects, valids)):
        ax.text(i, a + 2.0, f"{a:.1f}%\n({c}/{v})", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Correct chance level: with 3 options (Option A, Option B, Indifferent), pure chance = 33.3%
    ax.axhline(33.3, color="gray", linestyle=":", linewidth=1, alpha=0.7, label="33.3% (3-option chance)")
    ax.axhline(50.0, color="lightgray", linestyle="--", linewidth=0.8, alpha=0.5, label="50.0% (binary chance)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.set_ylabel("C4 Numerical Dominance Accuracy (%)")
    ax.set_ylim(0, 125)
    ax.legend(fontsize=8.0, loc="lower right")
    ax.text(0.98, 0.97,
            "Values show: % acc (correct / valid trials)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color="gray", style="italic")

    fig.tight_layout()
    _save(fig, "FigE_NumericalControl")


# ══════════════════════════════════════════════════════════════════════════════
# FIG F — 2-D Behavioral Profile: Commitment rate vs Conditional accuracy
# ══════════════════════════════════════════════════════════════════════════════
def fig_f():
    # Exact empirical rates verified from raw trial-level CSVs:
    # Qwen:  135 / 161 = 83.9%
    # Llama: 679 / 903 = 75.2%
    # Phi:   701 / 940 = 74.6%
    # Gemma: 955 / 960 = 99.5%
    profiles = {
        "Qwen 2.5 3B":     {"commit_rate": 14.7, "cond_acc": 83.9, "offset": (8, -2),   "ha": "left",  "va": "center"},
        "Llama 3.2 3B":    {"commit_rate": 97.8, "cond_acc": 75.2, "offset": (-10, 8),  "ha": "right", "va": "bottom"},
        "Phi-4 Mini 3.8B": {"commit_rate": 98.7, "cond_acc": 74.6, "offset": (-10, -14), "ha": "right", "va": "top"},
        "Gemma 3 4B":      {"commit_rate": 100.0,"cond_acc": 99.5, "offset": (-10, 6),  "ha": "right", "va": "bottom"},
    }

    fig, ax = plt.subplots(figsize=(5.8, 4.2))

    for m in MODELS:
        d = profiles[m["label"]]
        ax.scatter(d["commit_rate"], d["cond_acc"],
                   color=m["color"], marker=m["marker"], s=180, zorder=4,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(m["label"],
                    xy=(d["commit_rate"], d["cond_acc"]),
                    xytext=d["offset"],
                    textcoords="offset points",
                    fontsize=8.5, ha=d["ha"], va=d["va"], color=m["color"], fontweight="bold")

    ax.set_xlabel("Stage-1 Commitment Rate (% of valid trials)", fontsize=10)
    ax.set_ylabel(r"Stage-2 Conditional Dominant Accuracy (%)", fontsize=10)
    ax.set_xlim(-5, 112)
    ax.set_ylim(58, 108)
    ax.axvline(50, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.axhline(75, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(108, 60, "Descriptive observable rates;\nnot a model ranking.",
            fontsize=7, color="gray", ha="right", style="italic")

    fig.tight_layout()
    _save(fig, "FigF_BehavioralProfile")


# ── Helper ────────────────────────────────────────────────────────────────────
def _save(fig, name):
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Figure A: Response Distribution...")
    fig_a()
    print("Generating Figure B: Commitment Sensitivity...")
    fig_b()
    print("Generating Figure C: Conditional Direction...")
    fig_c()
    print("Generating Figure D: Position Moderation...")
    fig_d()
    print("Generating Figure E: C4 Numerical Control...")
    fig_e()
    print("Generating Figure F: Behavioral Profile...")
    fig_f()
    print("\nAll 6 figures generated successfully.")
