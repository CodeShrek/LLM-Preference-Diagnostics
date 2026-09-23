"""
M3 v2 Analysis — Goal-Conditioned Construct-Validity Test

Primary Analysis: Multinomial Logit (Add, Subtract, Indifferent)
Secondary Analysis: Hurdle model (Commitment, Conditional Direction)
Inference: Dilemma-level Cluster Bootstrap
"""
from __future__ import annotations

import argparse
import sys
import warnings
import math
from pathlib import Path

import numpy as np
import pandas as pd

MIN_EFFECT_SIZE = 0.5

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    _HAS_SM = True
except ImportError:
    _HAS_SM = False
    warnings.warn("statsmodels not installed.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _HAS_PLOT = True
except ImportError:
    _HAS_PLOT = False
    warnings.warn("matplotlib/seaborn not installed for plotting.")


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"underlying_choice", "log_Q_ratio", "add_is_option_a", "dilemma_id", "Q_ratio_label"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Missing columns: {missing}")
    
    # Recode underlying_choice strictly
    df = df[df["underlying_choice"].isin(["add", "subtract", "INDIFFERENT"])]
    df["pos_add_first"] = df["add_is_option_a"].astype(int)
    
    # Target variable for MNLogit: 0=Indifferent, 1=Add, 2=Subtract
    df["Y_multi"] = df["underlying_choice"].map({"INDIFFERENT": 0, "add": 1, "subtract": 2})
    
    # Hurdle Stage 1: Committed (1) vs Indifferent (0)
    df["Y_commit"] = (df["underlying_choice"] != "INDIFFERENT").astype(int)
    
    # Hurdle Stage 2: Dominant (1) vs Dominated (0) conditional on commitment
    # Only applicable for non-parity ratios
    def get_dominant(row):
        if row["underlying_choice"] == "INDIFFERENT" or row["log_Q_ratio"] == 0:
            return np.nan
        if row["log_Q_ratio"] > 0: # Add is dominant
            return 1 if row["underlying_choice"] == "add" else 0
        else: # Subtract is dominant
            return 1 if row["underlying_choice"] == "subtract" else 0
            
    df["Y_dominant"] = df.apply(get_dominant, axis=1)
    
    # Q magnitude for Stage 2 (since log_Q_ratio is signed, we want the absolute strength for dominant model)
    df["abs_log_Q"] = df["log_Q_ratio"].abs()
    
    return df


def cluster_bootstrap(df: pd.DataFrame, formula: str, target_col: str, model_type: str = "logit", n_boot: int = 500) -> dict:
    """
    Performs dilemma-level cluster bootstrap to estimate robust CIs for coefficients.
    """
    if not _HAS_SM:
        return {}
        
    dilemmas = df["dilemma_id"].unique()
    coefs = []
    
    for _ in range(n_boot):
        # Sample dilemmas with replacement
        boot_dilemmas = np.random.choice(dilemmas, size=len(dilemmas), replace=True)
        
        # Build bootstrap sample
        boot_df_list = []
        for i, d in enumerate(boot_dilemmas):
            d_df = df[df["dilemma_id"] == d].copy()
            d_df["dilemma_id"] = f"{d}_{i}"  # unique IDs for fixed effects
            boot_df_list.append(d_df)
            
        boot_df = pd.concat(boot_df_list)
        
        try:
            if model_type == "logit":
                res = smf.logit(formula, data=boot_df).fit(disp=False)
            elif model_type == "mnlogit":
                res = smf.mnlogit(formula, data=boot_df).fit(disp=False)
            coefs.append(res.params)
        except Exception:
            pass # Skip failed convergence
            
    if not coefs:
        return {}
        
    # Aggregate
    if model_type == "logit":
        boot_params = pd.DataFrame(coefs)
        summary = {}
        for col in boot_params.columns:
            vals = boot_params[col].dropna()
            if len(vals) < n_boot * 0.5:
                continue
            summary[col] = {
                "mean": vals.mean(),
                "std": vals.std(),
                "ci_lower": vals.quantile(0.025),
                "ci_upper": vals.quantile(0.975),
                "p_val": 2 * min((vals > 0).mean(), (vals < 0).mean())
            }
        return summary
    else:
        # mnlogit params are DataFrames
        return {"note": "MNLogit bootstrap completed but aggregation omitted for brevity."}


def permutation_test(df: pd.DataFrame, formula: str, target_col: str, n_perm: int = 500) -> dict:
    """
    Randomization test preserving experimental blocking structure.
    Permutes log_Q_ratio within (dilemma_id, phrasing_id, pos_add_first) blocks.
    """
    if not _HAS_SM:
        return {}
        
    try:
        # Fit observed
        orig_res = smf.logit(formula, data=df).fit(disp=False)
        orig_coef = orig_res.params.get("abs_log_Q", None)
        if orig_coef is None:
            return {}
            
        greater_count = 0
        df_perm = df.copy()
        block_cols = ["dilemma_id", "phrasing_id", "pos_add_first"]
        
        for _ in range(n_perm):
            # Permute abs_log_Q within blocks
            df_perm["abs_log_Q"] = df.groupby(block_cols)["abs_log_Q"].transform(np.random.permutation)
            
            perm_res = smf.logit(formula, data=df_perm).fit(disp=False)
            perm_coef = perm_res.params.get("abs_log_Q", 0.0)
            
            if abs(perm_coef) >= abs(orig_coef):
                greater_count += 1
                
        p_val = greater_count / n_perm
        return {"observed_coef": orig_coef, "p_val": p_val}
        
    except Exception as e:
        print(f"Permutation test failed: {e}")
        return {}

def analyze_models(df: pd.DataFrame) -> dict:
    print("=" * 60)
    print("STATISTICAL MODELING")
    print("=" * 60)
    
    harness_res = {}
    
    if not _HAS_SM:
        print("statsmodels not installed, skipping regression.")
        return harness_res
        
    main_df = df[df["condition"] == "main"].copy() if "condition" in df.columns else df.copy()
    if len(main_df) == 0:
        print("No 'main' condition data found.")
        return harness_res

    # 1. Primary Analysis: MNLogit (Y = Add/Sub/Indiff)
    print("--- Primary Analysis: MNLogit (Add/Sub/Indifferent) ---")
    formula_mn = "Y_multi ~ log_Q_ratio * pos_add_first + C(dilemma_id)"
    try:
        mn_model = smf.mnlogit(formula_mn, data=main_df).fit(disp=False)
        print("Note: Reference category is 0 (Indifferent)")
        print(mn_model.summary().tables[1])
    except Exception as e:
        print(f"MNLogit failed: {e}")
    print()

    # 2. Stage 1: Commitment
    print("--- H1: Commitment Sensitivity (Commit vs Indifferent) ---")
    formula_commit = "Y_commit ~ abs_log_Q * pos_add_first + C(dilemma_id)"
    try:
        commit_model = smf.logit(formula_commit, data=main_df).fit(disp=False)
        print(commit_model.summary().tables[1])
        
        # Bootstrap
        print("\nCluster Bootstrap (500 resamples):")
        boot_res = cluster_bootstrap(main_df, formula_commit, "Y_commit", "logit", 500)
        if "abs_log_Q" in boot_res:
            b = boot_res["abs_log_Q"]
            print(f"  abs_log_Q: β = {b['mean']:.3f}, 95% CI [{b['ci_lower']:.3f}, {b['ci_upper']:.3f}], bootstrap p={b['p_val']:.3f}")
            
        # Permutation Test
        print("\nBlocked Permutation Test (500 iterations):")
        perm_res = permutation_test(main_df, formula_commit, "Y_commit", 500)
        if perm_res:
            print(f"  abs_log_Q: p={perm_res['p_val']:.3f} (observed β={perm_res['observed_coef']:.3f})")
            
        # Categorical Robustness
        print("\nCategorical Q-ratio Robustness Check:")
        cat_formula = "Y_commit ~ C(abs_log_Q) * pos_add_first + C(dilemma_id)"
        cat_model = smf.logit(cat_formula, data=main_df).fit(disp=False)
        print("Model fitted. Log-Likelihood:", cat_model.llf)
    except Exception as e:
        print(f"Stage 1 failed: {e}")
    print()

    # 3. Stage 2: Conditional Direction
    print("--- H2: Directional Sensitivity (Dominant vs Dominated | Committed) ---")
    stage2_df = main_df.dropna(subset=["Y_dominant"]).copy()
    formula_dir = "Y_dominant ~ abs_log_Q * pos_add_first + C(dilemma_id)"
    if len(stage2_df) > 0:
        try:
            dir_model = smf.logit(formula_dir, data=stage2_df).fit(disp=False)
            print(dir_model.summary().tables[1])
            
            # Bootstrap
            print("\nCluster Bootstrap (500 resamples):")
            boot_res2 = cluster_bootstrap(stage2_df, formula_dir, "Y_dominant", "logit", 500)
            if "abs_log_Q" in boot_res2:
                b2 = boot_res2["abs_log_Q"]
                print(f"  abs_log_Q: β = {b2['mean']:.3f}, 95% CI [{b2['ci_lower']:.3f}, {b2['ci_upper']:.3f}], bootstrap p={b2['p_val']:.3f}")
                
            # Permutation Test
            print("\nBlocked Permutation Test (500 iterations):")
            perm_res2 = permutation_test(stage2_df, formula_dir, "Y_dominant", 500)
            if perm_res2:
                print(f"  abs_log_Q: p={perm_res2['p_val']:.3f} (observed β={perm_res2['observed_coef']:.3f})")
                
            # Categorical Robustness
            print("\nCategorical Q-ratio Robustness Check:")
            cat_formula2 = "Y_dominant ~ C(abs_log_Q) * pos_add_first + C(dilemma_id)"
            cat_model2 = smf.logit(cat_formula2, data=stage2_df).fit(disp=False)
            print("Model fitted. Log-Likelihood:", cat_model2.llf)
        except Exception as e:
            print(f"Stage 2 failed: {e}")
    else:
        print("Not enough committed non-parity trials for Stage 2.")
    print()

    # Legacy fallback for synthetic_validation.py
    main_df["Y_add"] = (main_df["underlying_choice"] == "add").astype(int)
    legacy_df = main_df[main_df["underlying_choice"].isin(["add", "subtract"])].copy()
    if len(legacy_df) > 0:
        try:
            leg = smf.logit("Y_add ~ log_Q_ratio + pos_add_first", data=legacy_df).fit(disp=False)
            harness_res = {
                "beta_Q": leg.params.get("log_Q_ratio", 0),
                "beta_pos": leg.params.get("pos_add_first", 0),
                "ci_excludes_zero": (leg.conf_int().loc["log_Q_ratio", 0] > 0) or (leg.conf_int().loc["log_Q_ratio", 1] < 0),
                "beta_Q_ci_low": leg.conf_int().loc["log_Q_ratio", 0],
                "beta_Q_ci_high": leg.conf_int().loc["log_Q_ratio", 1],
                "decision": "OUTCOME-SENSITIVE" if abs(leg.params.get("log_Q_ratio", 0)) > MIN_EFFECT_SIZE else "INDETERMINATE"
            }
        except Exception:
            pass

    return harness_res


def generate_plots(df: pd.DataFrame, output_dir: Path) -> None:
    if not _HAS_PLOT:
        return
        
    main_df = df[df["condition"] == "main"].copy() if "condition" in df.columns else df.copy()
    if len(main_df) == 0:
        return

    # Figure 1: P(Add), P(Sub), P(Indiff) vs log_Q_ratio by Position
    sns.set_theme(style="whitegrid")
    fig1, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    for i, (pos_val, ax, title) in enumerate([(1, axes[0], "Add Displayed First"), (0, axes[1], "Subtract Displayed First")]):
        sub = main_df[main_df["pos_add_first"] == pos_val]
        grp = sub.groupby("log_Q_ratio")["underlying_choice"].value_counts(normalize=True).unstack(fill_value=0)
        
        for col in ["add", "subtract", "INDIFFERENT"]:
            if col not in grp.columns:
                grp[col] = 0.0
                
        ax.plot(grp.index, grp["add"], marker='o', label="P(Add)", color="blue")
        ax.plot(grp.index, grp["subtract"], marker='s', label="P(Subtract)", color="red")
        ax.plot(grp.index, grp["INDIFFERENT"], marker='^', label="P(Indifferent)", color="gray")
        
        ax.set_title(title)
        ax.set_xlabel("log(Q_add / Q_sub)")
        if i == 0:
            ax.set_ylabel("Probability")
            ax.legend()

    fig1.suptitle("Figure 1: Choice Probabilities vs Q-Ratio")
    fig1.tight_layout()
    fig1.savefig(output_dir / "m3v2_fig1.png", dpi=300)
    plt.close(fig1)

    # Figure 2: Commitment vs Direction Decomposition
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: P(Commit) vs Q-ratio strength (abs_log_Q)
    grp_commit = main_df.groupby("abs_log_Q")["Y_commit"].mean()
    axes[0].plot(grp_commit.index, grp_commit.values, marker='o', color="purple")
    axes[0].set_title("A. P(Committed) vs Absolute Q-Ratio Strength")
    axes[0].set_xlabel("|log(Q_add / Q_sub)|")
    axes[0].set_ylabel("Probability of Committing")
    axes[0].set_ylim(0, 1)

    # Panel B: P(Dominant | Commit) vs Q-ratio strength
    stage2_df = main_df.dropna(subset=["Y_dominant"])
    grp_dir = stage2_df.groupby("abs_log_Q")["Y_dominant"].mean()
    axes[1].plot(grp_dir.index, grp_dir.values, marker='o', color="green")
    axes[1].set_title("B. P(Dominant | Committed) vs Absolute Q-Ratio Strength")
    axes[1].set_xlabel("|log(Q_add / Q_sub)|")
    axes[1].set_ylabel("Probability of Choosing Dominant")
    axes[1].set_ylim(0, 1)

    fig2.suptitle("Figure 2: Commitment vs Direction Decomposition")
    fig2.tight_layout()
    fig2.savefig(output_dir / "m3v2_fig2.png", dpi=300)
    plt.close(fig2)


def print_c4_summary(df: pd.DataFrame) -> None:
    if "condition" not in df.columns:
        return
    c4 = df[df["condition"] == "C4"].copy()
    if len(c4) == 0:
        return
        
    print("=" * 60)
    print("C4: NUMERICAL COMPREHENSION CONTROL")
    print("=" * 60)
    
    # Correct dominance recognition
    def c4_correct(row):
        if row["log_Q_ratio"] == 0:
            return row["underlying_choice"] == "INDIFFERENT"
        elif row["log_Q_ratio"] > 0:
            return row["underlying_choice"] == "add"
        else:
            return row["underlying_choice"] == "subtract"
            
    c4["correct"] = c4.apply(c4_correct, axis=1)
    acc = c4["correct"].mean()
    print(f"Overall C4 accuracy: {acc*100:.1f}% (n={len(c4)})")
    
    grp = c4.groupby("Q_ratio_label")["correct"].mean()
    print("\nAccuracy by ratio:")
    print(grp)
    print()


def analyze(csv_path: str) -> dict:
    df = load(csv_path)
    
    print("=" * 60)
    print(f"M3 v2 ANALYSIS: {csv_path}")
    print("=" * 60)
    
    print_c4_summary(df)
    res = analyze_models(df)
    
    out_dir = Path(csv_path).parent
    generate_plots(df, out_dir)
    print(f"Plots saved to {out_dir}")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M3 v2 statistical analysis (Post-Review).")
    parser.add_argument("csv_path", help="Path to m3v2 results CSV")
    args = parser.parse_args()
    analyze(args.csv_path)
