import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

MODELS = ["qwen25_3b", "llama32_3b", "phi4mini_38b", "gemma3_4b"]
MODEL_NAMES = {
    "qwen25_3b": "Qwen 2.5 3B",
    "llama32_3b": "Llama 3.2 3B",
    "phi4mini_38b": "Phi-4 Mini 3.8B",
    "gemma3_4b": "Gemma 3 4B"
}

def load_and_prep(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["underlying_choice"].isin(["add", "subtract", "INDIFFERENT"])].copy()
    df["pos_add_first"] = df["add_is_option_a"].astype(int)
    df["Y_multi"] = df["underlying_choice"].map({"INDIFFERENT": 0, "add": 1, "subtract": 2})
    df["Y_commit"] = (df["underlying_choice"] != "INDIFFERENT").astype(int)
    
    def get_dominant(row):
        if row["underlying_choice"] == "INDIFFERENT" or row["log_Q_ratio"] == 0:
            return np.nan
        if row["log_Q_ratio"] > 0:
            return 1 if row["underlying_choice"] == "add" else 0
        else:
            return 1 if row["underlying_choice"] == "subtract" else 0
            
    df["Y_dominant"] = df.apply(get_dominant, axis=1)
    df["abs_log_Q"] = df["log_Q_ratio"].abs()
    return df

def cluster_bootstrap(df: pd.DataFrame, formula: str, target_col: str, n_boot: int = 500):
    dilemmas = df["dilemma_id"].unique()
    coefs = []
    
    for _ in range(n_boot):
        boot_dilemmas = np.random.choice(dilemmas, size=len(dilemmas), replace=True)
        boot_df_list = []
        for i, d in enumerate(boot_dilemmas):
            d_df = df[df["dilemma_id"] == d].copy()
            d_df["dilemma_id"] = f"{d}_{i}"
            boot_df_list.append(d_df)
        boot_df = pd.concat(boot_df_list)
        try:
            res = smf.logit(formula, data=boot_df).fit(disp=False)
            coefs.append(res.params["abs_log_Q"])
        except:
            pass
            
    if not coefs: return None, None
    vals = pd.Series(coefs).dropna()
    if len(vals) < n_boot * 0.5: return None, None
    return vals.quantile(0.025), vals.quantile(0.975)

def analyze_all():
    summary_data = []
    all_main_dfs = []
    all_c4_dfs = []
    
    # Process each model separately
    for mk in MODELS:
        print(f"--- Analyzing {mk} ---")
        
        main_df = load_and_prep(f"results/{mk}/primary.csv")
        main_df["model_id"] = mk
        all_main_dfs.append(main_df)
        
        c4_df = load_and_prep(f"results/{mk}/c4.csv")
        all_c4_dfs.append(c4_df)
        
        # Sample Size Accounting
        n_total = len(main_df)
        n_indiff = len(main_df[main_df["Y_commit"] == 0])
        n_directional = len(main_df[main_df["Y_commit"] == 1])
        n_parity_directional = len(main_df[(main_df["Y_commit"] == 1) & (main_df["log_Q_ratio"] == 0)])
        n_stage2 = len(main_df.dropna(subset=["Y_dominant"]))
        
        # C4 Accuracy
        def c4_correct(row):
            if row["log_Q_ratio"] == 0: return row["underlying_choice"] == "INDIFFERENT"
            elif row["log_Q_ratio"] > 0: return row["underlying_choice"] == "add"
            else: return row["underlying_choice"] == "subtract"
        c4_df["correct"] = c4_df.apply(c4_correct, axis=1)
        c4_acc = c4_df["correct"].mean()
        c4_correct_n = int(c4_df["correct"].sum())
        c4_total_n = len(c4_df)
        
        # Stage 1: Commitment
        formula_commit = "Y_commit ~ abs_log_Q * pos_add_first + C(dilemma_id)"
        commit_b = np.nan; commit_p = np.nan
        try:
            # We cluster-robust se might fail due to strict statsmodels requirement on matching groups.
            # But we can at least get the fixed effects MLE.
            res1 = smf.logit(formula_commit, data=main_df).fit(disp=False)
            commit_b = res1.params["abs_log_Q"]
            commit_p = res1.pvalues["abs_log_Q"]
        except: pass
        
        # Stage 2: MLE & Bootstrap
        stage2_df = main_df.dropna(subset=["Y_dominant"]).copy()
        formula_dir = "Y_dominant ~ abs_log_Q * pos_add_first + C(dilemma_id)"
        formula_dir_stabilized = "Y_dominant ~ abs_log_Q + pos_add_first" # Regularized to avoid separation
        dir_b = np.nan; dir_p = np.nan; ci_low = np.nan; ci_high = np.nan
        if len(stage2_df) > 0:
            try:
                res2 = smf.logit(formula_dir, data=stage2_df).fit(disp=False)
                dir_b = res2.params["abs_log_Q"]
                dir_p = res2.pvalues["abs_log_Q"]
            except: pass
            try:
                ci_low, ci_high = cluster_bootstrap(stage2_df, formula_dir_stabilized, "Y_dominant", n_boot=1000) # Increased boot for better CI
            except: pass
            
        # Threshold Effect
        thresh_df = load_and_prep(f"results/{mk}/threshold.csv")
        thresh_stage2 = thresh_df.dropna(subset=["Y_dominant"]).copy()
        thresh_b = np.nan; thresh_p = np.nan
        if len(thresh_stage2) > 0:
            try:
                # Use stabilized model to avoid separation in threshold sample too
                res_t = smf.logit(formula_dir_stabilized, data=thresh_stage2).fit(disp=False)
                thresh_b = res_t.params["abs_log_Q"]
                thresh_p = res_t.pvalues["abs_log_Q"]
            except: pass
            
        # Position Effect (Interaction in Stage 1 & 2)
        pos_int_s1 = np.nan; pos_int_s2 = np.nan
        try: pos_int_s1 = res1.params["abs_log_Q:pos_add_first"]
        except: pass
        try: pos_int_s2 = res2.params["abs_log_Q:pos_add_first"]
        except: pass
        
        summary_data.append({
            "model_id": mk,
            "N_Total": n_total,
            "N_Indiff": n_indiff,
            "N_Directional": n_directional,
            "N_Parity_Directional": n_parity_directional,
            "N_Stage2": n_stage2,
            "C4_correct": c4_correct_n,
            "C4_total": c4_total_n,
            "C4_accuracy": c4_acc,
            "Stage1_Commit_Beta": commit_b,
            "Stage1_Commit_P": commit_p,
            "Stage2_Direction_Beta": dir_b,
            "Stage2_Direction_P": dir_p,
            "Stage2_Boot_CI_Low": ci_low,
            "Stage2_Boot_CI_High": ci_high,
            "Threshold_Beta": thresh_b,
            "Threshold_P": thresh_p,
            "Pos_Interaction_S1": pos_int_s1,
            "Pos_Interaction_S2": pos_int_s2
        })
        
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv("results/multimodel_summary.csv", index=False)
    
    with open("results/multimodel_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
        
    print(df_summary)
    
    # Combine data for plotting
    combined_main = pd.concat(all_main_dfs, ignore_index=True)
    
    # ---------------------------------------------------------
    # Generate Figures
    # ---------------------------------------------------------
    sns.set_theme(style="whitegrid")
    
    # Figure A: Stacked Response Distribution by Model
    plt.figure(figsize=(8, 6))
    counts = combined_main.groupby(["model_id", "underlying_choice"]).size().unstack(fill_value=0)
    # Order models
    counts = counts.loc[MODELS]
    # Reorder columns
    cols = []
    for c in ["add", "subtract", "INDIFFERENT"]:
        if c in counts.columns: cols.append(c)
    counts = counts[cols]
    
    counts_pct = counts.div(counts.sum(axis=1), axis=0) * 100
    counts_pct.index = counts_pct.index.map(MODEL_NAMES)
    counts_pct.plot(kind="bar", stacked=True, color=["#2ca02c", "#d62728", "#7f7f7f"], figsize=(8, 6))
    # plt.title("A. Response Distribution by Model")
    plt.ylabel("Percentage of Valid Trials")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.legend(title="Choice", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("results/FigA_ResponseDist.pdf")
    plt.savefig("results/FigA_ResponseDist.png")
    plt.close()

    # Figure B: Commitment Sensitivity
    plt.figure(figsize=(8, 6))
    for mk in MODELS:
        sub = combined_main[combined_main["model_id"] == mk]
        grp = sub.groupby("abs_log_Q")["Y_commit"].mean()
        plt.plot(grp.index, grp.values, marker='s', label=MODEL_NAMES[mk])
    # plt.title("B. Cross-Model: Commitment Sensitivity")
    plt.xlabel("|log(Q_add / Q_sub)|")
    plt.ylabel("P(Commit)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/FigB_Commitment.pdf")
    plt.savefig("results/FigB_Commitment.png")
    plt.close()
    
    # Figure C: Conditional Directional Sensitivity
    plt.figure(figsize=(8, 6))
    for mk in MODELS:
        sub = combined_main[(combined_main["model_id"] == mk) & (combined_main["Y_commit"] == 1)].dropna(subset=["Y_dominant"])
        if len(sub) > 0:
            grp = sub.groupby("abs_log_Q")["Y_dominant"].mean()
            plt.plot(grp.index, grp.values, marker='^', label=MODEL_NAMES[mk])
    # plt.title("C. Cross-Model: Conditional Directional Sensitivity")
    plt.xlabel("|log(Q_add / Q_sub)|")
    plt.ylabel("P(Dominant | Committed)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/FigC_Conditional.pdf")
    plt.savefig("results/FigC_Conditional.png")
    plt.close()
    
    # Figure D: Signed Position Interaction
    plt.figure(figsize=(8, 5))
    models_sorted = df_summary["model_id"].map(MODEL_NAMES)
    int_s2 = df_summary["Pos_Interaction_S2"]
    sns.barplot(x=models_sorted, y=int_s2, hue=models_sorted, palette="coolwarm", legend=False)
    plt.axhline(0, color='black', linewidth=1)
    # plt.title("D. Presentation Position Effect (Interaction $\\beta$)")
    plt.ylabel("Stage 2 Position Interaction Coefficient")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig("results/FigD_PositionBias.pdf")
    plt.savefig("results/FigD_PositionBias.png")
    plt.close()

    # Figure E: Exact counts C4 Accuracy
    plt.figure(figsize=(8, 5))
    acc = df_summary["C4_accuracy"] * 100
    ax = sns.barplot(x=models_sorted, y=acc, hue=models_sorted, palette="mako", legend=False)
    plt.axhline(100, color='red', linestyle='--', alpha=0.7)
    # plt.title("E. Numerical Comprehension (C4) Accuracy")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Model")
    plt.ylim(0, 115)
    
    for i, p in enumerate(ax.patches):
        c4_c = df_summary["C4_correct"].iloc[i]
        c4_t = df_summary["C4_total"].iloc[i]
        ax.annotate(f"{c4_c}/{c4_t}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10, xytext=(0, 5), textcoords='offset points')
                    
    plt.tight_layout()
    plt.savefig("results/FigE_NumericalControl.pdf")
    plt.savefig("results/FigE_NumericalControl.png")
    plt.close()

    # Figure F: 2D Behavioral Profile
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=df_summary, x="Stage1_Commit_Beta", y="Stage2_Direction_Beta", 
                    hue=models_sorted, s=200, palette="husl", legend=False)
    for i, row in df_summary.iterrows():
        plt.text(row["Stage1_Commit_Beta"], row["Stage2_Direction_Beta"] + 0.1, 
                 f'{MODEL_NAMES[row["model_id"]]}\nC4: {row["C4_accuracy"]*100:.1f}%', 
                 ha='center', va='bottom', fontsize=10)
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    # plt.title("F. Two-Dimensional Behavioral Profile")
    plt.xlabel("Stage 1 Commitment Sensitivity ($\\beta$)")
    plt.ylabel("Stage 2 Conditional Directional Sensitivity ($\\beta$)")
    xlims = plt.xlim()
    ylims = plt.ylim()
    plt.xlim(xlims[0]-0.5, xlims[1]+0.5)
    plt.ylim(ylims[0]-0.5, ylims[1]+0.5)
    plt.tight_layout()
    plt.savefig("results/FigF_BehavioralProfile.pdf")
    plt.savefig("results/FigF_BehavioralProfile.png")
    plt.close()

if __name__ == "__main__":
    analyze_all()
