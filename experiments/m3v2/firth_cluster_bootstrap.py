#!/usr/bin/env python3
"""
Cluster-Aware Firth Logistic Regression Bootstrap
Paper: Stress-Testing Preference Elicitation in Instruction-Tuned Language Models (OJCS-2026-09-0983)
Addresses: Reviewer 2 Point 2 (Separation-robust fitting with dilemma-aware uncertainty)
           Gemma valid bootstrap CI under quasi-complete separation.

Resamples the 20 structural dilemmas with replacement (B = 2,000 resamples),
refits Firth (1993) penalized logistic regression on each resample, and
computes 95% percentile confidence intervals alongside the asymptotic penalized Wald CIs.
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))
from experiments.m3v2.firth_stage2_robustness import (
    load_model_data,
    fit_firth,
    prepare_stage2_matrix,
    MODELS,
    MODEL_NAMES,
    RESULTS_DIR
)

N_BOOT = 2000
RANDOM_SEED = 42

def run_firth_cluster_bootstrap(n_boot: int = N_BOOT, seed: int = RANDOM_SEED):
    print("=" * 80)
    print("CLUSTER-AWARE FIRTH PENALIZED LOGISTIC REGRESSION: 2,000 DILEMMA RESAMPLES")
    print("=" * 80)
    print(f"Resamples per model: {n_boot}")
    print(f"Cluster unit: 20 structural dilemmas (resampled with replacement)")
    print(f"Estimator: Firth (1993) Jeffreys penalized likelihood")
    print(f"Random seed: {seed}")
    print("=" * 80)

    np.random.seed(seed)
    all_results = {}

    t_global_start = time.time()

    for model_key in MODELS:
        m_name = MODEL_NAMES[model_key]
        print(f"\nEvaluating Model: {m_name} ({model_key})")
        print("-" * 60)

        df = load_model_data(model_key)
        s2 = df[
            (df["Y_commit"] == 1) & 
            (df["abs_log_Q"] > 0) & 
            df["Y_dominant"].notna()
        ].copy()

        dilemmas = np.sort(df["dilemma_id"].unique())
        n_dilemmas = len(dilemmas)
        stage2_n = len(s2)

        print(f"  Total Valid Dilemmas: {n_dilemmas}")
        print(f"  Stage 2 Committed Trials: {stage2_n}")

        # 1. Baseline full-sample Firth fit (with dilemma FE)
        X_full, y_full, colnames, _ = prepare_stage2_matrix(df, include_interaction=False)
        beta_full, se_full, conv_full = fit_firth(X_full, y_full)

        payoff_idx = colnames.index("abs_log_Q")
        b_hat = float(beta_full[payoff_idx])
        se_hat = float(se_full[payoff_idx])
        wald_ci_lo = b_hat - 1.96 * se_hat
        wald_ci_hi = b_hat + 1.96 * se_hat

        print(f"  Full-Sample Firth Estimate: beta = {b_hat:.4f}, SE = {se_hat:.4f}")
        print(f"  95% Penalized Wald CI: [{wald_ci_lo:.4f}, {wald_ci_hi:.4f}]")

        # 2. Cluster Bootstrap: Resample dilemmas with replacement
        boot_coefs_fe = []
        boot_coefs_nofe = []
        conv_failures = 0

        t0 = time.time()
        for b in range(n_boot):
            sampled_d = np.random.choice(dilemmas, size=n_dilemmas, replace=True)

            boot_chunks = []
            for i, d_id in enumerate(sampled_d):
                sub = s2[s2["dilemma_id"] == d_id].copy()
                if len(sub) > 0:
                    sub["cluster_inst"] = f"{d_id}_{i}"
                    boot_chunks.append(sub)

            if not boot_chunks:
                continue

            boot_df = pd.concat(boot_chunks, ignore_index=True)
            y_boot = boot_df["Y_dominant"].values

            # Design matrix WITH dilemma fixed effects (for unique dilemmas in resample)
            u_dil = sorted(boot_df["dilemma_id"].unique())
            fe_cols = [(boot_df["dilemma_id"] == d).astype(float).values for d in u_dil[1:]]
            if fe_cols:
                X_boot_fe = np.column_stack([
                    np.ones(len(boot_df)),
                    boot_df["abs_log_Q"].values,
                    boot_df["pos_effect"].values
                ] + fe_cols)
            else:
                X_boot_fe = np.column_stack([
                    np.ones(len(boot_df)),
                    boot_df["abs_log_Q"].values,
                    boot_df["pos_effect"].values
                ])

            try:
                b_fe, _, success_fe = fit_firth(X_boot_fe, y_boot)
                if success_fe and not np.isnan(b_fe[1]):
                    boot_coefs_fe.append(float(b_fe[1]))
                else:
                    conv_failures += 1
            except Exception:
                conv_failures += 1

            # Design matrix WITHOUT dilemma fixed effects (stabilized spec baseline)
            X_boot_nofe = np.column_stack([
                np.ones(len(boot_df)),
                boot_df["abs_log_Q"].values,
                boot_df["pos_effect"].values
            ])
            try:
                b_nofe, _, success_nofe = fit_firth(X_boot_nofe, y_boot)
                if success_nofe and not np.isnan(b_nofe[1]):
                    boot_coefs_nofe.append(float(b_nofe[1]))
            except Exception:
                pass

            if (b + 1) % 500 == 0 or (b + 1) == n_boot:
                elapsed = time.time() - t0
                rate = (b + 1) / elapsed
                print(f"    Resample {b+1:4d}/{n_boot} complete ({elapsed:.1f}s, {rate:.1f} resamples/s)")

        # 3. Percentile Confidence Intervals
        boot_fe_arr = np.array(boot_coefs_fe)
        ci_fe_lo = float(np.percentile(boot_fe_arr, 2.5))
        ci_fe_hi = float(np.percentile(boot_fe_arr, 97.5))
        mean_fe = float(np.mean(boot_fe_arr))
        std_fe = float(np.std(boot_fe_arr))
        median_fe = float(np.median(boot_fe_arr))

        boot_nofe_arr = np.array(boot_coefs_nofe)
        ci_nofe_lo = float(np.percentile(boot_nofe_arr, 2.5))
        ci_nofe_hi = float(np.percentile(boot_nofe_arr, 97.5))

        print(f"  --> Cluster-Bootstrap Percentile CI (with FE): [{ci_fe_lo:.4f}, {ci_fe_hi:.4f}] (mean={mean_fe:.4f}, sd={std_fe:.4f})")
        print(f"  --> Cluster-Bootstrap Percentile CI (no FE):   [{ci_nofe_lo:.4f}, {ci_nofe_hi:.4f}]")
        print(f"  --> Convergence Success: {len(boot_coefs_fe)}/{n_boot} (Failures: {conv_failures})")

        all_results[model_key] = {
            "model_name": m_name,
            "stage2_n": stage2_n,
            "n_boot": n_boot,
            "successful_replicates": len(boot_coefs_fe),
            "firth_beta": b_hat,
            "firth_se": se_hat,
            "penalized_wald_ci": [round(wald_ci_lo, 4), round(wald_ci_hi, 4)],
            "cluster_boot_percentile_ci_fe": [round(ci_fe_lo, 4), round(ci_fe_hi, 4)],
            "cluster_boot_percentile_ci_nofe": [round(ci_nofe_lo, 4), round(ci_nofe_hi, 4)],
            "boot_mean": round(mean_fe, 4),
            "boot_sd": round(std_fe, 4),
            "boot_median": round(median_fe, 4),
            "replicate_draws_fe": [round(x, 4) for x in boot_coefs_fe]
        }

    total_time = round(time.time() - t_global_start, 1)
    print(f"\n[DONE] Completed 2,000 resamples across all 4 models in {total_time}s")

    # 4. Save results to JSON
    out_json = RESULTS_DIR / "firth_cluster_bootstrap_2000.json"
    with open(out_json, "w") as f:
        # Save without raw draws for compact view
        summary_to_save = {}
        for k, v in all_results.items():
            d = dict(v)
            d.pop("replicate_draws_fe", None)
            summary_to_save[k] = d
        json.dump(summary_to_save, f, indent=2)
    print(f"[SAVED] Saved summary results to {out_json}")

    # Save full results with draws
    out_full = RESULTS_DIR / "firth_cluster_bootstrap_2000_full.json"
    with open(out_full, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[SAVED] Saved full distribution to {out_full}")

    # 5. Print Final Publication Comparison Table
    print("\n" + "=" * 90)
    print("FINAL COMPARISON: FIRTH POINT ESTIMATES, WALD CIs, AND CLUSTER BOOTSTRAP CIs")
    print("=" * 90)
    print(f"{'Model':<20} {'N':>6} {'Firth Beta':>12} {'95% Penalized Wald':>24} {'95% Cluster Boot CI':>24}")
    print("-" * 90)
    for model_key in MODELS:
        r = all_results[model_key]
        wald_str = f"[{r['penalized_wald_ci'][0]:.3f}, {r['penalized_wald_ci'][1]:.3f}]"
        boot_str = f"[{r['cluster_boot_percentile_ci_fe'][0]:.3f}, {r['cluster_boot_percentile_ci_fe'][1]:.3f}]"
        print(f"{r['model_name']:<20} {r['stage2_n']:>6} {r['firth_beta']:>12.4f} {wald_str:>24} {boot_str:>24}")
    print("=" * 90)

    return all_results

if __name__ == "__main__":
    run_firth_cluster_bootstrap()
