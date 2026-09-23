import os
import sys
import argparse
import subprocess
import json
import hashlib
from pathlib import Path
import pandas as pd
import shutil

MODELS = {
    "qwen25_3b": "qwen2.5:3b-instruct-q4_K_M",
    "llama32_3b": "llama3.2:3b-instruct-q4_K_M",
    "phi4mini_38b": "phi4-mini:3.8b-q4_K_M",
    "gemma3_4b": "gemma3:4b-it-q4_K_M"
}

def get_design_matrix(csv_path):
    df = pd.read_csv(csv_path)
    return df[['dilemma_id', 'phrasing_id', 'Q_ratio_label', 'add_is_option_a']].sort_values(
        ['dilemma_id', 'phrasing_id', 'Q_ratio_label', 'add_is_option_a']
    ).reset_index(drop=True)

def run_experiment(model_key, ollama_tag, condition, stimuli_path, output_csv, is_mock=False):
    print(f"\n--- Running {condition} for {model_key} ---")
    cmd = [
        sys.executable, "experiments/m3v2/m3v2_runner.py",
        "--condition", condition,
        "--stimuli", stimuli_path,
        "--output", output_csv,
        "--model-key", model_key,
        "--ollama-model", ollama_tag,
        "--backend", "ollama",
        "--sleep-seconds", "0.0" if is_mock else "0.1"
    ]
    if is_mock:
        cmd.append("--mock")
    
    subprocess.run(cmd, check=True)
    print(f"Finished {condition} for {model_key}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock-only", action="store_true")
    args = parser.parse_args()

    # 1. MOCK RUN & FACTORIAL VALIDATION
    print("=== PHASE 3: MOCK RUN & FACTORIAL VALIDATION ===")
    os.makedirs("results/mock_runs", exist_ok=True)
    mock_csvs = {}
    
    # We will test "main" for the prompt hashes, since it contains both T1 and T2
    for mk, tag in MODELS.items():
        csv_path = f"results/mock_runs/{mk}_main.csv"
        # Always remove mock run CSVs first to ensure fresh deterministic runs
        if os.path.exists(csv_path):
            os.remove(csv_path)
        run_experiment(mk, tag, "main", "experiments/m3v2/m3v2_stimuli.json", csv_path, is_mock=True)
        mock_csvs[mk] = csv_path

    # Compare design matrices
    base_mk = "qwen25_3b"
    base_matrix = get_design_matrix(mock_csvs[base_mk])
    
    assert len(base_matrix) == 1120, f"Expected 1120 trials, got {len(base_matrix)}"
    
    for mk in MODELS:
        if mk == base_mk: continue
        comp_matrix = get_design_matrix(mock_csvs[mk])
        if not base_matrix.equals(comp_matrix):
            raise AssertionError(f"Design matrix mismatch between {base_mk} and {mk}")
        print(f"Design matrix exactly matches: {base_mk} == {mk}")

    # Check cell counts for base
    counts_ratio = base_matrix['Q_ratio_label'].value_counts()
    for ratio, count in counts_ratio.items():
        assert count == 160, f"Expected 160 for {ratio}, got {count}"
    
    counts_pos = base_matrix['add_is_option_a'].value_counts()
    assert counts_pos[True] == 560
    assert counts_pos[False] == 560
    print("Factorial cells perfectly balanced (160 per ratio, 560 per position).")

    # Hash equality check
    print("Checking rendered prompt hashes across models (Qwen == Llama == Phi == Gemma)...")
    base_df = pd.read_csv(mock_csvs[base_mk]).sort_values("trial_id").reset_index(drop=True)
    
    for mk in MODELS:
        if mk == base_mk: continue
        comp_df = pd.read_csv(mock_csvs[mk]).sort_values("trial_id").reset_index(drop=True)
        
        # Check T1 hashes
        t1_diffs = base_df['t1_prompt_hash'] != comp_df['t1_prompt_hash']
        if t1_diffs.any():
            raise AssertionError(f"T1 Prompt Hash mismatch between {base_mk} and {mk}")
            
        # Check T2 hashes (only for trials where T2 was actually called)
        # We need to only check non-skipped T2s
        mask_t2 = base_df['t2_type'] != 'skipped'
        t2_diffs = base_df.loc[mask_t2, 't2_prompt_hash'] != comp_df.loc[mask_t2, 't2_prompt_hash']
        if t2_diffs.any():
            raise AssertionError(f"T2 Prompt Hash mismatch between {base_mk} and {mk}")
            
        print(f"Prompt hashes for T1 & T2 exactly match: {base_mk} == {mk}")

    if args.mock_only:
        print("Mock validation passed. Exiting.")
        return

    # 2. FULL EXECUTION
    print("\n=== PHASE 4-6: FULL EXECUTION ===")
    
    # Generate threshold stimuli
    thresh_stimuli_path = 'experiments/followups/threshold_relevance/m3v2_stimuli_thresh10.json'
    if not os.path.exists(thresh_stimuli_path):
        subprocess.run([sys.executable, "experiments/followups/threshold_relevance/run_threshold_experiment.py"], check=True)

    # Immutable Qwen Baseline Check
    os.makedirs("results/qwen25_3b", exist_ok=True)
    if os.path.exists("results/m3v2_main.csv") and not os.path.exists("results/qwen25_3b/primary.csv"):
        shutil.copy("results/m3v2_main.csv", "results/qwen25_3b/primary.csv")
    if os.path.exists("results/m3v2_C4.csv") and not os.path.exists("results/qwen25_3b/c4.csv"):
        shutil.copy("results/m3v2_C4.csv", "results/qwen25_3b/c4.csv")
    if os.path.exists("results/followups/threshold_relevance_main.csv") and not os.path.exists("results/qwen25_3b/threshold.csv"):
        shutil.copy("results/followups/threshold_relevance_main.csv", "results/qwen25_3b/threshold.csv")

    for mk, tag in MODELS.items():
        os.makedirs(f"results/{mk}", exist_ok=True)
        
        # Primary
        run_experiment(mk, tag, "main", "experiments/m3v2/m3v2_stimuli.json", f"results/{mk}/primary.csv", is_mock=False)
        
        # C4
        run_experiment(mk, tag, "C4", "experiments/m3v2/m3v2_stimuli.json", f"results/{mk}/c4.csv", is_mock=False)
        
        # Threshold
        run_experiment(mk, tag, "main", thresh_stimuli_path, f"results/{mk}/threshold.csv", is_mock=False)

if __name__ == "__main__":
    main()
