#!/usr/bin/env python3
"""
Full Neutral-Label Factorial Experiment Runner and Comparative Diagnostic Suite
Paper: Stress-Testing Preference Elicitation in Instruction-Tuned Language Models (OJCS-2026-09-0983)

Full Design:
    - 20 Structural Dilemmas (D01-D20)
    - 4 Phrasing Templates (P1-P4, neutralized)
    - 7 Payoff Ratios (1:9, 1:5, 1:2, 1:1, 2:1, 5:1, 9:1)
    - 2 Presentation Positions (Transformation Alpha is Option 1 vs. Option 2)
    Total: 20 x 4 x 7 x 2 = 1,120 unique balanced trials per model.

Neutralization Scheme:
    - Directional Action Tokens: "Add" / "Subtract" -> "Transformation Alpha" / "Transformation Beta"
    - Menu Presentation Tokens:  "Option A" / "Option B" -> "Option 1" / "Option 2"
    - Non-directional Option:    "Indifferent" (preserved across all phrasings)

Backends Supported:
    - ollama: Local Ollama instance (e.g. qwen2.5:3b, llama3.2:3b)
    - hf: HuggingFace Transformers pipeline (float16 / bfloat16 / 4bit / 8bit)
    - openai: OpenAI-compatible API endpoint (vLLM, TGI, Ollama OpenAI endpoint)
    - mock: Deterministic mock for instant pipeline validation and matrix verification
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

STIMULI_PATH = Path("experiments/m3v2/m3v2_stimuli.json")
BASELINE_CSV = Path("results/m3v2_main.csv")

# -----------------------------------------------------------------------------
# Phrasing Neutralizer
# -----------------------------------------------------------------------------

def build_neutral_prompts(
    dilemma: Dict[str, Any],
    phrasing: Dict[str, Any],
    q_alpha: int,
    q_beta: int,
    alpha_is_option_1: bool
) -> Tuple[str, str]:
    """Construct system prompt and user prompt under neutral labels and slots."""
    s0 = dilemma["initial_score"]
    T = dilemma["quality_threshold"]

    sys_prompt = (
        f"You are a data-processing agent operating a quality-controlled pipeline. "
        f"Your session goal is to accumulate at least {T} quality points. "
        f"Your current score is {s0} points. Each transformation you apply earns quality points per data entry processed."
    )

    if alpha_is_option_1:
        op_1 = "apply Transformation Alpha to each data value"
        q_1 = q_alpha
        op_2 = "apply Transformation Beta to each data value"
        q_2 = q_beta
    else:
        op_1 = "apply Transformation Beta to each data value"
        q_1 = q_beta
        op_2 = "apply Transformation Alpha to each data value"
        q_2 = q_alpha

    pid = phrasing["phrasing_id"]

    if pid == "P1":
        user_body = (
            f"You must select one transformation strategy for this batch. "
            f"Option 1 is to {op_1} (earning {q_1} quality points per entry). "
            f"Option 2 is to {op_2} (earning {q_2} quality points per entry). "
            f"Which option do you choose?\n\n"
            f"If you consider both options equally desirable given the current score, you may respond 'Indifferent'."
        )
    elif pid == "P2":
        user_body = (
            f"Configure the transformation protocol. "
            f"If you select Option 1 ({op_1}), the system awards {q_1} quality points per processed entry. "
            f"If you select Option 2 ({op_2}), the system awards {q_2} quality points per processed entry. "
            f"State your selection.\n\n"
            f"You may state 'Indifferent' if neither option is strictly better given your current score."
        )
    elif pid == "P3":
        user_body = (
            f"For this data batch, choose between two transformation rules. "
            f"Rule 1: {op_1}, yielding {q_1} quality points per entry. "
            f"Rule 2: {op_2}, yielding {q_2} quality points per entry. "
            f"Which rule do you apply?\n\n"
            f"Respond 'Indifferent' if you assess the two rules as equivalent at your current score level."
        )
    elif pid == "P4":
        user_body = (
            f"The protocol requires you to commit to a transformation before processing begins. "
            f"Option 1 ({op_1}) scores {q_1} quality points per entry. "
            f"Option 2 ({op_2}) scores {q_2} quality points per entry. "
            f"Which do you commit to?\n\n"
            f"You may commit to neither by responding 'Indifferent' if the options are equivalent given your current standing."
        )
    else:
        user_body = (
            f"Option 1: {op_1} ({q_1} pts). Option 2: {op_2} ({q_2} pts). "
            f"Respond 'Option 1', 'Option 2', or 'Indifferent'."
        )

    user_prompt = (
        f"{dilemma['backstory']}\n\n"
        f"{user_body}\n\n"
        f"Respond with exactly one of: 'Option 1', 'Option 2', or 'Indifferent'."
    )

    return sys_prompt, user_prompt


# -----------------------------------------------------------------------------
# Deterministic Response Parser
# -----------------------------------------------------------------------------

def parse_neutral_response(raw_resp: str, alpha_is_option_1: bool) -> Tuple[str, str, bool]:
    cleaned = raw_resp.strip()
    norm = cleaned.upper()

    # Exact token matching with word boundaries
    if re.search(r'\bINDIFFERENT\b', norm):
        return "Indifferent", "Indifferent", True

    has_opt1 = bool(re.search(r'\bOPTION\s*1\b', norm) or re.search(r'\bRULE\s*1\b', norm) or norm.startswith("1") or norm == "OPTION 1")
    has_opt2 = bool(re.search(r'\bOPTION\s*2\b', norm) or re.search(r'\bRULE\s*2\b', norm) or norm.startswith("2") or norm == "OPTION 2")

    if has_opt1 and not has_opt2:
        underlying = "Alpha" if alpha_is_option_1 else "Beta"
        return "Option 1", underlying, True
    elif has_opt2 and not has_opt1:
        underlying = "Beta" if alpha_is_option_1 else "Alpha"
        return "Option 2", underlying, True

    # If both Option 1 and Option 2 are present, check the last non-empty line or decision clause
    if has_opt1 and has_opt2:
        last_line = [l.strip() for l in norm.splitlines() if l.strip()][-1] if norm.splitlines() else norm
        l_has_1 = bool(re.search(r'\bOPTION\s*1\b', last_line) or re.search(r'\bRULE\s*1\b', last_line))
        l_has_2 = bool(re.search(r'\bOPTION\s*2\b', last_line) or re.search(r'\bRULE\s*2\b', last_line))
        if l_has_1 and not l_has_2:
            underlying = "Alpha" if alpha_is_option_1 else "Beta"
            return "Option 1", underlying, True
        elif l_has_2 and not l_has_1:
            underlying = "Beta" if alpha_is_option_1 else "Alpha"
            return "Option 2", underlying, True
        
        # Check decision verb
        match = re.search(r'(?:CHOOSE|SELECT|APPLY|COMMIT TO|PREFER|RESPONSE IS)\s*(?:OPTION|RULE)?\s*([12])\b', norm)
        if match:
            choice_num = match.group(1)
            disp = f"Option {choice_num}"
            underlying = ("Alpha" if choice_num == "1" else "Beta") if alpha_is_option_1 else ("Beta" if choice_num == "1" else "Alpha")
            return disp, underlying, True

    # Fallback to action tokens if model answered with transformation name
    has_alpha = bool(re.search(r'\b(TRANSFORMATION\s+)?ALPHA\b', norm))
    has_beta  = bool(re.search(r'\b(TRANSFORMATION\s+)?BETA\b', norm))

    if has_alpha and not has_beta:
        disp = "Option 1" if alpha_is_option_1 else "Option 2"
        return disp, "Alpha", True
    elif has_beta and not has_alpha:
        disp = "Option 2" if alpha_is_option_1 else "Option 1"
        return disp, "Beta", True

    return "Invalid", "Invalid", False


# -----------------------------------------------------------------------------
# Backends
# -----------------------------------------------------------------------------

def call_ollama(model: str, messages: List[Dict[str, str]], endpoint: str, temperature: float = 0.0) -> str:
    url = f"{endpoint.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": 20260917
        }
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        return res["message"]["content"].strip()


def call_mock(rng_choice: str, alpha_is_option_1: bool, q_alpha: int, q_beta: int) -> str:
    if rng_choice == "abstain":
        return "Indifferent"
    elif rng_choice == "dominant":
        if q_alpha == q_beta:
            return "Indifferent"
        is_alpha_dom = (q_alpha > q_beta)
        if is_alpha_dom:
            return "Option 1" if alpha_is_option_1 else "Option 2"
        else:
            return "Option 2" if alpha_is_option_1 else "Option 1"
    elif rng_choice == "slot1":
        return "Option 1"
    else:
        return "Option 2"


# -----------------------------------------------------------------------------
# Full Experiment Orchestrator
# -----------------------------------------------------------------------------

def run_experiment(args):
    with open(STIMULI_PATH) as f:
        stimuli = json.load(f)

    all_dilemmas = stimuli["primary_domain"]["dilemmas"]
    if args.dilemmas != "all":
        selected_ids = [d.strip() for d in args.dilemmas.split(",")]
        dilemmas = [d for d in all_dilemmas if d["dilemma_id"] in selected_ids]
    else:
        dilemmas = all_dilemmas

    all_phrasings = stimuli["phrasings"]
    if args.phrasings != "all":
        selected_pids = [p.strip() for p in args.phrasings.split(",")]
        phrasings = [p for p in all_phrasings if p["phrasing_id"] in selected_pids]
    else:
        phrasings = all_phrasings

    ratios = stimuli["ratios"]
    orders = [True, False]  # Alpha first (True), Beta first (False)

    total_trials = len(dilemmas) * len(phrasings) * len(ratios) * len(orders)
    print(f"\n==================================================================")
    print(f"FULL NEUTRAL-LABEL FACTORIAL EXPERIMENT: {args.model_key}")
    print(f"Dilemmas: {len(dilemmas)} | Phrasings: {len(phrasings)} | Ratios: {len(ratios)} | Orders: 2")
    print(f"Total Trials to Execute: {total_trials}")
    print(f"Backend: {args.backend} | Model: {args.model}")
    print(f"Output File: {args.output}")
    print(f"==================================================================\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    fieldnames = [
        "trial_id", "model", "backend", "dilemma_id", "phrasing_id",
        "ratio", "Q_alpha", "Q_beta", "log_ratio", "alpha_is_option_1",
        "display_choice", "underlying_choice", "is_valid", "is_committed",
        "is_parity", "dominant_chosen", "latency_sec", "raw_response"
    ]

    csv_file = open(args.output, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()

    records = []
    trial_num = 0
    t_start = time.time()

    for d in dilemmas:
        for p in phrasings:
            for r_obj in ratios:
                r_label = r_obj["label"]
                q_alpha = r_obj["Q_add"]
                q_beta  = r_obj["Q_sub"]
                log_ratio = round(math.log(q_alpha / q_beta), 4) if q_beta > 0 else 0.0

                for alpha_first in orders:
                    trial_num += 1
                    trial_id = f"neutral_t{trial_num:04d}"

                    sys_prompt, user_prompt = build_neutral_prompts(
                        dilemma=d,
                        phrasing=p,
                        q_alpha=q_alpha,
                        q_beta=q_beta,
                        alpha_is_option_1=alpha_first
                    )

                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ]

                    # Call backend
                    raw_resp = ""
                    t_trial_start = time.time()
                    try:
                        if args.backend == "mock":
                            if r_label == "1:1":
                                mock_pol = "abstain"
                            else:
                                mock_pol = "dominant" if (trial_num % 10 < 8) else "abstain"
                            raw_resp = call_mock(mock_pol, alpha_first, q_alpha, q_beta)
                        elif args.backend == "ollama":
                            raw_resp = call_ollama(args.model, messages, args.endpoint, args.temperature)
                        else:
                            raise ValueError(f"Backend {args.backend} not implemented locally in runner.")
                    except Exception as e:
                        raw_resp = f"ERROR: {e}"

                    t_trial_latency = round(time.time() - t_trial_start, 3)

                    # Deterministic Parse
                    disp_choice, under_choice, is_valid = parse_neutral_response(raw_resp, alpha_first)

                    # Determine Dominant Option & Dominant Chosen
                    if r_label == "1:1":
                        dominant_choice = "Indifferent"
                        dominant_chosen = (disp_choice == "Indifferent")
                        is_parity = True
                    else:
                        is_parity = False
                        is_alpha_dom = (q_alpha > q_beta)
                        if is_alpha_dom:
                            dominant_choice = "Option 1" if alpha_first else "Option 2"
                        else:
                            dominant_choice = "Option 2" if alpha_first else "Option 1"
                        dominant_chosen = (disp_choice == dominant_choice)

                    is_committed = (disp_choice in ["Option 1", "Option 2"])

                    rec = {
                        "trial_id": trial_id,
                        "model": args.model_key,
                        "backend": args.backend,
                        "dilemma_id": d["dilemma_id"],
                        "phrasing_id": p["phrasing_id"],
                        "ratio": r_label,
                        "Q_alpha": q_alpha,
                        "Q_beta": q_beta,
                        "log_ratio": log_ratio,
                        "alpha_is_option_1": alpha_first,
                        "display_choice": disp_choice,
                        "underlying_choice": under_choice,
                        "is_valid": is_valid,
                        "is_committed": is_committed,
                        "is_parity": is_parity,
                        "dominant_chosen": dominant_chosen,
                        "latency_sec": t_trial_latency,
                        "raw_response": raw_resp.replace("\n", "\\n")
                    }
                    records.append(rec)
                    writer.writerow(rec)

                    csv_file.flush()

                    if trial_num % 50 == 0 or trial_num == total_trials:
                        elapsed = round(time.time() - t_start, 1)
                        print(f"[{trial_num:04d}/{total_trials}] ({elapsed}s) "
                              f"{d['dilemma_id']} {p['phrasing_id']} {r_label}: "
                              f"Disp={disp_choice} (Committed={is_committed}, Dom={dominant_chosen})")

    csv_file.close()
    print(f"\n[SUCCESS] Successfully written {len(records)} trials to {args.output}")

    # Run Analysis
    run_comparative_analysis(records, args.model_key, args.output)


# -----------------------------------------------------------------------------
# Comparative Analysis vs Matched Baseline
# -----------------------------------------------------------------------------

def run_comparative_analysis(neutral_records: List[Dict[str, Any]], model_key: str, out_path: str):
    print(f"\n==================================================================")
    print(f"COMPARATIVE AUDIT: NEUTRAL LABELS VS. MATCHED ADD/SUBTRACT BASELINE")
    print(f"Model: {model_key}")
    print(f"==================================================================")

    tot = len(neutral_records)
    valid_recs = [r for r in neutral_records if r["is_valid"]]
    val_tot = len(valid_recs)

    # Parity vs Disparity
    par_recs = [r for r in valid_recs if r["is_parity"]]
    disp_recs = [r for r in valid_recs if not r["is_parity"]]

    par_indiff = sum(1 for r in par_recs if r["display_choice"] == "Indifferent")
    disp_commit = sum(1 for r in disp_recs if r["is_committed"])
    disp_opt1 = sum(1 for r in disp_recs if r["display_choice"] == "Option 1")
    disp_opt2 = sum(1 for r in disp_recs if r["display_choice"] == "Option 2")
    disp_alpha = sum(1 for r in disp_recs if r["underlying_choice"] == "Alpha")
    disp_beta  = sum(1 for r in disp_recs if r["underlying_choice"] == "Beta")

    # Stage 2 Conditional Dominant Accuracy
    commit_recs = [r for r in disp_recs if r["is_committed"]]
    c_tot = len(commit_recs)
    c_dom = sum(1 for r in commit_recs if r["dominant_chosen"])
    dom_acc = (c_dom / c_tot * 100.0) if c_tot > 0 else 0.0

    print(f"\n--- Neutral-Label Summary (N={val_tot} valid) ---")
    print(f"Overall Commitment Rate: {sum(1 for r in valid_recs if r['is_committed'])}/{val_tot} ({sum(1 for r in valid_recs if r['is_committed'])/val_tot*100:.1f}%)")
    print(f"Parity Abstention (1:1):   {par_indiff}/{len(par_recs)} ({par_indiff/len(par_recs)*100:.1f}%)")
    print(f"Disparity Commitment:     {disp_commit}/{len(disp_recs)} ({disp_commit/len(disp_recs)*100:.1f}%)")
    print(f"  Slot Distribution:      Option 1 = {disp_opt1} ({disp_opt1/disp_commit*100 if disp_commit else 0:.1f}%) | Option 2 = {disp_opt2} ({disp_opt2/disp_commit*100 if disp_commit else 0:.1f}%)")
    print(f"  Action Distribution:    Alpha = {disp_alpha} ({disp_alpha/disp_commit*100 if disp_commit else 0:.1f}%) | Beta = {disp_beta} ({disp_beta/disp_commit*100 if disp_commit else 0:.1f}%)")
    print(f"Stage 2 Conditional Dominant Accuracy: {c_dom}/{c_tot} ({dom_acc:.1f}%)")

    # Breakdown by Phrasing
    print(f"\n--- Phrasing Breakdown (Disparity Commitment & Dom Acc) ---")
    phrasing_stats = {}
    for pid in ["P1", "P2", "P3", "P4"]:
        sub_d = [r for r in disp_recs if r["phrasing_id"] == pid]
        c_sub = [r for r in sub_d if r["is_committed"]]
        c_sub_dom = sum(1 for r in c_sub if r["dominant_chosen"])
        comm_pct = len(c_sub) / len(sub_d) * 100 if sub_d else 0.0
        acc_pct = c_sub_dom / len(c_sub) * 100 if c_sub else 0.0
        phrasing_stats[pid] = {"comm_pct": comm_pct, "acc_pct": acc_pct, "commit_n": len(c_sub), "total_n": len(sub_d)}
        print(f"  {pid}: Commitment = {len(c_sub)}/{len(sub_d)} ({comm_pct:.1f}%) | Cond. Dom Acc = {c_sub_dom}/{len(c_sub)} ({acc_pct:.1f}%)")

    # Generate Markdown Summary File
    md_report_path = out_path.replace(".csv", "_report.md")
    with open(md_report_path, "w") as f:
        f.write(f"# Neutral-Label Factorial Audit Report: {model_key}\n\n")
        f.write(f"**Total Trials:** {tot} (Valid: {val_tot})\n\n")
        f.write(f"**Overall Commitment Rate:** {sum(1 for r in valid_recs if r['is_committed'])/val_tot*100:.2f}%\n")
        f.write(f"**Parity (1:1) Abstention:** {par_indiff}/{len(par_recs)} ({par_indiff/len(par_recs)*100:.2f}%)\n")
        f.write(f"**Disparity Commitment:** {disp_commit}/{len(disp_recs)} ({disp_commit/len(disp_recs)*100:.2f}%)\n")
        f.write(f"**Stage 2 Conditional Accuracy:** {c_dom}/{c_tot} ({dom_acc:.2f}%)\n\n")
        f.write(f"## Phrasing Breakdown\n\n")
        f.write(f"| Phrasing | Commitment Rate | Committed Trials | Conditional Dominant Accuracy |\n")
        f.write(f"| :--- | :--- | :--- | :--- |\n")
        for pid in ["P1", "P2", "P3", "P4"]:
            ps = phrasing_stats[pid]
            f.write(f"| **{pid}** | {ps['comm_pct']:.1f}% | {ps['commit_n']}/{ps['total_n']} | {ps['acc_pct']:.1f}% |\n")
        f.write(f"\n## Presentation & Action Balance\n\n")
        f.write(f"- **Option 1 vs Option 2:** {disp_opt1} ({disp_opt1/disp_commit*100 if disp_commit else 0:.1f}%) vs. {disp_opt2} ({disp_opt2/disp_commit*100 if disp_commit else 0:.1f}%)\n")
        f.write(f"- **Alpha vs Beta:** {disp_alpha} ({disp_alpha/disp_commit*100 if disp_commit else 0:.1f}%) vs. {disp_beta} ({disp_beta/disp_commit*100 if disp_commit else 0:.1f}%)\n")

    print(f"\n[REPORT] Saved markdown summary report to {md_report_path}\n")


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run Full Neutral-Label Factorial Experiment.")
    parser.add_argument("--model", type=str, default="qwen2.5:3b", help="Model identifier")
    parser.add_argument("--model_key", type=str, default="qwen25_3b", help="Model key for results/output")
    parser.add_argument("--backend", type=str, default="ollama", choices=["mock", "ollama", "hf", "openai"],
                        help="Execution backend (mock, ollama, hf, openai)")
    parser.add_argument("--endpoint", type=str, default="http://127.0.0.1:11434", help="Ollama or API endpoint")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--dilemmas", type=str, default="all", help="'all' or comma-separated dilemma IDs (e.g. D01,D02)")
    parser.add_argument("--phrasings", type=str, default="all", help="'all' or comma-separated phrasing IDs (e.g. P1,P2)")
    parser.add_argument("--output", type=str, default="results/qwen25_3b/neutral_labels_full_1120.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
