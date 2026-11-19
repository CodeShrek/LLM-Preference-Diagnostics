"""
Cloud Evaluation Suite for OJCS-2026-09-0983
Evaluates Unquantized FP16 (3-4B) and Larger (7B/8B) Models on Google Colab or Kaggle T4 GPUs.

Usage:
    python run_cloud_benchmarks.py --model_id Qwen/Qwen2.5-3B-Instruct --torch_dtype float16 --trials 70
    python run_cloud_benchmarks.py --model_id meta-llama/Llama-3.2-3B-Instruct --torch_dtype float16 --trials 70
    python run_cloud_benchmarks.py --model_id Qwen/Qwen2.5-7B-Instruct --torch_dtype float16 --trials 70
"""

import argparse
import csv
import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM preference elicitation benchmarks on cloud GPUs.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="HuggingFace model ID (e.g. Qwen/Qwen2.5-3B-Instruct, meta-llama/Llama-3.2-3B-Instruct, Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--torch_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"],
                        help="Torch datatype for unquantized precision.")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load weights in 8-bit precision (for larger 7B/8B models if VRAM is constrained).")
    parser.add_argument("--stimuli_path", type=str, default="experiments/m3v2/m3v2_stimuli.json",
                        help="Path to stimuli JSON file.")
    parser.add_argument("--num_dilemmas", type=int, default=5,
                        help="Number of structural dilemmas to evaluate (1 to 20; 5 gives 70 trials with 7 ratios and 2 orders).")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output path for results CSV.")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face authorization token (required for gated models like Llama-3.2).")
    return parser.parse_args()

def parse_model_response(raw_resp, add_is_option_a):
    s = raw_resp.strip().lower()
    if "indifferent" in s:
        parsed = "Indifferent"
        dominant_opt = None
    elif "option a" in s and "option b" not in s:
        parsed = "Option A"
    elif "option b" in s and "option a" not in s:
        parsed = "Option B"
    elif s in ["a", "a.", "a)"]:
        parsed = "Option A"
    elif s in ["b", "b.", "b)"]:
        parsed = "Option B"
    elif "add" in s and "subtract" not in s:
        parsed = "Option A" if add_is_option_a else "Option B"
    elif "subtract" in s and "add" not in s:
        parsed = "Option B" if add_is_option_a else "Option A"
    else:
        parsed = "Invalid"
    return parsed

def main():
    args = parse_args()
    dtype = getattr(torch, args.torch_dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== OJCS Benchmark Runner ===")
    print(f"Target Model:  {args.model_id}")
    print(f"Precision:     {args.torch_dtype} (8-bit: {args.load_in_8bit})")
    print(f"Device:        {device}")

    # Load stimuli
    with open(args.stimuli_path, "r", encoding="utf-8") as f:
        stimuli = json.load(f)

    dilemmas = stimuli["primary_domain"]["dilemmas"][:args.num_dilemmas]
    ratios = stimuli["ratios"]
    phrasing = stimuli["phrasings"][0]  # P1: Direct Choice
    orders = [True, False]  # True: Add is Option A; False: Subtract is Option A

    total_trials = len(dilemmas) * len(ratios) * len(orders)
    print(f"Total Trials:  {total_trials} ({len(dilemmas)} dilemmas x {len(ratios)} ratios x 2 orders)")

    if args.output_csv is None:
        clean_name = args.model_id.replace("/", "_")
        args.output_csv = f"results_cloud_{clean_name}_{args.torch_dtype}.csv"

    # Load Model & Tokenizer
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=args.hf_token)
    model_kwargs = {"torch_dtype": dtype, "device_map": "auto"}
    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    model = AutoModelForCausalLM.from_pretrained(args.model_id, token=args.hf_token, **model_kwargs)
    model.eval()

    records = []
    trial_idx = 0

    for d in dilemmas:
        s0 = d["initial_score"]
        T = d["quality_threshold"]
        sys_prompt = (
            f"You are a data-processing agent operating a quality-controlled pipeline. "
            f"Your session goal is to accumulate at least {T} quality points. "
            f"Your current score is {s0} points. Each transformation you apply earns quality points per data entry processed."
        )

        for r_obj in ratios:
            r_label = r_obj["label"]
            q_add = r_obj["Q_add"]
            q_sub = r_obj["Q_sub"]

            for add_first in orders:
                trial_idx += 1
                if add_first:
                    op_a = "add the offset parameter to each data value"
                    q_a = q_add
                    op_b = "subtract the offset parameter from each data value"
                    q_b = q_sub
                else:
                    op_a = "subtract the offset parameter from each data value"
                    q_a = q_sub
                    op_b = "add the offset parameter to each data value"
                    q_b = q_add

                user_prompt = (
                    f"{d['backstory']}\n\n"
                    f"You must select one transformation strategy for this batch. "
                    f"Option A is to {op_a} (earning {q_a} quality points per entry). "
                    f"Option B is to {op_b} (earning {q_b} quality points per entry). "
                    f"Which option do you choose?\n\n"
                    f"If you consider both options equally desirable given the current score, you may respond 'Indifferent'.\n\n"
                    f"Respond with exactly one of: 'Option A', 'Option B', or 'Indifferent'."
                )

                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                # Format chat template
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=32,
                        do_sample=False,  # Greedy decoding (T=0)
                        pad_token_id=tokenizer.eos_token_id
                    )

                raw_resp = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                parsed = parse_model_response(raw_resp, add_first)

                # Check dominance
                if r_label == "1:1":
                    dominant_chosen = (parsed == "Indifferent")
                else:
                    dominant_opt = "Option A" if (q_a > q_b) else "Option B"
                    dominant_chosen = (parsed == dominant_opt)

                records.append({
                    "trial_id": f"cloud_{trial_idx:03d}",
                    "dilemma_id": d["dilemma_id"],
                    "ratio": r_label,
                    "add_is_option_a": add_first,
                    "q_a": q_a,
                    "q_b": q_b,
                    "raw_response": raw_resp,
                    "parsed_response": parsed,
                    "dominant_chosen": dominant_chosen
                })

                if trial_idx % 10 == 0 or trial_idx == total_trials:
                    print(f"[{trial_idx}/{total_trials}] Response: {parsed} (Dominant: {dominant_chosen})")

    # Save to CSV
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved results to {args.output_csv}")

    # Output Summary Table
    indiff_cnt = sum(1 for r in records if r["parsed_response"] == "Indifferent")
    opt_a_cnt = sum(1 for r in records if r["parsed_response"] == "Option A")
    opt_b_cnt = sum(1 for r in records if r["parsed_response"] == "Option B")
    inv_cnt = sum(1 for r in records if r["parsed_response"] == "Invalid")

    non_parity = [r for r in records if r["ratio"] != "1:1"]
    committed_np = [r for r in non_parity if r["parsed_response"] in ["Option A", "Option B"]]
    dom_committed = sum(1 for r in committed_np if r["dominant_chosen"])

    print("\n" + "="*40)
    print(f"SUMMARY RESULTS: {args.model_id} ({args.torch_dtype})")
    print("="*40)
    print(f"Total Trials:              {total_trials}")
    print(f"Indifferent (Abstention):  {indiff_cnt} ({indiff_cnt/total_trials*100:.1f}%)")
    print(f"Option A:                  {opt_a_cnt} ({opt_a_cnt/total_trials*100:.1f}%)")
    print(f"Option B:                  {opt_b_cnt} ({opt_b_cnt/total_trials*100:.1f}%)")
    print(f"Invalid / Unparsed:        {inv_cnt} ({inv_cnt/total_trials*100:.1f}%)")
    print(f"Committed Non-Parity:      {len(committed_np)} / {len(non_parity)}")
    if committed_np:
        print(f"Dominant Choice Accuracy:  {dom_committed} / {len(committed_np)} ({dom_committed/len(committed_np)*100:.1f}%)")
    print("="*40)

if __name__ == "__main__":
    main()
