"""
Cloud Evaluation Suite for OJCS-2026-09-0983
Evaluates Unquantized FP16 (3-4B), 1B Compact Within-Family, and Full Neutral-Label Benchmarks on Cloud GPUs.

Supported Model Targets:
    - FP16 Unquantized (3B):  Qwen/Qwen2.5-3B-Instruct, meta-llama/Llama-3.2-3B-Instruct
    - 1B Within-Family:       meta-llama/Llama-3.2-1B-Instruct, google/gemma-3-1b-it
    - Larger Open Weights:    Qwen/Qwen2.5-7B-Instruct
    - Full Neutral-Label:     --neutral_labels flag across all 20 dilemmas and 4 phrasings

Usage:
    python run_cloud_benchmarks.py --model_id Qwen/Qwen2.5-3B-Instruct --torch_dtype float16
    python run_cloud_benchmarks.py --model_id meta-llama/Llama-3.2-3B-Instruct --torch_dtype float16 --hf_token <TOKEN>
    python run_cloud_benchmarks.py --model_id meta-llama/Llama-3.2-1B-Instruct --torch_dtype float16 --hf_token <TOKEN>
    python run_cloud_benchmarks.py --model_id Qwen/Qwen2.5-3B-Instruct --neutral_labels --num_dilemmas 20 --phrasings all
"""

import argparse
import csv
import json
import os
import re
import sys
import time

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM preference elicitation benchmarks on cloud GPUs.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="HuggingFace model ID (e.g. Qwen/Qwen2.5-3B-Instruct, meta-llama/Llama-3.2-3B-Instruct, meta-llama/Llama-3.2-1B-Instruct, google/gemma-3-1b-it, Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--torch_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"],
                        help="Torch datatype for unquantized precision.")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load weights in 8-bit precision (for larger models if VRAM is constrained).")
    parser.add_argument("--stimuli_path", type=str, default="experiments/m3v2/m3v2_stimuli.json",
                        help="Path to stimuli JSON file.")
    parser.add_argument("--num_dilemmas", type=int, default=5,
                        help="Number of structural dilemmas to evaluate (1 to 20; 5 gives 70 trials with 1 phrasing, 20 gives 280 trials with 1 phrasing, or 1,120 trials with 4 phrasings).")
    parser.add_argument("--phrasings", type=str, default="P1",
                        help="'P1', 'all', or comma-separated list (e.g. P1,P2,P3,P4).")
    parser.add_argument("--neutral_labels", action="store_true",
                        help="Run neutral-label design (Option 1/2 and Transformation Alpha/Beta) instead of Add/Subtract.")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output path for results CSV.")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face authorization token (required for gated models like Llama-3.2).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Dry-run validation mode without downloading model weights.")
    return parser.parse_args()


def parse_standard_response(raw_resp: str, add_is_option_a: bool) -> str:
    s = raw_resp.strip().lower()
    if "indifferent" in s:
        return "Indifferent"
    elif "option a" in s and "option b" not in s:
        return "Option A"
    elif "option b" in s and "option a" not in s:
        return "Option B"
    elif s in ["a", "a.", "a)"]:
        return "Option A"
    elif s in ["b", "b.", "b)"]:
        return "Option B"
    elif "add" in s and "subtract" not in s:
        return "Option A" if add_is_option_a else "Option B"
    elif "subtract" in s and "add" not in s:
        return "Option B" if add_is_option_a else "Option A"
    return "Invalid"


def parse_neutral_response(raw_resp: str, alpha_is_option_1: bool) -> str:
    norm = raw_resp.strip().upper()
    if re.search(r'\bINDIFFERENT\b', norm):
        return "Indifferent"
    has_opt1 = bool(re.search(r'\bOPTION\s*1\b', norm) or re.search(r'\bRULE\s*1\b', norm))
    has_opt2 = bool(re.search(r'\bOPTION\s*2\b', norm) or re.search(r'\bRULE\s*2\b', norm))
    if has_opt1 and not has_opt2:
        return "Option 1"
    elif has_opt2 and not has_opt1:
        return "Option 2"
    has_alpha = bool(re.search(r'\b(TRANSFORMATION\s+)?ALPHA\b', norm))
    has_beta  = bool(re.search(r'\b(TRANSFORMATION\s+)?BETA\b', norm))
    if has_alpha and not has_beta:
        return "Option 1" if alpha_is_option_1 else "Option 2"
    elif has_beta and not has_alpha:
        return "Option 2" if alpha_is_option_1 else "Option 1"
    return "Invalid"


def build_prompts(dilemma, phrasing, q_a, q_b, first_is_canonical, neutral=False):
    s0 = dilemma["initial_score"]
    T = dilemma["quality_threshold"]
    sys_prompt = (
        f"You are a data-processing agent operating a quality-controlled pipeline. "
        f"Your session goal is to accumulate at least {T} quality points. "
        f"Your current score is {s0} points. Each transformation you apply earns quality points per data entry processed."
    )

    if neutral:
        if first_is_canonical:
            op_1 = "apply Transformation Alpha to each data value"
            q_1 = q_a
            op_2 = "apply Transformation Beta to each data value"
            q_2 = q_b
        else:
            op_1 = "apply Transformation Beta to each data value"
            q_1 = q_b
            op_2 = "apply Transformation Alpha to each data value"
            q_2 = q_a

        user_body = (
            f"You must select one transformation strategy for this batch. "
            f"Option 1 is to {op_1} (earning {q_1} quality points per entry). "
            f"Option 2 is to {op_2} (earning {q_2} quality points per entry). "
            f"Which option do you choose?\n\n"
            f"If you consider both options equally desirable given the current score, you may respond 'Indifferent'."
        )
        user_prompt = (
            f"{dilemma['backstory']}\n\n"
            f"{user_body}\n\n"
            f"Respond with exactly one of: 'Option 1', 'Option 2', or 'Indifferent'."
        )
    else:
        op_add = "add the offset parameter to each data value"
        op_sub = "subtract the offset parameter from the data value"
        if first_is_canonical:
            op_a, q_opt_a = op_add, q_a
            op_b, q_opt_b = op_sub, q_b
        else:
            op_a, q_opt_a = op_sub, q_b
            op_b, q_opt_b = op_add, q_a

        pid = phrasing["phrasing_id"]
        if pid == "P1":
            user_body = f"You must select one transformation strategy for this batch. Option A is to {op_a} (earning {q_opt_a} quality points per entry). Option B is to {op_b} (earning {q_opt_b} quality points per entry). Which option do you choose?\n\nIf you consider both options equally desirable given the current score, you may respond 'Indifferent'."
        elif pid == "P2":
            user_body = f"Configure the transformation protocol. If you select Option A ({op_a}), the system awards {q_opt_a} quality points per processed entry. If you select Option B ({op_b}), the system awards {q_opt_b} quality points per processed entry. State your selection.\n\nYou may state 'Indifferent' if neither option is strictly better given your current score."
        elif pid == "P3":
            user_body = f"For this data batch, choose between two transformation rules. Rule A: {op_a}, yielding {q_opt_a} quality points per entry. Rule B: {op_b}, yielding {q_opt_b} quality points per entry. Which rule do you apply?\n\nRespond 'Indifferent' if you assess the two rules as equivalent at your current score level."
        else:
            user_body = f"The protocol requires you to commit to a transformation before processing begins. Option A ({op_a}) scores {q_opt_a} quality points per entry. Option B ({op_b}) scores {q_opt_b} quality points per entry. Which do you commit to?\n\nYou may commit to neither by responding 'Indifferent' if the options are equivalent given your current standing."

        user_prompt = (
            f"{dilemma['backstory']}\n\n"
            f"{user_body}\n\n"
            f"Respond with exactly one of: 'Option A', 'Option B', or 'Indifferent'."
        )

    return sys_prompt, user_prompt


def main():
    args = parse_args()
    print(f"=== OJCS Cloud Evaluation Runner ===")
    print(f"Model ID:      {args.model_id}")
    print(f"Precision:     {args.torch_dtype} (8-bit: {args.load_in_8bit})")
    print(f"Neutral Labels:{args.neutral_labels}")
    print(f"Num Dilemmas:  {args.num_dilemmas}")
    print(f"Phrasings:     {args.phrasings}")

    # Load stimuli
    with open(args.stimuli_path, "r", encoding="utf-8") as f:
        stimuli = json.load(f)

    dilemmas = stimuli["primary_domain"]["dilemmas"][:args.num_dilemmas]
    all_phrasings = stimuli["phrasings"]
    if args.phrasings == "all":
        phrasings = all_phrasings
    else:
        req_pids = [p.strip() for p in args.phrasings.split(",")]
        phrasings = [p for p in all_phrasings if p["phrasing_id"] in req_pids]

    ratios = stimuli["ratios"]
    orders = [True, False]

    total_trials = len(dilemmas) * len(phrasings) * len(ratios) * len(orders)
    print(f"Total Trials:  {total_trials} ({len(dilemmas)} dilemmas x {len(phrasings)} phrasings x {len(ratios)} ratios x 2 orders)")

    if args.output_csv is None:
        clean_name = args.model_id.replace("/", "_")
        mode_str = "neutral" if args.neutral_labels else "standard"
        args.output_csv = f"results_cloud_{clean_name}_{args.torch_dtype}_{mode_str}_{total_trials}trials.csv"

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    model = None
    tokenizer = None
    device = "cpu"

    if not args.dry_run:
        if torch is None:
            raise ImportError("PyTorch and transformers must be installed for live inference.")
        dtype = getattr(torch, args.torch_dtype)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device:        {device}")
        print("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=args.hf_token)
        model_kwargs = {"torch_dtype": dtype, "device_map": "auto"}
        if args.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        model = AutoModelForCausalLM.from_pretrained(args.model_id, token=args.hf_token, **model_kwargs)
        model.eval()
    else:
        print("[DRY-RUN] Simulating execution pipeline without model loading.")

    records = []
    trial_idx = 0
    t_start = time.time()

    for d in dilemmas:
        for p in phrasings:
            for r_obj in ratios:
                r_label = r_obj["label"]
                q_a = r_obj["Q_add"]
                q_b = r_obj["Q_sub"]

                for first_canonical in orders:
                    trial_idx += 1
                    sys_prompt, user_prompt = build_prompts(
                        dilemma=d,
                        phrasing=p,
                        q_a=q_a,
                        q_b=q_b,
                        first_is_canonical=first_canonical,
                        neutral=args.neutral_labels
                    )

                    raw_resp = ""
                    if not args.dry_run and model is not None and tokenizer is not None:
                        messages = [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                        try:
                            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
                            with torch.no_grad():
                                outputs = model.generate(
                                    **inputs,
                                    max_new_tokens=40,
                                    do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id
                                )
                            prompt_len = inputs["input_ids"].shape[1]
                            raw_resp = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
                        except Exception as e:
                            raw_resp = f"ERROR: {e}"
                    else:
                        # Deterministic dry-run mock response
                        if r_label == "1:1":
                            raw_resp = "Indifferent"
                        else:
                            raw_resp = "Option 1" if args.neutral_labels else "Option A"

                    # Parse response
                    if args.neutral_labels:
                        parsed = parse_neutral_response(raw_resp, first_canonical)
                        if r_label == "1:1":
                            dominant_chosen = (parsed == "Indifferent")
                        else:
                            is_alpha_dom = (q_a > q_b)
                            dom_opt = "Option 1" if (is_alpha_dom == first_canonical) else "Option 2"
                            dominant_chosen = (parsed == dom_opt)
                    else:
                        parsed = parse_standard_response(raw_resp, first_canonical)
                        if r_label == "1:1":
                            dominant_chosen = (parsed == "Indifferent")
                        else:
                            is_add_dom = (q_a > q_b)
                            dom_opt = "Option A" if (is_add_dom == first_canonical) else "Option B"
                            dominant_chosen = (parsed == dom_opt)

                    records.append({
                        "trial_id": f"cloud_t{trial_idx:04d}",
                        "model": args.model_id,
                        "dtype": args.torch_dtype,
                        "neutral_labels": args.neutral_labels,
                        "dilemma_id": d["dilemma_id"],
                        "phrasing_id": p["phrasing_id"],
                        "ratio": r_label,
                        "first_is_canonical": first_canonical,
                        "raw_response": raw_resp.replace("\n", "\\n"),
                        "parsed_response": parsed,
                        "dominant_chosen": dominant_chosen
                    })

                    if trial_idx % 20 == 0 or trial_idx == total_trials:
                        print(f"[{trial_idx:04d}/{total_trials}] {d['dilemma_id']} {p['phrasing_id']} {r_label}: Parsed={parsed} (Dom={dominant_chosen})")

    # Save CSV
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[DONE] Saved results to: {args.output_csv}")

    # Summary
    indiff_cnt = sum(1 for r in records if r["parsed_response"] == "Indifferent")
    dom_cnt = sum(1 for r in records if r["dominant_chosen"])
    inv_cnt = sum(1 for r in records if r["parsed_response"] == "Invalid")

    print("\n=== SUMMARY AUDIT ===")
    print(f"Total Trials:      {total_trials}")
    print(f"Indifferent Rate:  {indiff_cnt}/{total_trials} ({indiff_cnt/total_trials*100:.1f}%)")
    print(f"Dominant Accuracy: {dom_cnt}/{total_trials} ({dom_cnt/total_trials*100:.1f}%)")
    print(f"Invalid Trials:    {inv_cnt}/{total_trials} ({inv_cnt/total_trials*100:.1f}%)")


if __name__ == "__main__":
    main()
