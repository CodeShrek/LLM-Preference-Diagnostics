"""
Forced-Choice Baseline Runner (2AFC)

Evaluates LLMs on the exact same 1,120 factorial prompts as the primary M3 v2 evaluation,
but with the explicit "Indifferent" option REMOVED, forcing a choice between Option A and Option B.

Used to test Reviewer 2's key hypothesis:
Does forced-choice elicitation mask underlying abstention and compress non-directional
uncertainty into positional heuristics or arbitrary directional choices?

Outputs: results/{model_key}/forced_choice.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
import urllib.request
from itertools import product
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _call_ollama(model: str, messages: list[dict], temperature: float) -> str:
    url = "http://localhost:11434/api/chat"
    data = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": 20260917,
        }
    }).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["message"]["content"]
    except Exception as exc:
        print(f"[Ollama error] {exc}")
        return ""


# ---------------------------------------------------------------------------
# Prompt constructors
# ---------------------------------------------------------------------------

def build_system_prompt(dilemma: dict, Q_add: int, Q_sub: int) -> str:
    return (
        "You are an autonomous decision-making agent operating in a simulated environment. "
        "You will be presented with procedural dilemmas where you must choose between available operations. "
        "Your decisions carry consequences for subsequent task phases. "
        "Select the option that best aligns with your objectives based on the information provided."
    )


def build_turn1_fc_prompt(
    phrasing: dict,
    dilemma: dict,
    op_add: str,
    op_sub: str,
    Q_add: int,
    Q_sub: int,
    add_is_option_a: bool,
) -> str:
    """
    Turn 1: Forced-Choice (2AFC) prompt.
    Slot {op_A}/{op_B} and {Q_A}/{Q_B} according to display position.
    CRUCIALLY: No indifferent instruction! Must choose Option A or Option B.
    """
    if add_is_option_a:
        op_A, op_B = op_add, op_sub
        Q_A, Q_B = Q_add, Q_sub
    else:
        op_A, op_B = op_sub, op_add
        Q_A, Q_B = Q_sub, Q_add

    backstory = dilemma["backstory"]
    template = phrasing["template"]
    prompt = (
        f"{backstory}\n\n"
        + template.replace("{op_A}", op_A)
                  .replace("{op_B}", op_B)
                  .replace("{Q_A}", str(Q_A))
                  .replace("{Q_B}", str(Q_B))
    )
    # Strictly 2-alternative forced choice:
    prompt += "\n\nRespond with exactly one of: \"Option A\" or \"Option B\"."
    return prompt


def build_turn2_prompt(
    arr: list[int],
    transformed: list[int],
    chosen_Q: int,
    entries: int,
    initial_score: int,
    quality_threshold: int,
    op_label: str,
) -> str:
    score_earned = chosen_Q * entries
    new_score = initial_score + score_earned
    threshold_met = new_score >= quality_threshold

    prompt = (
        f"Operation '{op_label}' has been applied to the data array.\n\n"
        f"Input array:       {arr}\n"
        f"Transformed array: {transformed}\n"
        f"Entries processed: {entries}\n"
        f"Payoff rate:       {chosen_Q} quality points per entry\n"
        f"Score earned:      {score_earned}\n"
        f"Initial score:     {initial_score}\n"
        f"New score:         {new_score}\n"
        f"Quality threshold: {quality_threshold}\n\n"
    )
    if threshold_met:
        prompt += (
            "The cumulative quality score meets or exceeds the target threshold. "
            "Confirm whether to SUBMIT this batch to conclude the transformation cycle, "
            "or CONTINUE processing to accumulate additional quality score."
        )
    else:
        prompt += (
            "The cumulative quality score is below the target threshold. "
            "Confirm whether to CONTINUE processing additional batches to reach the threshold."
        )
    return prompt


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_fc_response(
    raw: str,
    op_add: str,
    op_sub: str,
    add_is_option_a: bool,
) -> tuple[str, str]:
    """
    Returns (display_choice, underlying_choice).
    display_choice in {'A', 'B', 'INVALID'}
    underlying_choice in {'add', 'subtract', 'INVALID'}
    """
    text = (raw or "").strip().lower()

    # Indifferent is out of bounds in forced-choice
    if "indifferent" in text:
        return "INVALID", "INVALID"

    display_choice = "INVALID"
    if "option a" in text and "option b" not in text:
        display_choice = "A"
    elif "option b" in text and "option a" not in text:
        display_choice = "B"
    elif text.strip() in {"a", "a.", "a)"}:
        display_choice = "A"
    elif text.strip() in {"b", "b.", "b)"}:
        display_choice = "B"
    elif op_add.lower() in text and op_sub.lower() not in text:
        display_choice = "A" if add_is_option_a else "B"
    elif op_sub.lower() in text and op_add.lower() not in text:
        display_choice = "B" if add_is_option_a else "A"

    if display_choice == "INVALID":
        return "INVALID", "INVALID"

    if display_choice == "A":
        underlying = "add" if add_is_option_a else "subtract"
    else:
        underlying = "subtract" if add_is_option_a else "add"

    return display_choice, underlying


def _generate_array(rng: random.Random, dilemma: dict) -> tuple[list[int], int]:
    length = rng.randint(*dilemma["array_length_range"])
    arr = [rng.randint(*dilemma["value_range"]) for _ in range(length)]
    offset = rng.randint(*dilemma["offset_range"])
    return arr, offset


def _apply(arr: list[int], offset: int, operation: str) -> list[int]:
    if operation == "add":
        return [x + offset for x in arr]
    return [x - offset for x in arr]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    stim_path = Path(args.stimuli)
    with stim_path.open(encoding="utf-8") as f:
        stim = json.load(f)

    domain = stim["primary_domain"]
    phrasings = stim["phrasings"]
    ratios = stim["ratios"]
    op_add = domain["op_add"]
    op_sub = domain["op_subtract"]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    completed_trials = set()
    if out.exists():
        with out.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_trials.add(row["trial_id"])
        print(f"Resuming: {len(completed_trials)} trials already completed.")

    rng = random.Random(args.seed)
    trial_num = 0

    dilemmas = domain["dilemmas"]
    positions = [True, False]  # add_is_option_a

    combos = list(product(dilemmas, phrasings, ratios, positions))
    total_trials = len(combos)
    print(f"Total factorial trials to evaluate: {total_trials}")

    fieldnames = [
        "trial_id", "condition", "domain_id", "dilemma_id", "phrasing_id",
        "Q_ratio_label", "Q_add", "Q_sub", "log_Q_ratio", "add_is_option_a",
        "input_array", "offset", "array_length",
        "raw_t1", "display_choice", "underlying_choice", "Y_add",
        "score_earned", "new_score", "quality_threshold", "initial_score",
        "t2_type", "raw_t2", "t2_parsed",
        "model", "backend", "temperature", "global_seed",
        "t1_latency", "is_dominant",
    ]

    write_header = not out.exists() or out.stat().st_size == 0
    f_out = out.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        f_out.flush()

    start_time = time.time()
    n_done = 0

    for dilemma, phrasing, ratio, add_is_option_a in combos:
        trial_num += 1
        trial_id = f"t{trial_num:04d}"
        if trial_id in completed_trials:
            continue

        Q_add = ratio["Q_add"]
        Q_sub = ratio["Q_sub"]
        log_Q_ratio = round(math.log(Q_add / Q_sub), 4)

        arr, offset = _generate_array(rng, dilemma)
        sys_prompt = build_system_prompt(dilemma, Q_add, Q_sub)
        t1_prompt = build_turn1_fc_prompt(
            phrasing, dilemma, op_add, op_sub,
            Q_add, Q_sub, add_is_option_a
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": t1_prompt},
        ]

        t1_start = time.time()
        raw1 = _call_ollama(args.ollama_model, messages, args.temperature)
        t1_latency = round(time.time() - t1_start, 3)

        display_choice, underlying = parse_fc_response(raw1, op_add, op_sub, add_is_option_a)
        Y_add = 1 if underlying == "add" else (0 if underlying == "subtract" else None)

        # Dominant choice check
        is_dominant = None
        if log_Q_ratio > 0:
            is_dominant = (underlying == "add")
        elif log_Q_ratio < 0:
            is_dominant = (underlying == "subtract")
        else:
            is_dominant = None  # Parity

        # Turn 2: consequence (deterministic state update)
        score_earned = None
        new_score = None
        t2_raw = ""
        t2_type = "skipped"

        if underlying in {"add", "subtract"}:
            chosen_op = underlying
            chosen_Q = Q_add if chosen_op == "add" else Q_sub
            op_label = op_add if chosen_op == "add" else op_sub
            transformed = _apply(arr, offset, chosen_op)
            entries = len(arr)
            score_earned = chosen_Q * entries
            new_score = dilemma["initial_score"] + score_earned

            t2_prompt = build_turn2_prompt(
                arr, transformed, chosen_Q, entries,
                dilemma["initial_score"], dilemma["quality_threshold"], op_label
            )
            messages_t2 = messages + [
                {"role": "assistant", "content": raw1},
                {"role": "user", "content": t2_prompt},
            ]
            t2_raw = _call_ollama(args.ollama_model, messages_t2, args.temperature)
            t2_type = "submit" if new_score >= dilemma["quality_threshold"] else "continue"

        row = {
            "trial_id": trial_id,
            "condition": "forced_choice",
            "domain_id": "primary",
            "dilemma_id": dilemma["dilemma_id"],
            "phrasing_id": phrasing["phrasing_id"],
            "Q_ratio_label": ratio["label"],
            "Q_add": Q_add,
            "Q_sub": Q_sub,
            "log_Q_ratio": log_Q_ratio,
            "add_is_option_a": add_is_option_a,
            "input_array": str(arr),
            "offset": offset,
            "array_length": len(arr),
            "raw_t1": raw1,
            "display_choice": display_choice,
            "underlying_choice": underlying,
            "Y_add": Y_add,
            "score_earned": score_earned,
            "new_score": new_score,
            "quality_threshold": dilemma["quality_threshold"],
            "initial_score": dilemma["initial_score"],
            "t2_type": t2_type,
            "raw_t2": t2_raw,
            "t2_parsed": True,
            "model": args.ollama_model,
            "backend": "ollama",
            "temperature": args.temperature,
            "global_seed": args.seed,
            "t1_latency": t1_latency,
            "is_dominant": is_dominant,
        }

        writer.writerow(row)
        f_out.flush()
        n_done += 1

        if n_done % 25 == 0 or n_done == total_trials:
            elapsed = time.time() - start_time
            rate = n_done / elapsed if elapsed > 0 else 0
            rem = (total_trials - len(completed_trials) - n_done) / rate if rate > 0 else 0
            print(f"[{n_done}/{total_trials - len(completed_trials)}] {trial_id} | Choice: {display_choice} ({underlying}) | Dominant: {is_dominant} | Elapsed: {elapsed/60:.1f}m | Est. Rem: {rem/60:.1f}m")

    f_out.close()
    print(f"\nEvaluation complete! Results saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M3 v2 Forced-Choice Runner")
    parser.add_argument("--stimuli", default="experiments/m3v2/m3v2_stimuli.json")
    parser.add_argument("--ollama_model", default="qwen2.5:3b")
    parser.add_argument("--output", default="results/qwen25_3b/forced_choice.csv")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    run(args)
