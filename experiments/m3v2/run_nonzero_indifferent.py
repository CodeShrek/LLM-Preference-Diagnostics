"""
Non-Zero Indifferent Payoff Experiment (Reviewer 3.1)

Evaluates Qwen 2.5 3B when the 'Indifferent' option awards a non-zero payoff
equal to min(Q_A, Q_B) points per entry.
At 1:1 parity, all three options (Option A, Option B, Indifferent) tie with identical positive payoffs.
Under disparities (1:2, 1:5, 1:9, etc.), Indifferent matches the lower of the two payoffs.

Tests whether Stage 1 abstention is an artifact of zero-payoff defaults
or reflects genuine score-relative equivalence / non-directional abstention.
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


def build_system_prompt(dilemma: dict, Q_add: int, Q_sub: int) -> str:
    return (
        f"You are a data-processing agent operating a quality-controlled pipeline. "
        f"Your session goal is to accumulate at least {dilemma['quality_threshold']} quality points. "
        f"Your current score is {dilemma['initial_score']} points. "
        f"Each transformation you apply earns quality points per data entry processed."
    )


def build_turn1_nonzero_prompt(
    phrasing: dict,
    dilemma: dict,
    op_add: str,
    op_sub: str,
    Q_add: int,
    Q_sub: int,
    add_is_option_a: bool,
) -> tuple[str, int]:
    if add_is_option_a:
        op_A, op_B = op_add, op_sub
        Q_A, Q_B = Q_add, Q_sub
    else:
        op_A, op_B = op_sub, op_add
        Q_A, Q_B = Q_sub, Q_add

    Q_indiff = min(Q_A, Q_B)

    backstory = dilemma["backstory"]
    template = phrasing["template"]
    prompt = (
        f"{backstory}\n\n"
        + template.replace("{op_A}", op_A)
                  .replace("{op_B}", op_B)
                  .replace("{Q_A}", str(Q_A))
                  .replace("{Q_B}", str(Q_B))
    )

    if Q_A == Q_B:
        indiff_info = (
            f"Selecting 'Indifferent' processes the batch without transformation and also earns "
            f"{Q_indiff} quality points per entry (matching both options). "
            f"{phrasing['indifferent_instruction']}"
        )
    else:
        indiff_info = (
            f"Selecting 'Indifferent' processes the batch without transformation and earns "
            f"{Q_indiff} quality points per entry (matching the lower-payoff option). "
            f"{phrasing['indifferent_instruction']}"
        )

    prompt += f"\n\n{indiff_info}"
    prompt += '\n\nRespond with exactly one of: "Option A", "Option B", or "Indifferent".'
    return prompt, Q_indiff


def parse_response(
    raw: str,
    op_add: str,
    op_sub: str,
    add_is_option_a: bool,
) -> tuple[str, str]:
    text = (raw or "").strip().lower()

    if "indifferent" in text:
        return "INDIFFERENT", "INDIFFERENT"

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
    positions = [True, False]

    if args.tiny:
        dilemmas = dilemmas[:2]
        phrasings = phrasings[:2]
        ratios = [r for r in ratios if r["label"] in {"1:9", "1:1", "9:1"}]
    elif args.max_dilemmas:
        dilemmas = dilemmas[:args.max_dilemmas]
    elif args.parity_only:
        ratios = [r for r in ratios if r["label"] == "1:1"]

    combos = list(product(dilemmas, phrasings, ratios, positions))
    total_trials = len(combos)
    print(f"Total trials to evaluate: {total_trials}")

    fieldnames = [
        "trial_id", "condition", "domain_id", "dilemma_id", "phrasing_id",
        "Q_ratio_label", "Q_add", "Q_sub", "Q_indiff", "log_Q_ratio", "add_is_option_a",
        "raw_t1", "display_choice", "underlying_choice", "is_dominant",
        "model", "backend", "temperature", "global_seed", "t1_latency"
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

        sys_prompt = build_system_prompt(dilemma, Q_add, Q_sub)
        t1_prompt, Q_indiff = build_turn1_nonzero_prompt(
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

        display_choice, underlying = parse_response(raw1, op_add, op_sub, add_is_option_a)

        is_dominant = None
        if log_Q_ratio > 0:
            is_dominant = (underlying == "add")
        elif log_Q_ratio < 0:
            is_dominant = (underlying == "subtract")
        else:
            is_dominant = (display_choice == "INDIFFERENT")

        row = {
            "trial_id": trial_id,
            "condition": "nonzero_indifferent",
            "domain_id": "primary",
            "dilemma_id": dilemma["dilemma_id"],
            "phrasing_id": phrasing["phrasing_id"],
            "Q_ratio_label": ratio["label"],
            "Q_add": Q_add,
            "Q_sub": Q_sub,
            "Q_indiff": Q_indiff,
            "log_Q_ratio": log_Q_ratio,
            "add_is_option_a": add_is_option_a,
            "raw_t1": raw1,
            "display_choice": display_choice,
            "underlying_choice": underlying,
            "is_dominant": is_dominant,
            "model": args.ollama_model,
            "backend": "ollama",
            "temperature": args.temperature,
            "global_seed": args.seed,
            "t1_latency": t1_latency,
        }

        writer.writerow(row)
        f_out.flush()
        n_done += 1

        if n_done % 20 == 0 or n_done == total_trials:
            elapsed = time.time() - start_time
            rate = n_done / elapsed if elapsed > 0 else 0
            rem = (total_trials - len(completed_trials) - n_done) / rate if rate > 0 else 0
            print(f"[{n_done}/{total_trials - len(completed_trials)}] {trial_id} | Choice: {display_choice} | Rem: {rem/60:.1f}m")

    f_out.close()
    print(f"\nCompleted! Saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli", default="experiments/m3v2/m3v2_stimuli.json")
    parser.add_argument("--ollama_model", default="qwen2.5:3b")
    parser.add_argument("--output", default="results/qwen25_3b/nonzero_indifferent.csv")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--parity_only", action="store_true")
    parser.add_argument("--max_dilemmas", type=int, default=None)
    args = parser.parse_args()
    run(args)
