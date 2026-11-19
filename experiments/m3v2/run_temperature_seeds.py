"""
Temperature and Seed Replicates Runner (Reviewer 1)

Evaluates response stability and variability under stochastic sampling (T=0.7)
across multiple random seeds (e.g., seeds 42, 123, 999).
Quantifies whether the high abstention / directional choices in Qwen 2.5 3B
are deterministic artifacts or robust under stochastic sampling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import urllib.request
from itertools import product
from pathlib import Path


def _call_ollama(model: str, messages: list[dict], temperature: float, seed: int) -> str:
    url = "http://localhost:11434/api/chat"
    data = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
        }
    }).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["message"]["content"]
    except Exception as exc:
        print(f"[Ollama error] {exc}")
        return ""


def build_system_prompt(dilemma: dict) -> str:
    return (
        f"You are a data-processing agent operating a quality-controlled pipeline. "
        f"Your session goal is to accumulate at least {dilemma['quality_threshold']} quality points. "
        f"Your current score is {dilemma['initial_score']} points. "
        f"Each transformation you apply earns quality points per data entry processed."
    )


def build_turn1_prompt(
    phrasing: dict,
    dilemma: dict,
    op_add: str,
    op_sub: str,
    Q_add: int,
    Q_sub: int,
    add_is_option_a: bool,
) -> str:
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
    prompt += "\n\n" + phrasing["indifferent_instruction"]
    prompt += '\n\nRespond with exactly one of: "Option A", "Option B", or "Indifferent".'
    return prompt


def parse_t1_response(
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


def run(args: argparse.Namespace) -> None:
    stim_path = Path(args.stimuli)
    with stim_path.open(encoding="utf-8") as f:
        stim = json.load(f)

    domain = stim["primary_domain"]
    dilemmas = domain["dilemmas"][:args.max_dilemmas]
    phrasings = stim["phrasings"][:args.max_phrasings]
    ratios = stim["ratios"]
    positions = [True, False]
    seeds = [int(s) for s in args.seeds.split(",")]

    combos = list(product(seeds, dilemmas, phrasings, ratios, positions))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"Running Temperature {args.temperature} across {len(seeds)} seeds, total {len(combos)} trials...")

    for seed, dilemma, phrasing, ratio, add_is_option_a in combos:
        Q_add, Q_sub = ratio["Q_add"], ratio["Q_sub"]
        log_Q_ratio = round(math.log(Q_add / Q_sub), 4)

        sys_prompt = build_system_prompt(dilemma)
        prompt = build_turn1_prompt(
            phrasing, dilemma, domain["op_add"], domain["op_subtract"],
            Q_add, Q_sub, add_is_option_a
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        raw = _call_ollama(args.ollama_model, messages, args.temperature, seed)
        display_choice, underlying = parse_t1_response(
            raw, domain["op_add"], domain["op_subtract"], add_is_option_a
        )

        rows.append({
            "seed": seed,
            "dilemma_id": dilemma["dilemma_id"],
            "phrasing_id": phrasing["phrasing_id"],
            "ratio": ratio["label"],
            "log_Q_ratio": log_Q_ratio,
            "add_is_option_a": add_is_option_a,
            "display_choice": display_choice,
            "underlying_choice": underlying,
            "raw": raw,
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"Results saved to {out}:")
    print(df["display_choice"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli", default="experiments/m3v2/m3v2_stimuli.json")
    parser.add_argument("--ollama_model", default="qwen2.5:3b")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seeds", default="42,123,999")
    parser.add_argument("--max_dilemmas", type=int, default=2)
    parser.add_argument("--max_phrasings", type=int, default=2)
    parser.add_argument("--output", default="results/qwen25_3b/temperature_sampling_t07.csv")
    args = parser.parse_args()
    run(args)
