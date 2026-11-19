"""
Label and Slot Manipulation Checks for Qwen 2.5 3B (Reviewer 2.6)

Tests whether abstention is sensitive to:
1. Label variations: "Neither" and "No preference" vs "Indifferent"
2. Slot variations: Placing abstention in the first slot vs terminal slot.

Evaluated on balanced subsets across parity (1:1) and disparities (1:9, 9:1).
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


def build_system_prompt(dilemma: dict) -> str:
    return (
        f"You are a data-processing agent operating a quality-controlled pipeline. "
        f"Your session goal is to accumulate at least {dilemma['quality_threshold']} quality points. "
        f"Your current score is {dilemma['initial_score']} points. "
        f"Each transformation you apply earns quality points per data entry processed."
    )


def build_prompt_variant(
    variant_type: str,  # 'neither', 'no_preference', 'first_slot'
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

    if variant_type == "neither":
        template = phrasing["template"]
        prompt = (
            f"{backstory}\n\n"
            + template.replace("{op_A}", op_A)
                      .replace("{op_B}", op_B)
                      .replace("{Q_A}", str(Q_A))
                      .replace("{Q_B}", str(Q_B))
        )
        prompt += "\n\nIf you consider both options equally desirable given the current score, you may respond 'Neither'."
        prompt += '\n\nRespond with exactly one of: "Option A", "Option B", or "Neither".'

    elif variant_type == "no_preference":
        template = phrasing["template"]
        prompt = (
            f"{backstory}\n\n"
            + template.replace("{op_A}", op_A)
                      .replace("{op_B}", op_B)
                      .replace("{Q_A}", str(Q_A))
                      .replace("{Q_B}", str(Q_B))
        )
        prompt += "\n\nIf you consider both options equally desirable given the current score, you may respond 'No preference'."
        prompt += '\n\nRespond with exactly one of: "Option A", "Option B", or "No preference".'

    elif variant_type == "first_slot":
        # Slot 1 is Abstain, Slot 2 is Option B, Slot 3 is Option C
        prompt = (
            f"{backstory}\n\n"
            f"You must select one transformation strategy for this batch.\n"
            f"Option A: No transformation (abstain if you assess rules as equivalent).\n"
            f"Option B: {op_A} (earning {Q_A} quality points per entry).\n"
            f"Option C: {op_B} (earning {Q_B} quality points per entry).\n\n"
            f"Which option do you choose? Respond with exactly one of: \"Option A\", \"Option B\", or \"Option C\"."
        )

    return prompt


def parse_variant_response(raw: str, variant_type: str) -> str:
    text = (raw or "").strip().lower()
    if variant_type == "neither":
        if "neither" in text:
            return "ABSTAIN"
        if "option a" in text and "option b" not in text:
            return "A"
        if "option b" in text and "option a" not in text:
            return "B"
    elif variant_type == "no_preference":
        if "no preference" in text:
            return "ABSTAIN"
        if "option a" in text and "option b" not in text:
            return "A"
        if "option b" in text and "option a" not in text:
            return "B"
    elif variant_type == "first_slot":
        if "option a" in text and "option b" not in text and "option c" not in text:
            return "ABSTAIN"
        if "option b" in text and "option a" not in text and "option c" not in text:
            return "DIR_1"
        if "option c" in text and "option a" not in text and "option b" not in text:
            return "DIR_2"
    return "INVALID"


def run(args: argparse.Namespace) -> None:
    stim_path = Path(args.stimuli)
    with stim_path.open(encoding="utf-8") as f:
        stim = json.load(f)

    domain = stim["primary_domain"]
    dilemmas = domain["dilemmas"][:args.max_dilemmas]
    phrasings = stim["phrasings"]
    ratios = [r for r in stim["ratios"] if r["label"] in {"1:9", "1:1", "9:1"}]
    positions = [True, False]

    combos = list(product(dilemmas, phrasings, ratios, positions))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"Running Variant {args.variant} across {len(combos)} trials...")

    for dilemma, phrasing, ratio, add_is_option_a in combos:
        Q_add, Q_sub = ratio["Q_add"], ratio["Q_sub"]
        sys_prompt = build_system_prompt(dilemma)
        prompt = build_prompt_variant(
            args.variant, phrasing, dilemma,
            domain["op_add"], domain["op_subtract"],
            Q_add, Q_sub, add_is_option_a
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        raw = _call_ollama(args.ollama_model, messages, args.temperature)
        parsed = parse_variant_response(raw, args.variant)

        rows.append({
            "variant": args.variant,
            "dilemma_id": dilemma["dilemma_id"],
            "phrasing_id": phrasing["phrasing_id"],
            "ratio": ratio["label"],
            "add_is_option_a": add_is_option_a,
            "raw": raw,
            "parsed": parsed,
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"Results for {args.variant}:")
    print(df["parsed"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli", default="experiments/m3v2/m3v2_stimuli.json")
    parser.add_argument("--ollama_model", default="qwen2.5:3b")
    parser.add_argument("--variant", choices=["neither", "no_preference", "first_slot"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_dilemmas", type=int, default=3)
    args = parser.parse_args()
    run(args)
