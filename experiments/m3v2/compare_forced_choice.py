"""
Comparative Analysis: 3-Option Abstention-Aware vs. 2-Option Forced-Choice Baseline

Directly answers Reviewer 2, Point 6:
Quantifies the exact behavioral distortions introduced by forced-choice preference elicitation
on the identical 1,120 factorial stimulus set for Qwen 2.5 3B.
"""

import csv
import json
import math
import os
import numpy as np
from collections import defaultdict

BASE = "/Users/karan/Desktop/APART"
PRIMARY_CSV = os.path.join(BASE, "results/qwen25_3b/primary.csv")
FORCED_CSV  = os.path.join(BASE, "results/qwen25_3b/forced_choice.csv")

def analyze():
    if not os.path.exists(FORCED_CSV):
        print(f"Error: {FORCED_CSV} does not exist yet.")
        return

    primary_rows = list(csv.DictReader(open(PRIMARY_CSV)))
    forced_rows  = list(csv.DictReader(open(FORCED_CSV)))

    print(f"Loaded Primary (3-Option): {len(primary_rows)} rows")
    print(f"Loaded Forced (2-Option):  {len(forced_rows)} rows\n")

    # Align by trial_id
    prim_map = {r["trial_id"]: r for r in primary_rows}
    fc_map   = {r["trial_id"]: r for r in forced_rows}
    common_ids = sorted(set(prim_map.keys()) & set(fc_map.keys()))

    print(f"Common aligned trials: {len(common_ids)}\n")

    # 1. Response breakdown in Forced Choice
    fc_choices = defaultdict(int)
    fc_underlying = defaultdict(int)
    for tid in common_ids:
        r = fc_map[tid]
        fc_choices[r["display_choice"]] += 1
        fc_underlying[r["underlying_choice"]] += 1

    print("=" * 60)
    print("FORCED-CHOICE (2AFC) OVERALL RESPONSE BREAKDOWN")
    print("=" * 60)
    total_fc = len(common_ids)
    for k, v in fc_choices.items():
        print(f"  Display Choice '{k}': {v} ({100*v/total_fc:.1f}%)")
    for k, v in fc_underlying.items():
        print(f"  Action '{k}': {v} ({100*v/total_fc:.1f}%)")

    # 2. What happens to the trials where Qwen chose "Indifferent" in 3-option?
    indiff_ids = [tid for tid in common_ids if prim_map[tid]["underlying_choice"].strip().lower() == "indifferent"]
    print("\n" + "=" * 60)
    print(f"FATE OF THE {len(indiff_ids)} 'INDIFFERENT' TRIALS WHEN FORCED TO CHOOSE")
    print("=" * 60)
    fate_display = defaultdict(int)
    fate_underlying = defaultdict(int)
    fate_dominant = defaultdict(int)

    for tid in indiff_ids:
        r = fc_map[tid]
        fate_display[r["display_choice"]] += 1
        fate_underlying[r["underlying_choice"]] += 1
        lq = float(r["log_Q_ratio"])
        if lq != 0:
            dom = r["is_dominant"]
            fate_dominant[dom] += 1

    print(f"Display option selected on abstained trials:")
    for k, v in fate_display.items():
        print(f"  {k}: {v} ({100*v/len(indiff_ids):.1f}%)")

    print(f"\nDominant vs Dominated choice on non-parity abstained trials (N={sum(fate_dominant.values())}):")
    for k, v in fate_dominant.items():
        print(f"  Dominant = {k}: {v} ({100*v/sum(fate_dominant.values()):.1f}%)")

    # 3. Parity condition (1:1 ratio) analysis
    parity_ids = [tid for tid in common_ids if float(prim_map[tid]["log_Q_ratio"]) == 0]
    print("\n" + "=" * 60)
    print(f"PARITY CONDITION (1:1, N={len(parity_ids)}) COMPARISON")
    print("=" * 60)
    prim_par_indiff = sum(1 for tid in parity_ids if prim_map[tid]["underlying_choice"].strip().lower() == "indifferent")
    print(f"  3-Option: Indifferent = {prim_par_indiff}/{len(parity_ids)} ({100*prim_par_indiff/len(parity_ids):.1f}%)")

    fc_par_optA = sum(1 for tid in parity_ids if fc_map[tid]["display_choice"] == "A")
    fc_par_optB = sum(1 for tid in parity_ids if fc_map[tid]["display_choice"] == "B")
    print(f"  2-Option Forced: Option A = {fc_par_optA} ({100*fc_par_optA/len(parity_ids):.1f}%), Option B = {fc_par_optB} ({100*fc_par_optB/len(parity_ids):.1f}%)")

    # 4. Dominant accuracy across payoff ratios
    print("\n" + "=" * 60)
    print("DOMINANT CHOICE ACCURACY BY PAYOFF RATIO")
    print("=" * 60)
    ratios = ["1:9", "1:5", "1:2", "2:1", "5:1", "9:1"]
    print(f"{'Ratio':<8} | {'3-Option P(Commit)':<20} | {'3-Option P(Dom|Commit)':<22} | {'2-Option Forced P(Dom)':<22}")
    print("-" * 80)
    for rat in ratios:
        r_ids = [tid for tid in common_ids if prim_map[tid]["Q_ratio_label"] == rat]
        # 3-option
        prim_dir = [tid for tid in r_ids if prim_map[tid]["underlying_choice"].strip().lower() in ("add", "subtract")]
        p_commit = len(prim_dir) / len(r_ids) if r_ids else 0
        
        # 3-option dominant
        prim_dom = sum(1 for tid in prim_dir if (float(prim_map[tid]["log_Q_ratio"]) > 0 and prim_map[tid]["underlying_choice"] == "add") or
                                                (float(prim_map[tid]["log_Q_ratio"]) < 0 and prim_map[tid]["underlying_choice"] == "subtract"))
        p_prim_dom = prim_dom / len(prim_dir) if prim_dir else 0

        # forced-choice dominant
        fc_dom = sum(1 for tid in r_ids if str(fc_map[tid]["is_dominant"]).strip().lower() in ("true", "1"))
        p_fc_dom = fc_dom / len(r_ids) if r_ids else 0

        print(f"{rat:<8} | {p_commit*100:>5.1f}% ({len(prim_dir):>2}/{len(r_ids)})      | {p_prim_dom*100:>5.1f}% ({prim_dom:>2}/{len(prim_dir):>2})           | {p_fc_dom*100:>5.1f}% ({fc_dom:>3}/{len(r_ids)})")

if __name__ == "__main__":
    analyze()
