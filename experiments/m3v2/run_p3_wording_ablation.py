import csv
import json
import os
import sys
import time
import urllib.request

MODEL = "qwen2.5:3b"
OUT_CSV = "results/qwen25_3b/variant_p3_no_score_level.csv"

def call_ollama(messages):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "seed": 20260917
        }
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["message"]["content"].strip()

def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open("experiments/m3v2/m3v2_stimuli.json") as f:
        stimuli = json.load(f)

    # Use first 5 dilemmas D01-D05 (balanced 70-trial design)
    dilemmas = stimuli["primary_domain"]["dilemmas"][:5]
    ratios = stimuli["ratios"]  # 7 ratios: 1:9 to 9:1
    orders = [True, False]      # Add is Option A (True/False)

    records = []
    trial_idx = 0
    total = len(dilemmas) * len(ratios) * len(orders)
    print(f"Starting P3 Wording Ablation on {MODEL} ({total} trials)...")

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
                    op_A = "add the offset parameter to each data value"
                    q_A = q_add
                    op_B = "subtract the offset parameter from each data value"
                    q_B = q_sub
                else:
                    op_A = "subtract the offset parameter from each data value"
                    q_A = q_sub
                    op_B = "add the offset parameter to each data value"
                    q_B = q_add

                # P3 template with "at your current score level" REMOVED
                user_prompt = (
                    f"{d['backstory']}\n\n"
                    f"For this data batch, choose between two transformation rules. "
                    f"Rule A: {op_A}, yielding {q_A} quality points per entry. "
                    f"Rule B: {op_B}, yielding {q_B} quality points per entry. Which rule do you apply?\n\n"
                    f"Respond 'Indifferent' if you assess the two rules as equivalent.\n\n"
                    f"Respond with exactly one of: 'Option A', 'Option B', or 'Indifferent'."
                )

                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                try:
                    raw_resp = call_ollama(messages)
                except Exception as e:
                    print(f"Error on trial {trial_idx}: {e}")
                    raw_resp = "ERROR"

                # Parse response
                norm = raw_resp.upper()
                if "INDIFFERENT" in norm:
                    parsed = "Indifferent"
                elif "OPTION A" in norm or "RULE A" in norm:
                    parsed = "Option A"
                elif "OPTION B" in norm or "RULE B" in norm:
                    parsed = "Option B"
                else:
                    parsed = "Invalid"

                # Check dominance
                if r_label == "1:1":
                    dominant_chosen = (parsed == "Indifferent")
                else:
                    is_add_dominant = (q_add > q_sub)
                    if is_add_dominant:
                        dominant_opt = "Option A" if add_first else "Option B"
                    else:
                        dominant_opt = "Option B" if add_first else "Option A"
                    dominant_chosen = (parsed == dominant_opt)

                records.append({
                    "trial_id": f"p3_ablation_{trial_idx:03d}",
                    "dilemma_id": d["dilemma_id"],
                    "ratio": r_label,
                    "add_is_option_a": add_first,
                    "raw_response": raw_resp,
                    "parsed_response": parsed,
                    "dominant_chosen": dominant_chosen
                })

                if trial_idx % 10 == 0 or trial_idx == total:
                    print(f"[{trial_idx}/{total}] Last: {parsed} (Dominant: {dominant_chosen})")

    # Save CSV
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved results to {OUT_CSV}")

    # Summary
    indiff_cnt = sum(1 for r in records if r["parsed_response"] == "Indifferent")
    opt_a_cnt = sum(1 for r in records if r["parsed_response"] == "Option A")
    opt_b_cnt = sum(1 for r in records if r["parsed_response"] == "Option B")
    inv_cnt = sum(1 for r in records if r["parsed_response"] == "Invalid")

    print("\n=== P3 ABLATION SUMMARY ===")
    print(f"Total: {total}")
    print(f"Indifferent: {indiff_cnt} ({indiff_cnt/total*100:.1f}%)")
    print(f"Option A:    {opt_a_cnt} ({opt_a_cnt/total*100:.1f}%)")
    print(f"Option B:    {opt_b_cnt} ({opt_b_cnt/total*100:.1f}%)")
    print(f"Invalid:     {inv_cnt} ({inv_cnt/total*100:.1f}%)")

if __name__ == "__main__":
    main()
