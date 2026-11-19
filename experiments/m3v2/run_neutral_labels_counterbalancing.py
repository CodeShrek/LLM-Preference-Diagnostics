import csv
import json
import os
import sys
import time
import urllib.request

MODEL = "qwen2.5:3b"
OUT_CSV = "results/qwen25_3b/variant_neutral_labels.csv"

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
    orders = [True, False]      # Alpha is Option 1 (True/False)

    records = []
    trial_idx = 0
    total = len(dilemmas) * len(ratios) * len(orders)
    print(f"Starting Neutral-Label Counterbalancing on {MODEL} ({total} trials)...")

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
            q_alpha = r_obj["Q_add"]
            q_beta = r_obj["Q_sub"]

            for alpha_first in orders:
                trial_idx += 1
                if alpha_first:
                    op_1 = "apply Transformation Alpha to each data value"
                    q_1 = q_alpha
                    op_2 = "apply Transformation Beta to each data value"
                    q_2 = q_beta
                else:
                    op_1 = "apply Transformation Beta to each data value"
                    q_1 = q_beta
                    op_2 = "apply Transformation Alpha to each data value"
                    q_2 = q_alpha

                # Neutralized prompt with Option 1 / Option 2 and Transformation Alpha / Beta
                user_prompt = (
                    f"{d['backstory']}\n\n"
                    f"You must select one transformation strategy for this batch. "
                    f"Option 1 is to {op_1} (earning {q_1} quality points per entry). "
                    f"Option 2 is to {op_2} (earning {q_2} quality points per entry). "
                    f"Which option do you choose?\n\n"
                    f"If you consider both options equally desirable given the current score, you may respond 'Indifferent'.\n\n"
                    f"Respond with exactly one of: 'Option 1', 'Option 2', or 'Indifferent'."
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
                elif "OPTION 1" in norm or "OPTION1" in norm:
                    parsed = "Option 1"
                elif "OPTION 2" in norm or "OPTION2" in norm:
                    parsed = "Option 2"
                elif "ALPHA" in norm:
                    parsed = "Option 1" if alpha_first else "Option 2"
                elif "BETA" in norm:
                    parsed = "Option 2" if alpha_first else "Option 1"
                else:
                    parsed = "Invalid"

                # Check dominance: Alpha corresponds to Q_add, Beta to Q_sub
                if r_label == "1:1":
                    dominant_chosen = (parsed == "Indifferent")
                else:
                    is_alpha_dominant = (q_alpha > q_beta)
                    if is_alpha_dominant:
                        dominant_opt = "Option 1" if alpha_first else "Option 2"
                    else:
                        dominant_opt = "Option 2" if alpha_first else "Option 1"
                    dominant_chosen = (parsed == dominant_opt)

                records.append({
                    "trial_id": f"neutral_{trial_idx:03d}",
                    "dilemma_id": d["dilemma_id"],
                    "ratio": r_label,
                    "alpha_is_option_1": alpha_first,
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
    opt_1_cnt = sum(1 for r in records if r["parsed_response"] == "Option 1")
    opt_2_cnt = sum(1 for r in records if r["parsed_response"] == "Option 2")
    inv_cnt = sum(1 for r in records if r["parsed_response"] == "Invalid")

    print("\n=== NEUTRAL LABELS SUMMARY ===")
    print(f"Total: {total}")
    print(f"Indifferent: {indiff_cnt} ({indiff_cnt/total*100:.1f}%)")
    print(f"Option 1:    {opt_1_cnt} ({opt_1_cnt/total*100:.1f}%)")
    print(f"Option 2:    {opt_2_cnt} ({opt_2_cnt/total*100:.1f}%)")
    print(f"Invalid:     {inv_cnt} ({inv_cnt/total*100:.1f}%)")

if __name__ == "__main__":
    main()
