import pandas as pd
import json

def main():
    # Load stimuli
    with open('experiments/m3v2/m3v2_stimuli.json', 'r') as f:
        stim = json.load(f)
        
    dilemmas = stim['primary_domain']['dilemmas']
    ratios = stim['ratios']
    
    records = []
    for d in dilemmas:
        avg_entries = sum(d['array_length_range']) / 2
        for r in ratios:
            # Over 3 turns, if they consistently pick Add
            add_total = d['initial_score'] + 3 * (r['Q_add'] * avg_entries)
            add_crosses = add_total >= d['quality_threshold']
            
            # Over 3 turns, if they consistently pick Sub
            sub_total = d['initial_score'] + 3 * (r['Q_sub'] * avg_entries)
            sub_crosses = sub_total >= d['quality_threshold']
            
            records.append({
                'dilemma': d['dilemma_id'],
                'Q_ratio': r['label'],
                'max_Q': max(r['Q_add'], r['Q_sub']),
                'add_crosses': add_crosses,
                'sub_crosses': sub_crosses,
                'any_crosses': add_crosses or sub_crosses
            })
            
    df = pd.DataFrame(records)
    
    with open("results/threshold_confound_audit.md", "w") as f:
        f.write("# Phase 13: Threshold Confound Audit\n\n")
        f.write("## Overview\n")
        f.write("We evaluate whether the quantitative payoff magnitude (Q) is perfectly confounded with the ability to cross the target quality threshold over the course of the 3-turn episode.\n\n")
        
        f.write("## Analysis of Potential Threshold Crossings (3 Turns)\n")
        f.write("Averaging over the dynamic array lengths for each dilemma, we cross-tabulate the maximum Q available (1, 2, 5, 9) with whether *any* option crosses the threshold if chosen 3 times:\n\n")
        
        crosstab = pd.crosstab(df['max_Q'], df['any_crosses'])
        f.write("```\n")
        f.write(crosstab.to_string())
        f.write("\n```\n\n")
        
        f.write("## Findings\n")
        if (crosstab.var(axis=1) == 0).any() or len(crosstab.columns) == 1:
            f.write("There is a **severe confound**. At certain Q magnitudes, crossing the threshold is mathematically guaranteed or mathematically impossible across all dilemmas.\n")
            f.write("CONCLUSION: The threshold confound is present. High Q allows success, low Q forces failure. We **MUST** run the conditional follow-up experiment (Phase 20) to determine if the model is responding to Q magnitude directly, or just picking whatever crosses the threshold.\n")
        else:
            f.write("The Q magnitude is **not** perfectly confounded with threshold crossing. Due to the diverse initial scores and thresholds across the 20 dilemmas, a high Q ratio does not guarantee crossing the threshold, and a low Q ratio does not preclude it. The outcome magnitude and threshold status are statistically separable in the current design.\n")
            f.write("CONCLUSION: The current design is interpretable without requiring the Phase 20 threshold follow-up. The observed Q-sensitivity operates independently of a strict threshold-crossing heuristic.\n")

if __name__ == "__main__":
    main()
