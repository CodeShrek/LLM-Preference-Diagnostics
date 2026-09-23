# Final Pre-Paper Audit (Phase 1)

**Date:** September 17, 2026
**Status:** M3 v2 MAIN IS FROZEN AND COMPLETE.

This document serves as the absolute baseline before executing the final pre-paper statistical verifications, control audits, and external validations.

## 1. Exact Repository State
The M3 v2 experiment logic resides exclusively in `experiments/m3v2/`. Follow-up configurations have been segmented into `experiments/followups/` to preserve the frozen core. Output data lives in `results/`.

### 1.1 Frozen Scientific Protocol Files
These files strictly dictate the M3 v2 experiment and **MUST NOT** be mutated:
- `experiments/m3v2/m3v2_stimuli.json` (20 primary dilemmas, 4 phrasings, 7 ratios, 5 controls, 5 held-out)
- `experiments/m3v2/m3v2_runner.py` (Execution engine handling the state updates and logging)
- `experiments/m3v2/m3v2_analysis.py` (Primary statistical scripts, Hurdle model, and visualization)

### 1.2 Data Files (Frozen Results)
- `results/m3v2_main.csv` (1120 trials of the main condition)
- `results/m3v2_C1.csv`
- `results/m3v2_C2.csv`
- `results/m3v2_C3_minus.csv`
- `results/m3v2_C3_plus.csv`
- `results/m3v2_C4.csv`
- `results/m3v2_combined.csv` (Concatenation of all conditions, used for final analysis)

## 2. Execution Metadata
- **Model:** `qwen2.5:3b-instruct-q4_K_M`
- **Inference Backend:** Ollama (local)
- **Temperature:** `0.0` (Greedy decoding)
- **Seed:** Not explicitly forced on generation (as T=0.0 is deterministic), but `20260917` is the default script seed. Bootstraps use a fixed `42` seed for reproducibility.

**Exact Execution Commands:**
```bash
# Generated via run_experiments.py wrapper for m3v2_runner.py
python experiments/m3v2/m3v2_runner.py --condition main --output results/m3v2_main.csv
# Followed by C1-C4 conditions
python experiments/m3v2/m3v2_analysis.py results/m3v2_combined.csv > results/m3v2_final_report.txt
```

## 3. Current Statistical Outputs and Figures
- **Primary Report:** `results/m3v2_final_report.txt` (Contains MNLogit and Stage 1/Stage 2 coefficients).
- **Secondary Audit:** `results/primary_statistics.txt` (Contains 10k bootstrap and non-linearity checks).
- **Figures:** `results/m3v2_fig1.png` (Multinomial probability surface), `results/m3v2_fig2.png` (Commitment & Direction decomposition).
- **Known Inconsistencies:** 
  1. The Stage 2 $\beta = 3.319$ does not align with the 10k bootstrap CI `[0.845, 2.161]`. The former included dilemma random/fixed effects, while the latter removed them to avoid quasi-separation during resampling. This must be formally resolved in Phase 3.
  2. The sample size for Stage 2 is reported as $N=160$, but earlier notes mentioned $N=180$ committed trials. This must be formally reconciled in Phase 4.

## 4. Current Literature Files
- `literature_review.md` (Updated to reflect 2026 publication dates)
- `novelty_audit.md` (Updated to soften causal claims to observational evidence)
- `claim_evidence_matrix.md` (Updated to reflect exact wording constraints)

No further changes to the frozen protocol will be tolerated. The paper preparation will proceed exactly from this verified state.
