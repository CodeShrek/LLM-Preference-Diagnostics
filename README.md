# Stress-Testing Preference Elicitation in Instruction-Tuned Language Models: Separating Commitment, Conditional Direction, and Payoff Sensitivity Across Small Model Families

[![Open Access Repository](https://img.shields.io/badge/Repository-GitHub-blue)](https://github.com/CodeShrek/LLM-Preference-Diagnostics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)

This repository contains the complete experimental framework, frozen trial records, stimulus specifications, deterministic response parsers, econometric modeling scripts, and publication figures for the study:

> **"Stress-Testing Preference Elicitation in Instruction-Tuned Language Models: Separating Commitment, Conditional Direction, and Payoff Sensitivity Across Small Model Families"**  
> *Target Venue: IEEE Open Journal of the Computer Society (OJCS)*  
> *Original Manuscript ID: OJCS-2026-09-0983 (Revised and Resubmitted as New)*  
> *Permanent Open-Access Archive: [https://github.com/CodeShrek/LLM-Preference-Diagnostics](https://github.com/CodeShrek/LLM-Preference-Diagnostics)*

---

## 🔬 Scientific Summary & Analytical Decomposition

Standard evaluations of preferences in Large Language Models (LLMs) commonly employ **two-alternative forced-choice (2AFC)** protocols. However, forcing an agent to select between two alternatives conflates two distinct behavioral mechanisms:
1. **Directional Commitment (Stage 1 / Whether to act):** Does the model commit to an operational action or choose an explicitly offered non-directional exit (*Indifferent*)?
2. **Conditional Dominance Selection (Stage 2 / Which action to choose):** Conditional on committing, does the model select the mathematically dominant (higher-payoff) alternative, and how does this sensitivity scale with payoff magnitude and presentation order?

We evaluate this two-stage hurdle decomposition within an operational, multi-turn task environment (**Data-Quality Pipeline**) across **1,120 balanced factorial trials per model** across four locally deployed instruction-tuned model families in the 3–4B parameter class (Qwen 2.5 3B, Llama 3.2 3B, Phi-4 Mini 3.8B, Gemma 3 4B; 4,480 total primary trials).

---

## 📊 Experimental Design & Factorial Structure

The evaluation uses a fully crossed, balanced factorial design:
$$\text{20 Structural Dilemmas} \times \text{4 Phrasing Templates (P1--P4)} \times \text{7 Payoff Ratios} \times \text{2 Presentation Positions} = 1,120 \text{ trials/model}.$$

- **Structural Dilemmas ($N=20$, D01--D20):** Parameterized by baseline data arrays, offset parameters, initial cumulative scores ($s_0 \in [40, 120]$), and aspirational session targets ($T \in [90, 220]$).
- **Operations & Consequential Protocol:**
  - Turn 1: Model chooses between *Option A*, *Option B*, or *Indifferent*.
  - Turn 2: Scripted deterministic environment update ($s_{\text{new}} = s_0 + Q \cdot \ell$).
  - Turn 3: Follow-on decision (continue or submit batch).
- **Payoff Ratios ($N=7$):** Reciprocally paired conditions: $1:9, 1:5, 1:2, 1:1, 2:1, 5:1, 9:1$. Signed disparity ratio is $r = Q_{\mathrm{add}}/Q_{\mathrm{sub}}$; disparity magnitude is $m = \max(Q)/\min(Q)$, with $\log m = |\log r|$.
- **Presentation Positions ($N=2$):** $\text{Pos}=+1$ when *Add* is Option~A; $\text{Pos}=-1$ when *Subtract* is Option~A. Dominant-first indicator is $\text{DominantFirst} = \operatorname{sign}(\log r) \times \text{Pos} \in \{+1, -1\}$.
- **Evaluated Models:**
  - `Qwen/Qwen2.5-3B-Instruct` (Ollama tag: `qwen2.5:3b-instruct-q4_K_M`)
  - `meta-llama/Llama-3.2-3B-Instruct` (Ollama tag: `llama3.2:3b-instruct-q4_K_M`)
  - `microsoft/Phi-4-mini-instruct` (Ollama tag: `phi4-mini:3.8b-q4_K_M`)
  - `google/gemma-3-4b-it` (Ollama tag: `gemma3:4b-it-q4_K_M`)
  - Decoding: Deterministic greedy decoding ($T=0.0$).

---

## 🧪 Comprehensive Controls and Robustness Suite

In direct response to reviewer feedback, this repository includes dedicated empirical controls:
1. **Numerical-Dominance Comprehension Control (C4, $N=1,120$/model):** Tests isolated arithmetic comparison without procedural narrative framing. Accuracies: Qwen 100.0%, Gemma 98.3%, Phi 71.0%, Llama 53.8%.
2. **Non-Binding Threshold Control ($T=10, s_0 \ge 40$, $N=1,120$/model):** Proves mathematically and empirically that payoff sensitivity is not an artifact of active threshold-seeking heuristics ($s_0 \ge 40 > 10$ unconditionally).
3. **Matched Two-Alternative Forced-Choice Baseline (2AFC, $N=1,120$ on Qwen):** Omitting the *Indifferent* option reveals severe protocol-induced bias at parity (79.4% Option B vs. 20.6% Option A) and massive position bias ($\hat\beta_{\text{DominantFirst}} = -2.267, p < 10^{-8}$).
4. **Non-Zero Indifferent Payoff Robustness ($N=280$ on Qwen):** Setting $Q_{\text{Indiff}} = \min(Q_A, Q_B)$ proves that 95.0% parity abstention persists when indifference awards positive points, ruling out zero-payoff aversion as the sole explanation.
5. **Indifferent Menu-Slot & Label Ablations ($N=72$/condition on Qwen):** Evaluates alternative non-directional phrasings ("Neither" [76.4%], "No preference" [97.2%]) and menu-slot positioning (Option A slot extinguishes abstention to 0.0%, confirming menu recency).
6. **P3 Phrasing Score-Level Wording Ablation ($N=70$ on Qwen):** Excising *"at your current score level"* preserves 100.0% abstention (70/70), demonstrating that score-level wording is not strictly necessary for universal abstention.
7. **Neutral-Label Counterbalancing Experiment ($N=70$ on Qwen):** Replacing "Add"/"Subtract" with "Transformation Alpha"/"Beta" across "Option 1"/"Option 2" eliminates semantic prior disparity (25 Alpha vs. 26 Beta) while maintaining 100.0% dominant accuracy on committed choices.
8. **Stochastic Sampling & Random Seed Replicates ($T=0.7$, 3 seeds, $N=168$ on Qwen):** Confirms consistent abstention (83.9%, 83.9%, 82.1%; 40/56 prompt conditions unanimous across seeds), verifying that abstention is not a greedy decoding artifact.
9. **Second Domain Replication: Resource Scheduling (H01--H05, $N=280$/model on Qwen & Llama):** Evaluates server quota allocation dilemmas, replicating Qwen's high abstention (98.6%) and Llama's high commitment (98.8%).
10. **Cloud GPU Benchmark Suite for Unquantized FP16 & 7B/8B Models:** Ready-to-run Colab/Kaggle notebook and script for evaluating unquantized checkpoints on free NVIDIA T4 GPUs (16 GB VRAM).

---

## 📁 Repository Directory Structure

```text
.
├── README.md                                 # Comprehensive project overview and reproduction guide
├── requirements.txt                          # Python package dependencies
├── experiments/
│   ├── m3v2/                                 # Primary M3 v2 experimental framework
│   │   ├── m3v2_runner.py                    # Primary experimental runner with deterministic parser
│   │   ├── m3v2_stimuli.json                 # Verbatim 20 structural dilemma definitions
│   │   ├── run_all_models.py                 # Multi-model batch orchestration script
│   │   ├── forced_choice_runner.py           # Matched 2AFC runner (omitting Indifferent)
│   │   ├── compare_forced_choice.py          # 2AFC vs. 3-option comparative statistical analysis
│   │   ├── firth_stage2_robustness.py        # Firth penalized likelihood logistic regression
│   │   ├── run_nonzero_indifferent.py        # Non-zero indifferent payoff robustness runner
│   │   ├── run_label_slot_checks.py          # Menu-slot and non-directional label ablations
│   │   ├── run_p3_wording_ablation.py        # P3 phrasing score-level wording ablation runner
│   │   ├── run_neutral_labels_counterbalancing.py # Neutral-label counterbalancing runner
│   │   ├── run_temperature_seeds.py          # Stochastic decoding sampling runner (T=0.7)
│   │   └── multimodel_analysis.py            # Primary econometric analysis and figure generator
│   ├── followups/
│   │   ├── heldout_resource/                 # Second task domain (Resource Scheduling, H01-H05)
│   │   │   ├── m3v2_holdout_main.csv         # Qwen 2.5 3B holdout trials (N=280)
│   │   │   └── m3v2_holdout_llama.csv        # Llama 3.2 3B holdout trials (N=280)
│   │   └── threshold_relevance/              # Non-binding threshold follow-up (T=10)
│   │       ├── m3v2_stimuli_thresh10.json    # Threshold follow-up dilemma specifications
│   │       ├── run_threshold_experiment.py   # Threshold follow-up runner
│   │       └── analyze_threshold.py          # Threshold follow-up analysis
│   └── cloud_eval/                           # Cloud GPU benchmarks for unquantized & larger models
│       ├── run_cloud_benchmarks.py           # Self-contained HuggingFace/PyTorch FP16 execution script
│       └── OJCS_FP16_and_7B_Benchmarks.ipynb # Interactive one-click Google Colab / Kaggle notebook
├── results/                                  # Strictly frozen raw trial records (CSVs)
│   ├── qwen25_3b/                            # Qwen 2.5 3B primary (1,120), C4, T=10, 2AFC, and ablations
│   ├── llama32_3b/                           # Llama 3.2 3B primary (1,120), C4, and T=10
│   ├── phi4mini_38b/                         # Phi-4 Mini 3.8B primary (1,120), C4, and T=10
│   └── gemma3_4b/                            # Gemma 3 4B primary (1,120), C4, and T=10
└── paper/                                    # Manuscript, supplement, and reviewer responses
    ├── main.tex                              # Primary IEEE OJ-CS manuscript LaTeX source
    ├── main.txt                              # Exact plain-text mirror of main.tex
    ├── supplementary.tex                     # Comprehensive Supplementary Material (Sections I-XVIII)
    ├── references.bib                        # Cleaned, verified BibTeX references (0 missing keys)
    ├── response_to_reviewers.tex             # Point-by-point formal response to reviewers (LaTeX)
    ├── response_to_reviewers.md              # Point-by-point response to reviewers (Markdown)
    └── figures/                              # Generated high-resolution publication figures (PDF & PNG)
```

---

## 🚀 Step-by-Step Reproduction Guide

### 1. Environment Setup

Clone this repository and create a clean Python virtual environment (Python 3.10+):

```bash
git clone https://github.com/CodeShrek/LLM-Preference-Diagnostics.git
cd LLM-Preference-Diagnostics
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Reproduce All Econometric Models, Tables, and Figures

To reproduce the complete statistical results, Stage 1 hurdle models, Stage 2 Firth regressions, dilemma cluster bootstraps, and generate all eight publication figures from the frozen raw data:

```bash
# Generate primary figures (Figs A through F) and statistical summaries:
python experiments/m3v2/multimodel_analysis.py

# Run Firth's penalized likelihood regression for Stage-2 separation robustness:
python experiments/m3v2/firth_stage2_robustness.py

# Run matched 2AFC comparison analysis (Figure G):
python experiments/m3v2/compare_forced_choice.py
```

Generated publication figures are saved directly to `paper/figures/`.

### 3. Re-Running Model Inference from Scratch (Local Ollama Backend)

To re-run inference across the four local models, install [Ollama](https://ollama.com) and pull the exact model tags:

```bash
ollama serve &

ollama pull qwen2.5:3b-instruct-q4_K_M
ollama pull llama3.2:3b-instruct-q4_K_M
ollama pull phi4-mini:3.8b-q4_K_M
ollama pull gemma3:4b-it-q4_K_M
```

Execute the full 4,480-trial evaluation pipeline:

```bash
# Run primary 1,120-trial factorial evaluation across all four models:
python experiments/m3v2/run_all_models.py

# Run C4 numerical-dominance comprehension control:
python experiments/m3v2/run_controls.py --condition C4

# Run non-binding threshold follow-up (T=10):
python experiments/followups/threshold_relevance/run_threshold_experiment.py

# Run matched 2AFC baseline on Qwen 2.5 3B:
python experiments/m3v2/forced_choice_runner.py

# Run robustness checks (Non-zero Indifferent, slot/label ablations, neutral labels):
python experiments/m3v2/run_nonzero_indifferent.py
python experiments/m3v2/run_label_slot_checks.py
python experiments/m3v2/run_p3_wording_ablation.py
python experiments/m3v2/run_neutral_labels_counterbalancing.py
python experiments/m3v2/run_temperature_seeds.py
```

### 4. Running Cloud GPU Benchmarks (Unquantized FP16 & Larger 7B Models)

To verify behavior on unquantized 16-bit floating-point weights and larger models without local hardware memory constraints, open `experiments/cloud_eval/OJCS_FP16_and_7B_Benchmarks.ipynb` in Google Colab or Kaggle (with a free NVIDIA T4 GPU), or run directly:

```bash
pip install torch transformers accelerate
python experiments/cloud_eval/run_cloud_benchmarks.py --model Qwen/Qwen2.5-3B-Instruct --precision fp16
```

### 5. Compiling Manuscript and Supplementary Materials

The LaTeX documents can be compiled via `pdflatex` or `latexmk`:

```bash
cd paper

# Compile Main Manuscript (Strictly <= 12 pages):
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Compile Supplementary Material:
pdflatex supplementary.tex
pdflatex supplementary.tex

# Compile Formal Response to Reviewers:
pdflatex response_to_reviewers.tex
pdflatex response_to_reviewers.tex
```

---

## 🔍 Deterministic Parser Specification

Responses are parsed deterministically by `experiments/m3v2/m3v2_runner.py` using strict ordered rules applied to lowercase response text $s$:
1. If substring `"indifferent"` is in $s$: classify as **INDIFFERENT**.
2. If `"option a"` is in $s$ and `"option b"` is not: classify as **Option A**.
3. If `"option b"` is in $s$ and `"option a"` is not: classify as **Option B**.
4. If stripped string is in `{"a", "a.", "a)"}`: classify as **Option A**.
5. If stripped string is in `{"b", "b.", "b)"}`: classify as **Option B**.
6. If the assigned Add operation label appears in $s$ and Subtract does not: classify as option assigned to Add.
7. If the assigned Subtract operation label appears in $s$ and Add does not: classify as option assigned to Subtract.
8. Otherwise: classify as **INVALID**.

Any trial classified as `INVALID` is excluded from primary estimation (43 Llama trials [3.8%], 16 Phi trials [1.4%], 0 Qwen/Gemma trials). Bounding and lenient re-parsing sensitivity analyses demonstrate that conditional dominant accuracy is robust within $\le 3.2$ percentage points (see manuscript Section~VIII and Supplementary Section~V).

---

## 📜 Citation

```bibtex
@article{apart_preference_diagnostics_2026,
  title={Stress-Testing Preference Elicitation in Instruction-Tuned Language Models: Separating Commitment, Conditional Direction, and Payoff Sensitivity Across Small Model Families},
  author={Anonymous Authors},
  journal={IEEE Open Journal of the Computer Society},
  year={2026},
  note={Under peer review; revised resubmission OJCS-2026-09-0983}
}
```
