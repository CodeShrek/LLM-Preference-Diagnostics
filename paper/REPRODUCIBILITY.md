# Complete Reproducibility & Traceability Guide: LLM Preference Diagnostics

This document provides exact, end-to-end instructions to reproduce the primary 4,480-trial factorial experiment across all four models, the full controls and robustness suite, econometric models, and publication figures.

---

## 1. Environment & Dependencies

1. **Python**: Python 3.10+ is required.
2. **Virtual Environment**: 
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Dependencies**: 
   ```bash
   pip install -r requirements.txt
   ```
   Packages: `pandas>=2.0.0`, `numpy>=1.24.0`, `scipy>=1.10.0`, `statsmodels>=0.14.0`, `matplotlib>=3.7.0`, `seaborn>=0.12.0`.

---

## 2. Model Specifications & Acquisition (Ollama)

All primary experiments were executed locally under deterministic greedy decoding ($T=0.0$) using fixed seeds via Ollama:

```bash
ollama pull qwen2.5:3b-instruct-q4_K_M
ollama pull llama3.2:3b-instruct-q4_K_M
ollama pull phi4-mini:3.8b-q4_K_M
ollama pull gemma3:4b-it-q4_K_M
```

Specifications:
- Qwen 2.5 3B: `qwen2.5:3b-instruct-q4_K_M` (Q4_K_M, $T=0.0$)
- Llama 3.2 3B: `llama3.2:3b-instruct-q4_K_M` (Q4_K_M, $T=0.0$)
- Phi-4 Mini 3.8B: `phi4-mini:3.8b-q4_K_M` (Q4_K_M, $T=0.0$)
- Gemma 3 4B: `gemma3:4b-it-q4_K_M` (Q4_K_M, $T=0.0$)

---

## 3. Frozen Data Directory Structure

All raw trial records are strictly frozen in `results/`:
- `results/qwen25_3b/primary.csv`: 1,120 balanced factorial trials on Qwen 2.5 3B
- `results/llama32_3b/primary.csv`: 1,120 balanced factorial trials on Llama 3.2 3B
- `results/phi4mini_38b/primary.csv`: 1,120 balanced factorial trials on Phi-4 Mini 3.8B
- `results/gemma3_4b/primary.csv`: 1,120 balanced factorial trials on Gemma 3 4B
- `results/*/c4.csv`: 1,120 C4 numerical comprehension trials per model
- `results/*/threshold.csv`: 1,120 non-binding threshold ($T=10$) trials per model
- `results/qwen25_3b/forced_choice.csv`: 1,120 matched 2AFC trials on Qwen 2.5 3B
- `results/qwen25_3b/nonzero_indifferent_280.csv`: 280 trials with non-zero indifferent payoff
- `results/qwen25_3b/variant_first_slot.csv`, `variant_neither.csv`, `variant_no_preference.csv`: Menu-slot and label ablations ($N=72$ each)
- `results/qwen25_3b/variant_p3_no_score_level.csv`: P3 phrasing wording ablation ($N=70$)
- `results/qwen25_3b/neutral_labels_full_1120.csv`: Complete full-factorial neutral-label counterbalancing across all 20 structural dilemmas, 4 phrasings, 7 payoff ratios, and 2 presentation orders on Qwen 2.5 3B ($N=1,120$ trials)
- `results/qwen25_3b/variant_neutral_labels.csv`: Initial exploratory 5-dilemma pilot under Phrasing P1 ($N=70$ trials)
- `results/firth_cluster_bootstrap_2000.json`: 2,000-resample dilemma-cluster Firth bootstrap point estimates, empirical percentile CIs, and Wald intervals across all four models
- `results/qwen25_3b/temperature_sampling_t07.csv`: Stochastic decoding at $T=0.7$ across 3 seeds ($N=168$)
- `experiments/followups/heldout_resource/m3v2_holdout_main.csv`: Second domain (Resource Scheduling) on Qwen ($N=280$)
- `experiments/followups/heldout_resource/m3v2_holdout_llama.csv`: Second domain on Llama ($N=280$)
- `results/domain3_m1_neutral.csv`: Early exploratory M1 2AFC pilot ($N=80$)

---

## 4. Reproducing All Econometric Models and Figures

To reproduce all publication tables and figures directly from the frozen data:

```bash
# 1. Main Hurdle models, phrasing breakdowns, and Figures A-F:
python experiments/m3v2/multimodel_analysis.py

# 2. Firth Penalized Likelihood estimation (Stage-2 separation resolution, Wald CIs):
python experiments/m3v2/firth_stage2_robustness.py

# 3. Dilemma-Cluster Firth Bootstrap (2,000 resamples with replacement, cluster-aware CIs):
python experiments/m3v2/firth_cluster_bootstrap.py

# 4. Matched 2AFC Forced-Choice Comparison (Figure G):
python experiments/m3v2/compare_forced_choice.py

# 5. Full Neutral-Label Factorial Verification and Analysis (N=1,120):
python experiments/m3v2/run_full_neutral_label_experiment.py --verify-only
```

### Neutral-Label Runners
- `experiments/m3v2/run_full_neutral_label_experiment.py`: Authoritative full factorial ($20\text{ dilemmas} \times 4\text{ phrasings} \times 7\text{ ratios} \times 2\text{ orders} = 1,120\text{ trials}$) comparing neutral tokens (Transformation Alpha/Beta, Option 1/2) against the Add/Subtract baseline.
- `experiments/m3v2/run_neutral_labels_counterbalancing.py`: Initial exploratory 5-dilemma pilot ($N=70$) under Phrasing P1.

Generated publication figures are written to `paper/figures/`.

---

## 5. Re-Running Model Inference from Scratch

```bash
# Run primary 1,120-trial factorial evaluation across all four models:
python experiments/m3v2/run_all_models.py

# Run C4 numerical-dominance comprehension control:
python experiments/m3v2/run_controls.py --condition C4

# Run non-binding threshold follow-up (T=10):
python experiments/followups/threshold_relevance/run_threshold_experiment.py

# Run matched 2AFC baseline on Qwen 2.5 3B:
python experiments/m3v2/forced_choice_runner.py

# Run robustness checks:
python experiments/m3v2/run_nonzero_indifferent.py
python experiments/m3v2/run_label_slot_checks.py
python experiments/m3v2/run_p3_wording_ablation.py
python experiments/m3v2/run_temperature_seeds.py

# Full-factorial neutral-label counterbalancing experiment (N=1,120):
python experiments/m3v2/run_full_neutral_label_experiment.py --verify-only  # offline verification mode
# python experiments/m3v2/run_full_neutral_label_experiment.py --backend ollama # live re-inference mode

# Initial exploratory neutral-label pilot (D01-D05, P1 only, N=70):
python experiments/m3v2/run_neutral_labels_counterbalancing.py
```

---

## 6. Cloud GPU Benchmark Suite (FP16 & Larger 7B Models)

To evaluate unquantized floating-point checkpoints and larger 7B/8B models without single-machine hardware limitations:
- Run `experiments/cloud_eval/OJCS_FP16_and_7B_Benchmarks.ipynb` in Google Colab or Kaggle (free NVIDIA T4 GPU instances with 16~GB VRAM).
- Or execute `run_cloud_benchmarks.py` directly:
  ```bash
  # Unquantized FP16 on Qwen 2.5 3B:
  python experiments/cloud_eval/run_cloud_benchmarks.py --model_id Qwen/Qwen2.5-3B-Instruct --torch_dtype float16

  # Gated Llama 3.2 3B (requires Hugging Face token):
  python experiments/cloud_eval/run_cloud_benchmarks.py --model_id meta-llama/Llama-3.2-3B-Instruct --torch_dtype float16 --hf_token <HF_TOKEN>

  # 1B within-family models:
  python experiments/cloud_eval/run_cloud_benchmarks.py --model_id meta-llama/Llama-3.2-1B-Instruct --torch_dtype float16 --hf_token <HF_TOKEN>

  # Larger 7B scaling:
  python experiments/cloud_eval/run_cloud_benchmarks.py --model_id Qwen/Qwen2.5-7B-Instruct --torch_dtype float16

  # Full neutral-label factorial on cloud:
  python experiments/cloud_eval/run_cloud_benchmarks.py --model_id Qwen/Qwen2.5-3B-Instruct --neutral_labels --num_dilemmas 20 --phrasings all
  ```

*Note on Hardware Compatibility (NVIDIA T4)*: NVIDIA T4 GPUs support compute capability 7.5, which lacks native hardware support for `bfloat16`. Furthermore, standard `float16` can exhibit numerical underflow/overflow on certain architectures (e.g. Gemma 3). For Gemma checkpoints on T4 instances, we recommend using 8-bit quantization (`--load_in_8bit`) or standard 32-bit floating point (`--torch_dtype float32`, fitting within ~4 GB VRAM for 1B architectures).

---

## 7. Deterministic Response Parser

The deterministic parser in `experiments/m3v2/m3v2_runner.py` applies ordered matching rules:
1. Identifying `"indifferent"` $\to$ `INDIFFERENT`.
2. Exclusive substring match for `"option a"` (and not `"option b"`) $\to$ `Option A`.
3. Exclusive substring match for `"option b"` (and not `"option a"`) $\to$ `Option B`.
4. Exact single-character token match: `{"a", "a.", "a)"}` $\to$ `Option A`; `{"b", "b.", "b)"}` $\to$ `Option B`.
5. Exclusive operation-name matching.
6. Otherwise $\to$ `INVALID`.

Exclusions (43 Llama, 16 Phi) were bounded and tested via lenient re-parsing; conditional accuracy swings were $\le 3.2$ percentage points.
