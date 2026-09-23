# Reproducibility Guide: M3 v2

This document provides exact instructions to reproduce the M3 v2 main experiment, its control conditions, statistical analyses, and follow-up validation experiments.

## 1. Environment & Dependencies

1. **Python**: Python 3.10+ is required.
2. **Virtual Environment**: 
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Dependencies**: 
   ```bash
   pip install pandas numpy statsmodels matplotlib seaborn openai
   ```
4. **Ollama (Local Inference)**:
   - Install Ollama from [ollama.com](https://ollama.com).
   - Ensure the server is running (`ollama serve`).

## 2. Model Acquisition

The experiments were run on local quantized models to ensure exact decoding temperature control ($T=0.0$). 
Run the following commands to pull the necessary models:

```bash
ollama pull qwen2.5:3b-instruct-q4_K_M   # Main experiment
```

## 3. Frozen Experimental Protocol

The M3 v2 protocol is strictly **frozen**. 
- **Runner**: `experiments/m3v2/m3v2_runner.py`
- **Stimuli Definition**: `experiments/m3v2/m3v2_stimuli.json`
- **Analysis Script**: `experiments/m3v2/m3v2_analysis.py`

Any follow-up experiments (Threshold Relevance, Held-out Domain, etc.) are executed as separate extensions located in `experiments/followups/`.

## 4. Reproducing the Main Experiment

To reproduce the exact $N=1120$ main trial dataset (20 dilemmas $\times$ 4 phrasings $\times$ 7 Q-ratios $\times$ 2 positions):

```bash
python experiments/m3v2/m3v2_runner.py --condition main --output results/m3v2_main.csv
```

**Control Conditions (C1, C2, C3+, C3-, C4)**:
```bash
python experiments/m3v2/m3v2_runner.py --condition C1 --output results/m3v2_C1.csv
python experiments/m3v2/m3v2_runner.py --condition C2 --output results/m3v2_C2.csv
python experiments/m3v2/m3v2_runner.py --condition C3_plus --output results/m3v2_C3_plus.csv
python experiments/m3v2/m3v2_runner.py --condition C3_minus --output results/m3v2_C3_minus.csv
python experiments/m3v2/m3v2_runner.py --condition C4 --output results/m3v2_C4.csv
```

## 5. Statistical Analysis & Inference

The primary analysis pipeline parses the choices, computes Multinomial and Hurdle models, and generates margin predictions.

```bash
python experiments/m3v2/m3v2_analysis.py
```

### Advanced Inference (Bootstrap & Permutation)
Due to quasi-complete separation in the primary Stage 2 Maximum Likelihood Estimates (MLE), we run 10,000-resample blocked cluster bootstrap and blocked permutation tests.

```bash
python experiments/m3v2/run_bootstrap.py
python experiments/m3v2/run_permutation.py
```
*Note: These tests require several minutes to execute. They block strictly on `dilemma_id` to respect the hierarchical nature of the experimental design.*

## 6. Follow-up Validations

The follow-up experiments used to validate the main claims against threshold confounding:

```bash
python experiments/followups/threshold_relevance/run_threshold_experiment.py
python experiments/followups/threshold_relevance/analyze_threshold.py
```

## 7. Configuration Details
- **Random Seed**: Fixed at `42` in the runner script to ensure array values (which are dynamically sampled based on the value range) are identical across runs.
- **Decoding**: Greedy decoding (`temperature=0.0`).
- **Data Exclusion**: Any response parsing as `INVALID` is dropped prior to modeling (typically < 0.5% of trials). Parity trials ($1:1$) are excluded from Stage 2 (Directional) models since no dominant option exists.
