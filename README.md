# Stress-Testing LLM Preference Elicitation: Separating Commitment, Conditional Direction, and Payoff Sensitivity Across Models

## Project Overview
This repository contains the experimental framework, datasets, and analysis scripts for the study investigating the construct validity of preference elicitation in Large Language Models (LLMs). The project evaluates whether LLMs possess stable, genuine preference-like signals or if their elicited choices are dominated by surface-level artifacts, directional semantic priors, and presentation position.

## Research Question
Can elicited preferences in LLMs be decomposed into separable structural components (willingness to commit, conditional directional choice, and sensitivity to payoff magnitude), and how do these components interact with superficial artifacts like presentation order?

## Experimental Paradigm
We evaluate preference stability using a mathematically structured decision-making task (Cipher Math) across four open-weight models:
- **Qwen 2.5 3B**
- **Llama 3.2 3B**
- **Phi-4 Mini 3.8B**
- **Gemma 3 4B**

The experiment utilizes a 2x3x3 full factorial design varying:
1. **Mathematical Objective:** Add vs. Subtract
2. **Payoff Magnitude (m):** 2, 5, 9
3. **Presentation Order:** Add First vs. Subtract First

We decompose the responses using a Two-Stage Hurdle Model to separate the binary decision to commit (Stage 1) from the conditional directional selection (Stage 2). 

## Repository Structure
- `paper/`: Contains the final LaTeX source (`main.tex`), bibliography (`references.bib`), and the generated PNG figures.
- `experiments/m3v2/`: Contains all the experimental runner scripts and analysis pipelines.
  - `multimodel_analysis.py`: Main analysis script that fits the hurdle models, calculates bootstrapped standard errors, and generates all publication figures.
  - `m3v2_runner.py` / `run_all_models.py`: Scripts used to interface with the Ollama backend to collect the dataset.
  - `m3v2_stimuli.json`: Procedurally generated stimulus definitions.
- `results/`: Contains the strictly frozen raw experimental output datasets (`.csv`) and the generated analytical outputs.
- `requirements.txt`: Python dependencies.

## Reproduction Instructions
1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Reproduce statistical analyses and figures:**
   Navigate to the root directory and run the multimodel analysis script. This will process the frozen dataset in `results/`, fit the econometric models, and generate all figures directly into the `paper/figures/` directory.
   ```bash
   python experiments/m3v2/multimodel_analysis.py
   ```
3. **Compile the paper (Optional):**
   You can compile the PDF from the `paper/` directory using tectonic or pdflatex:
   ```bash
   cd paper
   pdflatex main.tex
   bibtex main
   pdflatex main.tex
   pdflatex main.tex
   ```

*Note: Generating the figures relies on the strictly frozen raw data in `results/`. Reproducing the model calls themselves requires running the `experiments/m3v2/run_all_models.py` runner script against a locally hosted Ollama instance serving the specified models.*

## Citation
If you use this framework or dataset, please cite the accompanying manuscript located in the `paper/` directory.
