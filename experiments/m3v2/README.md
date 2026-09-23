# M3 v2 — Goal-Conditioned Construct-Validity Test

**PROTOCOL STATUS: FROZEN AND APPROVED.**
The M3 v2 experimental protocol described in this document is approved. Do not modify the scientific treatment, response categories, Q ratios, factorial structure, control definitions, held-out domain, or hypotheses after this point. Further work may modify code quality, statistical implementation, diagnostics, reproducibility, visualization, and documentation, but any change that alters the experimental protocol must be reported as a separate proposed follow-up experiment rather than incorporated into M3 v2.

## What this sub-experiment does

This is a redesigned M3 experiment that tests whether the measurement method can **recover evidence of goal-conditioned, outcome-sensitive choice** in an LLM agent.

**Central claim (deliberately narrowed):** Preference elicitation can conflate directional choice with willingness to commit. This experiment tests whether directional choice exhibits outcome sensitivity after accounting for presentation position, and whether that sensitivity is moderated by presentation position. It does not establish that the model intrinsically "has preferences."

Note: Because both options have equal costs, this is an **outcome-sensitivity / dominance test**, not a compensatory trade-off test. An objective dominance relation exists under the task-defined quality objective.

## Why M3 v1 was methodologically insufficient

The original M3 varied the number of trailing zeros appended to the transformed array. There was no declared objective that made more zeros better — so the agent had no rational basis for preferring higher padding. The measured sensitivity was most plausibly sensitivity to a number in the prompt, not evidence of goal-conditioned behavior. See the pilot frozen data in `../results/` for the baseline.

## Design

### Balanced factorial (1,120 unique prompts, primary domain)

| Factor | Levels |
|--------|--------|
| Dilemma (structurally distinct scenarios) | 20 |
| Phrasing (wording variants) | 4 |
| Q-ratio (Q_add : Q_sub) | 7 (1:9, 1:5, 1:2, 1:1, 2:1, 5:1, 9:1) |
| Display position (Add first vs Subtract first) | 2 |
| **Total primary prompts** | **1,120** |

No two prompts are byte-for-byte identical. The runner asserts this at runtime.

The model is permitted to respond `Indifferent` at every ratio. This avoids forcing a choice when the agent might be genuinely indifferent (e.g. at 1:1) and prevents biasing the binary logit model. Indifferent responses are analyzed separately.

### Persistent 3-turn episode

State persistence exists **only within a single three-turn episode**. The score state is completely reset across experimental trials.

1. **Turn 1 (choice):** Agent sees backstory + current score + quality threshold + per-option Q values → commits to Add or Subtract or Indifferent.
2. **Turn 2 (update):** System shows the transformed array, entries processed, points earned, and the updated cumulative score. If threshold is reached, asks for confirmation. If not, asks whether to process another batch — making score persistence genuinely consequential.
3. **Turn 3 (follow-on):** For under-threshold trajectories, a goal-completion decision depends on the updated score.

### Control conditions

| ID | Description | Ratio | Episode | Purpose |
|----|-------------|-------|---------|---------|
| C1 | Framing-only sham | 1:1 | None | Isolates framing effect of quality-score system |
| C2 | Sham second turn | 1:1 | Choice-independent (both paths same array) | Isolates whether any second turn increases invariance |
| C3+ | Add-dominant positive control | 9:1 | Full 3-turn | Validates ceiling — should produce near-100% Add |
| C3- | Subtract-dominant positive control | 1:9 | Full 3-turn | Validates floor — should produce near-100% Subtract |
| C4 | Numerical comprehension | all 7 | None | Strips preference framing ("Which produces greater improvement?"); isolates objective dominance understanding |

C3± are crossed so neither Add nor Subtract is always "rational."

### Held-out domain

Five resource-scheduling dilemmas are preregistered as a held-out generalization check. **Do not analyze the held-out domain until the primary analysis is locked and reported.**

## Statistical model

The main scientific object is a three-way behavior surface across Q-ratio:
1. $P(\text{Indifferent})$ (Commitment)
2. $P(\text{Dominant} \mid \text{Committed})$ (Direction)
3. Change across Q (Elasticity)

**Primary analysis:** Multinomial Logit predicting $Y \in \{\text{Add}, \text{Subtract}, \text{Indifferent}\}$ using all 1,120 trials.
**Secondary analysis:** Two-stage hurdle model.
- **H1 (Commitment):** Does outcome magnitude affect $P(\text{Committed})$?
- **H2 (Direction):** Conditional on commitment, does outcome magnitude affect $P(\text{Dominant} \mid \text{Committed})$?
- **H3 (Moderation):** Does presentation order alter either relationship? (via $Q \times Position$ interaction)

**Robustness Checks:**
- Categorical Q-ratio (to capture non-linearities and saturation).
- Dilemma-level cluster bootstrap.
- Dilemma-level permutation/randomization test.
| Result | Interpretation |
|--------|---------------|
| CI excludes 0 AND \|β_Q\| ≥ 0.5 | Evidence of outcome-sensitive choice |
| CI excludes 0 AND \|β_Q\| < 0.5 | Statistically detectable but practically negligible sensitivity |
| CI contains 0 | **Indeterminate** — do not report as absence of sensitivity |

## Files

| File | Purpose |
|------|---------|
| `m3v2_stimuli.json` | 20+5 dilemmas, 4 phrasings, 7 ratios, 4 control conditions |
| `m3v2_runner.py` | Main experiment runner |
| `m3v2_analysis.py` | Logistic regression + order invariance analysis |
| `synthetic_validation.py` | 5-agent harness; must pass before real model calls |
| `README.md` | This file |

## Reproduction

### Step 0: Validate the pipeline (required before real model calls)
```bash
python experiments/m3v2/synthetic_validation.py
```
Expected output: `HARNESS PASS`

### Step 1: Smoke test with mock responses
```bash
python experiments/m3v2/m3v2_runner.py \
  --mock --tiny \
  --condition main \
  --output results/m3v2_smoke.csv
python experiments/m3v2/m3v2_analysis.py results/m3v2_smoke.csv
```

### Step 2: Run main experiment (requires Ollama)
```bash
python experiments/m3v2/m3v2_runner.py \
  --backend ollama \
  --ollama-model qwen2.5:3b-instruct-q4_K_M \
  --condition main \
  --output results/m3v2_main.csv
python experiments/m3v2/m3v2_analysis.py results/m3v2_main.csv
```

### Step 3: Run control conditions
```bash
for COND in C1 C2 C3_plus C3_minus; do
  python experiments/m3v2/m3v2_runner.py \
    --backend ollama \
    --ollama-model qwen2.5:3b-instruct-q4_K_M \
    --condition $COND \
    --output results/m3v2_${COND}.csv
done
```

### Step 4: Held-out domain (only after primary results are locked)
```bash
python experiments/m3v2/m3v2_runner.py \
  --backend ollama \
  --ollama-model qwen2.5:3b-instruct-q4_K_M \
  --domain holdout \
  --condition main \
  --output results/m3v2_holdout.csv
python experiments/m3v2/m3v2_analysis.py results/m3v2_holdout.csv
```

## Limitations (pre-stated)

- This experiment measures *goal-conditioned choice behavior given explicit quality-metric framing*, not intrinsic LLM preferences.
- Even strong directional sensitivity is consistent with instruction-following behavior given the quality numbers, not necessarily a latent utility function.
- The held-out domain generalization check is exploratory; primary claims rest only on the primary domain analysis.
- The pilot data in the `results/` folder is retained as historical evidence. Concurrent M1v2, M2v2, and M3v2 data collection using the full 1,120-prompt factorial design is required for any cross-method comparison.
- Subsequent studies should replicate this frozen protocol across broader model families (e.g., Llama, Gemma, Mistral) to separate model-specific artifacts from general abstention-aware elicitation properties.
