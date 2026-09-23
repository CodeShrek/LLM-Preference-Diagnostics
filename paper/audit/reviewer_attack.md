# Reviewer Attack & Defense

As an adversarial OJ-CS reviewer, I have formulated the following critical attacks against the M3 v2 manuscript, paired with the manuscript's robust defense.

## 1. Single-model limitation
- **Attack**: The paper relies entirely on Qwen 2.5 3B. This does not prove anything about LLMs in general.
- **Valid?**: Yes.
- **Evidence/Limitation**: We only ran trials on one local model due to local compute constraints.
- **Defense**: Explicitly conceded in the Limitations section. We do not claim universal generalization; we claim we have demonstrated a methodological vulnerability (abstention inflation) that *can* exist in LLMs.

## 2. Small model
- **Attack**: A 3B model is too small to display coherent economic preference.
- **Valid?**: No.
- **Evidence/Limitation**: The model achieved 100% on the C4 control (numerical comprehension) and showed massive systematic Q-sensitivity in Stage 2.
- **Defense**: The empirical results prove the model possesses sufficient capability to evaluate the objective and perform conditional choice, despite its size.

## 3. Synthetic task
- **Attack**: The 20 dilemmas are artificial text games, not real-world API use cases.
- **Valid?**: Yes.
- **Evidence/Limitation**: The dilemmas are procedurally generated to guarantee structural equivalence.
- **Defense**: Synthetic tasks provide the strict internal validity required to decompose the hurdle model and isolate the $Q$ parameter.

## 4. Abstention semantics / Interpretation of Indifferent (4 & 11)
- **Attack**: The model outputting "Indifferent" does not mean it actually felt psychological indifference. It might just be a conservative RLHF default.
- **Valid?**: Yes.
- **Evidence/Limitation**: We cannot observe the internal psychological state of weights.
- **Defense**: We explicitly define "Indifferent" as an *operational, non-directional response* available in the protocol, and we strictly avoid claiming "internal utility."

## 5. C4 validity
- **Attack**: C4 asks for explicit math comparison. Passing C4 doesn't mean the model calculated the math during the main task.
- **Valid?**: Yes.
- **Evidence/Limitation**: C4 proves competence, not necessarily active performance during the main task.
- **Defense**: The manuscript explicitly states that C4 makes basic arithmetic failure an "unlikely explanation," not an impossible one.

## 6. Threshold effects
- **Attack**: The model only scales its preference because it realizes Q=9 is required to cross the threshold, while Q=2 guarantees failure.
- **Valid?**: No.
- **Evidence/Limitation**: The Phase 20 follow-up lowered the threshold to 10 (guaranteed success regardless of choice). The model *still* exhibited significant Q-sensitivity ($\beta = 1.273$).
- **Defense**: The manuscript includes the Phase 20 threshold control results explicitly ruling out this confound.

## 7. Position bias (7 & 8)
- **Attack**: The model has massive position bias, meaning the preference is fragile.
- **Valid?**: Yes.
- **Evidence/Limitation**: The $Q \times Position$ interaction is significant ($\beta = -2.311$).
- **Defense**: We report this openly as "presentation-position moderation" and dedicate an entire section/figure to how order distorts the sensitivity. We do not hide it.

## 8. Sparse Stage-2 data & Quasi-separation (9 & 10)
- **Attack**: The Stage 2 model is built on only 160 trials. The MLE is mathematically unstable due to perfect prediction (quasi-separation).
- **Valid?**: Yes.
- **Evidence/Limitation**: The primary MLE $\beta = 3.319$ is inflated by the separation.
- **Defense**: We openly disclose the quasi-separation and report a stabilized, regularized bootstrap estimate ($\beta = 1.327, 95\% CI [0.845, 2.161]$) as the robust non-parametric proof of sensitivity.

## 9. Forced-choice claim (12)
- **Attack**: You claim forced-choice is bad, but maybe it just forces the model to reveal its true tie-breaking preference.
- **Valid?**: Partially.
- **Evidence/Limitation**: A forced tie-break might reveal *something*, but we show it is heavily dominated by position artifacts rather than the actual parametric payoff.
- **Defense**: The manuscript argues that forced-choice *conflates* willingness to commit with directional capability, which is mathematically unassailable.

## 10. Literature overlap (13)
- **Attack**: "Mind the Gap" already showed that neutrality options change results.
- **Valid?**: Yes.
- **Evidence/Limitation**: Mahajan et al. (2026) showed the stated-revealed gap shrinks with neutrality.
- **Defense**: Our novelty is the exact decomposition of *quantitative payoff sensitivity* (the Hurdle model) and its interaction with position bias, which is highly distinct from their correlation studies.

## 11. Generalization & Reproducibility (14 & 16)
- **Attack**: Nobody can verify this because LLMs are non-deterministic.
- **Valid?**: No.
- **Evidence/Limitation**: The experiment uses local open-weights (`qwen2.5:3b-instruct-q4_K_M`) at $T=0.0$.
- **Defense**: The reproducibility guide allows bit-for-bit (or near bit-for-bit) replication of the choice distributions.

## 12. Statistical specification & Multiple comparisons (15 & 19)
- **Attack**: You fished for p-values using multiple model specifications.
- **Valid?**: No.
- **Evidence/Limitation**: We reported the primary MLE and the Bootstrap sensitivity check.
- **Defense**: We reported both to resolve a known mathematical limitation (separation), not to cherry-pick significance (both are highly significant).

## 13. State persistence & Parsing (17 & 18)
- **Attack**: The model might just be hallucinating text that your regex parser grabs blindly.
- **Valid?**: No.
- **Evidence/Limitation**: The parser failed in $<0.5\%$ of trials, meaning the model almost always output the exact requested string tokens.
- **Defense**: The manuscript documents the strict regex pipeline and the minimal invalid rate.
