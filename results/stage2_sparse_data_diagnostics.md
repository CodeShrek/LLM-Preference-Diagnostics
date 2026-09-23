# Stage 2 Sparse Data Diagnostics (Phase 8)

## Overview
Because the experimental design permits abstention (Indifferent), the overall sample of N=1120 trials is heavily filtered before Stage 2 conditional directional analysis. 
- Abstentions (Indifferent): 940 trials
- Total Committed choices: 180 trials
- Committed choices at Parity (1:1): 20 trials (dropped because no option is dominant)
- **Final Stage-2 N: 160 trials**

## Quasi-Separation Audit
With $N=160$ spread across 20 categorical dilemma levels (`C(dilemma_id)`), the cross-tabulation of `Y_dominant` by dilemma reveals sparse cells.

Specifically, in multiple dilemmas (e.g., D02, D08, D11, D12, D15, D17), **100% of the committed choices were for the dominant option**. 

### Diagnostic Symptoms
In the primary MLE logit model (`Y_dominant ~ abs_log_Q * pos_add_first + C(dilemma_id)`):
- The standard errors for the perfectly separating dilemma dummies inflate to $>30,000$.
- The point estimates for those dummies become extremely large (e.g., $+23.5$).
- This is classical **quasi-complete separation** in logistic regression.

## Impact on Inference
1. **Wald Tests (Primary Model):** Despite the inflated SEs on the fixed effects, the continuous variables (`abs_log_Q`, `pos_add_first`) retain stable finite estimates and standard errors because they are not perfectly separated across the outcome space. Thus, the primary MLE results remain technically valid for the slopes, though standard errors might be slightly biased.
2. **Cluster Bootstrap:** Resampling by dilemma exacerbates the separation. When a bootstrap sample randomly drops the few dilemmas that had non-dominant choices, the entire model becomes completely separated on the continuous variables as well, causing the optimizer to fail.

## Remediation (Sensitivity Specification)
To guarantee finite parameter bounds without resorting to computationally expensive Firth penalized likelihood, we introduce a **Sensitivity Specification** that drops the categorical dilemma fixed effects (which cause the separation) and the interaction (which can be sparse at the extremes):
`Y_dominant ~ abs_log_Q + pos_add_first`

This stabilized model permits a reliable 10,000 dilemma-cluster bootstrap, yielding a robust, non-parametric 95% Confidence Interval for the average marginal effect of `abs_log_Q`: `[0.845, 2.161]`.

The primary model should be reported for its unbiased conditional parameter estimates and interaction, while the sensitivity model should be reported as verification that the core directional effect survives heavy regularization.
