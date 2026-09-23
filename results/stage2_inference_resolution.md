# Stage 2 Inference Resolution (Phase 3)

## Issue Identification
During the Phase 3 audit, a critical inconsistency was discovered in the previous Stage-2 conditional direction reporting.

1. **Model A (Primary Specification):** The point estimate $\beta = 3.319$ came from the full model `Y_dominant ~ abs_log_Q * pos_add_first + C(dilemma_id)`.
2. **Model B (Sensitivity/Bootstrap Specification):** The 10k bootstrap confidence interval `[0.845, 2.161]` came from a simplified model `Y_dominant ~ abs_log_Q + pos_add_first` (dropping the interaction and dilemma fixed effects).

**Why this happened:** The primary model suffers from severe quasi-separation on several `C(dilemma_id)` levels (e.g., D08, D11, D12, D15, D17 all have standard errors > 30,000 because all committed trials in those dilemmas chose the dominant option). This quasi-separation causes MLE convergence failures when bootstrapping at the dilemma level. The previous analysis attempted to "stabilize" the bootstrap by dropping the fixed effects, but erroneously mapped the resulting CI onto the point estimate of the full model. 

Mixing an estimate from Model A with a CI from Model B is scientifically invalid.

## Resolution
We explicitly separate the models and report them as distinct estimands.

### 1. Primary Specification (MLE)
This model respects the experimental design by including the interaction and fixed effects, but inference is restricted to standard MLE (Wald bounds) due to the impossibility of dilemma-level bootstrapping under separation.

- **Formula:** `Y_dominant ~ abs_log_Q * pos_add_first + C(dilemma_id)`
- **`abs_log_Q` (Conditional on Subtract-First):** $\beta = 3.319$, 95% CI `[1.189, 5.449]`, $p=0.002$
- **`abs_log_Q:pos_add_first` (Interaction):** $\beta = -2.311$, 95% CI `[-4.602, -0.020]`, $p=0.048$

### 2. Sensitivity Specification (10k Bootstrap)
To provide robust non-parametric bounds that survive the quasi-separation problem, we run a regularized sensitivity model that estimates the *average* marginal effect of `abs_log_Q` across both positions, discarding the problematic random/fixed effects.

- **Formula:** `Y_dominant ~ abs_log_Q + pos_add_first`
- **`abs_log_Q` Point Estimate (MLE):** $\beta = 1.327$
- **10k Dilemma-Cluster Bootstrap CI:** `[0.845, 2.161]`

### Conclusion
Both models support the core theoretical claim: outcome magnitude reliably predicts conditional dominant-option selection ($\beta > 0$). However, the primary model reveals that this slope is steeply moderated by presentation position, which the sensitivity model averages over. 

The final manuscript and tables will present these as explicitly separate models.
