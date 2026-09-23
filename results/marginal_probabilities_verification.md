# Phase 10: Marginal Probabilities Verification

## 1. Stage 2 Conditional Direction P(Dominant | Commit)
Calculated from the primary MLE specification: `Y_dominant ~ abs_log_Q * pos_add_first + C(dilemma_id)`.
Probabilities are marginalized (averaged) over the empirical distribution of dilemmas in the sample.

| Q Ratio | P(Dominant), Add-First | P(Dominant), Subtract-First |
|---------|------------------------|-----------------------------|
| 1:2 / 2:1 | 0.747 | 0.389 |
| 1:5 / 5:1 | 0.863 | 0.821 |
| 1:9 / 9:1 | 0.914 | 0.964 |

## 2. Parity (1:1) Conditional Probabilities
At parity, 'Dominant' is undefined. We use the full MNLogit model to estimate P(Add | Commit) and P(Subtract | Commit).

- **P(Indifferent)** at Parity: 0.893
- **P(Add | Commit)** at Parity: 0.382
- **P(Subtract | Commit)** at Parity: 0.618
