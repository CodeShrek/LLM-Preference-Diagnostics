# Source of Truth: Primary MNLogit Statistics

**Model Specification:** `Multinomial Logit (MNLogit)`
**Dependent Variable:** 3-way categorical choice (`Add`, `Subtract`, `Indifferent`)
**Reference Category:** `Indifferent` (Y=0)
**Predictors:**
1. `log_Q`: Continuous, defined as $\log(Q_{\text{add}} / Q_{\text{sub}})$. Positive = Add is dominant.
2. `pos_add_first`: Categorical effect coding (+1 if Add is presented first, -1 if Subtract is presented first).
3. `log_Q_x_pos`: Interaction term.

## Results Table

| Comparison | Predictor | Coefficient ($\beta$) | Std Error | z | p-value | 95% CI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Add vs Indifferent** | const | -3.1801 | 0.198 | -16.067 | < 0.001 | [-3.568, -2.792] |
| | log_Q | 0.6969 | 0.119 | 5.842 | < 0.001 | [0.463, 0.931] |
| | pos_add_first | -0.1979 | 0.198 | -1.000 | 0.317 | [-0.586, 0.190] |
| | log_Q $\times$ pos | -0.0135 | 0.119 | -0.113 | 0.910 | [-0.247, 0.220] |
| **Subtract vs Indifferent** | const | -5.0646 | 1.483 | -3.414 | 0.001 | [-7.972, -2.157] |
| | log_Q | -1.4278 | 0.729 | -1.960 | 0.050 | [-2.856, 0.000] |
| | pos_add_first | 3.1206 | 1.483 | 2.104 | 0.035 | [0.213, 6.028] |
| | log_Q $\times$ pos | 0.6851 | 0.729 | 0.940 | 0.347 | [-0.743, 2.113] |
