"""
Firth Penalized Logistic Regression — Stage 2 Robustness Check
Addresses: R2 point 7, R3 point 2 (quasi-complete separation)

Implements Firth's (1993) penalized likelihood logistic regression.
The penalized log-likelihood is:
    L_firth(beta) = L_MLE(beta) + 0.5 * log(|I(beta)|)

This produces finite estimates under separation by pulling extreme
coefficients toward zero via the Jeffrey's prior penalty.

References:
  Firth D. (1993). Bias reduction of maximum likelihood estimates.
  Biometrika, 80(1), 27-38. doi:10.1093/biomet/80.1.27
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid function
from numpy.linalg import slogdet
import json

# ---------------------------------------------------------------------------
# Firth penalized logistic regression (from scratch)
# ---------------------------------------------------------------------------

def firth_loglikelihood(beta, X, y):
    """Firth's penalized log-likelihood with Jeffrey's prior."""
    p = expit(X @ beta)
    # Clip for numerical stability
    p = np.clip(p, 1e-10, 1 - 1e-10)
    
    # MLE log-likelihood
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    
    # Fisher information matrix: I = X' * W * X, W = diag(p*(1-p))
    W = p * (1 - p)
    XtWX = X.T @ (W[:, None] * X)
    
    # Firth penalty: 0.5 * log(det(I(beta)))
    sign, logdet = slogdet(XtWX)
    if sign <= 0:
        penalty = 0
    else:
        penalty = 0.5 * logdet
    
    return -(ll + penalty)  # negative because we minimize


def firth_gradient(beta, X, y):
    """Gradient of Firth's penalized negative log-likelihood."""
    p = expit(X @ beta)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    W = p * (1 - p)
    
    XtWX = X.T @ (W[:, None] * X)
    
    # Try to invert; if singular, skip penalty gradient
    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        return -(X.T @ (y - p))
    
    # Hat matrix diagonal h_i = x_i' * (X'WX)^{-1} * x_i * w_i
    h = np.einsum('ij,jk,ik->i', X, XtWX_inv, X) * W
    
    # Firth correction term: X' * (h * (0.5 - p))
    correction = X.T @ (h * (0.5 - p))
    
    grad_mle = X.T @ (y - p)
    return -(grad_mle + correction)


def fit_firth(X, y, max_iter=200, tol=1e-8):
    """
    Fit Firth penalized logistic regression.
    Returns: (coefficients, standard_errors, convergence_flag)
    """
    beta0 = np.zeros(X.shape[1])
    
    result = minimize(
        firth_loglikelihood,
        beta0,
        jac=firth_gradient,
        args=(X, y),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol}
    )
    
    beta_hat = result.x
    converged = result.success
    
    # Compute standard errors from observed Fisher information at beta_hat
    p = expit(X @ beta_hat)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    W = p * (1 - p)
    XtWX = X.T @ (W[:, None] * X)
    
    try:
        cov = np.linalg.inv(XtWX)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(len(beta_hat), np.nan)
    
    return beta_hat, se, converged


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

MODELS = ["qwen25_3b", "llama32_3b", "phi4mini_38b", "gemma3_4b"]
MODEL_NAMES = {
    "qwen25_3b": "Qwen 2.5 3B",
    "llama32_3b": "Llama 3.2 3B",
    "phi4mini_38b": "Phi-4 Mini 3.8B",
    "gemma3_4b": "Gemma 3 4B"
}

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


def load_model_data(model_key: str) -> pd.DataFrame:
    csv_path = RESULTS_DIR / model_key / "primary.csv"
    if not csv_path.exists():
        # Try the combined CSV
        combined = RESULTS_DIR / "m3v2_combined.csv"
        if combined.exists():
            df = pd.read_csv(combined)
            df = df[df["model"] == model_key].copy()
        else:
            raise FileNotFoundError(f"Cannot find data for {model_key}")
    else:
        df = pd.read_csv(csv_path)
    
    # Filter to valid trials
    df = df[df["underlying_choice"].isin(["add", "subtract", "INDIFFERENT"])].copy()
    
    # Prepare variables
    df["Y_commit"] = (df["underlying_choice"] != "INDIFFERENT").astype(int)
    df["abs_log_Q"] = df["log_Q_ratio"].abs()
    df["pos_effect"] = df["add_is_option_a"].apply(lambda x: 1 if x else -1)
    
    def get_dominant(row):
        if row["underlying_choice"] == "INDIFFERENT" or row["log_Q_ratio"] == 0:
            return np.nan
        if row["log_Q_ratio"] > 0:
            return 1 if row["underlying_choice"] == "add" else 0
        else:
            return 1 if row["underlying_choice"] == "subtract" else 0
    
    df["Y_dominant"] = df.apply(get_dominant, axis=1)
    
    return df


def prepare_stage2_matrix(df: pd.DataFrame, include_interaction: bool = False):
    """Prepare design matrix for Stage 2 with dilemma fixed effects (D02-D20)."""
    s2 = df[
        (df["Y_commit"] == 1) & 
        (df["abs_log_Q"] > 0) & 
        df["Y_dominant"].notna()
    ].copy()
    
    y = s2["Y_dominant"].values
    
    if include_interaction:
        X_cols = {
            "intercept": np.ones(len(s2)),
            "abs_log_Q": s2["abs_log_Q"].values,
            "pos_effect": s2["pos_effect"].values,
            "interaction": (s2["abs_log_Q"] * s2["pos_effect"]).values,
        }
    else:
        X_cols = {
            "intercept": np.ones(len(s2)),
            "abs_log_Q": s2["abs_log_Q"].values,
            "pos_effect": s2["pos_effect"].values,
        }
    
    # Add dilemma fixed effects (drop one for reference)
    dilemmas = sorted(s2["dilemma_id"].unique())
    ref = dilemmas[0]
    for d in dilemmas[1:]:
        X_cols[f"D_{d}"] = (s2["dilemma_id"] == d).astype(float).values
    
    X = np.column_stack(list(X_cols.values()))
    colnames = list(X_cols.keys())
    
    return X, y, colnames, len(s2)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_firth_analysis():
    results = {}
    
    for model_key in MODELS:
        name = MODEL_NAMES[model_key]
        print(f"\n{'='*60}")
        print(f"Model: {name}")
        print('='*60)
        
        try:
            df = load_model_data(model_key)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            results[model_key] = {"error": str(e)}
            continue
        
        # Stage 2 without interaction (stabilized spec)
        try:
            X, y, colnames, n = prepare_stage2_matrix(df, include_interaction=False)
            
            if len(y) == 0:
                print(f"  Stage 2: No observations (N=0)")
                results[model_key] = {"stage2_n": 0, "note": "No directional non-parity trials"}
                continue
            
            if np.std(y) == 0:
                print(f"  Stage 2: Outcome has zero variance (all same label)")
                results[model_key] = {"stage2_n": len(y), "note": "Perfect outcome — no variation"}
                continue
            
            beta, se, converged = fit_firth(X, y)
            
            # Extract payoff coefficient (abs_log_Q is index 1)
            payoff_idx = colnames.index("abs_log_Q")
            payoff_beta = beta[payoff_idx]
            payoff_se = se[payoff_idx]
            payoff_ci_lo = payoff_beta - 1.96 * payoff_se
            payoff_ci_hi = payoff_beta + 1.96 * payoff_se
            
            # Position coefficient
            pos_idx = colnames.index("pos_effect")
            pos_beta = beta[pos_idx]
            pos_se = se[pos_idx]
            
            print(f"  N (Stage 2): {n}")
            print(f"  Converged: {converged}")
            print(f"  Firth payoff coef (abs_log_Q): {payoff_beta:.4f} (SE={payoff_se:.4f})")
            print(f"  95% Penalized Wald CI: [{payoff_ci_lo:.4f}, {payoff_ci_hi:.4f}]")
            print(f"  Position coef: {pos_beta:.4f} (SE={pos_se:.4f})")
            
            results[model_key] = {
                "model_name": name,
                "stage2_n": n,
                "converged": converged,
                "firth_payoff_beta": float(payoff_beta),
                "firth_payoff_se": float(payoff_se),
                "firth_payoff_ci_lo": float(payoff_ci_lo),
                "firth_payoff_ci_hi": float(payoff_ci_hi),
                "firth_pos_beta": float(pos_beta),
                "firth_pos_se": float(pos_se),
                "colnames": colnames,
                "n_params": len(beta),
            }
            
        except Exception as e:
            print(f"  ERROR in Stage 2: {e}")
            results[model_key] = {"error": str(e)}
    
    return results


if __name__ == "__main__":
    print("Firth Penalized Logistic Regression — Stage 2 Robustness")
    print("Implements Firth (1993) Biometrika 80(1):27-38")
    
    results = run_firth_analysis()
    
    # Save results
    out_path = RESULTS_DIR / "firth_stage2_robustness.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")
    
    # Print summary table
    print("\n" + "="*70)
    print("FIRTH STAGE 2 SUMMARY TABLE")
    print("="*70)
    print(f"{'Model':<20} {'N':>6} {'Firth Beta':>12} {'95% CI':>22} {'Conv':>6}")
    print("-"*70)
    for model_key in MODELS:
        r = results.get(model_key, {})
        name = MODEL_NAMES[model_key]
        if "error" in r or "note" in r:
            note = r.get("error", r.get("note", "?"))
            print(f"{name:<20} {'—':>6} {'—':>12} {note[:22]:>22} {'—':>6}")
        elif "firth_payoff_beta" in r:
            ci = f"[{r['firth_payoff_ci_lo']:.3f}, {r['firth_payoff_ci_hi']:.3f}]"
            print(f"{name:<20} {r['stage2_n']:>6} {r['firth_payoff_beta']:>12.4f} {ci:>22} {str(r['converged']):>6}")
        else:
            print(f"{name:<20} {'?':>6} {'?':>12} {'?':>22} {'?':>6}")
