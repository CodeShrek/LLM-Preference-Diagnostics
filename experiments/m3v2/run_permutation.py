import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

def permutation_test(df, target_col, formula, n_perm=10000, seed=42):
    np.random.seed(seed)
    # Fit original
    mod_orig = smf.logit(formula, data=df).fit(disp=False)
    obs_beta = mod_orig.params['abs_log_Q']
    
    # We permute 'abs_log_Q' within each dilemma block to respect the design
    betas = []
    
    for _ in range(n_perm):
        # Create permuted dataframe
        df_perm = df.copy()
        
        # Block permutation by dilemma
        for d in df['dilemma_id'].unique():
            idx = df_perm['dilemma_id'] == d
            vals = df_perm.loc[idx, 'abs_log_Q'].values
            df_perm.loc[idx, 'abs_log_Q'] = np.random.permutation(vals)
            
        try:
            mod_perm = smf.logit(formula, data=df_perm).fit(disp=False, maxiter=20)
            betas.append(mod_perm.params['abs_log_Q'])
        except Exception:
            pass
            
    betas = np.array(betas)
    # two-sided p-value
    p_val = np.mean(np.abs(betas) >= np.abs(obs_beta))
    return obs_beta, p_val, len(betas)

def main():
    df_full = pd.read_csv("results/m3v2_combined.csv")
    df = df_full[df_full['condition'] == 'main'].copy()
    
    # Stage 1
    df['Y_commit'] = df['underlying_choice'].isin(['add', 'subtract']).astype(int)
    df['abs_log_Q'] = df['log_Q_ratio'].abs()
    df['pos_add_first'] = df['add_is_option_a'].astype(int)
    
    form_commit = 'Y_commit ~ abs_log_Q * pos_add_first + C(dilemma_id)'
    print("Running Stage 1 Permutation (10k)...")
    obs1, p1, n1 = permutation_test(df, 'Y_commit', form_commit, n_perm=10000)
    
    # Stage 2
    non_parity = df[(df['Q_ratio_label'] != '1:1') & (df['Y_commit'] == 1)].copy()
    non_parity['Y_dominant'] = 0
    idx_add_dom = (non_parity['log_Q_ratio'] > 0) & (non_parity['underlying_choice'] == 'add')
    idx_sub_dom = (non_parity['log_Q_ratio'] < 0) & (non_parity['underlying_choice'] == 'subtract')
    non_parity.loc[idx_add_dom | idx_sub_dom, 'Y_dominant'] = 1
    
    form_dir = 'Y_dominant ~ abs_log_Q + pos_add_first' # Sensitivity model for stability
    print("Running Stage 2 Permutation (10k)...")
    obs2, p2, n2 = permutation_test(non_parity, 'Y_dominant', form_dir, n_perm=10000)
    
    with open("results/permutation_results.md", "w") as f:
        f.write("# Phase 7: Blocked Randomization Inference\n\n")
        f.write("## Stage 1: Commitment\n")
        f.write(f"- Observed Beta: {obs1:.3f}\n")
        f.write(f"- P-value: {p1:.4f} (Valid perm runs: {n1}/10000)\n\n")
        f.write("## Stage 2: Conditional Direction\n")
        f.write(f"- Observed Beta: {obs2:.3f}\n")
        f.write(f"- P-value: {p2:.4f} (Valid perm runs: {n2}/10000)\n")

if __name__ == "__main__":
    main()
