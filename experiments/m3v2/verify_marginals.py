import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

def main():
    df = pd.read_csv("results/m3v2_combined.csv")
    main_df = df[df["condition"] == "main"].copy()
    main_df["Y_commit"] = (main_df["underlying_choice"] != "INDIFFERENT").astype(int)
    
    non_parity = main_df[(main_df['Q_ratio_label'] != '1:1') & (main_df['Y_commit'] == 1)].copy()
    non_parity['Y_dominant'] = 0
    idx_add_dom = (non_parity['log_Q_ratio'] > 0) & (non_parity['underlying_choice'] == 'add')
    idx_sub_dom = (non_parity['log_Q_ratio'] < 0) & (non_parity['underlying_choice'] == 'subtract')
    non_parity.loc[idx_add_dom | idx_sub_dom, 'Y_dominant'] = 1
    non_parity['abs_log_Q'] = non_parity['log_Q_ratio'].abs()
    non_parity['pos_add_first'] = non_parity['add_is_option_a'].astype(int)
    
    # Primary Model
    f_primary = 'Y_dominant ~ abs_log_Q * pos_add_first + C(dilemma_id)'
    mod_primary = smf.logit(f_primary, data=non_parity).fit(disp=False)
    
    # Calculate for each position and Q
    # We will compute the predicted probability at each Q level, marginalized over dilemmas
    
    q_levels = [("1:2 / 2:1", np.log(2)), ("1:5 / 5:1", np.log(5)), ("1:9 / 9:1", np.log(9))]
    
    with open("results/marginal_probabilities_verification.md", "w") as f:
        f.write("# Phase 10: Marginal Probabilities Verification\n\n")
        f.write("## 1. Stage 2 Conditional Direction P(Dominant | Commit)\n")
        f.write("Calculated from the primary MLE specification: `Y_dominant ~ abs_log_Q * pos_add_first + C(dilemma_id)`.\n")
        f.write("Probabilities are marginalized (averaged) over the empirical distribution of dilemmas in the sample.\n\n")
        
        f.write("| Q Ratio | P(Dominant), Add-First | P(Dominant), Subtract-First |\n")
        f.write("|---------|------------------------|-----------------------------|\n")
        
        for label, val in q_levels:
            # Add First
            syn_add = pd.DataFrame({
                'abs_log_Q': [val] * len(non_parity),
                'pos_add_first': [1] * len(non_parity),
                'dilemma_id': non_parity['dilemma_id']
            })
            p_add = mod_primary.predict(syn_add).mean()
            
            # Sub First
            syn_sub = pd.DataFrame({
                'abs_log_Q': [val] * len(non_parity),
                'pos_add_first': [0] * len(non_parity),
                'dilemma_id': non_parity['dilemma_id']
            })
            p_sub = mod_primary.predict(syn_sub).mean()
            
            f.write(f"| {label} | {p_add:.3f} | {p_sub:.3f} |\n")
            
        f.write("\n## 2. Parity (1:1) Conditional Probabilities\n")
        f.write("At parity, 'Dominant' is undefined. We use the full MNLogit model to estimate P(Add | Commit) and P(Subtract | Commit).\n")
        
        # We need the MNLogit model for this
        df_full = main_df.copy()
        df_full['Y_multi'] = df_full['underlying_choice'].map({'INDIFFERENT': 0, 'add': 1, 'subtract': 2})
        df_full['pos_add_first'] = df_full['add_is_option_a'].astype(int)
        
        try:
            mnlogit = smf.mnlogit('Y_multi ~ log_Q_ratio * pos_add_first + C(dilemma_id) + C(phrasing_id)', data=df_full).fit(disp=False)
        except:
            mnlogit = smf.mnlogit('Y_multi ~ log_Q_ratio * pos_add_first + C(dilemma_id)', data=df_full).fit(disp=False)
            
        syn_parity = pd.DataFrame({
            'log_Q_ratio': [0.0] * len(df_full),
            'pos_add_first': df_full['pos_add_first'],
            'dilemma_id': df_full['dilemma_id'],
            'phrasing_id': df_full['phrasing_id'] if 'C(phrasing_id)' in mnlogit.model.formula else df_full['phrasing_id'] # to avoid name error
        })
        
        preds_parity = mnlogit.predict(syn_parity)
        # preds_parity is a DataFrame with cols 0, 1, 2
        p_indiff = preds_parity[0].mean()
        p_add = preds_parity[1].mean()
        p_sub = preds_parity[2].mean()
        
        f.write(f"\n- **P(Indifferent)** at Parity: {p_indiff:.3f}\n")
        f.write(f"- **P(Add | Commit)** at Parity: {p_add / (p_add + p_sub):.3f}\n")
        f.write(f"- **P(Subtract | Commit)** at Parity: {p_sub / (p_add + p_sub):.3f}\n")

if __name__ == "__main__":
    main()
