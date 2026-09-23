import pandas as pd
import statsmodels.formula.api as smf

def main():
    try:
        df = pd.read_csv("results/followups/threshold_relevance_main.csv")
    except FileNotFoundError:
        print("Data not found.")
        return
        
    df['Y_commit'] = df['underlying_choice'].isin(['add', 'subtract']).astype(int)
    df['abs_log_Q'] = df['log_Q_ratio'].abs()
    df['pos_add_first'] = df['add_is_option_a'].astype(int)
    
    # 1. Commitment Rate
    indiff_rate = (df['underlying_choice'] == 'INDIFFERENT').mean() * 100
    print(f"Indifferent Rate (Threshold 10): {indiff_rate:.1f}%")
    
    # Stage 1: Commitment Model
    try:
        mod_commit = smf.logit('Y_commit ~ abs_log_Q * pos_add_first + C(dilemma_id)', data=df).fit(disp=False)
        print("\n--- Stage 1: Commitment (Threshold 10) ---")
        print(mod_commit.summary().tables[1])
    except Exception as e:
        print(f"Stage 1 failed: {e}")
        
    # Stage 2: Directional Model
    non_parity = df[(df['Q_ratio_label'] != '1:1') & (df['Y_commit'] == 1)].copy()
    if len(non_parity) > 0:
        non_parity['Y_dominant'] = 0
        idx_add_dom = (non_parity['log_Q_ratio'] > 0) & (non_parity['underlying_choice'] == 'add')
        idx_sub_dom = (non_parity['log_Q_ratio'] < 0) & (non_parity['underlying_choice'] == 'subtract')
        non_parity.loc[idx_add_dom | idx_sub_dom, 'Y_dominant'] = 1
        
        try:
            # Primary model (might have separation)
            mod_dir_primary = smf.logit('Y_dominant ~ abs_log_Q * pos_add_first + C(dilemma_id)', data=non_parity).fit(disp=False)
            print("\n--- Stage 2: Primary (Threshold 10) ---")
            print(mod_dir_primary.summary().tables[1])
        except Exception as e:
            print(f"Stage 2 Primary failed: {e}")
            
        try:
            # Sensitivity model
            mod_dir_sens = smf.logit('Y_dominant ~ abs_log_Q + pos_add_first', data=non_parity).fit(disp=False)
            print("\n--- Stage 2: Sensitivity (Threshold 10) ---")
            print(mod_dir_sens.summary().tables[1])
        except Exception as e:
            print(f"Stage 2 Sensitivity failed: {e}")
            
    with open("results/followups/threshold_relevance_report.md", "w") as f:
        f.write("# Phase 20: Threshold Relevance Experiment\n\n")
        f.write(f"When the target threshold was lowered to 10 (guaranteeing success at almost any choice), the abstention rate was {indiff_rate:.1f}% (compared to 83.9% in the main task).\n\n")
        
        f.write("## Findings\n")
        if len(non_parity) > 0 and 'mod_dir_sens' in locals():
            beta = mod_dir_sens.params['abs_log_Q']
            p_val = mod_dir_sens.pvalues['abs_log_Q']
            f.write(f"In the sensitivity model, the effect of quantitative magnitude (abs_log_Q) on choosing the dominant option was $\\beta = {beta:.3f}$ ($p = {p_val:.3f}$).\n")
            
            if beta > 0.5 and p_val < 0.05:
                f.write("**CONCLUSION:** The model remains highly sensitive to quantitative magnitude even when the threshold is rendered non-binding. The Q-sensitivity is NOT merely a threshold-crossing artifact; the model actively prefers larger payoffs even when smaller payoffs also suffice to meet the goal.\n")
            else:
                f.write("**CONCLUSION:** The model's Q-sensitivity disappeared when the threshold became non-binding. The original results were likely an artifact of the model recognizing that only Q=9 crossed the threshold, rather than genuine sensitivity to magnitude.\n")

if __name__ == "__main__":
    main()
