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

    # Original primary model (with dilemma fixed effects and interaction)
    f_primary = 'Y_dominant ~ abs_log_Q * pos_add_first + C(dilemma_id)'
    mod_primary = smf.logit(f_primary, data=non_parity).fit(disp=False)
    
    print("=== PRIMARY MODEL (Full interaction + Dilemma FE) ===")
    print(mod_primary.summary().tables[1])
    
    # Simple Sensitivity Model (No interaction, No dilemma FE)
    f_sens = 'Y_dominant ~ abs_log_Q + pos_add_first'
    mod_sens = smf.logit(f_sens, data=non_parity).fit(disp=False)
    
    print("\n=== SENSITIVITY MODEL (No interaction, No dilemma FE) ===")
    print(mod_sens.summary().tables[1])

if __name__ == "__main__":
    main()
