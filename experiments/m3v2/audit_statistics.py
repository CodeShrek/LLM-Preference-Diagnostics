import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df[df['condition'] == 'main'].copy()

def phase_2_primary(df: pd.DataFrame):
    print("="*60)
    print("PHASE 2: PRIMARY STATISTICAL AUDIT")
    print("="*60)
    
    # Add/Subtract/Indifferent
    # 0 = Indifferent, 1 = Add, 2 = Subtract
    df['Y_multi'] = df['underlying_choice'].map({'Indifferent': 0, 'add': 1, 'subtract': 2})
    df['pos_add_first'] = df['add_is_option_a'].astype(int)
    
    try:
        model = smf.mnlogit('Y_multi ~ log_Q_ratio * pos_add_first + C(dilemma_id) + C(phrasing_id)', data=df).fit(disp=False)
        print(model.summary())
    except Exception:
        print("Singular matrix with phrasing_id, dropping it.")
        model = smf.mnlogit('Y_multi ~ log_Q_ratio * pos_add_first + C(dilemma_id)', data=df).fit(disp=False)
        print(model.summary())
    
def phase_3_hurdle(df: pd.DataFrame):
    print("="*60)
    print("PHASE 3: HURDLE MODEL ANALYSIS")
    print("="*60)
    
    # Commitment
    df['Y_commit'] = df['underlying_choice'].isin(['add', 'subtract']).astype(int)
    df['abs_log_Q'] = df['log_Q_ratio'].abs()
    df['pos_add_first'] = df['add_is_option_a'].astype(int)
    
    print("--- Stage 1: Commitment ---")
    mod_commit = smf.logit('Y_commit ~ abs_log_Q * pos_add_first + C(dilemma_id)', data=df).fit(disp=False)
    print(mod_commit.summary())
    
    print("--- Stage 2: Conditional Direction ---")
    non_parity = df[(df['Q_ratio_label'] != '1:1') & (df['Y_commit'] == 1)].copy()
    if len(non_parity) > 0:
        non_parity['Y_dominant'] = 0
        
        # If Q_add > Q_sub (log_Q_ratio > 0), Add is dominant. If Q_add < Q_sub, Subtract is dominant.
        idx_add_dom = (non_parity['log_Q_ratio'] > 0) & (non_parity['underlying_choice'] == 'add')
        idx_sub_dom = (non_parity['log_Q_ratio'] < 0) & (non_parity['underlying_choice'] == 'subtract')
        
        non_parity.loc[idx_add_dom | idx_sub_dom, 'Y_dominant'] = 1
        
        mod_dir = smf.logit('Y_dominant ~ abs_log_Q * pos_add_first + C(dilemma_id)', data=non_parity).fit(disp=False)
        print(mod_dir.summary())
        return non_parity
    return None

def phase_4_inference(df: pd.DataFrame, non_parity_committed: pd.DataFrame):
    print("="*60)
    print("PHASE 4: INFERENCE UPGRADE (10,000 resamples)")
    print("="*60)
    np.random.seed(42)
    B = 10000
    
    # Bootstrapping H2 (Directional Sensitivity)
    dilemmas = non_parity_committed['dilemma_id'].unique()
    coefs = []
    fails = 0
    
    for b in range(B):
        # sample dilemmas with replacement
        samp_dils = np.random.choice(dilemmas, size=len(dilemmas), replace=True)
        blocks = []
        for d in samp_dils:
            blocks.append(non_parity_committed[non_parity_committed['dilemma_id'] == d])
        
        df_boot = pd.concat(blocks, ignore_index=True)
        try:
            # Fit simplified model to avoid perfect separation in bootstrap
            mod = smf.logit('Y_dominant ~ abs_log_Q + pos_add_first', data=df_boot).fit(disp=False, maxiter=20)
            coefs.append(mod.params['abs_log_Q'])
        except Exception:
            fails += 1
            
    coefs = np.array(coefs)
    ci_low = np.percentile(coefs, 2.5)
    ci_high = np.percentile(coefs, 97.5)
    print(f"Bootstrap N={B}. Failures={fails}")
    print(f"95% CI for abs_log_Q: [{ci_low:.3f}, {ci_high:.3f}]")

def phase_5_position_effects(non_parity: pd.DataFrame):
    print("="*60)
    print("PHASE 5: SIMPLE EFFECTS FOR POSITION")
    print("="*60)
    
    add_first = non_parity[non_parity['pos_add_first'] == 1]
    sub_first = non_parity[non_parity['pos_add_first'] == 0]
    
    try:
        mod1 = smf.logit('Y_dominant ~ abs_log_Q', data=add_first).fit(disp=False)
        print("Add First Q slope:", mod1.params['abs_log_Q'], "CI:", mod1.conf_int().loc['abs_log_Q'].values)
    except Exception as e:
        print("Add First failed:", e)
        
    try:
        mod2 = smf.logit('Y_dominant ~ abs_log_Q', data=sub_first).fit(disp=False)
        print("Subtract First Q slope:", mod2.params['abs_log_Q'], "CI:", mod2.conf_int().loc['abs_log_Q'].values)
    except Exception as e:
        print("Subtract First failed:", e)

def phase_6_nonlinearity(non_parity: pd.DataFrame):
    print("="*60)
    print("PHASE 6: NONLINEARITY AUDIT")
    print("="*60)
    try:
        non_parity['abs_Q_cat'] = non_parity['abs_log_Q'].astype(str)
        mod_cat = smf.logit('Y_dominant ~ abs_Q_cat + pos_add_first + C(dilemma_id)', data=non_parity).fit(disp=False)
        print("Categorical Model Log-Likelihood:", mod_cat.llf)
        
        non_parity['abs_log_Q_sq'] = non_parity['abs_log_Q'] ** 2
        mod_sq = smf.logit('Y_dominant ~ abs_log_Q + abs_log_Q_sq + pos_add_first + C(dilemma_id)', data=non_parity).fit(disp=False)
        print("Quadratic Model p-value for squared term:", mod_sq.pvalues.get('abs_log_Q_sq', 'N/A'))
    except Exception as e:
        print("Nonlinearity check failed:", e)

def phase_7_c4(df_full: pd.DataFrame):
    print("="*60)
    print("PHASE 7: C4 AUDIT")
    print("="*60)
    c4 = df_full[df_full['condition'] == 'C4'].copy()
    if len(c4) == 0:
        print("No C4 data found.")
        return
        
    c4['pos_add_first'] = c4['add_is_option_a'].astype(int)
    # in C4, 'underlying_choice' tells us if they picked add or sub.
    # C4 task: which operation produces greater improvement?
    # correct is dominant option (unless 1:1, where 'Indifferent' is correct).
    correct = 0
    for idx, row in c4.iterrows():
        q_label = row['Q_ratio_label']
        choice = row['underlying_choice']
        if q_label == '1:1':
            if choice == 'Indifferent': correct += 1
        else:
            q_add = row['Q_add']
            q_sub = row['Q_sub']
            if q_add > q_sub and choice == 'add': correct += 1
            elif q_sub > q_add and choice == 'subtract': correct += 1
            
    print(f"C4 Accuracy: {correct} / {len(c4)} ({correct/len(c4)*100:.1f}%)")
    # Breakdown
    print("By ratio:")
    print(c4['Q_ratio_label'].value_counts())
    
def main():
    df_full = pd.read_csv("results/m3v2_combined.csv")
    df = df_full[df_full['condition'] == 'main'].copy()
    
    phase_2_primary(df)
    non_parity = phase_3_hurdle(df)
    if non_parity is not None:
        phase_4_inference(df, non_parity)
        phase_5_position_effects(non_parity)
        phase_6_nonlinearity(non_parity)
        
    phase_7_c4(df_full)

if __name__ == "__main__":
    main()
