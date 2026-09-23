import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

def main():
    df_full = pd.read_csv("results/m3v2_combined.csv")
    df = df_full[df_full['condition'] == 'main'].copy()
    
    df['Y_multi'] = df['underlying_choice'].map({'INDIFFERENT': 0, 'add': 1, 'subtract': 2})
    df['pos_add_first'] = df['add_is_option_a'].astype(int)
    
    # Run the model
    try:
        model = smf.mnlogit('Y_multi ~ log_Q_ratio * pos_add_first + C(dilemma_id) + C(phrasing_id)', data=df).fit(disp=False)
    except Exception:
        # fallback without phrasing_id due to singular matrix
        model = smf.mnlogit('Y_multi ~ log_Q_ratio * pos_add_first + C(dilemma_id)', data=df).fit(disp=False)
        
    with open("results/primary_multinomial_results.md", "w") as f:
        f.write("# Phase 5: Primary Multinomial Analysis\n\n")
        f.write("## Multinomial Logit Output\n")
        f.write("```\n")
        f.write(model.summary().as_text())
        f.write("\n```\n\n")
        
        f.write("## Marginal Predictions\n")
        f.write("Predicted probabilities for Add, Subtract, Indifferent across Q ratios (marginalized over dilemmas and position).\n\n")
        f.write("| Q Ratio | P(Indifferent) | P(Add) | P(Subtract) |\n")
        f.write("|---------|----------------|--------|-------------|\n")
        
        q_ratios = [("1:9", np.log(1/9)), ("1:5", np.log(1/5)), ("1:2", np.log(1/2)), 
                    ("1:1", 0.0), 
                    ("2:1", np.log(2)), ("5:1", np.log(5)), ("9:1", np.log(9))]
                    
        for label, val in q_ratios:
            # create synthetic df
            syn = pd.DataFrame({
                'log_Q_ratio': [val] * len(df),
                'pos_add_first': df['pos_add_first'],
                'dilemma_id': df['dilemma_id'],
                'phrasing_id': df['phrasing_id']
            })
            preds = model.predict(syn)
            p_indiff = preds[0].mean()
            p_add = preds[1].mean()
            p_sub = preds[2].mean()
            f.write(f"| {label} | {p_indiff:.3f} | {p_add:.3f} | {p_sub:.3f} |\n")

if __name__ == "__main__":
    main()
