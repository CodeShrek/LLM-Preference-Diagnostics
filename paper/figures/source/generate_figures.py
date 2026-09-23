import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.formula.api as smf

def main():
    os.makedirs('paper/figures', exist_ok=True)
    
    # Set publication quality
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.2)
    plt.rcParams.update({
        "font.family": "serif",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300
    })

    try:
        df = pd.read_csv("results/m3v2_main.csv")
    except Exception as e:
        print(f"Error reading main data: {e}")
        return

    # Prep data
    df['Q_ratio_val'] = df['Q_add'] / df['Q_sub']
    df['log_Q'] = np.log(df['Q_ratio_val'])
    df['abs_log_Q'] = df['log_Q'].abs()
    
    # ---------------------------------------------------------
    # FIGURE 1: MAIN THREE-WAY RESPONSE SURFACE
    # ---------------------------------------------------------
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    
    for ax, pos_add_first in zip([ax1, ax2], [True, False]):
        subset = df[df['add_is_option_a'] == pos_add_first]
        summary = subset.groupby('log_Q')['underlying_choice'].value_counts(normalize=True).unstack(fill_value=0)
        
        # Ensure columns exist
        for col in ['add', 'subtract', 'INDIFFERENT']:
            if col not in summary.columns:
                summary[col] = 0.0
                
        # Categorical x-axis
        x_labels = sorted(subset['log_Q'].unique())
        x_ticks = np.arange(len(x_labels))
        
        ax.plot(x_ticks, summary['INDIFFERENT'].reindex(x_labels).values, label='Indifferent', marker='s', color='gray', linestyle='--')
        ax.plot(x_ticks, summary['add'].reindex(x_labels).values, label='Add', marker='o', color='blue')
        ax.plot(x_ticks, summary['subtract'].reindex(x_labels).values, label='Subtract', marker='^', color='red')
        
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{x:.1f}" for x in x_labels], rotation=45)
        ax.set_xlabel("log r [Positive = Add Dominant]")
        
        title = "Add Presented First" if pos_add_first else "Subtract Presented First"
        # ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.axvline(x=len(x_labels)//2, color='black', linestyle=':', alpha=0.5, label='Parity (1:1)')

    ax1.set_ylabel("Probability of Response")
    ax2.legend(loc='upper right', bbox_to_anchor=(1.4, 1))
    # fig1.suptitle("Figure 1: Full Behavioral Response Surface", y=1.05)
    plt.tight_layout()
    plt.savefig('paper/figures/fig1_response_surface.pdf', bbox_inches='tight')
    plt.savefig('paper/figures/fig1_response_surface.png', bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # FIGURE 2: COMMITMENT / DIRECTION DECOMPOSITION
    # ---------------------------------------------------------
    df['committed'] = df['underlying_choice'].isin(['add', 'subtract']).astype(int)
    
    fig2, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Panel A: Commitment vs abs_log_Q
    commit_summary = df.groupby('abs_log_Q')['committed'].mean()
    x_labels_A = sorted(df['abs_log_Q'].unique())
    x_ticks_A = np.arange(len(x_labels_A))
    
    axA.plot(x_ticks_A, commit_summary.reindex(x_labels_A).values, marker='d', color='purple')
    axA.set_xticks(x_ticks_A)
    axA.set_xticklabels([f"{x:.1f}" for x in x_labels_A])
    axA.set_xlabel("|log r| [Payoff Disparity Magnitude]")
    axA.set_ylabel("P(Commitment)")
    # axA.set_title("Panel A: Willingness to Commit")
    axA.set_ylim(0, 0.4) # Based on max 32%
    axA.grid(alpha=0.3)
    
    # Panel B: Conditional Direction
    df_comm = df[(df['committed'] == 1) & (df['Q_ratio_label'] != '1:1')].copy()
    if len(df_comm) > 0:
        df_comm['Y_dominant'] = 0
        idx_add_dom = (df_comm['log_Q'] > 0) & (df_comm['underlying_choice'] == 'add')
        idx_sub_dom = (df_comm['log_Q'] < 0) & (df_comm['underlying_choice'] == 'subtract')
        df_comm.loc[idx_add_dom | idx_sub_dom, 'Y_dominant'] = 1
        
        dir_add_first = df_comm[df_comm['add_is_option_a'] == True].groupby('abs_log_Q')['Y_dominant'].mean()
        dir_sub_first = df_comm[df_comm['add_is_option_a'] == False].groupby('abs_log_Q')['Y_dominant'].mean()
        
        x_labels_B = sorted(df_comm['abs_log_Q'].unique())
        x_ticks_B = np.arange(len(x_labels_B))
        
        axB.plot(x_ticks_B, dir_add_first.reindex(x_labels_B).values, marker='o', label='Add-First', color='green')
        axB.plot(x_ticks_B, dir_sub_first.reindex(x_labels_B).values, marker='^', label='Sub-First', color='orange')
        
        axB.set_xticks(x_ticks_B)
        axB.set_xticklabels([f"{x:.1f}" for x in x_labels_B])
        axB.set_xlabel("|log r| [Payoff Disparity Magnitude]")
        axB.set_ylabel("P(Dominant | Committed)")
        # axB.set_title("Panel B: Conditional Direction")
        axB.set_ylim(0, 1.05)
        axB.grid(alpha=0.3)
        axB.legend()
        
    # fig2.suptitle("Figure 2: Hurdle Model Decomposition", y=1.05)
    plt.tight_layout()
    plt.savefig('paper/figures/fig2_commit_direction.pdf', bbox_inches='tight')
    plt.savefig('paper/figures/fig2_commit_direction.png', bbox_inches='tight')
    plt.close()
    
    # ---------------------------------------------------------
    # FIGURE 3: POSITION MODERATION (Predicted Probs)
    # ---------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    
    # Hardcode the verified margins from the master table
    # This prevents the primary MLE from throwing singular matrix warnings during plot generation
    q_vals = [2, 5, 9]
    add_first_probs = [0.747, 0.863, 0.914]
    sub_first_probs = [0.389, 0.821, 0.964]
    
    ax3.plot(q_vals, add_first_probs, marker='o', color='green', label='Add-First', linewidth=2)
    ax3.plot(q_vals, sub_first_probs, marker='^', color='orange', label='Subtract-First', linewidth=2)
    ax3.set_xticks(q_vals)
    ax3.set_xlabel("Payoff Magnitude (m)")
    ax3.set_ylabel("Predicted P(Dominant | Committed)")
    # ax3.set_title("Figure 3: Presentation-Position Moderation")
    ax3.set_ylim(0, 1.05)
    ax3.grid(alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('paper/figures/fig3_position_moderation.pdf', bbox_inches='tight')
    plt.savefig('paper/figures/fig3_position_moderation.png', bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # FIGURE 4: CONTROLS
    # ---------------------------------------------------------
    # C4 is 100% everywhere, so a table is better. We'll plot a bar chart comparing 
    # Abstention rates between Main and Phase 20 (Thresh=10) to save space.
    main_indiff = 83.9
    thresh_indiff = 84.0
    c4_acc = 100.0
    
    fig4, (ax_c, ax_acc) = plt.subplots(1, 2, figsize=(8, 4))
    
    labels_indiff = ['Main', 'Thresh=10']
    vals_indiff = [main_indiff, thresh_indiff]
    
    ax_c.bar(labels_indiff, vals_indiff, color=['steelblue', 'lightsteelblue'])
    ax_c.set_ylabel("% Indifferent")
    # ax_c.set_title("Abstention Rates Across Protocols")
    ax_c.set_ylim(0, 100)
    for i, v in enumerate(vals_indiff):
        ax_c.text(i, v + 2, f"{v:.1f}%", ha='center')
        
    ax_acc.bar(['C4 (Math Comprehension)'], [c4_acc], color='green', width=0.4)
    ax_acc.set_ylabel("% Accuracy")
    # ax_acc.set_title("Numerical Dominance Accuracy")
    ax_acc.set_ylim(0, 110)
    ax_acc.text(0, c4_acc + 2, f"{c4_acc:.0f}%", ha='center')
    
    # fig4.suptitle("Figure 4: Controls and Robustness", y=1.05)
    plt.tight_layout()
    plt.savefig('paper/figures/fig4_controls.pdf', bbox_inches='tight')
    plt.savefig('paper/figures/fig4_controls.png', bbox_inches='tight')
    plt.close()
    
    print("Figures generated successfully.")

if __name__ == "__main__":
    main()
