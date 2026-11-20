import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_overview_diagram():
    fig = plt.figure(figsize=(16, 10.8), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10.8)
    ax.axis('off')

    c_blue = '#1f4e79'
    c_green = '#236b28'
    c_orange = '#b93815'
    c_purple = '#531b77'
    c_teal = '#005b52'

    def draw_card(x, y, w, h, title, subtitle, color):
        hdr_h = 0.55
        hdr = patches.FancyBboxPatch((x, y + h - hdr_h), w, hdr_h,
                                     boxstyle="round,pad=0.01,rounding_size=0.08",
                                     facecolor=color, edgecolor=color)
        ax.add_patch(hdr)
        ax.text(x + w/2, y + h - hdr_h/2, title, color='#ffffff', weight='bold',
                fontsize=11.5, ha='center', va='center', fontfamily='sans-serif')

        body = patches.FancyBboxPatch((x, y), w, h - hdr_h*0.8,
                                      boxstyle="round,pad=0.01,rounding_size=0.08",
                                      facecolor='#ffffff', edgecolor=color, linewidth=1.5)
        ax.add_patch(body)
        if subtitle:
            ax.text(x + w/2, y + h - hdr_h - 0.22, subtitle, color='#2c3e50', weight='bold',
                    fontsize=9.2, ha='center', va='center', fontfamily='sans-serif')
        return body

    # 1. Model Selection (x=0.5, y=5.8, w=3.2, h=4.5)
    draw_card(0.5, 5.8, 3.2, 4.5, "1. Evaluated Models", "4 Local Instruction-Tuned LLMs", c_blue)
    models_text = (
        "• Qwen 2.5 3B (3.1B, Q4_K_M)\n"
        "• Llama 3.2 3B (3.2B, Q4_K_M)\n"
        "• Phi-4 Mini (3.8B, Q4_K_M)\n"
        "• Gemma 3 4B (4.0B, Q4_K_M)\n\n"
        "Inference Protocol:\n"
        "• Deterministic greedy (T = 0.0)\n"
        "• Local Ollama deployment\n"
        "• Pre-trained open weights\n"
        "• Zero local training or fine-tuning\n"
        "• Independent trial state reset"
    )
    ax.text(0.7, 9.1, models_text, fontsize=8.4, va='top', ha='left', color='#263238', linespacing=1.35)

    # 2. Factorial Design (x=4.0, y=5.8, w=3.4, h=4.5)
    draw_card(4.0, 5.8, 3.4, 4.5, "2. Factorial Design", "1,120 Balanced Trials per Model", c_green)
    design_text = (
        "Fully Crossed Factors:\n"
        "• 20 Structural Dilemmas (D01–D20)\n"
        "• 4 Phrasing Templates (P1–P4)\n"
        "• 7 Payoff Ratios (Q_add : Q_sub):\n"
        "   1:9, 1:5, 1:2, 1:1, 2:1, 5:1, 9:1\n"
        "• 2 Presentation Positions (±1):\n"
        "   Add-first (A) vs. Subtract-first (B)\n\n"
        "Design Balance:\n"
        "20 × 4 × 7 × 2 = 1,120 / model\n"
        "Total Primary Factorial N = 4,480"
    )
    ax.text(4.2, 9.1, design_text, fontsize=8.4, va='top', ha='left', color='#263238', linespacing=1.35)

    # 3. Consequential Trial Protocol (x=7.7, y=5.3, w=7.8, h=5.0)
    draw_card(7.7, 5.3, 7.8, 5.0, "3. Consequential Trial Protocol & State Mechanics",
              "Multi-Turn Pipeline (Concrete Dilemma D01 Example)", c_orange)

    # Sub-box: Turn 1
    t1_box = patches.FancyBboxPatch((7.9, 7.55), 3.6, 1.8, boxstyle="round,pad=0.01,rounding_size=0.06",
                                   facecolor='#fff3e0', edgecolor='#ffb74d', linewidth=1.2)
    ax.add_patch(t1_box)
    ax.text(9.7, 9.1, "Turn 1: Action Selection (Decisive)", weight='bold', fontsize=8.5, ha='center', color='#b93815')
    t1_str = (
        "Input: Pipeline goal + Backstory + Phrasing\n"
        "Three Permitted Choices:\n"
        " • Option A: Add (+9 quality pts/entry)\n"
        " • Option B: Subtract (+1 quality pt/entry)\n"
        " • Indifferent: Explicit non-directional exit\n"
        "Primary estimands evaluated on Turn 1!"
    )
    ax.text(8.05, 8.88, t1_str, fontsize=7.7, va='top', ha='left', color='#37474f', linespacing=1.25)

    # Sub-box: Concrete D01 State
    ex_box = patches.FancyBboxPatch((11.7, 7.55), 3.6, 1.8, boxstyle="round,pad=0.01,rounding_size=0.06",
                                   facecolor='#fbe9e7', edgecolor='#ff8a65', linewidth=1.2)
    ax.add_patch(ex_box)
    ax.text(13.5, 9.1, "Concrete State (D01, 9:1, ℓ=5)", weight='bold', fontsize=8.5, ha='center', color='#b93815')
    ex_str = (
        "Baseline Score s₀ = 60, Target T = 120\n"
        "• If Add (+9): 60 + 9 × 5 = 105 [15 short]\n"
        "• If Subtract (+1): 60 + 1 × 5 = 65 [55 short]\n"
        "• If Indifferent: skips T2–T3 (score=60)\n"
        "Notice: Single batch CANNOT cross T;\n"
        "continuation is objectively rational policy!"
    )
    ax.text(11.85, 8.88, ex_str, fontsize=7.7, va='top', ha='left', color='#37474f', linespacing=1.25)

    # Sub-box: Turn 2
    t2_box = patches.FancyBboxPatch((7.9, 5.55), 3.6, 1.75, boxstyle="round,pad=0.01,rounding_size=0.06",
                                   facecolor='#ede7f6', edgecolor='#b39ddb', linewidth=1.2)
    ax.add_patch(t2_box)
    ax.text(9.7, 7.05, "Turn 2: Scripted Environment Update", weight='bold', fontsize=8.5, ha='center', color='#4a148c')
    t2_str = (
        "• Deterministically scripted by system runner\n"
        "• Reports updated score (e.g., 60+45=105 pts)\n"
        "• Model-independent: same for given choice\n"
        "• Skipped entirely if Indifferent chosen"
    )
    ax.text(8.05, 6.8, t2_str, fontsize=7.7, va='top', ha='left', color='#37474f', linespacing=1.25)

    # Sub-box: Turn 3
    t3_box = patches.FancyBboxPatch((11.7, 5.55), 3.6, 1.75, boxstyle="round,pad=0.01,rounding_size=0.06",
                                   facecolor='#e8f5e9', edgecolor='#a5d6a7', linewidth=1.2)
    ax.add_patch(t3_box)
    ax.text(13.5, 7.05, "Turn 3: Scripted Follow-On Action", weight='bold', fontsize=8.5, ha='center', color='#1b5e20')
    t3_str = (
        "• Prompts: Submit batch vs. Process more\n"
        "• Result: 100% 'Yes' (continue) across models\n"
        "• Zero empirical variance on Turn 3\n"
        "• Evaluates Turn 1 as primary choice point"
    )
    ax.text(11.85, 6.8, t3_str, fontsize=7.7, va='top', ha='left', color='#37474f', linespacing=1.25)

    # 4. Hurdle Decomposition (x=0.5, y=0.3, w=6.9, h=4.7)
    draw_card(0.5, 0.3, 6.9, 4.7, "4. Two-Stage Hurdle Decomposition",
              "Separating Commitment Propensity from Conditional Direction", c_purple)

    s1_box = patches.FancyBboxPatch((0.7, 2.5), 6.5, 1.65, boxstyle="round,pad=0.01,rounding_size=0.06",
                                   facecolor='#f3e5f5', edgecolor='#ce93d8', linewidth=1.2)
    ax.add_patch(s1_box)
    ax.text(3.95, 3.9, "Stage 1: Directional Commitment Hurdle", weight='bold', fontsize=9, ha='center', color='#4a148c')
    s1_str = (
        "Binary Outcome: Y_commit = 1 if (Add or Subtract) else 0 (Indifferent)\n"
        "Primary Additive Model: logit P(Commit = 1) ~ |log r| + Pos + Dilemma Fixed Effects\n"
        "Finding: Dramatic cross-model disparity in baseline propensity to commit:\n"
        " • Qwen 2.5 3B: 14.7% commitment (85.3% abstention; sensitive to disparity)\n"
        " • Llama 3.2 3B: 97.8% commitment; Phi-4 Mini: 98.7% commitment; Gemma 3 4B: 100.0%"
    )
    ax.text(0.9, 3.65, s1_str, fontsize=7.8, va='top', ha='left', color='#263238', linespacing=1.25)

    s2_box = patches.FancyBboxPatch((0.7, 0.5), 6.5, 1.75, boxstyle="round,pad=0.01,rounding_size=0.06",
                                   facecolor='#e8eaf6', edgecolor='#9fa8da', linewidth=1.2)
    ax.add_patch(s2_box)
    ax.text(3.95, 2.0, "Stage 2: Conditional Dominance Selection", weight='bold', fontsize=9, ha='center', color='#1a237e')
    s2_str = (
        "Conditional Outcome: P(Dominant = 1 | Commitment = 1) across non-parity trials\n"
        "Separation-Resistant Estimator: Firth's penalized likelihood [95% penalized Wald CIs]:\n"
        " • Qwen: β = 1.231 [0.50, 1.96]; Gemma: β = 1.531 [0.51, 2.55] (strong sensitivity)\n"
        " • Llama: β = 0.282 [0.04, 0.53]; Phi: β = 0.225 [-0.01, 0.46] (moderate sensitivity)\n"
        "Order Moderation: Accuracy drops severely when dominant option is in disfavored slot"
    )
    ax.text(0.9, 1.75, s2_str, fontsize=7.8, va='top', ha='left', color='#263238', linespacing=1.25)

    # 5. Controls & Robustness (x=7.7, y=0.3, w=7.8, h=4.7)
    draw_card(7.7, 0.3, 7.8, 4.7, "5. Control Analyses and Robustness Suite",
              "Validating Alternative Explanations & Construct Validity", c_teal)

    c4_box = patches.FancyBboxPatch((7.9, 2.5), 3.6, 1.65, boxstyle="round,pad=0.01,rounding_size=0.06",
                                   facecolor='#e0f2f1', edgecolor='#80cbc4', linewidth=1.2)
    ax.add_patch(c4_box)
    ax.text(9.7, 3.9, "C4 Numeracy Control", weight='bold', fontsize=8.5, ha='center', color='#004d40')
    c4_str = (
        "• Isolated payoff comparison prompt\n"
        "• Accuracy: Qwen 100%, Gemma 98.3%,\n"
        "  Phi 71.0%, Llama 53.8%\n"
        "• Non-monotone ranking vs. Stage 2\n"
        "• Shows arithmetic error co-occurs with bias"
    )
    ax.text(8.05, 3.65, c4_str, fontsize=7.7, va='top', ha='left', color='#263238', linespacing=1.25)

    th_box = patches.FancyBboxPatch((11.7, 2.5), 3.6, 1.65, boxstyle="round,pad=0.01,rounding_size=0.06",
                                   facecolor='#e0f7fa', edgecolor='#80deea', linewidth=1.2)
    ax.add_patch(th_box)
    ax.text(13.5, 3.9, "Non-Binding Goal (T = 10)", weight='bold', fontsize=8.5, ha='center', color='#006064')
    th_str = (
        "• Starting score s₀ ≥ 40 > T=10 unconditionally\n"
        "• Both choices cross goal before acting\n"
        "• Qwen payoff slope persists: β = 1.273\n"
        "• Gemma ceiling persists: 99.6% dominant\n"
        "• Tests threshold-insensitivity"
    )
    ax.text(11.85, 3.65, th_str, fontsize=7.7, va='top', ha='left', color='#263238', linespacing=1.25)

    fc_box = patches.FancyBboxPatch((7.9, 0.45), 3.6, 1.85, boxstyle="round,pad=0.01,rounding_size=0.06",
                                   facecolor='#fff8e1', edgecolor='#ffe082', linewidth=1.2)
    ax.add_patch(fc_box)
    ax.text(9.7, 2.05, "Matched 2AFC Baseline", weight='bold', fontsize=8.5, ha='center', color='#b78103')
    fc_str = (
        "• Forced choice without Indifferent exit\n"
        "• At Parity (1:1): Compelled 79.4% B vs. 20.6% A\n"
        "• Non-Parity: 86.5% latent dominant accuracy\n"
        "• Severe Slot Bias: 99.6% (B) vs. 72.9% (A)\n"
        "• Shows forced choice induces position bias"
    )
    ax.text(8.05, 1.85, fc_str, fontsize=7.7, va='top', ha='left', color='#263238', linespacing=1.25)

    rob_box = patches.FancyBboxPatch((11.7, 0.45), 3.6, 1.85, boxstyle="round,pad=0.01,rounding_size=0.06",
                                    facecolor='#ede7f6', edgecolor='#d1c4e9', linewidth=1.2)
    ax.add_patch(rob_box)
    ax.text(13.5, 2.05, "Robustness Extensions", weight='bold', fontsize=8.5, ha='center', color='#311b92')
    rob_str = (
        "• Non-Zero Indiff: 95.0% parity abstention\n"
        "• Slot Ablation: 1st slot extinguishes exit (0%)\n"
        "• Domain Replication: Resource Scheduling\n"
        "• Stochastic Replicates (T=0.7, 3 seeds)\n"
        "• Hardware Memory & Quantization Audit"
    )
    ax.text(11.85, 1.85, rob_str, fontsize=7.7, va='top', ha='left', color='#263238', linespacing=1.25)

    # Connectors
    ax.annotate("", xy=(3.95, 8.0), xytext=(3.72, 8.0), arrowprops=dict(arrowstyle="->", lw=2, color=c_blue))
    ax.annotate("", xy=(7.65, 8.0), xytext=(7.42, 8.0), arrowprops=dict(arrowstyle="->", lw=2, color=c_green))
    ax.annotate("", xy=(3.9, 5.0), xytext=(9.7, 5.25), arrowprops=dict(arrowstyle="->", lw=2, color=c_orange, connectionstyle="arc3,rad=-0.12"))
    ax.annotate("", xy=(11.5, 5.0), xytext=(11.5, 5.25), arrowprops=dict(arrowstyle="->", lw=2, color=c_orange))

    plt.tight_layout()
    plt.savefig('paper/figures/Fig1_MethodologyOverview.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper/figures/Fig1_MethodologyOverview.pdf', dpi=300, bbox_inches='tight')
    print("Clean Fig1_MethodologyOverview generated successfully.")

if __name__ == '__main__':
    create_overview_diagram()
