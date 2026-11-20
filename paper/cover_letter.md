# Cover Letter for Resubmission as New

**To:**  
Prof. Vincenzo Piuri, Editor-in-Chief  
*IEEE Open Journal of the Computer Society*  
Email: vincenzo.piuri@unimi.it  

**Date:** November 20, 2026  

**Subject:** Resubmission of Revised Manuscript (Original ID: **OJCS-2026-09-0983**)  

**Manuscript Title:**  
*Stress-Testing Preference Elicitation in Instruction-Tuned Language Models: Separating Commitment, Conditional Direction, and Payoff Sensitivity Across Small Model Families*  
*(Formerly: "Beyond Forced Choice: Separating Commitment and Payoff Sensitivity in LLM Preference Elicitation")*  

**Authors:**  
Shriyam Baloni, Achin Jain, Debojyoti Roy, Vanita Jain, and Massimo Donelli  

---

Dear Prof. Piuri and the Editorial Board,

Thank you for the opportunity to revise and resubmit our manuscript to the *IEEE Open Journal of the Computer Society*. In accordance with your decision letter dated November 6, 2026, we are resubmitting our manuscript as a new submission linked to our original manuscript ID: **OJCS-2026-09-0983**.

We have conducted a thorough, exhaustive overhaul of the paper in response to the valuable evaluations of the Associate Editor and the three reviewers. Specifically:

1. **Complete Task & Protocol Specification (R1.1, R2.1):** Added a dedicated Methods section (§ IV), an experimental workflow diagram (Fig. 1), formal state-transition equations, verbatim prompt phrasings (Table II), and complete turn-by-turn transcripts (Supplementary § III).
2. **Statistical Rigor & Separation-Robust Modeling (R2.2, R2.3, R3.2):** Explicitly formulated all joint multinomial and hurdle regression equations (Eqs. 3–7), implemented Firth penalized likelihood estimation to resolve separation on sparse cells (Table IV, Supp. Table IV), aligned all point estimates with matching dilemma-cluster bootstrap intervals, and verified asymptotic convergence properties.
3. **Threshold Follow-Up Mathematical Verification (R2.4, R3.3):** Mathematically proved that the follow-up threshold is non-binding ($s_0 + 5 \times 9 < T$ across all 20 dilemmas; § V-B, Supp. § X) and corrected illustrative figure parameters to match the actual experimental distribution.
4. **Matched 2AFC Baseline & Construct Validity (R2.6, R3.1):** Evaluated a paired two-alternative forced-choice (2AFC) baseline ($N=439$), non-zero indifferent payoffs ($N=280$), and a 70-trial counterbalanced neutral-label experiment (Supp. § XVI), demonstrating that simultaneously neutralizing action labels and presentation slots resolves directional disparities while preserving 100% payoff sensitivity.
5. **Phrasing Sensitivity & Hardware Audit (R1.2):** Added a full cross-model phrasing breakdown (Table VI), evaluated stochastic decoding across random seeds (Supp. § XV), documented local hardware memory constraints (Supp. § XI), and provided a self-contained execution suite and Jupyter notebook for evaluating unquantized FP16 and 7B models on free cloud GPUs (Supp. § XVII).
6. **Open Science & Transparency (R1.1, R2.1, R3.6):** Fully deposited all experimental runners, deterministic response parsers, stimulus specifications, and frozen raw CSV trial records in our public GitHub repository:  
   [https://github.com/CodeShrek/LLM-Preference-Diagnostics](https://github.com/CodeShrek/LLM-Preference-Diagnostics).

Along with this resubmission, we have provided:
* A **clean, unmarked main manuscript** (`main.pdf`), formatted strictly within the 12-page budget.
* A comprehensive **Supplementary Material package** (`supplementary.pdf`, 18 sections, 13 tables).
* A detailed, point-by-point **Response to Reviewers** document (`response_to_reviewers.pdf`).

We believe the manuscript is now mature, technically rigorous, and directly answers every concern raised by the reviewers. We look forward to your reconsideration.

Sincerely,

**Achin Jain** *(Corresponding Author)*  
Department of Computer Engineering, University of Delhi, Delhi, India  
Email: karanbaloni@ce.du.ac.in  

On behalf of co-authors:  
* Achin Jain (achin.mails@gmail.com)  
* Debojyoti Roy (roydebojyoti16@gmail.com)  
* Prof. Vanita Jain (vanitajain@fot.du.ac.in), Head, Department of Electronics and Communication Engineering, University of Delhi  
* Prof. Massimo Donelli (massimo.donelli@unitn.it), Department of Information Engineering and Computer Science, University of Trento  
