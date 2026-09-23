# Paper Readiness Checklist (Phase 23)

## Target Venue: IEEE Open Journal of the Computer Society (OJ-CS)

### 1. Scope & Framing Alignment
- [x] **Relevance:** Aligns with OJ-CS focus on emerging topics in computing, specifically AI evaluation and behavioral measurement of LLMs.
- [x] **Causal Language:** "Prove", "establish", and causal verbs have been systematically removed or softened to "evidence indicates", "shows behavior consistent with".
- [x] **Construct Validity Focus:** The paper explicitly frames the result not as "discovering what models prefer," but as diagnosing *how elicitation protocols affect observable choice behavior*.

### 2. Statistical & Methodological Rigor
- [x] **Sample Size & Design:** $N=1120$ strictly balanced factorial trials (Dilemma × Phrasing × Q-Ratio × Position).
- [x] **Inference Stability:** Dilemma-level random effects and cluster bootstrap successfully account for internal dependence.
- [x] **Separation Addressed:** Quasi-complete separation in Stage 2 is explicitly documented, and sensitivity modeling guarantees non-parametric robustness.
- [x] **Confounders Handled:** C4 completely rules out mathematical incompetence.
- [x] **Pending Follow-ups:** 
  - Cross-model replication (Phase 19)
  - Held-out domain (Phase 17)
  - Threshold Confound isolation (Phase 20)
  *Note: These are currently executing in the background and will finalize the results section.*

### 3. IEEE OJ-CS Required Statements
- [ ] **Data Availability Statement:** Must be added to manuscript. (Will state: "All raw generation logs, parsed choices, and statistical scripts are available in the supplementary open-source repository.")
- [ ] **Code Availability:** Must provide a GitHub or equivalent link for reproducibility.
- [ ] **Ethical Statement:** Must include a brief note on the use of local open-weight models (Qwen, Llama, Mistral) and the absence of human subjects.
- [ ] **Funding Acknowledgment:** Needs a placeholder.

### 4. Conclusion
The core manuscript is scientifically sound and methodologically defensive. Once the background follow-up experiments finish and are integrated, and the required IEEE statements are appended, the package will be submission-ready.
