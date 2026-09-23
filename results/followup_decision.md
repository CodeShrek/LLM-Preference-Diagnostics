# Follow-up Decision Log (Phase 21)

## Background
Earlier reviews suggested additional follow-ups, specifically:
- Testing if the model cares about the raw `Q` scalar vs the total points (`Q * array_length`).
- Testing if the model cares about the phrasing framing (already built-in via the 4 phrasings).
- Testing if the model handles semantic priors (already handled via the neutral domain).

## Decision: Reject Additional Follow-ups
We have determined that these additional follow-ups are **not required** for the IEEE OJ-CS submission, and would unnecessarily bloat the paper.

### Rationale
1. **Scope of Contribution:** The paper's core contribution is the methodological demonstration that LLMs possess an immense "indifference region" that masks parametric sensitivity to quantitative consequences under forced-choice conditions. We do not need to prove *how* the LLM calculates the consequence (scalar vs total), only that manipulating the objective payoff parametrically scales the probability of choosing the dominant option once committed.
2. **Methodological Focus:** IEEE OJ-CS favors clear, rigorous, well-scoped claims. Adding a deep dive into the model's internal arithmetic heuristics detracts from the broader construct-validity argument regarding preference elicitation.
3. **Sufficient Validation:** We already have:
   - A Held-out domain replication (Phase 17)
   - A Forced vs Expanded comparison (Phase 18)
   - Cross-model replication (Phase 19)
   - Threshold confounding resolution (Phase 20)

Adding further behavioral perturbations is exploratory and out of scope for the current manuscript.
