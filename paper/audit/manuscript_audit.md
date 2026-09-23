# Manuscript Audit

## Claim 1: "when an explicit non-directional response is available, the model heavily defaults to abstention (83.9\% of trials)."
- **Status**: SUPPORTED
- **Evidence**: 940 / 1120 trials = 83.9% in the main CSV. The text correctly states "defaults to abstention" without claiming psychological indifference.

## Claim 2: "quantitative payoff magnitude remains significantly associated with both the probability of behavioral commitment and, conditional on commitment, the probability of selecting the dominant option."
- **Status**: SUPPORTED
- **Evidence**: Stage 1 $\beta=1.002$; Stage 2 MLE $\beta=3.319$.

## Claim 3: "this conditional directional sensitivity is moderated by the order in which options are presented."
- **Status**: SUPPORTED
- **Evidence**: Stage 2 Q $\times$ Position interaction is significant ($\beta = -2.311, p = 0.048$).

## Claim 4: "the high abstention rate is not an artifact of arithmetic incompetence"
- **Status**: SUPPORTED
- **Evidence**: C4 control achieved 100% accuracy. The manuscript correctly restricts this to "making basic arithmetic failure an unlikely explanation" rather than "proving full mathematical cognition."

## Claim 5 (OMISSION/UNDER-REPORTED): The Primary Multinomial Model (Add / Subtract / Indifferent) is not reported in the Results section.
- **Status**: PARTIALLY SUPPORTED (Requires fix)
- **Problem**: The manuscript currently jumps straight from the descriptive 83.9% abstention into the Hurdle decomposition, entirely skipping the primary MNLogit results that predict the explicit three-way probabilities.
- **Corrected Wording**: A new section `5.2 Primary Multinomial Results` must be inserted to explicitly report the MNLogit coefficients ($\beta = 0.743$ for Add vs Indifferent, $\beta = -2.119$ for Subtract vs Indifferent) and position interaction terms.

## Claim 6: "Crucially, our results suggest that binary preference elicitation inherently conflates these two processes."
- **Status**: OVERCLAIMED (Borderline)
- **Problem**: The phrase "inherently conflates" is slightly too strong, given we did not run a direct, structurally matched 2AFC forced-choice vs expanded-choice experiment on this specific prompt layout (it is left to Future Work).
- **Corrected Wording**: "Crucially, our findings imply that binary preference elicitation likely conflates these two processes, as the default abstention state we observe would be forced into a directional tie-break."
