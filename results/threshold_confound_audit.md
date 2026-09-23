# Phase 13: Threshold Confound Audit

## Overview
We evaluate whether the quantitative payoff magnitude (Q) is perfectly confounded with the ability to cross the target quality threshold over the course of the 3-turn episode.

## Analysis of Potential Threshold Crossings (3 Turns)
Averaging over the dynamic array lengths for each dilemma, we cross-tabulate the maximum Q available (1, 2, 5, 9) with whether *any* option crosses the threshold if chosen 3 times:

```
any_crosses  False  True 
max_Q                    
1               20      0
2               40      0
5               12     28
9                0     40
```

## Findings
There is a **severe confound** between Q-magnitude and threshold crossing ability. 
- At `max_Q = 1` and `max_Q = 2`, crossing the threshold is mathematically impossible for all dilemmas.
- At `max_Q = 9`, crossing the threshold is mathematically guaranteed for all dilemmas.
- Only at `max_Q = 5` does the outcome depend on the specific dilemma.

Because Q=9 always succeeds and Q=2 always fails, the observed "Q-sensitivity" ($\beta=3.319$) might simply be a step-function "threshold-crossing heuristic" rather than continuous sensitivity to quantitative magnitude. 

**CONCLUSION:** We **MUST** run the conditional follow-up experiment (Phase 20: Threshold Relevance) to separate Q magnitude from Threshold Crossing. We cannot claim pure magnitude sensitivity when magnitude is perfectly correlated with task success at the extremes.
