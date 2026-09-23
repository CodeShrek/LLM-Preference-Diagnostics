# Novelty Audit

## Current Proposed Contribution
"An abstention-aware, consequential decomposition of observable LLM choice into willingness to commit and conditional directional choice, with systematically varied quantitative payoff magnitude and explicit analysis of presentation-position moderation."

## Falsification Attempt

**1. Is "abstention-aware preference elicitation" completely novel?**
- **NO.** The *Mind the Gap* (Mahajan et al., 2026) paper explicitly demonstrated that adding a neutrality option drastically changes preference distributions and shrinks the stated-revealed gap. Therefore, discovering that LLMs abstain when permitted is *not* a novel contribution of our paper.

**2. Is "order effects/position bias" completely novel?**
- **NO.** *Fragile Preferences* (Yin et al., 2026) thoroughly maps order effects in LLMs across multiple models and domains.

**3. Are "economic preferences/utility modeling in LLMs" completely novel?**
- **NO.** *The Innate Economic Preferences of Language Models* (Buchanan & Foster, 2026) and *What Do LLMs Want?* (Cook et al., 2026) both apply economic utility frameworks (like Random Utility Models and allocation games) to LLM choice.

## What is Genuinely Distinctive?
While the components (abstention, position bias, economic payoffs) are known in the literature, this paper's true novelty lies in the **synthesis and structural decomposition**:
1. **The Hurdle Model of LLM Choice**: No identified prior work explicitly models the *conflation* in forced-choice as a two-stage Hurdle process (Commitment vs. Conditional Direction) governed by a quantitative parameter ($|\log Q|$).
2. **Parametric Position Moderation**: Prior work showed that position bias *exists*. We demonstrate mathematically how presentation order *distorts sensitivity to objective outcome magnitudes* ($Q \times Position$ interaction).
3. **Consequentiality + Abstention**: Most prior economic preference elicitation uses forced-choice. Most prior abstention work (like *Mind the Gap*) uses abstract stated preferences without parametric stakes. We combine them.

## Required Claim Narrowing
- We must **never** claim to be the "first to discover LLMs abstain" or "first to discover position bias."
- We must frame the contribution as an *analytical decomposition* of these known artifacts against a parametric payoff signal. 
- The novelty is the *separation* of commitment and outcome sensitivity, not the discovery of either in isolation.

## Status
The novelty survives falsification when strictly narrowed to the **Hurdle decomposition and position-moderation of parametric outcome sensitivity**. The manuscript's current introduction (Lines 48-54) aligns well with this narrowed scope, but care must be taken in the discussion section not to overclaim.
