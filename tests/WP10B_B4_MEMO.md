# WP10B B4 decision memo: can the dual engine publish? (2026-11-27)

**The dual engine does not beat the baseline engine by the declared margin (criterion (i) fails), so it is NOT publishable by the declared rule (criteria 11.6), whatever its published share. It stays research-only, as before; no default and no switch changes.**

Every number: Python 3.12.3, repo pins, the same 60 S0 users and the single analysis as B3 (`wp10b_evidence/b3_analysis.txt`; the easy + medium users pooled, n = 40; configuration I600 frozen by the Part 4 rule; owner decision W10B-1: evaluated once).

## The rule (declared before the run)
Publishable iff (i) the mean over users of (held-out one-step mean of the dual fit minus that of the baseline engine) minus 3 times its standard error is above 0, AND (ii) the published share (converged-publishable or certified) is at least the level 0.60 (MET iff the Wilson lower end is at least 0.60).

## Result (RUN)
| part | value | result |
|---|---|---|
| (i) dual minus baseline engine, easy + medium, n = 40 | mean -0.464 nats per step, se 0.206, mean - 3 se = -1.082 | **fails** (needs above 0) |
| the dual is ahead of the baseline engine for | 10 of 40 users (median difference -0.554) | – |
| secondary: the same margin on the published subset (n = 31) | -0.725, se 0.208, mean - 3 se = -1.350 | fails |
| reported: dual minus the single Gaussian, n = 40 | +0.278, se 0.159, mean - 3 se = -0.198 | the dual is ahead of the Gaussian on average, not by 3 se |
| reported, hard stratum (n = 20) | dual minus baseline -0.311 (se 0.171) | no verdict |
| (ii) published share | 31 of 40 = 0.775, Wilson 0.625 to 0.877 | **MET** at 0.60 (lower end above it) |
Published share against the levels the memo has to show: 0.40 MET, 0.50 MET, 0.60 MET, **0.80 NOT MET** (point estimate 0.775, lower end 0.625). Of the 31 published fits all 31 are converged-publishable and none was certified; the 9 not published stopped on the iteration cap (4, one of them also with a degenerate duration) or on a rejected update (5), and the certificate was asked for each and refused each.
**Rule result: NOT MET** (one of two parts).

## What the owner may want from this (INFERENCE where marked)
- **The B2 fix did its job on convergence and not on prediction.** Publishable share went from 11% (B1, S0 settings, 36 development users) to 58% (B2 development, I600) and 78% here on easy + medium; but a converged dual fit predicts held-out days worse than the baseline engine by about half a nat per step on these streams, and the published subset is worse still (-0.73): convergence does not select good fits.
- **Not a share problem.** If the owner set the level at 0.40, 0.50 or 0.60 the share part would still be MET and the rule would still fail on (i); only a different margin rule (for example no margin against the baseline engine, or a margin on a different score) would change the verdict, and that would be a new rule, which I do not choose after seeing the result.
- **Where the comparison could mislead (INFERENCE):** the streams are the identity streams the production fit is made for, and the score is the held-out one-step log density, a quantity the baseline model is fitted to maximise in a similar family; the dual engine's claimed advantage (a fast timescale) may show on other generators, which this package did not test. I make no claim about those.
- **Decisions open to the owner:** (1) accept "research-only" as the status of the dual engine for the S0 setting; (2) whether any further work on the dual engine is wanted (the owner's call; a second frozen version needs the spare seeds 11901..11999 and the go-ahead, W10B-1); (3) whether a level other than 0.60 is wanted for the share (it would not change the outcome here).
- **Fail-closed unchanged:** the dual-to-dual and cross-engine lineage pairs stay uncalibrated (B5), so a published dual fit would not be given an identity ALIGNED anyway.

## Limits
One setting, one configuration, 40 easy + medium users, synthetic streams, no real users; the baseline margin is a paired comparison of one-step scores on 30 test days per user; cross-machine reproducibility is not claimed.
