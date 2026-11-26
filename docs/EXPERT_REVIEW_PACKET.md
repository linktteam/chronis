# Expert review packet: how CHRONIS decides that a regime is "the same as before" (WP9 A5, owner decision A7)

**Status: PENDING. No expert has reviewed anything in this packet.** Every lineage record carries `expert_review_status = "PENDING"` until the owner records a sign-off (`lineage.EXPERT_REVIEW_STATUS`, `docs/EXPERT_REVIEW_REGISTER.md`). Every place in `backbone/hssm/` that defines or uses the matching distance, a ladder weight or an alignment threshold is tagged `EXPERT_REVIEW(identity-distance)`; the register of those places (33 sites) is generated from the tags and a test (`tests/test_wp8_item7_expert_review.py`) fails when it is stale or when a function uses one of the watched names without the tag.

Labels used below: **RUN** = executed in this repository and the output is stored in `wp8_evidence/` or `wp9_evidence/`; **READ** = read in code; **REPORTED** = taken from a package document; **UNKNOWN** = not established.

## 1. What the reviewer is asked to look at, in one paragraph
A person's fitted model has "regimes" (phases). A refit on a newer window names its regimes with new numbers; the system must say whether regime 2 now is regime 2 before. It matches the regimes of the new fit to those of the last trusted fit ("the anchor") and **may say "same as before" only when the fit is identifiable and the match is unambiguous** (`continuity_state = ALIGNED`). In every other case it keeps the old names, says "cannot confirm this is the same as before", and lets the user choose new names. All of that rests on the distance, the gate and the thresholds below. They were chosen on synthetic users (the development seeds listed in each section), never on real people.

## 2. The items under review

### 2.1 The matching distance (READ: `lineage._vectors`, `alignment`, `alignment_from_cost`)
- Each regime is the vector of its **regime means in raw physical units** (not the fit's own z-scores; the scaler of every fit is stored), **divided feature by feature by the anchor's own scaler sd**.
- The cost of pairing old regime *i* with new regime *j* is the **squared Euclidean distance** between those vectors. The match is the Hungarian assignment (minimum total cost). `C1` = its total cost, `C2` = the cost of the best *other* one-to-one assignment (exhaustive for K up to 6).
- Optional extras, both **off by default**: a duration coordinate (`duration_weight`, the log expected dwell time as one more coordinate; the development study found it tied with "off", so the simpler one stayed) and a second, contextual channel (2.5).
- Margins stored with every alignment: `margin_diff = (C2 - C1) / K` and `margin_ratio = C1 / C2`.

### 2.2 The identifiability gate (READ: `separation_min`; `AlignmentConfig.min_separation = h = 2.5`; WP8 had 4.0; WP9 A2 lowered it by the declared rule on development seeds 9301..9360, plus a posterior-sharpness floor of 0.95)
- A fit can be matched only if its own regimes are far enough apart from each other. The statistic is the **smallest, over pairs of regimes, Euclidean norm of the difference of the two regime means divided per feature by the pooled within-regime sd of the days decoded into the regimes** (seven features, so a value of 4 is roughly 1.5 sd per feature and the current 2.5 roughly 0.95 sd per feature).
- A fit below `h` (or an anchor below `h`) is never ALIGNED (reasons `fit_not_identifiable`, `anchor_not_identifiable`).
- Why a gate and not only a margin: a fit that merged two real regimes can match its neighbour cleanly. In the WP8 development data the strict-wrong alignments occur from `margin_ratio` 0.013 upward (RUN: `wp8_evidence/item4_margin_analysis.txt`), so a good margin does not show that a fit is identifiable.

### 2.3 The acceptance rule (READ: `margin_ok`, `PairThreshold`; baseline to baseline: rule "ratio", `rho = 0.5`)
- ALIGNED needs a **strict** win (`C1 < C2`, a tie is never accepted) and `margin_ratio <= 0.5`. 0.5 is the WP5 rule, and the threshold may never be more lenient than that (enforced in `PairThreshold`).
- Evidence (RUN, 960 WP8 development fits): among alignments of two fits that both recovered the planted regimes there were 4 permutation errors, all at `margin_ratio >= 0.48` (0.48, 0.57, 0.61, 0.91). So matching itself starts to go wrong at about 0.48, next to the rule of 0.5. Among the 242 baseline alignments that passed the gate every one was correct and the largest ratio was 0.332 (`wp8_evidence/item4_margin_analysis.txt`).

### 2.4 The naming ladder and its weights (READ: `ordering.py`; used ONLY at the first accepted fit of a lineage)
- Orders the regimes from least to most active, so that regime 0 means "calmest" and a name carries a meaning. Score of a regime = sum over the rung's features of `w_f * direction_f * z`; `direction_f` is a **domain prior** from the feature registry (never learned); `w_f` proportional to `coverage_f * SNR_f / (1 + SNR_f)`, SNR = between-regime over within-regime variance of the feature over the decoded regimes (data quality only; no outcome label enters).
- Rungs: 1 all features with a direction; 2 movement and time moving; 3 movement only.
- A rung is **reliable** when its modal ordering appears in at least `min_modal_freq = 0.90` of 200 block-bootstrap resamples (7-day blocks, labels fixed, weights and scores recomputed) **and** the smallest gap between adjacent regime scores is at least `g_min = min_gap = 0.7`. The highest reliable rung names the fit if the other reliable rungs agree; otherwise the ordering is UNVERIFIED, ids follow first-fit order, and **nothing about activity may be said**.
- Evidence (RUN, 60 development fits, `tests/WP8_ITEM3_RESULTS.md`): 45 of 60 verified, correct among verified 45 of 45. `g_min = 0.7` comes from the declared rule (the largest gap among the wrong results was 0.624), and the 100% is a development number, not an accuracy estimate. The ladder abstains where the planted separation is low (verified 9 of 20 below separation 0.8).

### 2.5 The contextual channel (READ: `ChannelConfig`, `hellinger2_matrix`; off by default, `w2 = 0`)
- Optional second channel: one non-negative profile per regime (for example scene or topic shares) **supplied by the caller**; nothing in this package extracts them. Distance = squared Hellinger. Combined cost `w1 * d1 / s1 + w2 * d2 / s2` (`s` = the anchor's mean inter-regime distance in that channel). `context_threshold` is the acceptance rule of the second channel alone and is the provisional WP5 ratio rule.
- If both channels are confident on their own and disagree about the match, the state is `CONTINUITY_UNVERIFIED` (`channels_disagree`).
- **UNKNOWN:** whether such a channel helps on real users (no real descriptors exist here). Contextual descriptors are sensitive personal data; the contract (section 9.10) says they must come from a consented, aggregated, non-identifying pipeline.

### 2.6 Classes of pair with NO calibrated threshold (owner decision A6; engine reliability is WP10)
- **Dual to dual** and **baseline to dual / dual to baseline** pairs have `PairThreshold.calibrated = False`: they are **never ALIGNED** (reason `threshold_not_calibrated`), whatever the margins are. WP8 had fewer than 20 development alignments that passed the gate for these classes (4 of 216 and 16 of 480), so no value can be supported. This packet does not ask the expert for one; it asks whether the fail-closed default is acceptable until the dual engine's fits are good enough to calibrate (WP10).

## 3. What WP9 added that rests on the items above
- **FIT_UNSTABLE (A1).** A baseline fit whose three best healthy restarts do not agree (adjusted Rand index below 0.99) is flagged and never served as ALIGNED (`fit_unstable`, `anchor_fit_unstable`). Nothing about the selected model changes. Result on 48 synthetic evaluation users: no flagged fit was ever served (0 of 230), and when both seeds' fits are unflagged they give the same partition in 98.5% of windows; but two runs of the same user agree on whether to serve in only 67.6% of served cases (target 95%), mostly because one run is flagged and the other is not (section 4; `tests/WP9_A12_RESULTS.md`).
- **K change suspected (A3).** On a refit, a fit of the most recent 45 days at K - 1, K and K + 1 regimes is compared by BIC. A suspected change keeps the old names and the state `CONTINUITY_UNVERIFIED` with the reason `k_change_suspected` (never "same as before"); split and merge candidates now come from the posterior overlap with the old regimes, next to the old nearest-mean rule. **The signal did not detect planted splits in the evaluation (0 of 12) and detected one of 12 merges**; what holds is the safety rule (a suspected or real K change is never "same as before"), `tests/WP9_A3_RESULTS.md`.
- **Gate and coverage trade-off (A2).** See section 4.

## 4. The trade-off the expert is asked to weigh (A2)
**What is traded.** A lower gate `h` lets more fits be matched ("same as before" is offered more often) and lets more wrong matches through; a higher `h` offers less and is safer. Targets set by the owner: coverage >= 80% and wrong-and-unflagged <= 1%. Numbers below are RUN, synthetic users only (a third hard, a third medium, a third easy separation: an assumption of the study; the real mix is **UNKNOWN**).

**Development curve** (users 9301..9360, 540 alignments; used to choose; `wp9_evidence/a2_dev_analysis.txt`). Coverage = share of accepted refits that end ALIGNED; wrong = ALIGNED but not correct against the planted truth.

| gate h | consensus flag ON, sharpness floor 0.95: coverage / wrong | flag OFF: coverage / wrong |
|---|---|---|
| 0 - 1.0 | 46.5% / 2 | 85.7% / 29 |
| 1.5 | 46.5% / 2 | 85.4% / 28 |
| 2.0 | 46.3% / 2 | 76.9% / 9 |
| **2.5 (chosen)** | **45.9% / 0** | 68.9% / 0 |
| 3.0 | 45.4% / 0 | 63.3% / 0 |
| 3.5 | 43.3% / 0 | 52.8% / 0 |
| 4.0 (WP8 default) | 41.3% / 0 | 49.6% / 0 |
| 5.0 | 39.6% / 0 | 48.0% / 0 |

**Evaluation curve** (users 9401..9448, 432 alignments; run once at the frozen candidate, the curve is REPORTED, nothing was chosen from it; `wp9_evidence/a2_eval_analysis.txt`, `a2_eval_flagoff.txt`).

| gate h (10 restarts) | flag ON: coverage / wrong | flag OFF: coverage / wrong |
|---|---|---|
| 0 - 1.5 | 51.2% / 0 | 95.1% / 34 (h = 0) |
| 2.0 | 50.0% / 0 | 88.9% / 17 |
| **2.5 (frozen)** | **49.5% / 0** | 77.1% / 0 |
| 3.0 | 48.6% / 0 | 68.5% / 0 |
| 3.5 | 45.6% / 0 | not computed |
| 4.0 | 42.4% / 0 | 55.1% / 0 |
| 5.0 | 28.2% / 0 | not computed |

**Per separation stratum at the frozen gate** (evaluation, coverage / wrong): flag ON, 10 restarts: hard 14.6% / 0, medium 59.7% / 0, easy 74.3% / 0; flag ON, 20 restarts: hard 23.6%, medium 66.0%, easy 84.7% (0 wrong); flag OFF, 10 restarts: hard 45.8%, medium 85.4%, easy 100% (0 wrong).

**Reading, plainly.**
1. **80% coverage with at most 1% wrong are not achieved together on the pooled mix at the frozen design** (49.5% coverage, 0 of 432 wrong; Wilson upper bound of the wrong rate 0.9%). The wrong-and-unflagged level is met; the coverage level is not. No gate value with the flag on reaches 80%.
2. The flag is the big cost: about 28 coverage points at h = 2.5 (49.5% versus 77.1%), for the same 0 of 432 wrong on these users. What it buys is that a fit whose three best restarts disagree is not served; with the flag off the agreement between two runs of the same user was not what the criterion asked for (seed-to-seed agreement of the offer: see `wp9_evidence/a1_eval_flagoff_identity.txt` and `tests/WP9_A12_RESULTS.md`).
3. Below h = 2.5 the wrong-match rate rises quickly without the flag (3.9% at 2.0, 7.9% at 0), so the choice of the gate is not neutral: the unflagged curve has a knee between 2.0 and 2.5 that the grid does not resolve.
4. The hard stratum is the reason the pooled number is low: those fits do not recover the planted regimes, and no gate setting that stays at 0 wrong serves most of them.
5. 0 of 432 wrong is a point estimate on 48 synthetic users, not a demonstration that the wrong rate is below 1% for real people.

**The question for the expert:** is it acceptable to serve "same as before" to about half of the offered refits (and to most easy ones) while saying "cannot confirm" to the rest, or should the product accept more wrong claims for more coverage? The package's default is the safe side (flag on, h = 2.5).

## 5. What the synthetic evidence cannot say
- All users are generated (K = 3 planted regimes, planted separation in a stated range, seven features). Real separation, real regime counts, real dwell times and real drift are **UNKNOWN**. The strata (hard, medium, easy) are assumptions of the study, not measurements.
- Coverage numbers describe the generator's mix of easy and hard users; the share of real users who are identifiable is UNKNOWN.
- Thresholds were chosen on development seeds and judged on separate evaluation seeds once; 48 evaluation users per design is a small sample and several results are "met on the point estimate, not demonstrated".

## 6. Questions for the expert
1. **Distance.** Is Euclidean distance between regime means in raw units, scaled by the anchor's per-feature sd, an acceptable measure of "the same regime"? Would you scale by the regime's own spread, use a Mahalanobis distance, or include how long a regime lasts?
2. **Gate.** Is `h = 2.5` (seven features, smallest pairwise norm in within-regime sd; WP8 had 4.0) the right way to say that a fit is identifiable, and is the level defensible? The curve in section 4 shows what each level costs in coverage and in wrong-and-unflagged matches.
3. **Ratio rule.** Is a strict win with `C1 / C2 <= 0.5` an acceptable definition of an unambiguous match, given that errors began at about 0.48 in development?
4. **Ladder.** Are the registry directions (which feature means "more active") and the weight `coverage * SNR / (1 + SNR)` acceptable? Are `min_modal_freq = 0.90` and `g_min = 0.7` too lenient or too strict, knowing that the ladder abstains for weakly separated people?
5. **Contextual channel.** Under which consent and aggregation conditions may it be switched on, and what threshold should apply to it?
6. **Product targets.** The package asks for coverage of at least 80% with at most 1% wrong-and-unflagged matches. Is a 1% rate of saying "same as before" wrongly acceptable for this product, and is 80% the right ambition? (If the two cannot be met together, section 4 says so plainly.)
7. **Fail closed.** Is it acceptable that dual-to-dual and cross-engine matches are never served as "same as before" until WP10?
8. **K change.** Is keeping the old names with "cannot confirm this is the same as before" the right behaviour when the number of regimes appears to change, and should the user be offered split / merge candidates at all?

## 7. How to record the outcome
Record the review (who, when, which items, what was changed) in this file and in `docs/EXPERT_REVIEW_REGISTER.md`, change `lineage.EXPERT_REVIEW_STATUS`, and regenerate the register (`python3 integration_report/wp8/scripts/expert_review_register.py --write`). Until then every record says PENDING and every use of the distance, a ladder weight or a threshold stays tagged.
