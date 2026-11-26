# Expert review packet: how CHRONIS decides that a regime is "the same as before" (WP9 A5, owner decision A7)

**Status: PENDING. No expert has reviewed anything in this packet.** Every lineage record carries `expert_review_status = "PENDING"` until the owner records a sign-off (`lineage.EXPERT_REVIEW_STATUS`, `docs/EXPERT_REVIEW_REGISTER.md`). Every place in `backbone/hssm/` that defines or uses the matching distance, a ladder weight or an alignment threshold is tagged `EXPERT_REVIEW(identity-distance)`; the register of those places (32 sites) is generated from the tags and a test (`tests/test_wp8_item7_expert_review.py`) fails when it is stale or when a function uses one of the watched names without the tag.

Labels used below: **RUN** = executed in this repository and the output is stored in `wp8_evidence/` or `wp9_evidence/`; **READ** = read in code; **REPORTED** = taken from a package document; **UNKNOWN** = not established.

## 1. What the reviewer is asked to look at, in one paragraph
A person's fitted model has "regimes" (phases). A refit on a newer window names its regimes with new numbers; the system must say whether regime 2 now is regime 2 before. It matches the regimes of the new fit to those of the last trusted fit ("the anchor") and **may say "same as before" only when the fit is identifiable and the match is unambiguous** (`continuity_state = ALIGNED`). In every other case it keeps the old names, says "cannot confirm this is the same as before", and lets the user choose new names. All of that rests on the distance, the gate and the thresholds below. They were chosen on synthetic users (the development seeds listed in each section), never on real people.

## 2. The items under review

### 2.1 The matching distance (READ: `lineage._vectors`, `alignment`, `alignment_from_cost`)
- Each regime is the vector of its **regime means in raw physical units** (not the fit's own z-scores; the scaler of every fit is stored), **divided feature by feature by the anchor's own scaler sd**.
- The cost of pairing old regime *i* with new regime *j* is the **squared Euclidean distance** between those vectors. The match is the Hungarian assignment (minimum total cost). `C1` = its total cost, `C2` = the cost of the best *other* one-to-one assignment (exhaustive for K up to 6).
- Optional extras, both **off by default**: a duration coordinate (`duration_weight`, the log expected dwell time as one more coordinate; the development study found it tied with "off", so the simpler one stayed) and a second, contextual channel (2.5).
- Margins stored with every alignment: `margin_diff = (C2 - C1) / K` and `margin_ratio = C1 / C2`.

### 2.2 The identifiability gate (READ: `separation_min`; `AlignmentConfig.min_separation = h = 4.0`)
- A fit can be matched only if its own regimes are far enough apart from each other. The statistic is the **smallest, over pairs of regimes, Euclidean norm of the difference of the two regime means divided per feature by the pooled within-regime sd of the days decoded into the regimes** (seven features, so a value of 4 is roughly 1.5 sd per feature).
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
- **FIT_UNSTABLE (A1).** A baseline fit whose three best healthy restarts do not agree (adjusted Rand index below 0.99) is flagged and never served as ALIGNED (`fit_unstable`, `anchor_fit_unstable`). Nothing about the selected model changes.
- **K change suspected (A3).** On a refit, a fit of the most recent 45 days at K - 1, K and K + 1 regimes is compared by BIC. A suspected change keeps the old names and the state `CONTINUITY_UNVERIFIED` with the reason `k_change_suspected` (never "same as before"); split and merge candidates now come from the posterior overlap with the old regimes, next to the old nearest-mean rule.
- **Gate and coverage trade-off (A2).** See section 4.

## 4. The trade-off the expert is asked to weigh (A2) ⟦filled from the A2 evaluation⟧
⟦A2-CURVE⟧

## 5. What the synthetic evidence cannot say
- All users are generated (K = 3 planted regimes, planted separation in a stated range, seven features). Real separation, real regime counts, real dwell times and real drift are **UNKNOWN**. The strata (hard, medium, easy) are assumptions of the study, not measurements.
- Coverage numbers describe the generator's mix of easy and hard users; the share of real users who are identifiable is UNKNOWN.
- Thresholds were chosen on development seeds and judged on separate evaluation seeds once; 48 evaluation users per design is a small sample and several results are "met on the point estimate, not demonstrated".

## 6. Questions for the expert
1. **Distance.** Is Euclidean distance between regime means in raw units, scaled by the anchor's per-feature sd, an acceptable measure of "the same regime"? Would you scale by the regime's own spread, use a Mahalanobis distance, or include how long a regime lasts?
2. **Gate.** Is `h = 4.0` (seven features, smallest pairwise norm in within-regime sd) the right way to say that a fit is identifiable, and is the level defensible? The curve in section 4 shows what each level costs in coverage and in wrong-and-unflagged matches.
3. **Ratio rule.** Is a strict win with `C1 / C2 <= 0.5` an acceptable definition of an unambiguous match, given that errors began at about 0.48 in development?
4. **Ladder.** Are the registry directions (which feature means "more active") and the weight `coverage * SNR / (1 + SNR)` acceptable? Are `min_modal_freq = 0.90` and `g_min = 0.7` too lenient or too strict, knowing that the ladder abstains for weakly separated people?
5. **Contextual channel.** Under which consent and aggregation conditions may it be switched on, and what threshold should apply to it?
6. **Product targets.** The package asks for coverage of at least 80% with at most 1% wrong-and-unflagged matches. Is a 1% rate of saying "same as before" wrongly acceptable for this product, and is 80% the right ambition? (If the two cannot be met together, section 4 says so plainly.)
7. **Fail closed.** Is it acceptable that dual-to-dual and cross-engine matches are never served as "same as before" until WP10?
8. **K change.** Is keeping the old names with "cannot confirm this is the same as before" the right behaviour when the number of regimes appears to change, and should the user be offered split / merge candidates at all?

## 7. How to record the outcome
Record the review (who, when, which items, what was changed) in this file and in `docs/EXPERT_REVIEW_REGISTER.md`, change `lineage.EXPERT_REVIEW_STATUS`, and regenerate the register (`python3 integration_report/wp8/scripts/expert_review_register.py --write`). Until then every record says PENDING and every use of the distance, a ladder weight or a threshold stays tagged.
