# WP10A A6: B4 (fresh and unusual people) and a prior that is provably harmful: result

Label for every number: Python 3.12, repo pins, generated users, production `fit_hssm` with `certify_prior_anchored=True`. Pre-declaration: `tests/PREDECLARED_CRITERIA_WP10A.md` Part 6 and 6b (committed before any B4 user beyond the pilot and before any harmful-prior fit). Real users: UNKNOWN throughout.

## Answer
- **B4 (RUN):** the production fit **refuses most of these users** (39 of 64 withheld). On the published users the favourable control is in band, and the misspecified condition (7 published users) is in band and not worse than the control; two conditions (5 published users each) cannot be read. No condition is worse than the favourable control by more than the declared 0.10. N17 stays an expected failure; `confidence_display_gate` stays `early_estimate`.
- **The harmful-prior control (RUN): the control can be built, and the certificate refuses what it should.** A prior with every intercept shifted by +16 was verified harmful on development by the declared rule (H1 and H2 were not), frozen, and evaluated once on fresh seeds: the control is **valid** (14 eligible fits worse than the single Gaussian; at least 12 required), **E1 MET** (0 of the 14 certified; exact one-sided 95% upper bound on the certification rate 19%), **E0 informative** (the certificate certifies 13 of the 25 eligible fits of the true prior), E3: of the 51 harmful fits that score below the Gaussian, **0 were published**. Whether a *milder* wrong prior would be refused stays **UNKNOWN** (see the limits below).

## Part 1: B4, fresh and unusual people (RUN, 64 users, seeds 10701..10764, one training length of about 90 days)
Label: Python 3.12, repo pins, generated users (`generate_population_user`), production dual fit with the shared prior (n0 = 5) and `certify_prior_anchored=True`. Measure: share of held-out days on which the true fast state, mapped into the fitted basis by the N17 per-true-regime affine map (**in sample**, so this is optimistic), lies within 1 sd and 2 sd of the engine's `m_t_std`, per latent dimension, against 0.6827 and 0.9545. The unit is the user. Real users: UNKNOWN.

**The fit refuses most of these users.** 39 of 64 fits were withheld (61%): 25 for `fit_state_not_converged`, 14 for `degenerate_duration`; 12 were certified as prior-anchored. By condition, published/16: C0 8, C1 5, C2 5, C3 7.

| condition (published users) | coverage 1 sd (d0 / d1) | coverage 2 sd (d0 / d1) | in band (+-0.10) | against C0 |
|---|---|---|---|---|
| C0 favourable control (8) | 0.69 / 0.70 | 0.90 / 0.90 | **YES** | n/a |
| C1 off-centre prior (5) | 0.57 / 0.58 (for the record) | 0.78 / 0.77 (for the record) | too few published fits to read (< 6) | not read |
| C2 atypical users (5) | 0.65 / 0.71 (for the record) | 0.90 / 0.84 (for the record) | too few to read | not read |
| C3 misspecified fit (7) | 0.70 / 0.72 | 0.98 / 0.94 | **YES** | 2-sd +0.07 / +0.04 (bootstrap 95% [-0.02, +0.21] / [-0.06, +0.17]): not worse |
Secondary, all 16 users per condition (labelled: a withheld fit has no user-facing uncertainty): C0 in band (2 sd 0.89 / 0.89); C1 NOT in band (1 sd 0.62 / 0.66, 2 sd 0.85 / 0.86; 2-sd misses -0.11 / -0.09); C2 in band (2 sd 0.87 / 0.88); C3 NOT in band (2 sd 0.84 / 0.85; misses -0.11 / -0.10). Against C0, no condition is worse by more than 0.10 (2-sd differences -0.04 / -0.02, -0.02 / -0.00, -0.05 / -0.04).

Reading:
- **The favourable control is in band on the published users** (8 users; the bootstrap 95% interval of the 1-sd coverage reaches 0.55 and 0.60, so the band is met by the point estimate, not demonstrated tightly). I had expected it to miss (N17); it did not on this measure. **N17 stays an expected failure** in its own test (cell B, dimension 0) whatever this shows, and `confidence_display_gate` stays `early_estimate`: the in-sample affine mapping, 8 users and a different generator are not the N17 test.
- **Two of the four conditions cannot be read** (5 published users each, below the declared minimum of 6): of 16 users with an off-centre prior or atypical spread, 11 each were withheld. The declared expectation "C1 and C3(d) will be worse than C0" is **not supported**: no condition is worse than C0 by more than 0.10, and C3(d) (K too small) has only 1 published fit of 4. In C3, Dmax = 8 published 0 of 4: the misspecified fits were mostly refused rather than published with bad uncertainty (3 of 4 K-too-small, 4 of 4 Dmax = 8, 1 of 4 each otherwise).
- Not run, as declared: the parametric-bootstrap comparison of the design note; K one too large. Compute was about 80 CPU-seconds per user.

## Part 2: the harmful-prior control (RUN)
Study: the WP9 B1 study (K = 2, F = 4, L = 2, lengths 30 / 60 / 120 days, 200 iterations, 1 restart, fast-test switch and cold-start gate off for the study only, n0 = 5, the public path with `certify_prior_anchored=True`); the harmful prior is the true prior with every intercept `d_obs` plus a shift.

**Development (WP9 B1 development users 9631..9642, 36 fits per candidate; the personal and Gaussian scores are the stored WP9 ones).**
| candidate | (a) mean(fit - personal), needs <= -0.25 | (b) below the single Gaussian, needs >= 6 | (c) eligible among them, needs >= 4 | verified |
|---|---|---|---|---|
| H1 (+4) | +7.525 NO | 11 ok | 3 NO | no |
| H2 (+8) | +6.013 NO | 18 ok | 5 ok | no |
| H3 (+16) | -2.977 ok | 32 ok | 7 ok | **YES, frozen** |
The declared rule froze H3 (first verified in the order H1, H2, H3). No criterion was changed; I had considered an amendment after the H1 and H2 readouts (restricting rule (a) to the 120-day fits) and **dropped it** when H3 passed the rule as declared; its code was never committed. A post hoc diagnostic (`wp10a_evidence/a6_harm_dev_posthoc.txt`, changes nothing) shows why H1 and H2 failed (a): the WP9 personal fits on 30- and 60-day streams are themselves below the single Gaussian in 11 of 12 and 7 of 12 cases, so any prior beats them, while at 120 days all three priors are worse than the personal fit.

**Evaluation (fresh seeds 10771..10794, 24 users x 3 lengths = 72 fits per arm, run once).**
- **Valid control:** arm H (the frozen +16 prior): 51 of 72 fits score below the single Gaussian on the held-out days, **14 of them eligible for the certificate** (EM state `REJECTED_UPDATE` or `MAX_ITER_REACHED`, no other withhold reason); required >= 12 -> **VALID**. Mean gap to the Gaussian -10.1 nats/step (median -0.47).
- **E1 (the certificate refuses a harmful fit): MET.** 0 of the 14 eligible harmful fits certified (exact one-sided 95% upper bound on the certification rate 19%).
- **E0 (informative): YES.** Arm T (the true prior): 13 of its 25 eligible fits are certified (CERTIFIED_PRIOR_ANCHORED), 38 of 72 converged (not the certificate's business), 31 of 72 published in all; the true prior scores above the Gaussian on 69 of 72 fits (median +1.29 nats/step). So "refuses the harmful ones" is not "refuses everything".
- **E3 (reported): 0 of 51** harmful fits that score below the Gaussian were published.

**How to read it (limits stated):**
- **The harmful prior was also stopped by rules other than the certificate.** All 72 arm-H fits ended `MAX_ITER_REACHED` (none converged), and 54 of the 72 carry another withhold reason (`degenerate_duration` in 53 fits, `collapsed_regime` in 7, some in both). The certificate was the deciding rule on only the 18 eligible fits (14 worse than the Gaussian); that is why E1 rests on 14 fits and its upper bound is 19%, not small.
- **+16 is a blunt control** (four times the separation of the true regimes). It shows the certificate refuses a grossly wrong prior on eligible fits; it does **not** show that a mildly wrong prior (the +1.5 of WP9 or the +4 and +8 candidates, which were not harmful on these measures) would be refused or harms anyone: **UNKNOWN**.
- Development and evaluation are generated users; real users: UNKNOWN. The WP9 personal and Gaussian scores of the development step were computed in WP9's environment (3.13, unpinned) and reused as stored, as declared.

## Seeds, order and what was looked at first
B4 seeds 10701..10764 (pilot user 10701 kept as it is); harmful-prior development used the WP9 development seeds 9631..9642, evaluation 10771..10794, each used once. Looked at first: the B4 pilot (declared in Part 6 section 16); the H1 development result before H2, and H2 before H3, as the declared order requires. Dated criterion changes: none after any result (Part 6b was dated before any harmful-prior fit).

## Files
`wp10a_evidence/a6_b4_fits.jsonl`, `a6_b4_analysis.txt/.json`, `a6_harm_dev_fits.jsonl`, `a6_harm_dev_analysis.txt/.json`, `a6_harm_dev_posthoc.txt`, `a6_harm_eval_fits.jsonl`, `a6_harm_eval_analysis.txt/.json`; `tests/DESIGN_FREEZE_WP10A_A6.txt`; scripts `integration_report/wp10a/scripts/a6_b4_run.py`, `a6_b4_analyze.py`, `a6_harm_run.py`, `a6_harm_analyze.py`, `a6_harm_posthoc.py`; tests `tests/test_wp10a_a6_b4.py` (29) and `tests/test_wp10a_a6_harm.py` (32).

## Mutation check
`mutation_wp10a_a6_b4.py` (the B4 design, the copied generator, the analysis): **56/56 killed** on the final head (`wp10a_evidence/mutation_wp10a_a6_b4_final.txt`; the first run was also 56/56). `mutation_wp10a_a6_harm.py` (the study settings, candidates, eligibility, validity rule, choice and evaluation levels): **51/51 killed** on the final head (`wp10a_evidence/mutation_wp10a_a6_harm_final.txt`). No survivors.
