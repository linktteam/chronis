# WP9 B4 (shrinkage uncertainty on fresh and unusual people): design note, NOT RUN in this package

Status: **not executed**. The brief ranks B4 last (after B6 and B5) and lets it fall to a design note when compute runs short; the compute went to the unmet WP8 criteria (A2 evaluation, 768 fits) and to B1, which B4 depends on (it uses certified fits). Nothing below has been run, so there is no result, no coverage number and no change to the N17 xfail (it stays).

## Question (owner decision B6)
Does the WP7-R result "the posterior sd of the fast state (`m_t_std`) reaches the 0.6827 / 0.9545 bands" hold outside the favourable case (typical person, prior at the population centre, correct K, F and Dmax)?

## Design for WP10 (seeds 9701..9724, reserved by the brief)
- **Data:** 24 users from `data/synthetic_population.generate_population_user`, 8 per condition. Conditions: (1) **off-centre prior** (the shared prior built from a population whose centre is shifted by 1 and 2 population sd); (2) **atypical users** (spread factor 3x); (3) **misspecification**: fitted K one too large and one too small, fitted F with one feature dropped, Dmax half and double the true mean dwell.
- **Fits:** the production dual fit with the shrinkage prior (n0 = 5) and `certify_prior_anchored=True`; only fits that are published or `CERTIFIED_PRIOR_ANCHORED` enter the coverage (a withheld fit has no user-facing uncertainty); the count of withheld fits per condition is reported, because a condition in which nothing certifies is itself the answer.
- **Measure:** for every published fit, the share of held-out days on which the true fast state lies within 1 and 2 posterior sd (`m_t_std`), against 0.6827 and 0.9545; per condition and pooled over users with a user-level bootstrap interval; the bootstrap on a small subset (8 users, 200 resamples of the training days) as the comparison the brief asks for.
- **Levels to declare before running:** coverage within +-0.10 of both nominal levels per condition for the favourable case reproduced from WP7-R as a control; for the other conditions the report states which conditions fall outside the band and by how much; no level is moved.
- **Cost estimate:** 24 users x about 4 fits x (3 certification refits) at about 3-13 minutes each: about 6-8 hours on the 2 cores of this environment.

## What B4 is expected to find (stated now so it cannot be moved)
I expect the bands to be missed for an off-centre prior (the shrinkage pulls the means toward the wrong centre, and the posterior sd does not know) and for K too small. N17 stays an expected failure; the display gate (B5) already returns `early_estimate` for every input while N17 is not passed.
