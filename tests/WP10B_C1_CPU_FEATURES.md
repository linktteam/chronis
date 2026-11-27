# WP10B C1(3): the cause of the WP10A F2 flip, and the cross-machine check of the baseline engine
Written 2026-11-27, after the development runs of C1(2)/C2 and the B1 development run, in answer to opa's message (below). Labels: RUN = I executed it here; READ = supplied by opa or read from a file; INFERENCE; UNKNOWN. Environment of every RUN: Python 3.12.3, repo pins (numpy 2.2.6), this machine = an Intel Xeon with AVX-512 (numpy enables AVX512F/CD/BW/DQ/VL/VNNI/IFMA/VBMI/FP16 and the dispatch groups SKX, CLX, CNL, ICL, SPR), OpenBLAS kernel SkylakeX, 2 cores. Evidence: `wp10b_evidence/c1_cpu_features.json`, `c1_cpu_features_localise.json`, `wp10b_evidence/cross_machine/*`.

## What opa reported (READ)
On the AMD EPYC 7763 runner (Python 3.12.16) `seed1_fallback` is True in every configuration; on this machine and on an EPYC 9V74 it is False. The hardened test file passed 80/80 and 40/40 on the 7763. AVX-512 is the only recorded CPU difference (INFERENCE by opa). Requested: (a) reproduce the flip here with NPY_DISABLE_CPU_FEATURES; (b) run the 12-user baseline bit-identity battery with and without AVX-512 numpy features and report what the product sees; (c) no claim of cross-machine reproducibility before (a) and (b) are reported. The B1 development run had already finished when the message arrived, so "before B1" could not be honoured literally; the B1 results are affected only in the way stated in section 4.

## (a) The flip is reproduced here (RUN)
Probe = the WP10A probe child (`wp10a/scripts/a2_probe.py --child`: the fits of `tests/test_wp8_item0_metadata.py`), one fresh process per configuration, `integration_report/wp10b/scripts/c1_cpu_features.py`.

| configuration (env) | numpy AVX-512 names still enabled | seed1_fallback | seed2_fallback | other discrete outcomes | floats vs default |
|---|---|---|---|---|---|
| default | AVX512F/CD/SKX/CLX/CNL/ICL/SPR + BW/DQ/VL/... | **False** | True | same | - |
| NPY_DISABLE_CPU_FEATURES = AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL | only BW/DQ/VL/VNNI/IFMA/VBMI/FP16/SPR | **True** | True | same | differ |
| the same + OPENBLAS_CORETYPE=ZEN (OpenBLAS then reports the Haswell kernel) | same | **True** | True | same | differ |
| only AVX512_SKX/CLX/CNL/ICL off (F and CD on) | F, CD, ... | False | True | same | bit-identical on every fit |
| AVX512CD/KNL/KNM/SKX/CLX/CNL/ICL off (F on) | F, ... | False | True | same | differ |

So the flag follows numpy's own AVX-512 dispatch, not the OpenBLAS kernel (the 18-kernel WP10A probe had already shown that). Within the AVX-512 set I did not isolate one name: the flip appeared only when AVX512F was turned off together with the others; turning off everything above F, or only the SKX group, did not flip it. The result on this machine shows that disabling numpy's AVX-512 paths is SUFFICIENT to reproduce your 7763 value (True); that the 7763's own behaviour has this cause is INFERENCE (I did not run on that CPU).

**Mechanism (RUN).** The duration M-step of the dual model minimises a weighted log-normal likelihood with `scipy.optimize.minimize(method="Nelder-Mead")` and falls back to the previous duration when `res.success` is False. In the seed-1 probe fit (80 optimiser calls in 40 EM iterations) all 80 calls succeed with AVX-512 on; with it off, call 24 ends "Maximum number of function evaluations has been exceeded" (nit 199, nfev 400) at an objective value of 1.46e-117: the regime has essentially no expected duration counts, the objective is numerically flat, and whether the simplex shrinks below the tolerance before the evaluation limit depends on rounding-level differences between numpy's AVX-512 and non-AVX-512 exp/log loops. `res.success` False raises inside the try block, the previous parameters are kept and `duration_fallback_used_` becomes True; `optimizer_success`, `duration_optimizer_success` follow. INFERENCE: the flat objective is the signature of a nearly empty regime; I did not look at which regime it is.

**Size of the differences (RUN; disabled set 1 and 2 above give the same picture).** Converged fixture: stored floats differ by at most 1.4e-12 absolute (6.3e-13 relative), no discrete field differs. Seed-2 fit: at most 3.8e-13 absolute (2.4e-11 relative), no discrete field differs. The two MAX_ITER_REACHED fits (not-converged fixture and the seed-1 fit): the duration parameters differ by up to 2.8e-3 absolute (8% relative on a sigma of about 0.03); in the seed-1 fit the four fields derived from the fallback (`duration_fallback_used`, `model.duration_fallback_used_`, `duration_optimizer_success`, `optimizer_success`) flip. The fit states and iteration counts are the same in all configurations.

**Hardened tests (RUN).** `tests/test_wp8_item0_metadata.py` and `tests/test_wp7r_item5_dual_honesty.py` (44 tests): 44 passed with AVX-512 on and 44 passed with it off (this machine), consistent with your 80/80 and 40/40 on the 7763.

## (b) The baseline engine across the same switch (RUN)
`integration_report/wp10b/scripts/c1_cross_machine.py`: the WP9 12-user battery (population users 5301..5312; `fast` = K 2, 3 restarts; `full` = K in (2, 3), 10 restarts, 60 iterations; public path `fit_hssm`), each in a fresh process of the same tree, configurations default / NO_AVX512 (set 2 above) / NO_AVX512_ZEN (set 3: the closest emulation of the 7763 that I can make on this CPU). Controls: the same configuration run twice gives 12 of 12 bit-identical (fast); a file compared with itself gives 12 of 12.

| comparison with default (12 users each) | bit-identical outputs | same fit state, K, Dmax, withheld reason, collapse flag, stability flag + reason | labels equal | largest abs differences (p_t_posterior / regime means / durations / BIC) | served ids equal | ALIGNED with the other configuration's fit (vs the same fit fed twice) |
|---|---|---|---|---|---|---|
| fast, NO_AVX512 | 0 | 12 | 12 | 1.3e-8 / 1.1e-9 / 2.0e-7 / 9.1e-8 | 12 | 12 (12) |
| fast, NO_AVX512_ZEN | 0 | 12 | 12 | 1.9e-8 / 8.3e-10 / 1.7e-7 / 1.3e-7 | 12 | 12 (12) |
| full, NO_AVX512 | 0 | 12 | 12 | 8.3e-8 / 1.8e-8 / 2.9e-7 / 1.3e-7 | 12 | 3 (3) |
| full, NO_AVX512_ZEN | 0 | 12 | 12 | 1.6e-7 / 9.9e-9 / 1.0e-6 / 1.3e-7 | 12 | 3 (3) |

What the table says. Fields that differ: every float of every fit (fitted posterior probabilities, regime means, duration parameters, BIC, BIC by K, and one `restart_summary` number, `gap_to_next_distinct`, by 6.9e-8); never a discrete product field: fit state (all CONVERGED), K, Dmax, withheld reason (none withheld), collapse flag, `fit_unstable` flag and reason, the decoded regime labels (identical sequences), the served stable ids and names of the lineage. In the `full` variant 9 of 12 users carry `fit_unstable` = True (`best_restarts_disagree`) in BOTH configurations, so their lineage state is CONTINUITY_UNVERIFIED in both, and also when the same fit is fed to the lineage twice: that is the production gate, not the CPU. The lineage continuity decision (ALIGNED or not) is identical between the configurations in every user (aligned-only-with-the-other-configuration: 0; aligned-only-with-itself: 0).

**Is the baseline engine affected? Plainly:** its numbers are NOT bit-reproducible across this switch (0 of 24 fits bit-identical; differences 1e-9 to 1e-6), but on these 12 users and 2 variants (24 fits) every discrete output the product acts on or shows was identical. Limits: 12 population users, one emulation on one CPU (not a second physical CPU; the OpenBLAS thread count was the default 2), the identity-stream lineage case was not run in this form. UNKNOWN: how often a near-tie (the best-restart choice has a gap of 1e-7 nats in places, a convergence tolerance, the consensus threshold, the 0.95 posterior flag) flips on other users or other CPUs; 24 fits cannot bound that. The dual engine is more exposed (section (a): a flag flipped in 1 of the 2 probe fits).

## (c) What is and is not claimed
- Claimed (RUN): the F2 outcome flip is caused by numpy's AVX-512 dispatch on this machine; mechanism as above; the hardened tests pass either way; the baseline engine's discrete outputs were identical in 24 of 24 fits.
- NOT claimed: bit-reproducibility of anything across CPUs (it does not hold); that no discrete outcome of the baseline engine can differ across machines (not bounded); anything about a 7763 beyond what opa measured there.
- Consequence for the earlier statements: every number in WP10A and WP10B was produced on an AVX-512 machine; a test or report that asserts a discrete dual-engine fit outcome (a fallback flag, an optimiser success flag) is machine dependent.

## 4. Dated addendum slot: B1 diagnosis replicate with AVX-512 off (started 2026-11-27, results added below when finished)

### Addendum, 2026-11-27 (written after the replicate finished; the earlier sections are unchanged)
**What was run (RUN).** The B1 diagnosis (36 S0 dual fits, users 11201..11236, `wp10b_evidence/b1_dev_fits.jsonl`) was run a second time on the same machine with `NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL"` set before numpy was imported (`wp10b_evidence/b1_dev_fits_noavx512.jsonl`, analysis `b1_dev_analysis_noavx512.txt/json`). Python 3.12, repo pins; no code was changed between the two runs.

**Result (RUN).** Discrete outcomes: identical in 36 of 36 fits for the fit state, the reason categories, the publishable flag and the iteration count. All counts of `b1_dev_analysis.txt` (states, categories, exclusive combinations, degenerate-duration counts, the extrapolations to +200/+400/+800/+1600 iterations, publishable 2/12, 2/12, 0/12) are the same in the replicate. Floats: not bit-identical. Largest differences over the 36 fits: held-out one-step mean of the dual fit 6.8e-4 (median 3.7e-14), of the baseline engine 8.2e-8, final log-likelihood 3.9e-3, a duration location parameter 1.3e-2; the median creep rate differs in the 10th digit. The only line of the analysis that differs materially is the median fit time of the easy stratum (59 s against 113 s): that is machine load (other jobs were running on the two cores during that part of the replicate), not a CPU-feature effect, and it is not a result.

**Reading.**
- (RUN) The B1 conclusion (dual fits do not publish because EM does not finish: iteration cap and rejected update; durations are not the cause) does not depend on the AVX-512 setting on this machine.
- (RUN) On these 36 fits the iteration cap was never moved across the line by the changed rounding, so the `seed1_fallback`-type flip of F2 is not visible in the S0 setting; the F2 flip needs the flat Nelder-Mead objective of the nearly empty regime and was found in the K = 2 population setting.
- (INFERENCE) The B2 to B5 numbers need no AVX-512 caveat for their discrete findings on this machine; they keep the caveat of section (c) for other CPUs.
- (UNKNOWN) Whether the 36 fits would agree on the EPYC 7763; no claim of cross-machine reproducibility is made.
