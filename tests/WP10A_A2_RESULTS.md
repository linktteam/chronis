# WP10A A2 (finding F2): the intermittent failure in `tests/test_wp8_item0_metadata.py`: result

Label for every number: Python 3.12.3, repo pins (numpy 2.2.6, scipy 1.15.3, pytest 9.1.1), this 2-core machine (OpenBLAS reports `SkylakeX`). Real runners other than this machine: UNKNOWN.

## Answer
**F2 was not reproduced here, so its cause is UNKNOWN. What is known: (1) the fits are bit-identical across OpenBLAS thread settings on one kernel; (2) they are NOT bit-identical across kernels (up to 0.8% relative on a non-converged fixture), although every discrete outcome was identical; (3) one test in the file depended on exactly the kind of knife edge that a kernel difference can flip, and I replaced that dependence with an injected optimiser verdict plus an invariant check; (4) an exact experiment for the failing runner type is ready.**

## 1. The repeat (RUN): the file run N times per BLAS setting, one process per run, `PYTHONHASHSEED` = the run index
**250 of 250 runs passed (23 tests each, 0 failures) in every configuration:**
| configuration | OpenBLAS threads | kernel (OPENBLAS_CORETYPE) | runs | passed | failed | one-sided 95% upper bound on the failure rate |
|---|---|---|---|---|---|---|
| t1 | 1 | default (SkylakeX) | 50 | 50 | 0 | 5.8% |
| t2 | 2 | default (SkylakeX) | 50 | 50 | 0 | 5.8% |
| tdef | unset (all 2 cores) | default (SkylakeX) | 50 | 50 | 0 | 5.8% |
| haswell | 1 | HASWELL | 50 | 50 | 0 | 5.8% |
| skylakex | 1 | SKYLAKEX | 50 | 50 | 0 | 5.8% |
| all | | | 250 | 250 | 0 | 1.2% |
Run i of every configuration used `PYTHONHASHSEED` = i (0 to 49). Each run took about 76 s (median). Two runs carry long recorded elapsed times (`tdef` run 5: 3 h 42 min; `t1` run 9: 14 min): I suspended the repeat twice to give the cores to other jobs, and the elapsed time of the run in progress includes the suspension; both passed 23 of 23. The failure the integration agent saw was 1 in 9 runs (about 11%); if this machine's failure rate were 11%, 50 passes in a row would happen with probability 0.3% per configuration, so **the failing condition is not present on this machine's kernels (SkylakeX and Haswell kernels, 1 to 2 threads); it says nothing about the runner CPU type that failed (UNKNOWN)**.


## 2. What the cross-kernel probe found (RUN; `wp10a_evidence/a2_probe_*.txt/.json`)
The fixture fits of the metadata file (one converged, one non-converged, plus the seed-1 and seed-2 duration-optimiser cases) were run under 5 distinct OpenBLAS kernels that this machine can force (default = SKYLAKEX = COOPERLAKE = SAPPHIRERAPIDS identical; HASWELL = ZEN; NEHALEM = BARCELONA = ATOM; SANDYBRIDGE = BULLDOZER = PILEDRIVER = STEAMROLLER = EXCAVATOR; PRESCOTT = CORE2 = DUNNINGTON = OPTERON) and thread settings 1, 2 and unset, 15 configurations in the first probe.
- **Within a kernel, across thread settings: bit-identical** (all 655 fit fields, every kernel tested). OpenBLAS with more than 2 threads was NOT tested (this machine has 2 cores).
- **Between kernels: not bit-identical.** Against the default kernel, HASWELL, SANDYBRIDGE and NEHALEM differ in about 308 of 655 stored fit fields; the largest differences are on the non-converged fixture (`dur_sigma`, absolute 1.0e-4 to 2.8e-4, **relative up to 0.82%**); the SKYLAKEX kernel is identical to the default. **Every discrete outcome was identical in every configuration** (converged state, non-converged state `MAX_ITER_REACHED`, seed 1 duration optimiser did not fall back, seed 2 fell back, no bound or degradation).
- So the statement the owner may use is: **"fits are bit-identical across OpenBLAS thread counts 1, 2 and default on one CPU kernel (this machine), and across SKYLAKEX, COOPERLAKE and SAPPHIRERAPIDS kernels; they are equal to within 0.8% relative (stored floats of the non-converged fixture) and equal in every discrete outcome across the five kernels tested."** It must not be said that fits are bit-identical across CPUs.

## 3. Audit of random-number paths and thread-dependent reductions (READ, `backbone/hssm`)
- Every random generator is built from an explicit seed (`np.random.default_rng(seed)` in `model.py`, `dual_model.py`, `neural_residual.py`, `ordering.py`, `shrinkage.py` for label scrambling only); no global `np.random` state is used; the fixtures pass seeds. No iteration over a `set` or an unordered mapping in the fitting path (`fitting.py`, `dual_model.py`, `model.py`, `ordering.py`); `PYTHONHASHSEED` was varied in the repeat (run index) and played no role in the runs.
- The thread-dependent part is the BLAS in matrix products; on this machine its result does not depend on the thread count (Section 2), but it does depend on the kernel.
- **The only discrete decision that depends on the last bits of a float is the duration optimiser:** `scipy.optimize.minimize(..., method="Nelder-Mead")` with SciPy's default caps (`dual_model.py` lines 718 and 758), whose `success` flag is False when it stops on its iteration or function-evaluation cap. Where a Nelder-Mead run stops, and whether it hits the cap, can move with the last bits of the E-step. This is the **hypothesis** for F2 (INFERENCE, not tested on a failing machine): a seed on which the optimiser sits at the cap on one kernel and just below it on another.

## 4. What changed (committed)
- `tests/test_wp8_item0_metadata.py`: `test_dual_optimizer_success_is_false_when_the_duration_optimiser_fell_back_and_true_otherwise` used two data seeds chosen because the optimiser fell back on one and not on the other, which is the knife edge above. It now **injects the optimiser's verdict** (real optimiser run, `success` flag set) so the model's own fallback path is what is tested and the result no longer depends on where Nelder-Mead stops; the natural fits are checked for the invariant (the reported flag is the negation of the fallback flag, whatever the optimiser did) for both seeds. That this test is the one that failed in the integration run is an **inference**: the failing test name was lost.
- `tests/test_wp10a_a2_cross_kernel.py`, `tests/test_wp10a_a2_experiment.py`: tests of the cross-kernel probe helpers and the runner experiment. No production code changed for A2.

## 5. The exact experiment for the integration agent (the failing runner type)
`PYTHONPATH=. python integration_report/wp10a/scripts/a2_runner_experiment.py OUT_DIR [--runs 20] [--kernels default,HASWELL,SKYLAKEX,ZEN,SANDYBRIDGE] [--threads 1,2,4,unset]` from the repository root with the pinned environment; everything is also written to OUT_DIR for upload as a CI artifact. Phase 0 records the CPU model, the BLAS libraries and the kernel each actually uses; phase 1 runs one fresh process per (kernel, threads), fingerprints the fixture fits and compares them with the reference fingerprint in `wp10a_evidence/a2_probe_drive.json` and prints the discrete outcomes that decide the test; phase 2 runs the test file `--runs` times in the runner's own environment with `PYTHONHASHSEED` = run index and records every failure with its test id and message. Reading: (a) a configuration whose discrete outcomes differ from the reference reproduces the failure mechanism (a CPU-kernel dependent fixture); (b) differing bits but equal discrete outcomes does not; (c) a phase-2 failure with equal outcomes in phase 1 means another cause, and the recorded test id and message are the starting point. It also tests thread counts above 2, which this machine could not.

## What was looked at first, and what was declared
This item has no pre-declared numeric level (the brief asks for counts, an audit, and either a deterministic result or a documented tolerance). Looked at first: the probe of one converged and one non-converged fixture across kernels and threads (Section 2) before the repeat; the hypothesis in Section 3 was formed from that probe and the code, before any repeat result. No criterion was changed. The repeat runs fixed hash seeds 0 to 49, not fresh random ones.

## What this does not say
It does not say that F2 is fixed or absent on the integration runners: the failing runner type was not available here and the failure was never seen (0 of 250). It does not say the hardened test is the one that failed (INFERENCE). Thread counts above 2 and other BLAS libraries were not tested.

## Files and checks
`wp10a_evidence/a2_repeat.jsonl` (250 runs), `a2_probe_*.txt/.json` (cross-kernel and thread probes), `mutation_item0e_final.txt`; scripts `integration_report/wp10a/scripts/a2_repeat.py`, `a2_probe.py`, `a2_probe_summary.py`, `a2_runner_experiment.py`; tests `tests/test_wp10a_a2_cross_kernel.py` and `tests/test_wp10a_a2_experiment.py`, and the hardened `tests/test_wp8_item0_metadata.py` (23 tests). Mutation: the 15-mutant set of the metadata test (`integration_report/wp8/scripts/mutation_item0e.py`) was rerun on the final head: **15/15 killed**, including the two mutants of the hardened test (`optimizer_success` constant True; inverted).
