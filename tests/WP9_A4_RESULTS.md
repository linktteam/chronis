# WP9 A4: a negative control that can fail, results

Design and levels: `tests/PREDECLARED_CRITERIA_WP9.md` Part 6 (sections 17 and 18, committed before any A4 fit); frozen: `tests/DESIGN_FREEZE_WP9_A4.txt` (`{"design": "A"}`). Labels: **RUN** = executed here, output in `wp9_evidence/`; **READ**; **UNKNOWN**. All data are generated (easy stratum, planted K = 3, 150-day streams, windows of 45 days). Real streams with such scale changes may not exist (**UNKNOWN**).

## The question
WP8's control (a), each fit compared in its own z-space instead of through the anchor's stored scaler, was **inert** (it did not fail). A4 asked for a control that actually fails, so that the statement "the stored scaler is what keeps the identity when windows differ" is measured, not assumed. The generator got an additive option `feature_scale_change` (default off; 8 default streams pinned bit-identical to WP8 in `tests/test_wp9_a4_generator.py`) that multiplies the within-regime noise of chosen features from a given day on, leaving the regime means (raw-unit identity) untouched. Method pair on the same fits: **R** = the production matching (raw means over the anchor's stored scaler), **Z** = the WP8 control.

## Development (RUN, users 9261..9276, 416 fits, `wp9_evidence/a4_dev_analysis.txt`)
Three designs: A (features 5 and 6, noise x 0.1 from day 60), B (features 3 to 6, x 0.1), C (features 0 to 2, x 4.0). **No design made Z fail** (0 wrong-and-unflagged for both methods in every design, and 0 alignments where Z was wrong and R right). By the declared rule the control **stays inert on every design tried**, no design was searched further, and the evaluation was run once on design A (the first in the order) so that the statement is measured. (The development replay used the default gate of that time, `min_separation` 4.0; the evaluation replay used the final 2.5, as declared: the evaluation runs after the A2 freeze.)

## Evaluation (RUN once, users 9281..9296, design A, 224 fits, `wp9_evidence/a4_eval_analysis.txt`)
| method | alignments | served | correct among served | wrong-and-unflagged |
|---|---|---|---|---|
| R, stored scaler (design A) | 112 | 34 (30%) | 34/34 = 100% | 0/112 |
| Z, own z-space (design A) | 112 | 33 (29%) | 33/33 = 100% | 0/112 |
| R, same users without the change | 112 | 16 (14%) | 16/16 = 100% | 0/112 |
| Z, same users without the change | 112 | 16 (14%) | 16/16 = 100% | 0/112 |

Paired on the 112 common alignments: Z wrong and R right **0**, R wrong and Z right **0**. No fit was withheld by the publish rule (128 of 128 fits per design accepted).

| level (declared before any fit) | result | verdict |
|---|---|---|
| (R) works: correct among served >= 95% and wrong-and-unflagged <= 1% over at least 40 alignments | 100%, 0 of 112, 112 alignments | the stream is one where the method works: **MET** |
| (Z) fails as required: correct among served < 95% or wrong-and-unflagged > 1% | 100%, 0 of 112 | **NOT MET: the control is INERT** (reported as inert, not as passed) |

## Reading, stated plainly
- **A4 is not achieved: the negative control did not fail on any of the four designs tried (A, B, C on development; A on evaluation).** I did not search for a design that makes it fail, because that would have been a search on the evaluation users; the declared rule said to stop and report "inert".
- What this shows: under these planted feature-scale changes the production matching and the z-space control give the same lineage in every alignment. So these experiments **do not show that the stored scaler is what protects the identity**; they show that the identity is not at risk from these changes at all (at this separation, with fresh 45-day windows).
- Reading (**not tested**): the Hungarian match uses all seven features, and the unchanged features still dominate the distance when the scale of two to four of them moves; the mixture standard deviation of a window is dominated by the between-regime spread, not the within-regime noise that was scaled. A control would have to change what the matching sees (for example regime means themselves, or a match against another user's regimes) to fail; that is WP10 design work and has not been tried.
- The A4 statement that the package can make is limited to: raw-unit matching on a stored scaler gave 0 wrong claims in 112 alignments (Wilson upper bound 3.3%) on streams with a planted scale change, and so did the control; the control cannot tell the two methods apart here.

## Mutation checks (RUN on the final head)
`mutation_a4.py` -> `wp9_evidence/mutation_a4.txt`: **18 of 18 killed** (mutants of the analysis script's levels, chain selection, paired counts, design choice and of the generator option `feature_scale_change`). The first run killed 10 of 18: the tests did not pin the inclusive and exclusive boundaries of the declared levels, the shared-window selection or the design-selection rule (D02 to D06, D09, D11, D12); I added boundary and selection tests and the re-run killed all 18.

## Artifacts
`wp9_evidence/a4_dev_fits.jsonl` (416), `a4_dev_analysis.txt/.json`, `a4_eval_fits.jsonl` (224), `a4_eval_analysis.txt`, `mutation_a4.txt`; scripts `integration_report/wp9/scripts/a4_run.py`, `a4_analyze.py`, `mutation_a4.py`; tests `tests/test_wp9_a4_generator.py`, `tests/test_wp9_a4_analyze.py`.
