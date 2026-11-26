# WP10A A4: a negative control that can fail: result (CLOSED on development; no evaluation run)

Label for every number: Python 3.12, repo pins, generated users, production `fit_hssm` (10 restarts), production lineage matching (R) and the fault/control (Z, each fit compared in its own z-space instead of through the anchor's stored scaler). Pre-declaration: `tests/PREDECLARED_CRITERIA_WP10A.md` Part 2 (committed before any A4 fit) and Part 7 (the closure).

## Answer
**No design made the control fail, so A4 is closed, by the rule declared in advance, as "the protective effect of the stored scaler cannot be demonstrated". No evaluation was run and the evaluation seeds (users 10321..10336) were not generated.** The stored scaler is neither shown to be needed nor shown to be unnecessary: on these streams Z never served a wrong answer where R served a right one (`zr` = 0 in every design). This is RUN on development only; real users are UNKNOWN.

## Numbers (development fits; alignments = accepted refits after the first of a chain)
| design | fits | R: served / correct among served / wrong-and-unflagged | Z: served / correct among served / wrong-and-unflagged | zr / rz | R works | control fails (new rule: zr >= 3 and zr > rz) | old WP9 criterion on Z |
|---|---|---|---|---|---|---|---|
| D scale change tied to a regime-mean change (users 10301..10316) | 128 (16 chains) | 10 of 111 / 9 of 10 = 90.0% / 1 of 111 = 0.9% | 37 of 111 / 32 of 37 = 86.5% / 5 of 111 = 4.5% | 0 / 0 | NO (90% < 95%, on only 10 served) | NO | would fail |
| D, same users without the planted change (reported only) | 128 | 37 of 112 / 37 of 37 / 0 | 37 of 112 / 37 of 37 / 0 | 0 / 0 | yes | NO | passes |
| E windows of very different length (users 10301..10316) | 144 (128 chains) | 9 of 123 / 9 of 9 / 0 | 9 of 123 / 8 of 9 = 88.9% / 1 of 123 = 0.8% | 0 / 0 | yes | NO | would fail |
| F the production protocol with the fault planted (A3 development users 10001..10048) | 480 (48 chains) | 228 of 432 / 228 of 228 / 0 | 228 of 432 / 228 of 228 / 0 | 0 / 0 | yes | NO | passes |

Design F is the direct statement of the claim (production windows, production protocol): with the stored scaler replaced by the per-window scaler, **0 of 432 alignments change** in any of the three strata (easy, medium, hard: 144 each, zr = rz = 0). I said in advance that I expected it to be inert; it was.

## What the data does and does not say (reading, not a criterion)
- In D the planted change does damage to R itself: R's coverage falls from 33% (without the change) to 9% and its correct-among-served is 9 of 10. R mostly withholds there. Z serves 37 and gets 5 wrong, and they are all alignments on which R did **not** serve (hence `zr` = 0: no alignment where Z is wrong and R is right). The two methods differ in the serve decision on 27 of 111 alignments. So in D the stored scaler appears to protect by **withholding** more, and the declared criterion (which asks for Z wrong where R served right) does not count that. The old WP9 criterion (Z correct among served < 95% or Z wrong-and-unflagged > 1%) is met in D (86.5%, 4.5%) and in E (8 of 9 = 88.9%), but it rests on 1 to 5 wrong alignments out of 111 to 123 and on 9 to 37 served, and I do not rest a claim on it.
- The new criterion was declared to be stricter than the old one for the reason in Part 2 section 7 (a Z that is wrong only where R also withholds shows nothing about R being right). It is applied as declared.
- Nothing here says the stored scaler is useless. It says that on three designs (a correlated scale-and-mean change, very different window lengths, and the production protocol) the replacement did not produce a wrong answer that R got right.

## What I looked at before the full run
A preview of the same analysis on the part of the A3 development fits that existed at the time (616 of 768) was read before the A3 development run finished; it showed `zr` = 0 in all designs. The full analysis above was then run once on all 768 A3 development fits and the 368 A4 fits. Criterion changes: none (Part 7 only records the closure and corrects a wording slip about seed ranges).

## Files
`wp10a_evidence/a4_dev_fits.jsonl` (368 fits), `a4_dev_analysis.txt`, `a4_dev_analysis.json`; scripts `integration_report/wp10a/scripts/a4_run.py`, `a4_analyze.py`; tests `tests/test_wp10a_a4_analyze.py` and `tests/test_wp10a_a4_generator.py` (38 in all, including the one added for the survivor M17).

## Mutation check
`mutation_wp10a_a4.py`: 51 mutants of the analysis, the run helpers and the control. First run 50/51 (`wp10a_evidence/mutation_wp10a_a4_run1.txt`; the survivor M17 was killed by a new test). **Rerun on the final head: 51/51 killed** (`wp10a_evidence/mutation_wp10a_a4_final.txt`).
