# WP5: n0 frozen (development seeds 6001..6012 only)

**Frozen n0 = 5 pseudo-steps per regime.** Chosen by the pre-declared rule of `tests/PREDECLARED_CRITERIA_WP5.md` section 6 (maximise, over the grid 5, 20, 80, 320, the minimum over the three lengths of `mean - 3 se` of `shrinkage - gauss` on the 12 development seeds; ties to the smaller n0). Evaluation seeds 7001..7024, atypical seeds 9001..9012, and the criterion-6 variant scores had not been touched when this file was written (the generator self-check of amendment A6 has not been run either).

| n0 | lower at 30 | lower at 60 | lower at 120 | minimum |
|---|---|---|---|---|
| **5** | **+0.105** | +0.569 | +0.886 | **+0.105** |
| 20 | -0.353 | +0.158 | +0.637 | -0.353 |
| 80 | -1.792 | -0.998 | -0.113 | -1.792 |
| 320 | -4.957 | -4.611 | -2.407 | -4.957 |

**Reading, stated before the evaluation:** the best value is the SMALLEST grid value, so the optimum may lie below 5 (the grid was pre-declared and is not extended). On the development seeds `shared` alone is far below `gauss` (about -6 to -8.5 nats per held-out step), `personal` and the `warm` ablation (n0 = 1e-6) have catastrophic seeds at 30 steps (one seed at about -14000 and -8000 per step), and 5 to 6 of 12 `shrinkage` fits at n0 = 5 were stopped early by the EM guard (REJECTED_UPDATE). Full tables (every arm, every length, health counts): `dev_report.txt` below. `dev_results.jsonl` sha256 85ff97d8419926ab6b176eda5c39dd2502de4232dba2229fdc29eea2af94cef6.

```
288 records, T_train by length: {30: [18, '..', 60], 60: [41, '..', 111], 120: [85, '..', 196]}
arm                                 30        60       120   (mean held-out step score; n seeds)
gauss                           -6.380    -5.832    -5.450   n=12
personal                     -1216.103    -6.294    -4.516   n=12
shared@main                    -12.259   -12.980   -13.919   n=12
shrink@main@1e-06             -991.264   -11.563    -4.519   n=12
shrink@main@20                  -5.642    -4.828    -4.309   n=12
shrink@main@320                 -8.828    -7.730    -6.229   n=12
shrink@main@5                   -5.363    -4.702    -4.121   n=12
shrink@main@80                  -6.761    -5.519    -4.766   n=12

paired difference: arm minus gauss   (mean, se, mean-3se, seeds favouring arm / n, dropped)
personal                     -1209.723 1196.670 -4799.734  2/12 d0 |  -0.461  0.425  -1.735  6/12 d0 |  +0.934  0.367  -0.166 10/12 d0
shared@main                   -5.878  1.429 -10.164  2/12 d0 |  -7.148  1.548 -11.793  1/12 d0 |  -8.468  1.624 -13.341  0/12 d0
shrink@main@1e-06            -984.884 718.626 -3140.760  1/12 d0 |  -5.731  3.237 -15.443  5/12 d0 |  +0.931  0.308  +0.007 10/12 d0
shrink@main@20                +0.739  0.364  -0.353  9/12 d0 |  +1.005  0.282  +0.158 11/12 d0 |  +1.141  0.168  +0.637 12/12 d0
shrink@main@320               -2.448  0.837  -4.957  2/12 d0 |  -1.898  0.905  -4.611  3/12 d0 |  -0.779  0.543  -2.407  6/12 d0
shrink@main@5                 +1.017  0.304  +0.105 11/12 d0 |  +1.131  0.187  +0.569 12/12 d0 |  +1.329  0.148  +0.886 12/12 d0
shrink@main@80                -0.381  0.470  -1.792  4/12 d0 |  +0.313  0.437  -0.998  9/12 d0 |  +0.684  0.266  -0.113  9/12 d0

paired difference: arm minus personal   (mean, se, mean-3se, seeds favouring arm / n, dropped)
gauss                        +1209.723 1196.670 -2380.288 10/12 d0 |  +0.461  0.425  -0.813  6/12 d0 |  -0.934  0.367  -2.035  2/12 d0
shared@main                  +1203.845 1196.460 -2385.535  7/12 d0 |  -6.687  1.615 -11.532  1/12 d0 |  -9.403  1.736 -14.611  0/12 d0
shrink@main@1e-06            +224.840 1470.005 -4185.176  6/12 d0 |  -5.270  3.277 -15.100  4/12 d0 |  -0.003  0.156  -0.471  5/12 d0
shrink@main@20               +1210.462 1196.662 -2379.523 10/12 d0 |  +1.466  0.549  -0.181  8/12 d0 |  +0.207  0.349  -0.840  5/12 d0
shrink@main@320              +1207.275 1196.557 -2382.397  9/12 d0 |  -1.436  1.001  -4.438  5/12 d0 |  -1.713  0.607  -3.534  2/12 d0
shrink@main@5                +1210.740 1196.668 -2379.264 11/12 d0 |  +1.592  0.508  +0.067  9/12 d0 |  +0.395  0.321  -0.569  6/12 d0
shrink@main@80               +1209.342 1196.665 -2380.654 10/12 d0 |  +0.775  0.642  -1.151  7/12 d0 |  -0.250  0.385  -1.405  4/12 d0

paired difference: arm minus shared@main   (mean, se, mean-3se, seeds favouring arm / n, dropped)
gauss                         +5.878  1.429  +1.592 10/12 d0 |  +7.148  1.548  +2.503 11/12 d0 |  +8.468  1.624  +3.595 12/12 d0
personal                     -1203.845 1196.460 -4793.225  5/12 d0 |  +6.687  1.615  +1.841 11/12 d0 |  +9.403  1.736  +4.194 12/12 d0
shrink@main@1e-06            -979.005 718.462 -3134.390  5/12 d0 |  +1.417  3.571  -9.296  9/12 d0 |  +9.399  1.751  +4.147 12/12 d0
shrink@main@20                +6.617  1.232  +2.920 12/12 d0 |  +8.153  1.368  +4.049 12/12 d0 |  +9.609  1.603  +4.799 12/12 d0
shrink@main@320               +3.430  0.748  +1.187 12/12 d0 |  +5.250  0.843  +2.721 12/12 d0 |  +7.689  1.374  +3.566 12/12 d0
shrink@main@5                 +6.896  1.317  +2.946 12/12 d0 |  +8.279  1.471  +3.867 12/12 d0 |  +9.797  1.676  +4.770 12/12 d0
shrink@main@80                +5.497  1.084  +2.244 12/12 d0 |  +7.461  1.206  +3.843 12/12 d0 |  +9.152  1.542  +4.527 12/12 d0

arm                           len   n degraded nonfinite error scoring-fallbacks REJECTED_UPDATE NUMERICAL_FAILURE
gauss                          30  12        0         0     0                 0               0                 0
gauss                          60  12        0         0     0                 0               0                 0
gauss                         120  12        0         0     0                 0               0                 0
personal                       30  12        0         0     0                 0               0                 0
personal                       60  12        0         0     0                 0               0                 0
personal                      120  12        0         0     0                 0               0                 0
shared@main                    30  12        0         0     0                 0               0                 0
shared@main                    60  12        0         0     0                 0               0                 0
shared@main                   120  12        0         0     0                 0               0                 0
shrink@main@1e-06              30  12        0         0     0                 0               0                 0
shrink@main@1e-06              60  12        0         0     0                 0               0                 0
shrink@main@1e-06             120  12        0         0     0                 0               0                 0
shrink@main@20                 30  12        0         0     0                 0               2                 0
shrink@main@20                 60  12        0         0     0                 0               2                 0
shrink@main@20                120  12        0         0     0                 0               5                 0
shrink@main@320                30  12        0         0     0                 0               0                 0
shrink@main@320                60  12        0         0     0                 0               0                 0
shrink@main@320               120  12        0         0     0                 0               1                 0
shrink@main@5                  30  12        0         0     0                 0               5                 0
shrink@main@5                  60  12        0         0     0                 0               6                 0
shrink@main@5                 120  12        0         0     0                 0               5                 0
shrink@main@80                 30  12        0         0     0                 0               0                 0
shrink@main@80                 60  12        0         0     0                 0               2                 0
shrink@main@80                120  12        0         0     0                 0               1                 0

```
