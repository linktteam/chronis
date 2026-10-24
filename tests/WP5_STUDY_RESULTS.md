# WP5 study results (N18-v2), n0 frozen at 5 (commit 80bfd9f) before these seeds were touched

Seeds: evaluation 7001..7024 (24 users, 3 lengths), atypical 9001..9012 (12 users, 3 lengths), prior built from 8001..8020. Arms and criteria: `tests/PREDECLARED_CRITERIA_WP5.md` (incl. amendments A1..A8). Lengths are 2, 4, 8 generator segments; the realised training lengths are in the first line of each table (nominal 30, 60, 120).
Raw per-seed records (not in the repo; in the evidence bundle): eval_results.jsonl sha256 35967a72b3d5a9bb98f067d0f648043e5d4f7bb8f86388a94d2153d9747dc22d, atyp_results.jsonl sha256 8d6504f101df80dd63b44ce34a9a9c821795eff6882a1e195d8c55fdd23e944f.

## Criteria 1-3 and criterion 6 (evaluate: `n18v2_study.py eval-criteria`)
```
criteria 1-3 for shrink@main@5: c1=True c2=True c3=True all=True
  vs_gauss                 30: mean +1.312 se 0.232 mean-3se +0.616 mean+1se +1.544 mean+3se +2.008 favour 22/24 dropped 0
  vs_gauss                 60: mean +1.480 se 0.091 mean-3se +1.206 mean+1se +1.572 mean+3se +1.755 favour 24/24 dropped 0
  vs_gauss                120: mean +1.478 se 0.091 mean-3se +1.204 mean+1se +1.569 mean+3se +1.752 favour 24/24 dropped 0
  vs_personal              30: mean +65.162 se 53.174 mean-3se -94.361 mean+1se +118.336 mean+3se +224.684 favour 23/24 dropped 0
  vs_personal              60: mean +3.656 se 1.077 mean-3se +0.425 mean+1se +4.733 mean+3se +6.886 favour 23/24 dropped 0
  vs_personal             120: mean +0.506 se 0.236 mean-3se -0.203 mean+1se +0.743 mean+3se +1.216 favour 17/24 dropped 0
  vs_shared                30: mean +6.711 se 1.132 mean-3se +3.315 mean+1se +7.843 mean+3se +10.107 favour 24/24 dropped 0
  vs_shared                60: mean +7.464 se 1.309 mean-3se +3.537 mean+1se +8.773 mean+3se +11.390 favour 24/24 dropped 0
  vs_shared               120: mean +6.915 se 0.964 mean-3se +4.022 mean+1se +7.879 mean+3se +9.808 favour 24/24 dropped 0
  atypical_vs_personal     30: mean +141.160 se 122.325 mean-3se -225.815 mean+1se +263.485 mean+3se +508.135 favour 10/12 dropped 0
  atypical_vs_personal     60: mean +2.427 se 1.919 mean-3se -3.331 mean+1se +4.346 mean+3se +8.184 favour 8/12 dropped 0
  atypical_vs_personal    120: mean -0.594 se 0.512 mean-3se -2.129 mean+1se -0.082 mean+3se +0.942 favour 6/12 dropped 0
variant shrink@main@1e-06: c1=False c2=False c3=True  discriminates=True
variant personal: c1=False c2=False c3=True  discriminates=True
variant shrink@main@1e6: c1=False c2=False c3=False  discriminates=True
variant shrink@scrambled@5: c1=False c2=True c3=True  discriminates=True
variant shrink@shifted@5: c1=False c2=True c3=True  discriminates=True

```

## Evaluation seeds: every arm at every length
```
720 records, T_train by length: {30: [17, '..', 52], 60: [40, '..', 109], 120: [77, '..', 196]}
arm                                 30        60       120   (mean held-out step score; n seeds)
gauss                           -5.897    -5.638    -5.207   n=24
personal                       -69.747    -7.814    -4.235   n=24
shared@main                    -11.297   -11.622   -10.644   n=24
shared@scrambled              -120.403  -136.771  -121.346   n=24
shared@shifted                 -12.726   -12.586   -11.823   n=24
shrink@main@1e-06              -15.309    -9.339    -3.914   n=24
shrink@main@1e6                -11.294   -11.617   -10.635   n=24
shrink@main@5                   -4.586    -4.158    -3.729   n=24
shrink@scrambled@5              -6.049    -4.657    -4.080   n=24
shrink@shifted@5                -5.925    -4.509    -4.112   n=24

paired difference: arm minus gauss   (mean, se, mean-3se, seeds favouring arm / n, dropped)
personal                     -63.850 53.193 -223.428  2/24 d0 |  -2.176  1.083  -5.424 13/24 d0 |  +0.972  0.270  +0.162 23/24 d0
shared@main                   -5.399  1.243  -9.128  1/24 d0 |  -5.984  1.323  -9.954  1/24 d0 |  -5.437  0.993  -8.417  1/24 d0
shared@scrambled             -114.506  8.901 -141.209  0/24 d0 | -131.133  8.823 -157.602  0/24 d0 | -116.139  9.017 -143.191  0/24 d0
shared@shifted                -6.829  1.371 -10.942  0/24 d0 |  -6.948  1.449 -11.295  0/24 d0 |  -6.616  1.062  -9.803  0/24 d0
shrink@main@1e-06             -9.412  2.718 -17.566  2/24 d0 |  -3.701  1.507  -8.222 10/24 d0 |  +1.293  0.276  +0.464 23/24 d0
shrink@main@1e6               -5.397  1.242  -9.124  1/24 d0 |  -5.978  1.322  -9.944  1/24 d0 |  -5.428  0.992  -8.403  1/24 d0
shrink@main@5                 +1.312  0.232  +0.616 22/24 d0 |  +1.480  0.091  +1.206 24/24 d0 |  +1.478  0.091  +1.204 24/24 d0
shrink@scrambled@5            -0.152  0.383  -1.300 15/24 d0 |  +0.981  0.113  +0.641 22/24 d0 |  +1.126  0.085  +0.870 24/24 d0
shrink@shifted@5              -0.028  0.269  -0.835 15/24 d0 |  +1.129  0.094  +0.846 24/24 d0 |  +1.095  0.102  +0.790 24/24 d0

paired difference: arm minus personal   (mean, se, mean-3se, seeds favouring arm / n, dropped)
gauss                        +63.850 53.193 -95.728 22/24 d0 |  +2.176  1.083  -1.073 11/24 d0 |  -0.972  0.270  -1.782  1/24 d0
shared@main                  +58.451 53.353 -101.608 16/24 d0 |  -3.808  1.201  -7.412  5/24 d0 |  -6.409  0.850  -8.959  0/24 d0
shared@scrambled             -50.656 53.482 -211.101  1/24 d0 | -128.957  9.012 -155.995  0/24 d0 | -117.111  9.053 -144.270  0/24 d0
shared@shifted               +57.021 53.378 -103.115 16/24 d0 |  -4.773  1.286  -8.629  5/24 d0 |  -7.588  0.926 -10.365  0/24 d0
shrink@main@1e-06            +54.438 53.502 -106.067 14/24 d0 |  -1.526  1.266  -5.324  9/24 d0 |  +0.321  0.115  -0.025 15/24 d0
shrink@main@1e6              +58.453 53.353 -101.605 16/24 d0 |  -3.803  1.200  -7.404  5/24 d0 |  -6.400  0.848  -8.945  0/24 d0
shrink@main@5                +65.162 53.174 -94.361 23/24 d0 |  +3.656  1.077  +0.425 23/24 d0 |  +0.506  0.236  -0.203 17/24 d0
shrink@scrambled@5           +63.699 53.229 -95.987 22/24 d0 |  +3.157  1.050  +0.007 19/24 d0 |  +0.155  0.278  -0.679 11/24 d0
shrink@shifted@5             +63.822 53.202 -95.783 22/24 d0 |  +3.304  1.060  +0.124 19/24 d0 |  +0.123  0.276  -0.704 10/24 d0

paired difference: arm minus shared@main   (mean, se, mean-3se, seeds favouring arm / n, dropped)
gauss                         +5.399  1.243  +1.671 23/24 d0 |  +5.984  1.323  +2.013 23/24 d0 |  +5.437  0.993  +2.457 23/24 d0
personal                     -58.451 53.353 -218.509  8/24 d0 |  +3.808  1.201  +0.204 19/24 d0 |  +6.409  0.850  +3.858 24/24 d0
shared@scrambled             -109.106  8.903 -135.817  0/24 d0 | -125.149  8.849 -151.696  0/24 d0 | -110.702  9.091 -137.976  0/24 d0
shared@shifted                -1.430  0.259  -2.207  2/24 d0 |  -0.964  0.262  -1.751  5/24 d0 |  -1.179  0.230  -1.870  3/24 d0
shrink@main@1e-06             -4.012  2.825 -12.487  9/24 d0 |  +2.283  1.363  -1.805 17/24 d0 |  +6.730  0.843  +4.199 24/24 d0
shrink@main@1e6               +0.002  0.001  -0.000 24/24 d0 |  +0.005  0.002  +0.000 24/24 d0 |  +0.009  0.002  +0.001 24/24 d0
shrink@main@5                 +6.711  1.132  +3.315 24/24 d0 |  +7.464  1.309  +3.537 24/24 d0 |  +6.915  0.964  +4.022 24/24 d0
shrink@scrambled@5            +5.248  1.248  +1.505 22/24 d0 |  +6.965  1.298  +3.071 24/24 d0 |  +6.563  0.962  +3.677 24/24 d0
shrink@shifted@5              +5.371  1.067  +2.171 24/24 d0 |  +7.113  1.287  +3.251 24/24 d0 |  +6.532  0.993  +3.553 24/24 d0

arm                           len   n degraded nonfinite error scoring-fallbacks REJECTED_UPDATE NUMERICAL_FAILURE
gauss                          30  24        0         0     0                 0               0                 0
gauss                          60  24        0         0     0                 0               0                 0
gauss                         120  24        0         0     0                 0               0                 0
personal                       30  24        0         0     0                 0               0                 0
personal                       60  24        0         0     0                 0               0                 0
personal                      120  24        0         0     0                 0               0                 0
shared@main                    30  24        0         0     0                 0               0                 0
shared@main                    60  24        0         0     0                 0               0                 0
shared@main                   120  24        0         0     0                 0               0                 0
shared@scrambled               30  24        0         0     0                 0               0                 0
shared@scrambled               60  24        0         0     0                 0               0                 0
shared@scrambled              120  24        0         0     0                 0               0                 0
shared@shifted                 30  24        0         0     0                 0               0                 0
shared@shifted                 60  24        0         0     0                 0               0                 0
shared@shifted                120  24        0         0     0                 0               0                 0
shrink@main@1e-06              30  24        0         0     0                 0               0                 0
shrink@main@1e-06              60  24        0         0     0                 0               0                 0
shrink@main@1e-06             120  24        0         0     0                 0               0                 0
shrink@main@1e6                30  24        0         0     0                 0               0                 0
shrink@main@1e6                60  24        0         0     0                 0               0                 0
shrink@main@1e6               120  24        0         0     0                 0               1                 0
shrink@main@5                  30  24        0         0     0                 0               8                 0
shrink@main@5                  60  24        0         0     0                 0               9                 0
shrink@main@5                 120  24        0         0     0                 0               8                 0
shrink@scrambled@5             30  24        0         0     0                 0               2                 0
shrink@scrambled@5             60  24        0         0     0                 0               0                 0
shrink@scrambled@5            120  24        0         0     0                 0               1                 0
shrink@shifted@5               30  24        0         0     0                 0               1                 0
shrink@shifted@5               60  24        0         0     0                 0               0                 0
shrink@shifted@5              120  24        0         0     0                 0              
```

## Atypical seeds: every arm at every length
```
360 records, T_train by length: {30: [16, '..', 60], 60: [32, '..', 120], 120: [64, '..', 230]}
arm                                 30        60       120   (mean held-out step score; n seeds)
gauss                           -8.593    -6.488    -6.334   n=12
personal                      -149.934    -7.951    -4.395   n=12
shared@main                   -120.516   -96.274   -99.077   n=12
shared@scrambled              -237.813  -224.675  -219.855   n=12
shared@shifted                -121.380   -97.384  -100.215   n=12
shrink@main@1e-06             -352.205  -138.289    -4.818   n=12
shrink@main@1e6               -120.278   -95.632   -98.031   n=12
shrink@main@5                   -8.775    -5.524    -4.989   n=12
shrink@scrambled@5             -10.567    -5.589    -5.295   n=12
shrink@shifted@5               -10.252    -5.870    -5.454   n=12

paired difference: arm minus gauss   (mean, se, mean-3se, seeds favouring arm / n, dropped)
personal                     -141.342 122.104 -507.655  2/12 d0 |  -1.463  1.880  -7.103  8/12 d0 |  +1.938  0.389  +0.773 12/12 d0
shared@main                  -111.924 26.465 -191.318  0/12 d0 | -89.786 19.113 -147.124  0/12 d0 | -92.743 23.503 -163.252  0/12 d0
shared@scrambled             -229.220 25.495 -305.705  0/12 d0 | -218.187 29.202 -305.791  0/12 d0 | -213.521 20.489 -274.989  0/12 d0
shared@shifted               -112.787 26.848 -193.332  0/12 d0 | -90.897 20.297 -151.789  0/12 d0 | -93.881 23.721 -165.044  0/12 d0
shrink@main@1e-06            -343.613 268.152 -1148.069  1/12 d0 | -131.801 124.318 -504.755  7/12 d0 |  +1.515  0.264  +0.724 11/12 d0
shrink@main@1e6              -111.685 26.369 -190.794  0/12 d0 | -89.144 18.834 -145.646  0/12 d0 | -91.697 23.090 -160.967  0/12 d0
shrink@main@5                 -0.182  1.044  -3.313  6/12 d0 |  +0.964  0.186  +0.406 12/12 d0 |  +1.345  0.245  +0.609 12/12 d0
shrink@scrambled@5            -1.974  1.289  -5.841  5/12 d0 |  +0.898  0.153  +0.440 12/12 d0 |  +1.038  0.375  -0.086 10/12 d0
shrink@shifted@5              -1.659  0.746  -3.896  5/12 d0 |  +0.618  0.177  +0.087 10/12 d0 |  +0.880  0.240  +0.161 11/12 d0

paired difference: arm minus personal   (mean, se, mean-3se, seeds favouring arm / n, dropped)
gauss                        +141.342 122.104 -224.971 10/12 d0 |  +1.463  1.880  -4.177  4/12 d0 |  -1.938  0.389  -3.104  0/12 d0
shared@main                  +29.418 129.127 -357.964  2/12 d0 | -88.323 18.944 -145.156  0/12 d0 | -94.682 23.739 -165.898  0/12 d0
shared@scrambled             -87.878 127.620 -470.737  2/12 d0 | -216.724 29.435 -305.027  0/12 d0 | -215.460 20.752 -277.716  0/12 d0
shared@shifted               +28.555 129.668 -360.449  2/12 d0 | -89.434 20.175 -149.959  0/12 d0 | -95.820 23.943 -167.649  0/12 d0
shrink@main@1e-06            -202.271 308.337 -1127.281  4/12 d0 | -130.339 124.725 -504.513  5/12 d0 |  -0.423  0.550  -2.073  7/12 d0
shrink@main@1e6              +29.656 129.094 -357.626  2/12 d0 | -87.682 18.658 -143.655  0/12 d0 | -93.635 23.327 -163.615  0/12 d0
shrink@main@5                +141.160 122.325 -225.815 10/12 d0 |  +2.427  1.919  -3.331  8/12 d0 |  -0.594  0.512  -2.129  6/12 d0
shrink@scrambled@5           +139.368 122.275 -227.458  9/12 d0 |  +2.361  1.921  -3.403  7/12 d0 |  -0.900  0.527  -2.481  6/12 d0
shrink@shifted@5             +139.683 122.223 -226.986  8/12 d0 |  +2.081  1.926  -3.697  6/12 d0 |  -1.059  0.436  -2.367  6/12 d0

paired difference: arm minus shared@main   (mean, se, mean-3se, seeds favouring arm / n, dropped)
gauss                        +111.924 26.465 +32.530 12/12 d0 | +89.786 19.113 +32.447 12/12 d0 | +92.743 23.503 +22.235 12/12 d0
personal                     -29.418 129.127 -416.800 10/12 d0 | +88.323 18.944 +31.490 12/12 d0 | +94.682 23.739 +23.466 12/12 d0
shared@scrambled             -117.297 24.036 -189.403  1/12 d0 | -128.401 25.788 -205.766  0/12 d0 | -120.778 13.968 -162.681  0/12 d0
shared@shifted                -0.863  2.380  -8.002  7/12 d0 |  -1.111  2.063  -7.299  8/12 d0 |  -1.138  2.089  -7.404  6/12 d0
shrink@main@1e-06            -231.689 255.473 -998.109  8/12 d0 | -42.016 110.095 -372.301 10/12 d0 | +94.259 23.490 +23.788 12/12 d0
shrink@main@1e6               +0.238  0.118  -0.115 12/12 d0 |  +0.641  0.312  -0.295 12/12 d0 |  +1.046  0.547  -0.596 12/12 d0
shrink@main@5                +111.741 25.877 +34.111 12/12 d0 | +90.750 19.098 +33.456 12/12 d0 | +94.088 23.383 +23.939 12/12 d0
shrink@scrambled@5           +109.950 25.799 +32.553 12/12 d0 | +90.684 19.100 +33.385 12/12 d0 | +93.782 23.349 +23.735 12/12 d0
shrink@shifted@5             +110.265 26.536 +30.657 12/12 d0 | +90.404 19.110 +33.073 12/12 d0 | +93.623 23.458 +23.251 12/12 d0

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
shared@scrambled               30  12        0         0     0                 0               0                 0
shared@scrambled               60  12        0         0     0                 0               0                 0
shared@scrambled              120  12        0         0     0                 0               0                 0
shared@shifted                 30  12        0         0     0                 0               0                 0
shared@shifted                 60  12        0         0     0                 0               0                 0
shared@shifted                120  12        0         0     0                 0               0                 0
shrink@main@1e-06              30  12        0         0     0                 0               0                 0
shrink@main@1e-06              60  12        0         0     0                 0               0                 0
shrink@main@1e-06             120  12        0         0     0                 0               0                 0
shrink@main@1e6                30  12        0         0     0                 0               0                 0
shrink@main@1e6                60  12        0         0     0                 0               0                 0
shrink@main@1e6               120  12        0         0     0                 0               0                 0
shrink@main@5                  30  12        0         0     0                 0               3                 0
shrink@main@5                  60  12        0         0     0                 0               1                 0
shrink@main@5                 120  12        0         0     0                 0               1                 0
shrink@scrambled@5             30  12        0         0     0                 0               1                 0
shrink@scrambled@5             60  12        0         0     0                 0               0                 0
shrink@scrambled@5            120  12        0         0     0                 0               1                 0
shrink@shifted@5               30  12        0         0     0                 0               2                 0
shrink@shifted@5               60  12        0         0     0                 0               4                 0
shrink@shifted@5              120  
```
