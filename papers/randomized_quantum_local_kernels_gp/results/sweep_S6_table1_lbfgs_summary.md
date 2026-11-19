# Sweep S6_table1_lbfgs: COMPLETE SWEEP

Candidate: `{"config": "table1_lbfgs", "feature_mode": "code_gaussian"}`  
Completed 15 / 15 planned runs. mean and sample std (ddof=1) over (data_seed, sobol_seed) repetitions.

| Kernel | Method | Centered alignment | Uncentered alignment | RMSE | Avg NLPD | 2σ coverage |
|---|---|---|---|---|---|---|
| matern32 | learned_mixture | 0.9862 ± 0.0072 | 0.8486 ± 0.0095 | 0.0976 ± 0.0122 | -0.9031 ± 0.1213 | 0.9429 ± 0.0302 |
| matern32 | uniform_mixture | 0.9644 ± 0.0014 | 0.8241 ± 0.0045 | 0.1273 ± 0.0356 | -0.6546 ± 0.2631 | 0.9589 ± 0.0301 |
| matern32 | best_single_kernel | 0.9324 ± 0.0398 | 0.8497 ± 0.0214 | 0.1324 ± 0.0255 | -0.5832 ± 0.1887 | 0.9413 ± 0.0430 |
| nonstat_ls | learned_mixture | 0.9772 ± 0.0032 | 0.8955 ± 0.0074 | 0.0513 ± 0.0016 | -1.4800 ± 0.0601 | 0.8773 ± 0.0239 |
| nonstat_ls | uniform_mixture | 0.9778 ± 0.0029 | 0.8913 ± 0.0084 | 0.0483 ± 0.0011 | -1.6071 ± 0.0253 | 0.9408 ± 0.0116 |
| nonstat_ls | best_single_kernel | 0.9389 ± 0.0292 | 0.9088 ± 0.0159 | 0.0540 ± 0.0060 | -1.4924 ± 0.0993 | 0.9648 ± 0.0282 |
| rbf | learned_mixture | 0.9811 ± 0.0100 | 0.8217 ± 0.0067 | 0.0493 ± 0.0043 | -1.5278 ± 0.1371 | 0.9045 ± 0.0419 |
| rbf | uniform_mixture | 0.9789 ± 0.0012 | 0.8243 ± 0.0047 | 0.0467 ± 0.0039 | -1.6404 ± 0.0843 | 0.9472 ± 0.0222 |
| rbf | best_single_kernel | 0.9464 ± 0.0175 | 0.8448 ± 0.0194 | 0.0511 ± 0.0050 | -1.5360 ± 0.0767 | 0.9728 ± 0.0165 |

Feature sup-norm error: 0.0586 ± 0.0007 (max 0.0595).
