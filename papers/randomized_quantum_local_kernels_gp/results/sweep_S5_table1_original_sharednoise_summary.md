# Sweep S5_table1_original_sharednoise: COMPLETE SWEEP

Candidate: `{"config": "table1_original_sharednoise", "feature_mode": "code_gaussian"}`  
Completed 15 / 15 planned runs. mean and sample std (ddof=1) over (data_seed, sobol_seed) repetitions.

| Kernel | Method | Centered alignment | Uncentered alignment | RMSE | Avg NLPD | 2σ coverage |
|---|---|---|---|---|---|---|
| matern32 | learned_mixture | 0.9879 ± 0.0092 | 0.8421 ± 0.0082 | 0.0982 ± 0.0111 | -0.8732 ± 0.0865 | 0.9285 ± 0.0479 |
| matern32 | uniform_mixture | 0.9644 ± 0.0014 | 0.8241 ± 0.0045 | 0.1273 ± 0.0356 | -0.6546 ± 0.2631 | 0.9589 ± 0.0301 |
| matern32 | best_single_kernel | 0.9324 ± 0.0398 | 0.8497 ± 0.0214 | 0.1324 ± 0.0255 | -0.5832 ± 0.1887 | 0.9413 ± 0.0430 |
| nonstat_ls | learned_mixture | 0.9585 ± 0.0293 | 0.8953 ± 0.0207 | 0.0501 ± 0.0024 | -1.5474 ± 0.0376 | 0.9531 ± 0.0458 |
| nonstat_ls | uniform_mixture | 0.9778 ± 0.0029 | 0.8913 ± 0.0084 | 0.0483 ± 0.0011 | -1.6071 ± 0.0253 | 0.9408 ± 0.0116 |
| nonstat_ls | best_single_kernel | 0.9408 ± 0.0298 | 0.9070 ± 0.0227 | 0.0538 ± 0.0062 | -1.4942 ± 0.1011 | 0.9653 ± 0.0286 |
| rbf | learned_mixture | 0.9717 ± 0.0192 | 0.8363 ± 0.0116 | 0.0474 ± 0.0039 | -1.5980 ± 0.1103 | 0.9451 ± 0.0402 |
| rbf | uniform_mixture | 0.9789 ± 0.0012 | 0.8243 ± 0.0047 | 0.0467 ± 0.0039 | -1.6404 ± 0.0843 | 0.9472 ± 0.0222 |
| rbf | best_single_kernel | 0.9504 ± 0.0153 | 0.8439 ± 0.0239 | 0.0515 ± 0.0051 | -1.5312 ± 0.0777 | 0.9707 ± 0.0165 |

Feature sup-norm error: 0.0586 ± 0.0007 (max 0.0595).
