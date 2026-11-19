# Sweep M2_table1_merlin_lbfgs: COMPLETE SWEEP

Candidate: `{"config": "table1_merlin_lbfgs", "feature_mode": "photon_sampled"}`  
Completed 15 / 15 planned runs. mean and sample std (ddof=1) over (data_seed, sobol_seed) repetitions.

| Kernel | Method | Centered alignment | Uncentered alignment | RMSE | Avg NLPD | 2σ coverage |
|---|---|---|---|---|---|---|
| matern32 | learned_mixture | 0.9597 ± 0.0153 | 0.7310 ± 0.0106 | 0.0986 ± 0.0147 | -0.8994 ± 0.1424 | 0.9509 ± 0.0251 |
| matern32 | uniform_mixture | 0.9529 ± 0.0027 | 0.7118 ± 0.0068 | 0.1444 ± 0.0534 | -0.5518 ± 0.3279 | 0.9573 ± 0.0374 |
| matern32 | best_single_kernel | 0.8073 ± 0.0724 | 0.7168 ± 0.0146 | 0.1312 ± 0.0262 | -0.6030 ± 0.2006 | 0.9488 ± 0.0339 |
| nonstat_ls | learned_mixture | 0.9623 ± 0.0087 | 0.8126 ± 0.0082 | 0.0492 ± 0.0024 | -1.5849 ± 0.0515 | 0.9339 ± 0.0190 |
| nonstat_ls | uniform_mixture | 0.9777 ± 0.0020 | 0.8034 ± 0.0112 | 0.0491 ± 0.0018 | -1.5578 ± 0.0185 | 0.9573 ± 0.0496 |
| nonstat_ls | best_single_kernel | 0.8916 ± 0.0816 | 0.8240 ± 0.0213 | 0.0558 ± 0.0065 | -1.4627 ± 0.1124 | 0.9600 ± 0.0267 |
| rbf | learned_mixture | 0.9747 ± 0.0139 | 0.7175 ± 0.0061 | 0.0472 ± 0.0035 | -1.6205 ± 0.0886 | 0.9344 ± 0.0282 |
| rbf | uniform_mixture | 0.9672 ± 0.0028 | 0.7096 ± 0.0070 | 0.0462 ± 0.0034 | -1.5787 ± 0.0735 | 0.9488 ± 0.0622 |
| rbf | best_single_kernel | 0.8212 ± 0.1099 | 0.7263 ± 0.0197 | 0.0526 ± 0.0061 | -1.5141 ± 0.1013 | 0.9675 ± 0.0158 |

Feature sup-norm error: 0.0492 ± 0.0031 (max 0.0563).
