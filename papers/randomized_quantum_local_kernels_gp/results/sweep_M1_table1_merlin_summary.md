# Sweep M1_table1_merlin: COMPLETE SWEEP

Candidate: `{"config": "table1_merlin", "feature_mode": "photon_sampled"}`  
Completed 15 / 15 planned runs. mean and sample std (ddof=1) over (data_seed, sobol_seed) repetitions.

| Kernel | Method | Centered alignment | Uncentered alignment | RMSE | Avg NLPD | 2σ coverage |
|---|---|---|---|---|---|---|
| matern32 | learned_mixture | 0.9142 ± 0.0807 | 0.7302 ± 0.0103 | 0.1057 ± 0.0139 | -0.8094 ± 0.1295 | 0.9429 ± 0.0434 |
| matern32 | uniform_mixture | 0.9529 ± 0.0027 | 0.7118 ± 0.0068 | 0.1444 ± 0.0534 | -0.5518 ± 0.3279 | 0.9573 ± 0.0374 |
| matern32 | best_single_kernel | 0.8073 ± 0.0724 | 0.7168 ± 0.0146 | 0.1312 ± 0.0262 | -0.6030 ± 0.2006 | 0.9488 ± 0.0339 |
| nonstat_ls | learned_mixture | 0.9028 ± 0.1119 | 0.8151 ± 0.0110 | 0.0545 ± 0.0114 | -1.4640 ± 0.2037 | 0.9536 ± 0.0513 |
| nonstat_ls | uniform_mixture | 0.9777 ± 0.0020 | 0.8034 ± 0.0112 | 0.0491 ± 0.0018 | -1.5578 ± 0.0185 | 0.9573 ± 0.0496 |
| nonstat_ls | best_single_kernel | 0.8880 ± 0.0798 | 0.8218 ± 0.0191 | 0.0558 ± 0.0065 | -1.4627 ± 0.1125 | 0.9611 ± 0.0280 |
| rbf | learned_mixture | 0.9481 ± 0.0257 | 0.7260 ± 0.0110 | 0.0477 ± 0.0039 | -1.5826 ± 0.0518 | 0.9824 ± 0.0122 |
| rbf | uniform_mixture | 0.9672 ± 0.0028 | 0.7096 ± 0.0070 | 0.0462 ± 0.0034 | -1.5787 ± 0.0735 | 0.9488 ± 0.0622 |
| rbf | best_single_kernel | 0.8250 ± 0.1127 | 0.7283 ± 0.0210 | 0.0528 ± 0.0059 | -1.5099 ± 0.0993 | 0.9680 ± 0.0160 |

Feature sup-norm error: 0.0492 ± 0.0031 (max 0.0563).
