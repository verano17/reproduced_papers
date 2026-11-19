# Sweep S3_table1_pauli_shadow: COMPLETE SWEEP

Candidate: `{"config": "table1_pauli_shadow", "feature_mode": "pauli_shadow"}`  
Completed 15 / 15 planned runs. mean and sample std (ddof=1) over (data_seed, sobol_seed) repetitions.

| Kernel | Method | Centered alignment | Uncentered alignment | RMSE | Avg NLPD | 2σ coverage |
|---|---|---|---|---|---|---|
| matern32 | learned_mixture | 0.9804 ± 0.0151 | 0.8457 ± 0.0094 | 0.1246 ± 0.0209 | -0.6531 ± 0.1712 | 0.9541 ± 0.0332 |
| matern32 | uniform_mixture | 0.9644 ± 0.0014 | 0.8232 ± 0.0045 | 0.1311 ± 0.0381 | -0.6444 ± 0.2626 | 0.9472 ± 0.0191 |
| matern32 | best_single_kernel | 0.9498 ± 0.0146 | 0.8574 ± 0.0207 | 0.1578 ± 0.0342 | -0.4217 ± 0.2143 | 0.9541 ± 0.0333 |
| nonstat_ls | learned_mixture | 0.9556 ± 0.0190 | 0.9068 ± 0.0209 | 0.0550 ± 0.0042 | -1.4604 ± 0.0698 | 0.9387 ± 0.0445 |
| nonstat_ls | uniform_mixture | 0.9777 ± 0.0029 | 0.8907 ± 0.0084 | 0.0524 ± 0.0029 | -1.4197 ± 0.0268 | 0.9952 ± 0.0073 |
| nonstat_ls | best_single_kernel | 0.9267 ± 0.0484 | 0.9114 ± 0.0244 | 0.0676 ± 0.0151 | -1.2700 ± 0.2376 | 0.9563 ± 0.0385 |
| rbf | learned_mixture | 0.9644 ± 0.0130 | 0.8503 ± 0.0213 | 0.0517 ± 0.0050 | -1.5170 ± 0.0923 | 0.9611 ± 0.0310 |
| rbf | uniform_mixture | 0.9789 ± 0.0012 | 0.8234 ± 0.0047 | 0.0498 ± 0.0043 | -1.4409 ± 0.0368 | 0.9947 ± 0.0065 |
| rbf | best_single_kernel | 0.9512 ± 0.0233 | 0.8500 ± 0.0296 | 0.0652 ± 0.0105 | -1.2982 ± 0.1844 | 0.9397 ± 0.0382 |

Feature sup-norm error: 0.1780 ± 0.0089 (max 0.1920).
