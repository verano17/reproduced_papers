# Sweep C3_table1_classical_rbfdict: COMPLETE SWEEP

Candidate: `{"config": "table1_classical_rbfdict", "feature_mode": "exact"}`  
Completed 15 / 15 planned runs. mean and sample std (ddof=1) over (data_seed, sobol_seed) repetitions.

| Kernel | Method | Centered alignment | Uncentered alignment | RMSE | Avg NLPD | 2σ coverage |
|---|---|---|---|---|---|---|
| matern32 | learned_mixture | 0.9826 ± 0.0185 | 0.9604 ± 0.0271 | 0.0499 ± 0.0016 | -1.5442 ± 0.0202 | 0.9872 ± 0.0066 |
| matern32 | uniform_mixture | 0.9684 ± 0.0011 | 0.9534 ± 0.0009 | 0.0562 ± 0.0106 | -1.4751 ± 0.0592 | 0.9856 ± 0.0097 |
| matern32 | best_single_kernel | 0.7933 ± 0.0312 | 0.7930 ± 0.0289 | 0.0537 ± 0.0030 | -1.5007 ± 0.0410 | 0.9792 ± 0.0124 |
| nonstat_ls | learned_mixture | 0.9336 ± 0.0526 | 0.9238 ± 0.0565 | 0.0471 ± 0.0019 | -1.6002 ± 0.0277 | 0.9888 ± 0.0084 |
| nonstat_ls | uniform_mixture | 0.9122 ± 0.0052 | 0.9509 ± 0.0031 | 0.0541 ± 0.0046 | -1.4728 ± 0.0591 | 0.9904 ± 0.0097 |
| nonstat_ls | best_single_kernel | 0.9494 ± 0.0183 | 0.9386 ± 0.0204 | 0.0470 ± 0.0019 | -1.6043 ± 0.0280 | 0.9872 ± 0.0135 |
| rbf | learned_mixture | 0.9748 ± 0.0285 | 0.9747 ± 0.0281 | 0.0453 ± 0.0032 | -1.6236 ± 0.0409 | 0.9856 ± 0.0097 |
| rbf | uniform_mixture | 0.9422 ± 0.0020 | 0.9423 ± 0.0010 | 0.0532 ± 0.0037 | -1.4826 ± 0.0413 | 0.9840 ± 0.0117 |
| rbf | best_single_kernel | 0.9981 ± 0.0034 | 0.9983 ± 0.0030 | 0.0454 ± 0.0032 | -1.6251 ± 0.0440 | 0.9856 ± 0.0097 |
