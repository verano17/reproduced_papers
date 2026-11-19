# Sweep S2_table1_exact_features: COMPLETE SWEEP

Candidate: `{"config": "table1_exact_features", "feature_mode": "exact"}`  
Completed 15 / 15 planned runs. mean and sample std (ddof=1) over (data_seed, sobol_seed) repetitions.

| Kernel | Method | Centered alignment | Uncentered alignment | RMSE | Avg NLPD | 2σ coverage |
|---|---|---|---|---|---|---|
| matern32 | learned_mixture | 0.9544 ± 0.0322 | 0.8449 ± 0.0129 | 0.0951 ± 0.0153 | -0.8926 ± 0.1442 | 0.9264 ± 0.0654 |
| matern32 | uniform_mixture | 0.9644 ± 0.0014 | 0.8242 ± 0.0045 | 0.1260 ± 0.0355 | -0.6662 ± 0.2674 | 0.9541 ± 0.0365 |
| matern32 | best_single_kernel | 0.9335 ± 0.0378 | 0.8524 ± 0.0227 | 0.1176 ± 0.0260 | -0.7147 ± 0.2150 | 0.9595 ± 0.0335 |
| nonstat_ls | learned_mixture | 0.9576 ± 0.0249 | 0.8960 ± 0.0292 | 0.0470 ± 0.0015 | -1.6056 ± 0.0220 | 0.9824 ± 0.0106 |
| nonstat_ls | uniform_mixture | 0.9778 ± 0.0029 | 0.8915 ± 0.0084 | 0.0472 ± 0.0013 | -1.6010 ± 0.0199 | 0.9872 ± 0.0090 |
| nonstat_ls | best_single_kernel | 0.9494 ± 0.0185 | 0.9109 ± 0.0236 | 0.0473 ± 0.0014 | -1.6007 ± 0.0202 | 0.9813 ± 0.0120 |
| rbf | learned_mixture | 0.9700 ± 0.0138 | 0.8414 ± 0.0209 | 0.0454 ± 0.0033 | -1.6254 ± 0.0451 | 0.9867 ± 0.0099 |
| rbf | uniform_mixture | 0.9790 ± 0.0012 | 0.8245 ± 0.0047 | 0.0455 ± 0.0034 | -1.6240 ± 0.0457 | 0.9861 ± 0.0111 |
| rbf | best_single_kernel | 0.9453 ± 0.0219 | 0.8442 ± 0.0196 | 0.0458 ± 0.0033 | -1.6208 ± 0.0444 | 0.9872 ± 0.0112 |

Feature sup-norm error: 0.0000 ± 0.0000 (max 0.0000).
