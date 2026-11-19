# Sweep S8_table1_exact_features_lbfgs_x2p5: COMPLETE SWEEP

Candidate: `{"config": "table1_exact_features_lbfgs_x2p5", "feature_mode": "exact"}`  
Completed 5 / 5 planned runs. mean and sample std (ddof=1) over (data_seed, sobol_seed) repetitions.

| Kernel | Method | Centered alignment | Uncentered alignment | RMSE | Avg NLPD | 2σ coverage |
|---|---|---|---|---|---|---|
| matern32 | learned_mixture | 0.8669 ± 0.0268 | 0.8134 ± 0.0136 | 0.0536 ± 0.0023 | -1.5048 ± 0.0421 | 0.9456 ± 0.0119 |
| matern32 | uniform_mixture | 0.9700 ± 0.0009 | 0.8338 ± 0.0058 | 0.0589 ± 0.0023 | -1.4131 ± 0.0410 | 0.9552 ± 0.0121 |
| matern32 | best_single_kernel | 0.7123 ± 0.0764 | 0.7868 ± 0.0178 | 0.0667 ± 0.0032 | -1.2661 ± 0.0657 | 0.9200 ± 0.0277 |
| nonstat_ls | learned_mixture | 0.9433 ± 0.0184 | 0.9012 ± 0.0135 | 0.0471 ± 0.0021 | -1.6352 ± 0.0458 | 0.9488 ± 0.0193 |
| nonstat_ls | uniform_mixture | 0.8644 ± 0.0073 | 0.8769 ± 0.0093 | 0.0479 ± 0.0025 | -1.5829 ± 0.0375 | 0.9856 ± 0.0104 |
| nonstat_ls | best_single_kernel | 0.8774 ± 0.0840 | 0.8899 ± 0.0168 | 0.0476 ± 0.0027 | -1.5950 ± 0.0404 | 0.9888 ± 0.0091 |
| rbf | learned_mixture | 0.9486 ± 0.0359 | 0.8465 ± 0.0230 | 0.0457 ± 0.0037 | -1.6617 ± 0.0787 | 0.9536 ± 0.0191 |
| rbf | uniform_mixture | 0.9396 ± 0.0019 | 0.8258 ± 0.0060 | 0.0458 ± 0.0036 | -1.6115 ± 0.0472 | 0.9856 ± 0.0104 |
| rbf | best_single_kernel | 0.9139 ± 0.0386 | 0.8489 ± 0.0205 | 0.0457 ± 0.0036 | -1.6216 ± 0.0497 | 0.9856 ± 0.0104 |

Feature sup-norm error: 0.0000 ± 0.0000 (max 0.0000).
