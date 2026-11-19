# Sweep C1_table1_classical_rff: COMPLETE SWEEP

Candidate: `{"config": "table1_classical_rff", "feature_mode": "exact"}`  
Completed 15 / 15 planned runs. mean and sample std (ddof=1) over (data_seed, sobol_seed) repetitions.

| Kernel | Method | Centered alignment | Uncentered alignment | RMSE | Avg NLPD | 2σ coverage |
|---|---|---|---|---|---|---|
| matern32 | learned_mixture | 0.9111 ± 0.0810 | 0.8306 ± 0.0213 | 0.0517 ± 0.0016 | -1.5286 ± 0.0190 | 0.9803 ± 0.0059 |
| matern32 | uniform_mixture | 0.9644 ± 0.0017 | 0.7807 ± 0.0056 | 0.0561 ± 0.0129 | -1.5009 ± 0.0730 | 0.9808 ± 0.0144 |
| matern32 | best_single_kernel | 0.4793 ± 0.0961 | 0.7394 ± 0.0167 | 0.0701 ± 0.0069 | -1.2027 ± 0.1275 | 0.9205 ± 0.0413 |
| nonstat_ls | learned_mixture | 0.8854 ± 0.0390 | 0.8721 ± 0.0383 | 0.0472 ± 0.0019 | -1.5979 ± 0.0274 | 0.9872 ± 0.0108 |
| nonstat_ls | uniform_mixture | 0.9131 ± 0.0054 | 0.8519 ± 0.0094 | 0.0516 ± 0.0037 | -1.5157 ± 0.0493 | 0.9877 ± 0.0085 |
| nonstat_ls | best_single_kernel | 0.8055 ± 0.1347 | 0.8727 ± 0.0264 | 0.0478 ± 0.0018 | -1.5931 ± 0.0265 | 0.9808 ± 0.0116 |
| rbf | learned_mixture | 0.9172 ± 0.0588 | 0.8452 ± 0.0270 | 0.0457 ± 0.0033 | -1.6184 ± 0.0445 | 0.9867 ± 0.0108 |
| rbf | uniform_mixture | 0.9383 ± 0.0026 | 0.7752 ± 0.0058 | 0.0491 ± 0.0028 | -1.5456 ± 0.0381 | 0.9872 ± 0.0120 |
| rbf | best_single_kernel | 0.8748 ± 0.0907 | 0.8307 ± 0.0240 | 0.0455 ± 0.0034 | -1.6245 ± 0.0457 | 0.9835 ± 0.0115 |

Feature sup-norm error: 0.0000 ± 0.0000 (max 0.0000).
