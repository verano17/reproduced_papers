# Sweep C2_table1_classical_rff_lbfgs: COMPLETE SWEEP

Candidate: `{"config": "table1_classical_rff_lbfgs", "feature_mode": "exact"}`  
Completed 15 / 15 planned runs. mean and sample std (ddof=1) over (data_seed, sobol_seed) repetitions.

| Kernel | Method | Centered alignment | Uncentered alignment | RMSE | Avg NLPD | 2σ coverage |
|---|---|---|---|---|---|---|
| matern32 | learned_mixture | 0.9250 ± 0.0581 | 0.8291 ± 0.0249 | 0.0516 ± 0.0019 | -1.5493 ± 0.0346 | 0.9424 ± 0.0187 |
| matern32 | uniform_mixture | 0.9644 ± 0.0017 | 0.7807 ± 0.0056 | 0.0561 ± 0.0129 | -1.5009 ± 0.0730 | 0.9808 ± 0.0144 |
| matern32 | best_single_kernel | 0.4494 ± 0.0913 | 0.7333 ± 0.0187 | 0.0680 ± 0.0062 | -1.2469 ± 0.1076 | 0.9296 ± 0.0342 |
| nonstat_ls | learned_mixture | 0.8952 ± 0.0484 | 0.8699 ± 0.0313 | 0.0473 ± 0.0024 | -1.6316 ± 0.0525 | 0.9493 ± 0.0207 |
| nonstat_ls | uniform_mixture | 0.9131 ± 0.0054 | 0.8519 ± 0.0094 | 0.0516 ± 0.0037 | -1.5157 ± 0.0493 | 0.9877 ± 0.0085 |
| nonstat_ls | best_single_kernel | 0.8104 ± 0.1052 | 0.8715 ± 0.0228 | 0.0475 ± 0.0020 | -1.5976 ± 0.0296 | 0.9845 ± 0.0130 |
| rbf | learned_mixture | 0.9211 ± 0.0657 | 0.8286 ± 0.0440 | 0.0454 ± 0.0033 | -1.6672 ± 0.0696 | 0.9552 ± 0.0198 |
| rbf | uniform_mixture | 0.9383 ± 0.0026 | 0.7752 ± 0.0058 | 0.0491 ± 0.0028 | -1.5456 ± 0.0381 | 0.9872 ± 0.0120 |
| rbf | best_single_kernel | 0.8751 ± 0.0911 | 0.8308 ± 0.0241 | 0.0455 ± 0.0033 | -1.6248 ± 0.0455 | 0.9835 ± 0.0115 |

Feature sup-norm error: 0.0000 ± 0.0000 (max 0.0000).
