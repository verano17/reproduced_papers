# Sweep M3_table1_merlin_mixed: COMPLETE SWEEP

Candidate: `{"config": "table1_merlin_mixed", "feature_mode": "photon_sampled"}`  
Completed 15 / 15 planned runs. mean and sample std (ddof=1) over (data_seed, sobol_seed) repetitions.

| Kernel | Method | Centered alignment | Uncentered alignment | RMSE | Avg NLPD | 2σ coverage |
|---|---|---|---|---|---|---|
| matern32 | learned_mixture | 0.9496 ± 0.0312 | 0.7310 ± 0.0087 | 0.0866 ± 0.0128 | -0.9815 ± 0.1187 | 0.9211 ± 0.0461 |
| matern32 | uniform_mixture | 0.9798 ± 0.0008 | 0.7256 ± 0.0067 | 0.1066 ± 0.0219 | -0.8030 ± 0.1904 | 0.9557 ± 0.0394 |
| matern32 | best_single_kernel | 0.8511 ± 0.0564 | 0.7341 ± 0.0143 | 0.1082 ± 0.0132 | -0.7883 ± 0.1467 | 0.9552 ± 0.0262 |
| nonstat_ls | learned_mixture | 0.9125 ± 0.0734 | 0.8177 ± 0.0133 | 0.0493 ± 0.0037 | -1.5581 ± 0.0539 | 0.9824 ± 0.0155 |
| nonstat_ls | uniform_mixture | 0.9759 ± 0.0031 | 0.8132 ± 0.0110 | 0.0477 ± 0.0016 | -1.5744 ± 0.0218 | 0.9419 ± 0.0555 |
| nonstat_ls | best_single_kernel | 0.8346 ± 0.0873 | 0.8211 ± 0.0188 | 0.0537 ± 0.0055 | -1.4970 ± 0.0907 | 0.9669 ± 0.0200 |
| rbf | learned_mixture | 0.9378 ± 0.0372 | 0.7359 ± 0.0142 | 0.0468 ± 0.0042 | -1.5749 ± 0.0810 | 0.9483 ± 0.0582 |
| rbf | uniform_mixture | 0.9893 ± 0.0010 | 0.7234 ± 0.0069 | 0.0456 ± 0.0035 | -1.5947 ± 0.0610 | 0.9168 ± 0.0463 |
| rbf | best_single_kernel | 0.8631 ± 0.0779 | 0.7292 ± 0.0203 | 0.0507 ± 0.0052 | -1.5432 ± 0.0778 | 0.9680 ± 0.0181 |

Feature sup-norm error: 0.0489 ± 0.0026 (max 0.0556).
