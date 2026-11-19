# Sweep M4_table1_merlin_mixed_lbfgs: COMPLETE SWEEP

Candidate: `{"config": "table1_merlin_mixed_lbfgs", "feature_mode": "photon_sampled"}`  
Completed 15 / 15 planned runs. mean and sample std (ddof=1) over (data_seed, sobol_seed) repetitions.

| Kernel | Method | Centered alignment | Uncentered alignment | RMSE | Avg NLPD | 2σ coverage |
|---|---|---|---|---|---|---|
| matern32 | learned_mixture | 0.9642 ± 0.0244 | 0.7317 ± 0.0079 | 0.0840 ± 0.0112 | -1.0548 ± 0.1317 | 0.9403 ± 0.0230 |
| matern32 | uniform_mixture | 0.9798 ± 0.0008 | 0.7256 ± 0.0067 | 0.1066 ± 0.0219 | -0.8030 ± 0.1904 | 0.9557 ± 0.0394 |
| matern32 | best_single_kernel | 0.8583 ± 0.0546 | 0.7355 ± 0.0134 | 0.1063 ± 0.0124 | -0.8118 ± 0.1330 | 0.9541 ± 0.0245 |
| nonstat_ls | learned_mixture | 0.9653 ± 0.0139 | 0.8153 ± 0.0112 | 0.0490 ± 0.0026 | -1.5816 ± 0.0645 | 0.9232 ± 0.0176 |
| nonstat_ls | uniform_mixture | 0.9759 ± 0.0031 | 0.8132 ± 0.0110 | 0.0477 ± 0.0016 | -1.5744 ± 0.0218 | 0.9419 ± 0.0555 |
| nonstat_ls | best_single_kernel | 0.8323 ± 0.0864 | 0.8199 ± 0.0139 | 0.0539 ± 0.0053 | -1.4945 ± 0.0884 | 0.9669 ± 0.0200 |
| rbf | learned_mixture | 0.9813 ± 0.0114 | 0.7248 ± 0.0075 | 0.0468 ± 0.0033 | -1.6249 ± 0.0862 | 0.9307 ± 0.0261 |
| rbf | uniform_mixture | 0.9893 ± 0.0010 | 0.7234 ± 0.0069 | 0.0456 ± 0.0035 | -1.5947 ± 0.0610 | 0.9168 ± 0.0463 |
| rbf | best_single_kernel | 0.8631 ± 0.0779 | 0.7292 ± 0.0203 | 0.0507 ± 0.0052 | -1.5432 ± 0.0778 | 0.9680 ± 0.0181 |

Feature sup-norm error: 0.0489 ± 0.0026 (max 0.0556).
