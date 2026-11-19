# Sweep S4_table1_papertext_data: COMPLETE SWEEP

Candidate: `{"config": "table1_papertext_data", "feature_mode": "code_gaussian"}`  
Completed 15 / 15 planned runs. mean and sample std (ddof=1) over (data_seed, sobol_seed) repetitions.

| Kernel | Method | Centered alignment | Uncentered alignment | RMSE | Avg NLPD | 2σ coverage |
|---|---|---|---|---|---|---|
| matern32 | learned_mixture | 0.9495 ± 0.0174 | 0.7710 ± 0.0153 | 0.1717 ± 0.0335 | -0.3235 ± 0.1781 | 0.9573 ± 0.0502 |
| matern32 | uniform_mixture | 0.8869 ± 0.0026 | 0.7258 ± 0.0040 | 0.2364 ± 0.0879 | -0.0611 ± 0.3575 | 0.9493 ± 0.0509 |
| matern32 | best_single_kernel | 0.9244 ± 0.0200 | 0.7794 ± 0.0231 | 0.2102 ± 0.0584 | -0.1486 ± 0.2663 | 0.9680 ± 0.0338 |
| nonstat_ls | learned_mixture | 0.9484 ± 0.0386 | 0.8995 ± 0.0353 | 0.0482 ± 0.0029 | -1.5795 ± 0.0472 | 0.9717 ± 0.0244 |
| nonstat_ls | uniform_mixture | 0.9965 ± 0.0007 | 0.9082 ± 0.0046 | 0.0479 ± 0.0027 | -1.6163 ± 0.0648 | 0.9323 ± 0.0165 |
| nonstat_ls | best_single_kernel | 0.9647 ± 0.0397 | 0.9156 ± 0.0237 | 0.0510 ± 0.0048 | -1.5388 ± 0.0725 | 0.9723 ± 0.0218 |
| rbf | learned_mixture | 0.9642 ± 0.0154 | 0.7593 ± 0.0111 | 0.0542 ± 0.0053 | -1.4762 ± 0.0816 | 0.9675 ± 0.0239 |
| rbf | uniform_mixture | 0.8988 ± 0.0027 | 0.7237 ± 0.0042 | 0.0520 ± 0.0068 | -1.5292 ± 0.1447 | 0.9461 ± 0.0283 |
| rbf | best_single_kernel | 0.9210 ± 0.0323 | 0.7705 ± 0.0261 | 0.0798 ± 0.0187 | -1.1085 ± 0.2282 | 0.9563 ± 0.0299 |

Feature sup-norm error: 0.0586 ± 0.0007 (max 0.0595).
