# Sweep C4_table1_classical_rbfdict_lbfgs: COMPLETE SWEEP

Candidate: `{"config": "table1_classical_rbfdict_lbfgs", "feature_mode": "exact"}`  
Completed 15 / 15 planned runs. mean and sample std (ddof=1) over (data_seed, sobol_seed) repetitions.

| Kernel | Method | Centered alignment | Uncentered alignment | RMSE | Avg NLPD | 2σ coverage |
|---|---|---|---|---|---|---|
| matern32 | learned_mixture | 0.9795 ± 0.0086 | 0.9600 ± 0.0326 | 0.0499 ± 0.0011 | -1.5770 ± 0.0236 | 0.9456 ± 0.0081 |
| matern32 | uniform_mixture | 0.9684 ± 0.0011 | 0.9534 ± 0.0009 | 0.0562 ± 0.0106 | -1.4751 ± 0.0592 | 0.9856 ± 0.0097 |
| matern32 | best_single_kernel | 0.7851 ± 0.0298 | 0.7850 ± 0.0275 | 0.0535 ± 0.0028 | -1.5033 ± 0.0390 | 0.9808 ± 0.0099 |
| nonstat_ls | learned_mixture | 0.9312 ± 0.0398 | 0.9058 ± 0.0535 | 0.0470 ± 0.0021 | -1.6381 ± 0.0454 | 0.9520 ± 0.0196 |
| nonstat_ls | uniform_mixture | 0.9122 ± 0.0052 | 0.9509 ± 0.0031 | 0.0541 ± 0.0046 | -1.4728 ± 0.0591 | 0.9904 ± 0.0097 |
| nonstat_ls | best_single_kernel | 0.9470 ± 0.0201 | 0.9360 ± 0.0221 | 0.0469 ± 0.0019 | -1.6050 ± 0.0276 | 0.9888 ± 0.0112 |
| rbf | learned_mixture | 0.9781 ± 0.0169 | 0.9215 ± 0.0877 | 0.0454 ± 0.0033 | -1.6679 ± 0.0689 | 0.9552 ± 0.0154 |
| rbf | uniform_mixture | 0.9422 ± 0.0020 | 0.9423 ± 0.0010 | 0.0532 ± 0.0037 | -1.4826 ± 0.0413 | 0.9840 ± 0.0117 |
| rbf | best_single_kernel | 0.9986 ± 0.0022 | 0.9988 ± 0.0020 | 0.0454 ± 0.0032 | -1.6251 ± 0.0439 | 0.9856 ± 0.0097 |
