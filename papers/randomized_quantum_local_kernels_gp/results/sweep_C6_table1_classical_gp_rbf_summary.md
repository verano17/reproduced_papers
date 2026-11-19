# Sweep C6_table1_classical_gp_rbf: COMPLETE SWEEP

Candidate: `{"config": "table1_classical_gp_rbf", "feature_mode": "code_gaussian"}`  
Completed 15 / 15 planned runs. mean and sample std (ddof=1) over (data_seed, sobol_seed) repetitions.

| Kernel | Method | Centered alignment | Uncentered alignment | RMSE | Avg NLPD | 2σ coverage |
|---|---|---|---|---|---|---|
| matern32 | parametric_gp | 0.7765 ± 0.0393 | 0.7767 ± 0.0373 | 0.0533 ± 0.0030 | -1.5134 ± 0.0526 | 0.9488 ± 0.0186 |
| nonstat_ls | parametric_gp | 0.9445 ± 0.0181 | 0.9332 ± 0.0182 | 0.0469 ± 0.0019 | -1.6396 ± 0.0408 | 0.9552 ± 0.0171 |
| rbf | parametric_gp | 0.9984 ± 0.0012 | 0.9986 ± 0.0010 | 0.0454 ± 0.0033 | -1.6674 ± 0.0688 | 0.9568 ± 0.0186 |
