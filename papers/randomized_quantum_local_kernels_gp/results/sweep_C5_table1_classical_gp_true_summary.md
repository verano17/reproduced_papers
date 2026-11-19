# Sweep C5_table1_classical_gp_true: COMPLETE SWEEP

Candidate: `{"config": "table1_classical_gp_true", "feature_mode": "code_gaussian"}`  
Completed 15 / 15 planned runs. mean and sample std (ddof=1) over (data_seed, sobol_seed) repetitions.

| Kernel | Method | Centered alignment | Uncentered alignment | RMSE | Avg NLPD | 2σ coverage |
|---|---|---|---|---|---|---|
| matern32 | parametric_gp | 0.9931 ± 0.0086 | 0.9926 ± 0.0093 | 0.0499 ± 0.0012 | -1.5775 ± 0.0235 | 0.9456 ± 0.0081 |
| nonstat_ls | parametric_gp | 0.9843 ± 0.0171 | 0.9800 ± 0.0250 | 0.0467 ± 0.0016 | -1.6445 ± 0.0361 | 0.9488 ± 0.0200 |
| rbf | parametric_gp | 0.9984 ± 0.0012 | 0.9986 ± 0.0010 | 0.0454 ± 0.0033 | -1.6674 ± 0.0688 | 0.9568 ± 0.0186 |
