# Sweep S7_table1_exact_features_lbfgs: COMPLETE SWEEP

Candidate: `{"config": "table1_exact_features_lbfgs", "feature_mode": "exact"}`  
Completed 15 / 15 planned runs. mean and sample std (ddof=1) over (data_seed, sobol_seed) repetitions.

| Kernel | Method | Centered alignment | Uncentered alignment | RMSE | Avg NLPD | 2σ coverage |
|---|---|---|---|---|---|---|
| matern32 | learned_mixture | 0.9687 ± 0.0203 | 0.8467 ± 0.0107 | 0.0908 ± 0.0121 | -0.9780 ± 0.1325 | 0.9445 ± 0.0254 |
| matern32 | uniform_mixture | 0.9644 ± 0.0014 | 0.8242 ± 0.0045 | 0.1260 ± 0.0355 | -0.6662 ± 0.2674 | 0.9541 ± 0.0365 |
| matern32 | best_single_kernel | 0.9260 ± 0.0463 | 0.8504 ± 0.0241 | 0.1183 ± 0.0268 | -0.7095 ± 0.2171 | 0.9563 ± 0.0330 |
| nonstat_ls | learned_mixture | 0.9566 ± 0.0201 | 0.8947 ± 0.0200 | 0.0469 ± 0.0018 | -1.6406 ± 0.0382 | 0.9520 ± 0.0198 |
| nonstat_ls | uniform_mixture | 0.9778 ± 0.0029 | 0.8915 ± 0.0084 | 0.0472 ± 0.0013 | -1.6010 ± 0.0199 | 0.9872 ± 0.0090 |
| nonstat_ls | best_single_kernel | 0.9410 ± 0.0268 | 0.9043 ± 0.0233 | 0.0473 ± 0.0014 | -1.6014 ± 0.0209 | 0.9819 ± 0.0115 |
| rbf | learned_mixture | 0.9594 ± 0.0258 | 0.8413 ± 0.0258 | 0.0454 ± 0.0033 | -1.6674 ± 0.0692 | 0.9531 ± 0.0198 |
| rbf | uniform_mixture | 0.9790 ± 0.0012 | 0.8245 ± 0.0047 | 0.0455 ± 0.0034 | -1.6240 ± 0.0457 | 0.9861 ± 0.0111 |
| rbf | best_single_kernel | 0.9452 ± 0.0214 | 0.8456 ± 0.0215 | 0.0460 ± 0.0034 | -1.6184 ± 0.0467 | 0.9856 ± 0.0114 |

Feature sup-norm error: 0.0000 ± 0.0000 (max 0.0000).
