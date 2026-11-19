# Table I comparison (test centered alignment, learned mixture)

| Sweep | Kernel | Mean ± std (n) | Paper | z | Seeds ≥ paper |
|---|---|---|---|---|---|
| S6_table1_lbfgs | matern32 | 0.9862 ± 0.0072 (15) | 0.9920 | +0.80 | 0.20 |
| S6_table1_lbfgs | nonstat_ls | 0.9772 ± 0.0032 (15) | 0.9538 | -7.30 | 1.00 |
| S6_table1_lbfgs | rbf | 0.9811 ± 0.0100 (15) | 0.9791 | -0.20 | 0.73 |
| M2_table1_merlin_lbfgs | matern32 | 0.9597 ± 0.0153 (15) | 0.9920 | +2.12 | 0.00 |
| M2_table1_merlin_lbfgs | nonstat_ls | 0.9623 ± 0.0087 (15) | 0.9538 | -0.97 | 0.87 |
| M2_table1_merlin_lbfgs | rbf | 0.9747 ± 0.0139 (15) | 0.9791 | +0.32 | 0.47 |

## Learned minus uniform mixture (paired)

| Sweep | Kernel | Metric | Mean diff [95% CI] | Wilcoxon p | #>0 / n |
|---|---|---|---|---|---|
| S6_table1_lbfgs | matern32 | alignment_centered | +0.0217 [+0.0181, +0.0246] | 6.1e-05 | 15/15 |
| S6_table1_lbfgs | matern32 | rmse | -0.0297 [-0.0461, -0.0157] | 6.1e-05 | 0/15 |
| S6_table1_lbfgs | matern32 | avg_nlpd | -0.2485 [-0.3612, -0.1502] | 6.1e-05 | 0/15 |
| S6_table1_lbfgs | nonstat_ls | alignment_centered | -0.0006 [-0.0019, +0.0008] | 0.524 | 6/15 |
| S6_table1_lbfgs | nonstat_ls | rmse | +0.0030 [+0.0024, +0.0036] | 6.1e-05 | 15/15 |
| S6_table1_lbfgs | nonstat_ls | avg_nlpd | +0.1271 [+0.1016, +0.1566] | 6.1e-05 | 15/15 |
| S6_table1_lbfgs | rbf | alignment_centered | +0.0021 [-0.0031, +0.0066] | 0.252 | 11/15 |
| S6_table1_lbfgs | rbf | rmse | +0.0026 [+0.0020, +0.0032] | 6.1e-05 | 15/15 |
| S6_table1_lbfgs | rbf | avg_nlpd | +0.1125 [+0.0836, +0.1425] | 6.1e-05 | 15/15 |
| M2_table1_merlin_lbfgs | matern32 | alignment_centered | +0.0068 [-0.0011, +0.0139] | 0.107 | 10/15 |
| M2_table1_merlin_lbfgs | matern32 | rmse | -0.0458 [-0.0720, -0.0243] | 6.1e-05 | 0/15 |
| M2_table1_merlin_lbfgs | matern32 | avg_nlpd | -0.3476 [-0.5054, -0.2101] | 6.1e-05 | 0/15 |
| M2_table1_merlin_lbfgs | nonstat_ls | alignment_centered | -0.0155 [-0.0198, -0.0111] | 0.000122 | 1/15 |
| M2_table1_merlin_lbfgs | nonstat_ls | rmse | +0.0001 [-0.0006, +0.0007] | 0.934 | 7/15 |
| M2_table1_merlin_lbfgs | nonstat_ls | avg_nlpd | -0.0271 [-0.0456, -0.0072] | 0.0181 | 4/15 |
| M2_table1_merlin_lbfgs | rbf | alignment_centered | +0.0075 [+0.0009, +0.0139] | 0.073 | 9/15 |
| M2_table1_merlin_lbfgs | rbf | rmse | +0.0010 [+0.0005, +0.0015] | 0.00537 | 11/15 |
| M2_table1_merlin_lbfgs | rbf | avg_nlpd | -0.0418 [-0.0656, -0.0171] | 0.00836 | 4/15 |

## Between sweeps (learned mixture, paired)

| Comparison | Kernel | Metric | Mean diff [95% CI] | Wilcoxon p |
|---|---|---|---|---|
| M2_table1_merlin_lbfgs_minus_S6_table1_lbfgs | matern32 | alignment_centered | -0.0265 [-0.0333, -0.0205] | 6.1e-05 |
| M2_table1_merlin_lbfgs_minus_S6_table1_lbfgs | matern32 | rmse | +0.0010 [-0.0053, +0.0071] | 0.89 |
| M2_table1_merlin_lbfgs_minus_S6_table1_lbfgs | matern32 | avg_nlpd | +0.0037 [-0.0593, +0.0622] | 0.934 |
| M2_table1_merlin_lbfgs_minus_S6_table1_lbfgs | nonstat_ls | alignment_centered | -0.0150 [-0.0200, -0.0100] | 0.000183 |
| M2_table1_merlin_lbfgs_minus_S6_table1_lbfgs | nonstat_ls | rmse | -0.0021 [-0.0035, -0.0008] | 0.0125 |
| M2_table1_merlin_lbfgs_minus_S6_table1_lbfgs | nonstat_ls | avg_nlpd | -0.1049 [-0.1495, -0.0637] | 0.000305 |
| M2_table1_merlin_lbfgs_minus_S6_table1_lbfgs | rbf | alignment_centered | -0.0064 [-0.0120, -0.0003] | 0.0554 |
| M2_table1_merlin_lbfgs_minus_S6_table1_lbfgs | rbf | rmse | -0.0021 [-0.0029, -0.0012] | 0.00061 |
| M2_table1_merlin_lbfgs_minus_S6_table1_lbfgs | rbf | avg_nlpd | -0.0927 [-0.1230, -0.0611] | 0.000305 |
