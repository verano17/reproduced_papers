# Table I comparison (test centered alignment, learned mixture)

| Sweep | Kernel | Mean ± std (n) | Paper | z | Seeds ≥ paper |
|---|---|---|---|---|---|
| S5_table1_original_sharednoise | matern32 | 0.9879 ± 0.0092 (15) | 0.9920 | +0.45 | 0.33 |
| S5_table1_original_sharednoise | nonstat_ls | 0.9585 ± 0.0293 (15) | 0.9538 | -0.16 | 0.73 |
| S5_table1_original_sharednoise | rbf | 0.9717 ± 0.0192 (15) | 0.9791 | +0.39 | 0.40 |
| M1_table1_merlin | matern32 | 0.9142 ± 0.0807 (15) | 0.9920 | +0.96 | 0.00 |
| M1_table1_merlin | nonstat_ls | 0.9028 ± 0.1119 (15) | 0.9538 | +0.46 | 0.27 |
| M1_table1_merlin | rbf | 0.9481 ± 0.0257 (15) | 0.9791 | +1.21 | 0.00 |

## Learned minus uniform mixture (paired)

| Sweep | Kernel | Metric | Mean diff [95% CI] | Wilcoxon p | #>0 / n |
|---|---|---|---|---|---|
| S5_table1_original_sharednoise | matern32 | alignment_centered | +0.0235 [+0.0184, +0.0269] | 0.000122 | 14/15 |
| S5_table1_original_sharednoise | matern32 | rmse | -0.0291 [-0.0457, -0.0152] | 6.1e-05 | 0/15 |
| S5_table1_original_sharednoise | matern32 | avg_nlpd | -0.2187 [-0.3425, -0.1099] | 0.00061 | 2/15 |
| S5_table1_original_sharednoise | nonstat_ls | alignment_centered | -0.0193 [-0.0359, -0.0078] | 0.00336 | 3/15 |
| S5_table1_original_sharednoise | nonstat_ls | rmse | +0.0018 [+0.0006, +0.0031] | 0.0215 | 11/15 |
| S5_table1_original_sharednoise | nonstat_ls | avg_nlpd | +0.0597 [+0.0415, +0.0807] | 6.1e-05 | 15/15 |
| S5_table1_original_sharednoise | rbf | alignment_centered | -0.0073 [-0.0170, +0.0018] | 0.359 | 6/15 |
| S5_table1_original_sharednoise | rbf | rmse | +0.0006 [+0.0001, +0.0012] | 0.0479 | 10/15 |
| S5_table1_original_sharednoise | rbf | avg_nlpd | +0.0424 [+0.0240, +0.0648] | 0.000122 | 14/15 |
| M1_table1_merlin | matern32 | alignment_centered | -0.0387 [-0.0805, -0.0035] | 0.303 | 7/15 |
| M1_table1_merlin | matern32 | rmse | -0.0387 [-0.0661, -0.0167] | 0.00116 | 2/15 |
| M1_table1_merlin | matern32 | avg_nlpd | -0.2576 [-0.4232, -0.1145] | 0.00427 | 3/15 |
| M1_table1_merlin | nonstat_ls | alignment_centered | -0.0750 [-0.1371, -0.0320] | 6.1e-05 | 0/15 |
| M1_table1_merlin | nonstat_ls | rmse | +0.0054 [+0.0006, +0.0110] | 0.135 | 10/15 |
| M1_table1_merlin | nonstat_ls | avg_nlpd | +0.0938 [+0.0047, +0.1973] | 0.277 | 8/15 |
| M1_table1_merlin | rbf | alignment_centered | -0.0190 [-0.0325, -0.0078] | 0.00336 | 3/15 |
| M1_table1_merlin | rbf | rmse | +0.0015 [+0.0007, +0.0025] | 0.00671 | 11/15 |
| M1_table1_merlin | rbf | avg_nlpd | -0.0039 [-0.0255, +0.0157] | 0.978 | 8/15 |

## Between sweeps (learned mixture, paired)

| Comparison | Kernel | Metric | Mean diff [95% CI] | Wilcoxon p |
|---|---|---|---|---|
| M1_table1_merlin_minus_S5_table1_original_sharednoise | matern32 | alignment_centered | -0.0737 [-0.1177, -0.0360] | 0.000427 |
| M1_table1_merlin_minus_S5_table1_original_sharednoise | matern32 | rmse | +0.0075 [+0.0008, +0.0146] | 0.0637 |
| M1_table1_merlin_minus_S5_table1_original_sharednoise | matern32 | avg_nlpd | +0.0639 [-0.0004, +0.1332] | 0.151 |
| M1_table1_merlin_minus_S5_table1_original_sharednoise | nonstat_ls | alignment_centered | -0.0557 [-0.1197, -0.0077] | 0.0256 |
| M1_table1_merlin_minus_S5_table1_original_sharednoise | nonstat_ls | rmse | +0.0044 [-0.0005, +0.0103] | 0.804 |
| M1_table1_merlin_minus_S5_table1_original_sharednoise | nonstat_ls | avg_nlpd | +0.0834 [-0.0041, +0.1870] | 0.599 |
| M1_table1_merlin_minus_S5_table1_original_sharednoise | rbf | alignment_centered | -0.0235 [-0.0360, -0.0117] | 0.00201 |
| M1_table1_merlin_minus_S5_table1_original_sharednoise | rbf | rmse | +0.0004 [-0.0007, +0.0016] | 0.978 |
| M1_table1_merlin_minus_S5_table1_original_sharednoise | rbf | avg_nlpd | +0.0154 [-0.0207, +0.0531] | 0.599 |
