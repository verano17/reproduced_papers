# Table I comparison (test centered alignment, learned mixture)

| Sweep | Kernel | Mean ± std (n) | Paper | z | Seeds ≥ paper |
|---|---|---|---|---|---|
| S4_table1_papertext_data | matern32 | 0.9495 ± 0.0174 (15) | 0.9920 | +2.44 | 0.00 |
| S4_table1_papertext_data | nonstat_ls | 0.9484 ± 0.0386 (15) | 0.9538 | +0.14 | 0.53 |
| S4_table1_papertext_data | rbf | 0.9642 ± 0.0154 (15) | 0.9791 | +0.97 | 0.07 |

## Learned minus uniform mixture (paired)

| Sweep | Kernel | Metric | Mean diff [95% CI] | Wilcoxon p | #>0 / n |
|---|---|---|---|---|---|
| S4_table1_papertext_data | matern32 | alignment_centered | +0.0626 [+0.0525, +0.0708] | 6.1e-05 | 15/15 |
| S4_table1_papertext_data | matern32 | rmse | -0.0648 [-0.1019, -0.0333] | 6.1e-05 | 0/15 |
| S4_table1_papertext_data | matern32 | avg_nlpd | -0.2624 [-0.4041, -0.1407] | 6.1e-05 | 0/15 |
| S4_table1_papertext_data | nonstat_ls | alignment_centered | -0.0482 [-0.0674, -0.0303] | 0.000122 | 1/15 |
| S4_table1_papertext_data | nonstat_ls | rmse | +0.0003 [-0.0003, +0.0009] | 0.208 | 10/15 |
| S4_table1_papertext_data | nonstat_ls | avg_nlpd | +0.0368 [+0.0215, +0.0530] | 0.00116 | 13/15 |
| S4_table1_papertext_data | rbf | alignment_centered | +0.0653 [+0.0567, +0.0731] | 6.1e-05 | 15/15 |
| S4_table1_papertext_data | rbf | rmse | +0.0022 [+0.0001, +0.0041] | 0.0554 | 11/15 |
| S4_table1_papertext_data | rbf | avg_nlpd | +0.0529 [+0.0080, +0.0965] | 0.0479 | 10/15 |

## Between sweeps (learned mixture, paired)

| Comparison | Kernel | Metric | Mean diff [95% CI] | Wilcoxon p |
|---|---|---|---|---|
