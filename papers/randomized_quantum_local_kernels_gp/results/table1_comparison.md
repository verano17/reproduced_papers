# Table I comparison (test centered alignment, learned mixture)

| Sweep | Kernel | Mean ± std (n) | Paper | z | Seeds ≥ paper |
|---|---|---|---|---|---|
| S1_table1_original | matern32 | 0.9758 ± 0.0137 (15) | 0.9920 | +1.18 | 0.00 |
| S1_table1_original | nonstat_ls | 0.9585 ± 0.0293 (15) | 0.9538 | -0.16 | 0.73 |
| S1_table1_original | rbf | 0.9728 ± 0.0202 (15) | 0.9791 | +0.31 | 0.47 |
| S2_table1_exact_features | matern32 | 0.9544 ± 0.0322 (15) | 0.9920 | +1.17 | 0.00 |
| S2_table1_exact_features | nonstat_ls | 0.9576 ± 0.0249 (15) | 0.9538 | -0.15 | 0.67 |
| S2_table1_exact_features | rbf | 0.9700 ± 0.0138 (15) | 0.9791 | +0.66 | 0.33 |
| S3_table1_pauli_shadow | matern32 | 0.9804 ± 0.0151 (15) | 0.9920 | +0.77 | 0.20 |
| S3_table1_pauli_shadow | nonstat_ls | 0.9556 ± 0.0190 (15) | 0.9538 | -0.09 | 0.53 |
| S3_table1_pauli_shadow | rbf | 0.9644 ± 0.0130 (15) | 0.9791 | +1.13 | 0.13 |
| S5_table1_original_sharednoise | matern32 | 0.9879 ± 0.0092 (15) | 0.9920 | +0.45 | 0.33 |
| S5_table1_original_sharednoise | nonstat_ls | 0.9585 ± 0.0293 (15) | 0.9538 | -0.16 | 0.73 |
| S5_table1_original_sharednoise | rbf | 0.9717 ± 0.0192 (15) | 0.9791 | +0.39 | 0.40 |
| S6_table1_lbfgs | matern32 | 0.9862 ± 0.0072 (15) | 0.9920 | +0.80 | 0.20 |
| S6_table1_lbfgs | nonstat_ls | 0.9772 ± 0.0032 (15) | 0.9538 | -7.30 | 1.00 |
| S6_table1_lbfgs | rbf | 0.9811 ± 0.0100 (15) | 0.9791 | -0.20 | 0.73 |

## Learned minus uniform mixture (paired)

| Sweep | Kernel | Metric | Mean diff [95% CI] | Wilcoxon p | #>0 / n |
|---|---|---|---|---|---|
| S1_table1_original | matern32 | alignment_centered | +0.0114 [+0.0046, +0.0174] | 0.0103 | 12/15 |
| S1_table1_original | matern32 | rmse | -0.0229 [-0.0389, -0.0094] | 0.000427 | 1/15 |
| S1_table1_original | matern32 | avg_nlpd | -0.1659 [-0.2773, -0.0709] | 0.00061 | 2/15 |
| S1_table1_original | nonstat_ls | alignment_centered | -0.0193 [-0.0359, -0.0078] | 0.00336 | 3/15 |
| S1_table1_original | nonstat_ls | rmse | +0.0018 [+0.0006, +0.0031] | 0.0215 | 11/15 |
| S1_table1_original | nonstat_ls | avg_nlpd | +0.0597 [+0.0415, +0.0807] | 6.1e-05 | 15/15 |
| S1_table1_original | rbf | alignment_centered | -0.0061 [-0.0165, +0.0034] | 0.524 | 7/15 |
| S1_table1_original | rbf | rmse | +0.0009 [+0.0004, +0.0015] | 0.00336 | 11/15 |
| S1_table1_original | rbf | avg_nlpd | +0.0480 [+0.0261, +0.0715] | 0.000854 | 12/15 |
| S2_table1_exact_features | matern32 | alignment_centered | -0.0100 [-0.0260, +0.0043] | 0.489 | 6/15 |
| S2_table1_exact_features | matern32 | rmse | -0.0309 [-0.0477, -0.0166] | 6.1e-05 | 0/15 |
| S2_table1_exact_features | matern32 | avg_nlpd | -0.2264 [-0.3585, -0.1069] | 0.00116 | 3/15 |
| S2_table1_exact_features | nonstat_ls | alignment_centered | -0.0202 [-0.0337, -0.0077] | 0.0215 | 5/15 |
| S2_table1_exact_features | nonstat_ls | rmse | -0.0003 [-0.0006, +0.0000] | 0.188 | 6/15 |
| S2_table1_exact_features | nonstat_ls | avg_nlpd | -0.0045 [-0.0097, +0.0000] | 0.0946 | 4/15 |
| S2_table1_exact_features | rbf | alignment_centered | -0.0090 [-0.0161, -0.0026] | 0.0554 | 5/15 |
| S2_table1_exact_features | rbf | rmse | -0.0001 [-0.0002, +0.0001] | 0.489 | 7/15 |
| S2_table1_exact_features | rbf | avg_nlpd | -0.0015 [-0.0036, +0.0004] | 0.252 | 6/15 |
| S3_table1_pauli_shadow | matern32 | alignment_centered | +0.0160 [+0.0080, +0.0224] | 0.00427 | 13/15 |
| S3_table1_pauli_shadow | matern32 | rmse | -0.0065 [-0.0218, +0.0084] | 0.639 | 7/15 |
| S3_table1_pauli_shadow | matern32 | avg_nlpd | -0.0087 [-0.1084, +0.0922] | 0.679 | 9/15 |
| S3_table1_pauli_shadow | nonstat_ls | alignment_centered | -0.0222 [-0.0313, -0.0126] | 0.00116 | 2/15 |
| S3_table1_pauli_shadow | nonstat_ls | rmse | +0.0027 [+0.0012, +0.0043] | 0.00836 | 10/15 |
| S3_table1_pauli_shadow | nonstat_ls | avg_nlpd | -0.0408 [-0.0656, -0.0139] | 0.0125 | 3/15 |
| S3_table1_pauli_shadow | rbf | alignment_centered | -0.0145 [-0.0209, -0.0087] | 0.000427 | 2/15 |
| S3_table1_pauli_shadow | rbf | rmse | +0.0019 [+0.0005, +0.0033] | 0.0554 | 9/15 |
| S3_table1_pauli_shadow | rbf | avg_nlpd | -0.0761 [-0.1112, -0.0434] | 0.000305 | 1/15 |
| S5_table1_original_sharednoise | matern32 | alignment_centered | +0.0235 [+0.0184, +0.0269] | 0.000122 | 14/15 |
| S5_table1_original_sharednoise | matern32 | rmse | -0.0291 [-0.0457, -0.0152] | 6.1e-05 | 0/15 |
| S5_table1_original_sharednoise | matern32 | avg_nlpd | -0.2187 [-0.3425, -0.1099] | 0.00061 | 2/15 |
| S5_table1_original_sharednoise | nonstat_ls | alignment_centered | -0.0193 [-0.0359, -0.0078] | 0.00336 | 3/15 |
| S5_table1_original_sharednoise | nonstat_ls | rmse | +0.0018 [+0.0006, +0.0031] | 0.0215 | 11/15 |
| S5_table1_original_sharednoise | nonstat_ls | avg_nlpd | +0.0597 [+0.0415, +0.0807] | 6.1e-05 | 15/15 |
| S5_table1_original_sharednoise | rbf | alignment_centered | -0.0073 [-0.0170, +0.0018] | 0.359 | 6/15 |
| S5_table1_original_sharednoise | rbf | rmse | +0.0006 [+0.0001, +0.0012] | 0.0479 | 10/15 |
| S5_table1_original_sharednoise | rbf | avg_nlpd | +0.0424 [+0.0240, +0.0648] | 0.000122 | 14/15 |
| S6_table1_lbfgs | matern32 | alignment_centered | +0.0217 [+0.0181, +0.0246] | 6.1e-05 | 15/15 |
| S6_table1_lbfgs | matern32 | rmse | -0.0297 [-0.0461, -0.0157] | 6.1e-05 | 0/15 |
| S6_table1_lbfgs | matern32 | avg_nlpd | -0.2485 [-0.3612, -0.1502] | 6.1e-05 | 0/15 |
| S6_table1_lbfgs | nonstat_ls | alignment_centered | -0.0006 [-0.0019, +0.0008] | 0.524 | 6/15 |
| S6_table1_lbfgs | nonstat_ls | rmse | +0.0030 [+0.0024, +0.0036] | 6.1e-05 | 15/15 |
| S6_table1_lbfgs | nonstat_ls | avg_nlpd | +0.1271 [+0.1016, +0.1566] | 6.1e-05 | 15/15 |
| S6_table1_lbfgs | rbf | alignment_centered | +0.0021 [-0.0031, +0.0066] | 0.252 | 11/15 |
| S6_table1_lbfgs | rbf | rmse | +0.0026 [+0.0020, +0.0032] | 6.1e-05 | 15/15 |
| S6_table1_lbfgs | rbf | avg_nlpd | +0.1125 [+0.0836, +0.1425] | 6.1e-05 | 15/15 |

## Between sweeps (learned mixture, paired)

| Comparison | Kernel | Metric | Mean diff [95% CI] | Wilcoxon p |
|---|---|---|---|---|
| S2_table1_exact_features_minus_S1_table1_original | matern32 | alignment_centered | -0.0214 [-0.0348, -0.0098] | 0.00116 |
| S2_table1_exact_features_minus_S1_table1_original | matern32 | rmse | -0.0093 [-0.0120, -0.0062] | 0.000305 |
| S2_table1_exact_features_minus_S1_table1_original | matern32 | avg_nlpd | -0.0722 [-0.1199, -0.0274] | 0.0103 |
| S2_table1_exact_features_minus_S1_table1_original | nonstat_ls | alignment_centered | -0.0009 [-0.0172, +0.0154] | 0.89 |
| S2_table1_exact_features_minus_S1_table1_original | nonstat_ls | rmse | -0.0031 [-0.0043, -0.0021] | 6.1e-05 |
| S2_table1_exact_features_minus_S1_table1_original | nonstat_ls | avg_nlpd | -0.0582 [-0.0760, -0.0431] | 6.1e-05 |
| S2_table1_exact_features_minus_S1_table1_original | rbf | alignment_centered | -0.0028 [-0.0111, +0.0061] | 0.252 |
| S2_table1_exact_features_minus_S1_table1_original | rbf | rmse | -0.0022 [-0.0030, -0.0015] | 0.000122 |
| S2_table1_exact_features_minus_S1_table1_original | rbf | avg_nlpd | -0.0331 [-0.0648, -0.0019] | 0.135 |
| S3_table1_pauli_shadow_minus_S1_table1_original | matern32 | alignment_centered | +0.0046 [-0.0071, +0.0155] | 0.359 |
| S3_table1_pauli_shadow_minus_S1_table1_original | matern32 | rmse | +0.0203 [+0.0127, +0.0294] | 6.1e-05 |
| S3_table1_pauli_shadow_minus_S1_table1_original | matern32 | avg_nlpd | +0.1673 [+0.1043, +0.2404] | 6.1e-05 |
| S3_table1_pauli_shadow_minus_S1_table1_original | nonstat_ls | alignment_centered | -0.0029 [-0.0150, +0.0139] | 0.121 |
| S3_table1_pauli_shadow_minus_S1_table1_original | nonstat_ls | rmse | +0.0050 [+0.0032, +0.0070] | 6.1e-05 |
| S3_table1_pauli_shadow_minus_S1_table1_original | nonstat_ls | avg_nlpd | +0.0869 [+0.0593, +0.1178] | 6.1e-05 |
| S3_table1_pauli_shadow_minus_S1_table1_original | rbf | alignment_centered | -0.0085 [-0.0174, +0.0012] | 0.121 |
| S3_table1_pauli_shadow_minus_S1_table1_original | rbf | rmse | +0.0041 [+0.0032, +0.0051] | 6.1e-05 |
| S3_table1_pauli_shadow_minus_S1_table1_original | rbf | avg_nlpd | +0.0753 [+0.0496, +0.1007] | 0.000305 |
| S5_table1_original_sharednoise_minus_S1_table1_original | matern32 | alignment_centered | +0.0121 [+0.0059, +0.0195] | 0.00123 |
| S5_table1_original_sharednoise_minus_S1_table1_original | matern32 | rmse | -0.0062 [-0.0089, -0.0038] | 0.000982 |
| S5_table1_original_sharednoise_minus_S1_table1_original | matern32 | avg_nlpd | -0.0528 [-0.1006, -0.0109] | 0.0219 |
| S5_table1_original_sharednoise_minus_S1_table1_original | nonstat_ls | alignment_centered | +0.0000 [+0.0000, +0.0000] | 1 |
| S5_table1_original_sharednoise_minus_S1_table1_original | nonstat_ls | rmse | +0.0000 [+0.0000, +0.0000] | 1 |
| S5_table1_original_sharednoise_minus_S1_table1_original | nonstat_ls | avg_nlpd | +0.0000 [+0.0000, +0.0000] | 1 |
| S5_table1_original_sharednoise_minus_S1_table1_original | rbf | alignment_centered | -0.0012 [-0.0065, +0.0034] | 0.345 |
| S5_table1_original_sharednoise_minus_S1_table1_original | rbf | rmse | -0.0002 [-0.0009, +0.0002] | 0.686 |
| S5_table1_original_sharednoise_minus_S1_table1_original | rbf | avg_nlpd | -0.0056 [-0.0279, +0.0131] | 0.686 |
| S6_table1_lbfgs_minus_S1_table1_original | matern32 | alignment_centered | +0.0104 [+0.0062, +0.0148] | 0.000122 |
| S6_table1_lbfgs_minus_S1_table1_original | matern32 | rmse | -0.0068 [-0.0093, -0.0043] | 6.1e-05 |
| S6_table1_lbfgs_minus_S1_table1_original | matern32 | avg_nlpd | -0.0827 [-0.1206, -0.0498] | 0.000122 |
| S6_table1_lbfgs_minus_S1_table1_original | nonstat_ls | alignment_centered | +0.0187 [+0.0074, +0.0351] | 0.00336 |
| S6_table1_lbfgs_minus_S1_table1_original | nonstat_ls | rmse | +0.0012 [-0.0004, +0.0027] | 0.107 |
| S6_table1_lbfgs_minus_S1_table1_original | nonstat_ls | avg_nlpd | +0.0674 [+0.0289, +0.1083] | 0.00427 |
| S6_table1_lbfgs_minus_S1_table1_original | rbf | alignment_centered | +0.0082 [-0.0015, +0.0193] | 0.389 |
| S6_table1_lbfgs_minus_S1_table1_original | rbf | rmse | +0.0017 [+0.0010, +0.0024] | 0.00061 |
| S6_table1_lbfgs_minus_S1_table1_original | rbf | avg_nlpd | +0.0645 [+0.0297, +0.1007] | 0.00262 |
