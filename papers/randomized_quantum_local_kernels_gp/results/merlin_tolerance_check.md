# Phase 4 tolerance check (generated from sweep summary.json files)

Rule (pre-declared in LOG.md Phase 4 Adaptation Plan): per kernel, |mean alignment difference| ≤ 0.02 and photonic RMSE within ±20% of qubit. M1/M2 use the Phase 4 circuit (layer-0 encoding inert); M3/M4 add the initial balanced brickwork (amendment 2026-11-19T16:00:08Z).

| Comparison | Kernel | Photonic alignment | Qubit alignment | Δ | RMSE ratio | Alignment OK | RMSE OK |
|---|---|---|---|---|---|---|---|
| M1_table1_merlin vs S5_table1_original_sharednoise | rbf | 0.9481 | 0.9717 | -0.0235 | 1.007 | no | yes |
| M1_table1_merlin vs S5_table1_original_sharednoise | matern32 | 0.9142 | 0.9879 | -0.0737 | 1.076 | no | yes |
| M1_table1_merlin vs S5_table1_original_sharednoise | nonstat_ls | 0.9028 | 0.9585 | -0.0557 | 1.089 | no | yes |
| M2_table1_merlin_lbfgs vs S6_table1_lbfgs | rbf | 0.9747 | 0.9811 | -0.0064 | 0.957 | yes | yes |
| M2_table1_merlin_lbfgs vs S6_table1_lbfgs | matern32 | 0.9597 | 0.9862 | -0.0265 | 1.010 | no | yes |
| M2_table1_merlin_lbfgs vs S6_table1_lbfgs | nonstat_ls | 0.9623 | 0.9772 | -0.0150 | 0.959 | yes | yes |
| M3_table1_merlin_mixed vs S5_table1_original_sharednoise (corrected encoding) | rbf | 0.9378 | 0.9717 | -0.0339 | 0.989 | no | yes |
| M3_table1_merlin_mixed vs S5_table1_original_sharednoise (corrected encoding) | matern32 | 0.9496 | 0.9879 | -0.0383 | 0.882 | no | yes |
| M3_table1_merlin_mixed vs S5_table1_original_sharednoise (corrected encoding) | nonstat_ls | 0.9125 | 0.9585 | -0.0460 | 0.984 | no | yes |
| M4_table1_merlin_mixed_lbfgs vs S6_table1_lbfgs (corrected encoding) | rbf | 0.9813 | 0.9811 | +0.0002 | 0.950 | yes | yes |
| M4_table1_merlin_mixed_lbfgs vs S6_table1_lbfgs (corrected encoding) | matern32 | 0.9642 | 0.9862 | -0.0219 | 0.861 | no | yes |
| M4_table1_merlin_mixed_lbfgs vs S6_table1_lbfgs (corrected encoding) | nonstat_ls | 0.9653 | 0.9772 | -0.0119 | 0.954 | yes | yes |
