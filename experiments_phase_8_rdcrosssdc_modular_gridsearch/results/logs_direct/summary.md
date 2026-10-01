# Testing res 汇总

指标为每个增量阶段的 overall accuracy（0–1），step 从 0 开始。
预期 seeds：42, 43, 44。complete 仅表示 Testing res 步骤齐全。
参数取自命令清单，有 Namespace 时核对；仅展示变化参数及损失类型。
每步按已有有效结果计算均值，n 表示该步实际参与的 seed 数。
n≥2 时计算样本标准差（ddof=1）；n=1 时保留单次结果，标准差为 —；无结果为 —。

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_direct_g001_lc0p1_a0p5_tol0p0_s1p0_h200

参数：`rd_cmr_penalty=direct`, `lam_cmr=0.1`, `rd_class_weight_alpha=0.5`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | error | — | — | — | — | — | — | — | — | — | — |
| 43 | error | — | — | — | — | — | — | — | — | — | — |
| 44 | complete | 0.772000 | 0.687000 | 0.622000 | 0.602500 | 0.564400 | 0.523333 | 0.468286 | 0.432750 | 0.400889 | 0.374400 |
| mean ± std | partial | 0.772000 ± — | 0.687000 ± — | 0.622000 ± — | 0.602500 ± — | 0.564400 ± — | 0.523333 ± — | 0.468286 ± — | 0.432750 ± — | 0.400889 ± — | 0.374400 ± — |
| n | 有效 seed 数 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

说明：seed 42: line 1: Namespace disagrees with commands: num_workers; seed 43: line 1: Namespace disagrees with commands: num_workers

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_direct_g002_lc0p1_a0p0_tol0p0_s1p0_h200

参数：`rd_cmr_penalty=direct`, `lam_cmr=0.1`, `rd_class_weight_alpha=0`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | complete | 0.782000 | 0.676000 | 0.634667 | 0.615000 | 0.578000 | 0.530000 | 0.483143 | 0.451000 | 0.416222 | 0.381400 |
| 43 | complete | 0.778000 | 0.673000 | 0.631333 | 0.621000 | 0.560000 | 0.525667 | 0.467143 | 0.431500 | 0.394000 | 0.367400 |
| 44 | complete | 0.772000 | 0.681000 | 0.611333 | 0.608500 | 0.560800 | 0.513667 | 0.473714 | 0.442250 | 0.410444 | 0.365200 |
| mean ± std | complete | 0.777333 ± 0.005033 | 0.676667 ± 0.004041 | 0.625778 ± 0.012620 | 0.614833 ± 0.006252 | 0.566267 ± 0.010169 | 0.523111 ± 0.008461 | 0.474667 ± 0.008042 | 0.441583 ± 0.009767 | 0.406889 ± 0.011530 | 0.371333 ± 0.008787 |
| n | 有效 seed 数 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_direct_g003_lc0p1_a1p0_tol0p0_s1p0_h200

参数：`rd_cmr_penalty=direct`, `lam_cmr=0.1`, `rd_class_weight_alpha=1`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | error | — | — | — | — | — | — | — | — | — | — |
| 43 | error | — | — | — | — | — | — | — | — | — | — |
| 44 | complete | 0.772000 | 0.690000 | 0.628000 | 0.607000 | 0.572000 | 0.523667 | 0.475714 | 0.441250 | 0.408889 | 0.373000 |
| mean ± std | partial | 0.772000 ± — | 0.690000 ± — | 0.628000 ± — | 0.607000 ± — | 0.572000 ± — | 0.523667 ± — | 0.475714 ± — | 0.441250 ± — | 0.408889 ± — | 0.373000 ± — |
| n | 有效 seed 数 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

说明：seed 42: line 1: Namespace disagrees with commands: num_workers; seed 43: line 1: Namespace disagrees with commands: num_workers

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_direct_g004_lc0p03_a0p5_tol0p0_s1p0_h200

参数：`rd_cmr_penalty=direct`, `lam_cmr=0.03`, `rd_class_weight_alpha=0.5`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | complete | 0.782000 | 0.670000 | 0.635333 | 0.633500 | 0.587200 | 0.546000 | 0.492286 | 0.457500 | 0.431556 | 0.377800 |
| 43 | complete | 0.778000 | 0.673000 | 0.633333 | 0.619000 | 0.575200 | 0.541000 | 0.499714 | 0.456000 | 0.428889 | 0.394600 |
| 44 | complete | 0.772000 | 0.677000 | 0.623333 | 0.626000 | 0.593200 | 0.543000 | 0.491143 | 0.457250 | 0.434222 | 0.392000 |
| mean ± std | complete | 0.777333 ± 0.005033 | 0.673333 ± 0.003512 | 0.630666 ± 0.006429 | 0.626167 ± 0.007251 | 0.585200 ± 0.009165 | 0.543333 ± 0.002517 | 0.494381 ± 0.004654 | 0.456917 ± 0.000804 | 0.431556 ± 0.002667 | 0.388133 ± 0.009043 |
| n | 有效 seed 数 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_direct_g005_lc0p03_a0p0_tol0p0_s1p0_h200

参数：`rd_cmr_penalty=direct`, `lam_cmr=0.03`, `rd_class_weight_alpha=0`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | error | — | — | — | — | — | — | — | — | — | — |
| 43 | error | — | — | — | — | — | — | — | — | — | — |
| 44 | complete | 0.772000 | 0.688000 | 0.624000 | 0.621000 | 0.582800 | 0.547667 | 0.496571 | 0.450750 | 0.431111 | 0.395600 |
| mean ± std | partial | 0.772000 ± — | 0.688000 ± — | 0.624000 ± — | 0.621000 ± — | 0.582800 ± — | 0.547667 ± — | 0.496571 ± — | 0.450750 ± — | 0.431111 ± — | 0.395600 ± — |
| n | 有效 seed 数 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

说明：seed 42: line 1: Namespace disagrees with commands: num_workers; seed 43: line 1: Namespace disagrees with commands: num_workers

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_direct_g006_lc0p03_a1p0_tol0p0_s1p0_h200

参数：`rd_cmr_penalty=direct`, `lam_cmr=0.03`, `rd_class_weight_alpha=1`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | complete | 0.782000 | 0.670000 | 0.630000 | 0.630500 | 0.591600 | 0.549333 | 0.485714 | 0.458250 | 0.433556 | 0.384600 |
| 43 | complete | 0.778000 | 0.679000 | 0.634667 | 0.612500 | 0.570400 | 0.531000 | 0.476000 | 0.437250 | 0.415778 | 0.378800 |
| 44 | complete | 0.772000 | 0.683000 | 0.622000 | 0.623000 | 0.592000 | 0.541667 | 0.494286 | 0.458750 | 0.432000 | 0.394000 |
| mean ± std | complete | 0.777333 ± 0.005033 | 0.677333 ± 0.006658 | 0.628889 ± 0.006406 | 0.622000 ± 0.009042 | 0.584667 ± 0.012357 | 0.540667 ± 0.009207 | 0.485333 ± 0.009149 | 0.451417 ± 0.012271 | 0.427111 ± 0.009846 | 0.385800 ± 0.007671 |
| n | 有效 seed 数 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_direct_g007_lc0p3_a0p5_tol0p0_s1p0_h200

参数：`rd_cmr_penalty=direct`, `lam_cmr=0.3`, `rd_class_weight_alpha=0.5`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | error | — | — | — | — | — | — | — | — | — | — |
| 43 | error | — | — | — | — | — | — | — | — | — | — |
| 44 | complete | 0.772000 | 0.695000 | 0.610667 | 0.606000 | 0.561200 | 0.509667 | 0.476286 | 0.431000 | 0.410444 | 0.385200 |
| mean ± std | partial | 0.772000 ± — | 0.695000 ± — | 0.610667 ± — | 0.606000 ± — | 0.561200 ± — | 0.509667 ± — | 0.476286 ± — | 0.431000 ± — | 0.410444 ± — | 0.385200 ± — |
| n | 有效 seed 数 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

说明：seed 42: line 1: Namespace disagrees with commands: num_workers; seed 43: line 1: Namespace disagrees with commands: num_workers

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_direct_g008_lc0p3_a0p0_tol0p0_s1p0_h200

参数：`rd_cmr_penalty=direct`, `lam_cmr=0.3`, `rd_class_weight_alpha=0`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | complete | 0.782000 | 0.672000 | 0.618667 | 0.609000 | 0.556400 | 0.520667 | 0.463714 | 0.435500 | 0.398222 | 0.372800 |
| 43 | complete | 0.778000 | 0.668000 | 0.618667 | 0.612500 | 0.548800 | 0.520000 | 0.468571 | 0.425250 | 0.398667 | 0.374800 |
| 44 | complete | 0.772000 | 0.692000 | 0.608000 | 0.608000 | 0.561200 | 0.507000 | 0.464571 | 0.433500 | 0.400444 | 0.374600 |
| mean ± std | complete | 0.777333 ± 0.005033 | 0.677333 ± 0.012858 | 0.615111 ± 0.006159 | 0.609833 ± 0.002363 | 0.555467 ± 0.006252 | 0.515889 ± 0.007705 | 0.465619 ± 0.002592 | 0.431417 ± 0.005433 | 0.399111 ± 0.001176 | 0.374067 ± 0.001102 |
| n | 有效 seed 数 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_direct_g009_lc0p3_a1p0_tol0p0_s1p0_h200

参数：`rd_cmr_penalty=direct`, `lam_cmr=0.3`, `rd_class_weight_alpha=1`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | complete | 0.782000 | 0.662000 | 0.616667 | 0.605500 | 0.572800 | 0.517333 | 0.465714 | 0.422500 | 0.402000 | 0.374200 |
| 43 | complete | 0.778000 | 0.668000 | 0.610000 | 0.608000 | 0.554800 | 0.517667 | 0.457143 | 0.424500 | 0.389556 | 0.373600 |
| 44 | complete | 0.772000 | 0.688000 | 0.602667 | 0.602500 | 0.560400 | 0.511667 | 0.459714 | 0.410750 | 0.386667 | 0.365200 |
| mean ± std | complete | 0.777333 ± 0.005033 | 0.672667 ± 0.013614 | 0.609778 ± 0.007003 | 0.605333 ± 0.002754 | 0.562667 ± 0.009212 | 0.515556 ± 0.003372 | 0.460857 ± 0.004398 | 0.419250 ± 0.007429 | 0.392741 ± 0.008148 | 0.371000 ± 0.005032 |
| n | 有效 seed 数 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
