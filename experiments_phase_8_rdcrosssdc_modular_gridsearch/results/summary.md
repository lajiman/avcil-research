# Testing res 汇总

指标为每个增量阶段的 overall accuracy（0–1），step 从 0 开始。
预期 seeds：42, 43, 44。complete 仅表示 Testing res 步骤齐全。
参数取自命令清单，有 Namespace 时核对；仅展示变化参数及损失类型。
每步按已有有效结果计算均值，n 表示该步实际参与的 seed 数。
n≥2 时计算样本标准差（ddof=1）；n=1 时保留单次结果，标准差为 —；无结果为 —。

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g001_lc0p1_a0p5_tol0p0_s5p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.1`, `rd_class_weight_alpha=0.5`, `rd_cmr_scale=5`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | partial | 0.780000 | 0.675000 | 0.624667 | 0.613500 | 0.565200 | 0.518333 | 0.465429 | 0.438500 | 0.404667 | — |
| 43 | partial | 0.770000 | 0.683000 | 0.642000 | 0.626000 | 0.572800 | 0.524000 | 0.471429 | 0.436500 | 0.402667 | — |
| 44 | complete | 0.758000 | 0.655000 | 0.618667 | 0.611000 | 0.562000 | 0.517667 | 0.467143 | 0.434500 | 0.418889 | 0.381800 |
| mean ± std | partial | 0.769333 ± 0.011015 | 0.671000 ± 0.014422 | 0.628445 ± 0.012117 | 0.616833 ± 0.008036 | 0.566667 ± 0.005547 | 0.520000 ± 0.003480 | 0.468000 ± 0.003091 | 0.436500 ± 0.002000 | 0.408741 ± 0.008845 | 0.381800 ± — |
| n | 有效 seed 数 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 1 |

说明：seed 42: missing steps: 9; seed 43: missing steps: 9

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g002_lc0p1_a0p5_tol0p0_s1p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.1`, `rd_class_weight_alpha=0.5`, `rd_cmr_scale=1`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | complete | 0.780000 | 0.684000 | 0.627333 | 0.605000 | 0.575600 | 0.528333 | 0.478857 | 0.441000 | 0.410222 | 0.363400 |
| 43 | complete | 0.770000 | 0.669000 | 0.635333 | 0.623500 | 0.580000 | 0.536000 | 0.476000 | 0.431000 | 0.407556 | 0.369600 |
| 44 | partial | 0.758000 | 0.655000 | 0.622667 | 0.624000 | 0.578400 | 0.530000 | 0.474571 | 0.445250 | — | — |
| mean ± std | partial | 0.769333 ± 0.011015 | 0.669333 ± 0.014503 | 0.628444 ± 0.006406 | 0.617500 ± 0.010828 | 0.578000 ± 0.002227 | 0.531444 ± 0.004032 | 0.476476 ± 0.002182 | 0.439083 ± 0.007316 | 0.408889 ± 0.001885 | 0.366500 ± 0.004384 |
| n | 有效 seed 数 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 2 | 2 |

说明：seed 44: missing steps: 8, 9

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g003_lc0p1_a0p5_tol0p0_s2p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.1`, `rd_class_weight_alpha=0.5`, `rd_cmr_scale=2`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | complete | 0.780000 | 0.680000 | 0.622000 | 0.599500 | 0.555200 | 0.518333 | 0.466571 | 0.437000 | 0.404222 | 0.370400 |
| 43 | complete | 0.770000 | 0.678000 | 0.632667 | 0.620500 | 0.575600 | 0.522000 | 0.474857 | 0.429750 | 0.406000 | 0.368200 |
| 44 | complete | 0.758000 | 0.660000 | 0.626667 | 0.620000 | 0.568000 | 0.525333 | 0.471143 | 0.446750 | 0.422444 | 0.379800 |
| mean ± std | complete | 0.769333 ± 0.011015 | 0.672667 ± 0.011015 | 0.627111 ± 0.005347 | 0.613333 ± 0.011983 | 0.566267 ± 0.010310 | 0.521889 ± 0.003501 | 0.470857 ± 0.004150 | 0.437833 ± 0.008531 | 0.410889 ± 0.010047 | 0.372800 ± 0.006161 |
| n | 有效 seed 数 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g004_lc0p1_a0p0_tol0p0_s5p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.1`, `rd_class_weight_alpha=0`, `rd_cmr_scale=5`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | complete | 0.780000 | 0.689000 | 0.622667 | 0.610000 | 0.573600 | 0.523333 | 0.471714 | 0.430500 | 0.395778 | 0.357600 |
| 43 | complete | 0.770000 | 0.687000 | 0.629333 | 0.617500 | 0.577600 | 0.531333 | 0.477143 | 0.441750 | 0.408444 | 0.369200 |
| 44 | partial | 0.758000 | 0.658000 | 0.624000 | 0.622500 | 0.572800 | 0.531333 | 0.476857 | 0.442500 | — | — |
| mean ± std | partial | 0.769333 ± 0.011015 | 0.678000 ± 0.017349 | 0.625333 ± 0.003527 | 0.616667 ± 0.006292 | 0.574667 ± 0.002572 | 0.528666 ± 0.004619 | 0.475238 ± 0.003055 | 0.438250 ± 0.006722 | 0.402111 ± 0.008956 | 0.363400 ± 0.008202 |
| n | 有效 seed 数 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 2 | 2 |

说明：seed 44: missing steps: 8, 9

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g005_lc0p1_a0p0_tol0p0_s1p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.1`, `rd_class_weight_alpha=0`, `rd_cmr_scale=1`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | complete | 0.780000 | 0.684000 | 0.631333 | 0.619500 | 0.583200 | 0.542333 | 0.479714 | 0.451000 | 0.412444 | 0.380400 |
| 43 | complete | 0.770000 | 0.688000 | 0.638000 | 0.623500 | 0.578400 | 0.532667 | 0.471714 | 0.431500 | 0.409333 | 0.376000 |
| 44 | partial | 0.758000 | 0.682000 | 0.637333 | 0.629000 | — | — | — | — | — | — |
| mean ± std | partial | 0.769333 ± 0.011015 | 0.684667 ± 0.003055 | 0.635555 ± 0.003672 | 0.624000 ± 0.004770 | 0.580800 ± 0.003394 | 0.537500 ± 0.006835 | 0.475714 ± 0.005657 | 0.441250 ± 0.013789 | 0.410888 ± 0.002200 | 0.378200 ± 0.003111 |
| n | 有效 seed 数 | 3 | 3 | 3 | 3 | 2 | 2 | 2 | 2 | 2 | 2 |

说明：seed 44: missing steps: 4, 5, 6, 7, 8, 9

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g006_lc0p1_a0p0_tol0p0_s2p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.1`, `rd_class_weight_alpha=0`, `rd_cmr_scale=2`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | complete | 0.780000 | 0.688000 | 0.618667 | 0.609500 | 0.574400 | 0.530000 | 0.474000 | 0.444500 | 0.413778 | 0.376000 |
| 43 | complete | 0.770000 | 0.674000 | 0.640000 | 0.624500 | 0.572400 | 0.532000 | 0.474571 | 0.428250 | 0.398667 | 0.362000 |
| 44 | complete | 0.758000 | 0.661000 | 0.619333 | 0.605500 | 0.570400 | 0.522333 | 0.471429 | 0.437500 | 0.406222 | 0.375600 |
| mean ± std | complete | 0.769333 ± 0.011015 | 0.674333 ± 0.013503 | 0.626000 ± 0.012129 | 0.613167 ± 0.010017 | 0.572400 ± 0.002000 | 0.528111 ± 0.005103 | 0.473333 ± 0.001674 | 0.436750 ± 0.008151 | 0.406222 ± 0.007556 | 0.371200 ± 0.007970 |
| n | 有效 seed 数 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g007_lc0p1_a1p0_tol0p0_s5p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.1`, `rd_class_weight_alpha=1`, `rd_cmr_scale=5`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | complete | 0.780000 | 0.682000 | 0.618667 | 0.602000 | 0.554800 | 0.512000 | 0.463714 | 0.430000 | 0.394444 | 0.359200 |
| 43 | complete | 0.770000 | 0.677000 | 0.634667 | 0.623000 | 0.565200 | 0.531000 | 0.476286 | 0.433000 | 0.400667 | 0.362000 |
| 44 | complete | 0.758000 | 0.660000 | 0.625333 | 0.616500 | 0.568800 | 0.533000 | 0.475429 | 0.441250 | 0.408444 | 0.371000 |
| mean ± std | complete | 0.769333 ± 0.011015 | 0.673000 ± 0.011533 | 0.626222 ± 0.008037 | 0.613833 ± 0.010751 | 0.562933 ± 0.007270 | 0.525333 ± 0.011590 | 0.471810 ± 0.007024 | 0.434750 ± 0.005826 | 0.401185 ± 0.007014 | 0.364067 ± 0.006165 |
| n | 有效 seed 数 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g008_lc0p1_a1p0_tol0p0_s1p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.1`, `rd_class_weight_alpha=1`, `rd_cmr_scale=1`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | complete | 0.780000 | 0.683000 | 0.615333 | 0.603000 | 0.563200 | 0.521667 | 0.474857 | 0.436500 | 0.413333 | 0.375200 |
| 43 | complete | 0.770000 | 0.680000 | 0.637333 | 0.629000 | 0.582000 | 0.533000 | 0.479143 | 0.436500 | 0.412222 | 0.376800 |
| 44 | complete | 0.758000 | 0.660000 | 0.637333 | 0.619000 | 0.576400 | 0.531667 | 0.477143 | 0.447000 | 0.414444 | 0.378400 |
| mean ± std | complete | 0.769333 ± 0.011015 | 0.674333 ± 0.012503 | 0.630000 ± 0.012702 | 0.617000 ± 0.013115 | 0.573867 ± 0.009653 | 0.528778 ± 0.006194 | 0.477048 ± 0.002145 | 0.440000 ± 0.006062 | 0.413333 ± 0.001111 | 0.376800 ± 0.001600 |
| n | 有效 seed 数 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g009_lc0p1_a1p0_tol0p0_s2p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.1`, `rd_class_weight_alpha=1`, `rd_cmr_scale=2`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | complete | 0.780000 | 0.675000 | 0.629333 | 0.612000 | 0.568400 | 0.525333 | 0.472286 | 0.440000 | 0.408889 | 0.371800 |
| 43 | complete | 0.770000 | 0.682000 | 0.643333 | 0.616500 | 0.574800 | 0.527000 | 0.474000 | 0.437000 | 0.406889 | 0.369800 |
| 44 | complete | 0.758000 | 0.661000 | 0.626667 | 0.616500 | 0.583200 | 0.534333 | 0.476857 | 0.437250 | 0.409778 | 0.380200 |
| mean ± std | complete | 0.769333 ± 0.011015 | 0.672667 ± 0.010693 | 0.633111 ± 0.008952 | 0.615000 ± 0.002598 | 0.575467 ± 0.007422 | 0.528889 ± 0.004788 | 0.474381 ± 0.002309 | 0.438083 ± 0.001665 | 0.408519 ± 0.001480 | 0.373933 ± 0.005518 |
| n | 有效 seed 数 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g010_lc0p03_a0p5_tol0p0_s5p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.03`, `rd_class_weight_alpha=0.5`, `rd_cmr_scale=5`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | complete | 0.780000 | 0.679000 | 0.641333 | 0.615000 | 0.580000 | 0.523333 | 0.482857 | 0.443500 | 0.420222 | 0.374600 |
| 43 | complete | 0.770000 | 0.682000 | 0.640667 | 0.631500 | 0.581600 | 0.547000 | 0.494000 | 0.449000 | 0.428222 | 0.400000 |
| 44 | complete | 0.758000 | 0.670000 | 0.640667 | 0.631000 | 0.593200 | 0.548000 | 0.498857 | 0.465500 | 0.436889 | 0.401200 |
| mean ± std | complete | 0.769333 ± 0.011015 | 0.677000 ± 0.006245 | 0.640889 ± 0.000385 | 0.625833 ± 0.009385 | 0.584933 ± 0.007204 | 0.539444 ± 0.013962 | 0.491905 ± 0.008203 | 0.452667 ± 0.011449 | 0.428444 ± 0.008336 | 0.391933 ± 0.015023 |
| n | 有效 seed 数 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g011_lc0p03_a0p5_tol0p0_s1p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.03`, `rd_class_weight_alpha=0.5`, `rd_cmr_scale=1`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | complete | 0.780000 | 0.663000 | 0.631333 | 0.620000 | 0.572800 | 0.531667 | 0.487429 | 0.442750 | 0.421556 | 0.379200 |
| 43 | complete | 0.770000 | 0.672000 | 0.632667 | 0.617000 | 0.580800 | 0.545667 | 0.494286 | 0.455500 | 0.422889 | 0.387000 |
| 44 | missing | — | — | — | — | — | — | — | — | — | — |
| mean ± std | partial | 0.775000 ± 0.007071 | 0.667500 ± 0.006364 | 0.632000 ± 0.000943 | 0.618500 ± 0.002121 | 0.576800 ± 0.005657 | 0.538667 ± 0.009899 | 0.490858 ± 0.004849 | 0.449125 ± 0.009016 | 0.422223 ± 0.000943 | 0.383100 ± 0.005515 |
| n | 有效 seed 数 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |

说明：seed 44: log not found

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g012_lc0p03_a0p5_tol0p0_s2p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.03`, `rd_class_weight_alpha=0.5`, `rd_cmr_scale=2`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | missing | — | — | — | — | — | — | — | — | — | — |
| 43 | missing | — | — | — | — | — | — | — | — | — | — |
| 44 | missing | — | — | — | — | — | — | — | — | — | — |
| mean ± std | partial | — | — | — | — | — | — | — | — | — | — |
| n | 有效 seed 数 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

说明：seed 42: log not found; seed 43: log not found; seed 44: log not found

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g013_lc0p03_a0p0_tol0p0_s5p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.03`, `rd_class_weight_alpha=0`, `rd_cmr_scale=5`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | missing | — | — | — | — | — | — | — | — | — | — |
| 43 | missing | — | — | — | — | — | — | — | — | — | — |
| 44 | missing | — | — | — | — | — | — | — | — | — | — |
| mean ± std | partial | — | — | — | — | — | — | — | — | — | — |
| n | 有效 seed 数 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

说明：seed 42: log not found; seed 43: log not found; seed 44: log not found

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g014_lc0p03_a0p0_tol0p0_s1p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.03`, `rd_class_weight_alpha=0`, `rd_cmr_scale=1`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | missing | — | — | — | — | — | — | — | — | — | — |
| 43 | missing | — | — | — | — | — | — | — | — | — | — |
| 44 | missing | — | — | — | — | — | — | — | — | — | — |
| mean ± std | partial | — | — | — | — | — | — | — | — | — | — |
| n | 有效 seed 数 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

说明：seed 42: log not found; seed 43: log not found; seed 44: log not found

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g015_lc0p03_a0p0_tol0p0_s2p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.03`, `rd_class_weight_alpha=0`, `rd_cmr_scale=2`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | missing | — | — | — | — | — | — | — | — | — | — |
| 43 | missing | — | — | — | — | — | — | — | — | — | — |
| 44 | missing | — | — | — | — | — | — | — | — | — | — |
| mean ± std | partial | — | — | — | — | — | — | — | — | — | — |
| n | 有效 seed 数 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

说明：seed 42: log not found; seed 43: log not found; seed 44: log not found

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g016_lc0p03_a1p0_tol0p0_s5p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.03`, `rd_class_weight_alpha=1`, `rd_cmr_scale=5`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | missing | — | — | — | — | — | — | — | — | — | — |
| 43 | missing | — | — | — | — | — | — | — | — | — | — |
| 44 | missing | — | — | — | — | — | — | — | — | — | — |
| mean ± std | partial | — | — | — | — | — | — | — | — | — | — |
| n | 有效 seed 数 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

说明：seed 42: log not found; seed 43: log not found; seed 44: log not found

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g017_lc0p03_a1p0_tol0p0_s1p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.03`, `rd_class_weight_alpha=1`, `rd_cmr_scale=1`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | missing | — | — | — | — | — | — | — | — | — | — |
| 43 | missing | — | — | — | — | — | — | — | — | — | — |
| 44 | missing | — | — | — | — | — | — | — | — | — | — |
| mean ± std | partial | — | — | — | — | — | — | — | — | — | — |
| n | 有效 seed 数 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

说明：seed 42: log not found; seed 43: log not found; seed 44: log not found

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g018_lc0p03_a1p0_tol0p0_s2p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.03`, `rd_class_weight_alpha=1`, `rd_cmr_scale=2`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | missing | — | — | — | — | — | — | — | — | — | — |
| 43 | missing | — | — | — | — | — | — | — | — | — | — |
| 44 | missing | — | — | — | — | — | — | — | — | — | — |
| mean ± std | partial | — | — | — | — | — | — | — | — | — | — |
| n | 有效 seed 数 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

说明：seed 42: log not found; seed 43: log not found; seed 44: log not found

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g019_lc0p3_a0p5_tol0p0_s5p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.3`, `rd_class_weight_alpha=0.5`, `rd_cmr_scale=5`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | missing | — | — | — | — | — | — | — | — | — | — |
| 43 | missing | — | — | — | — | — | — | — | — | — | — |
| 44 | missing | — | — | — | — | — | — | — | — | — | — |
| mean ± std | partial | — | — | — | — | — | — | — | — | — | — |
| n | 有效 seed 数 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

说明：seed 42: log not found; seed 43: log not found; seed 44: log not found

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g020_lc0p3_a0p5_tol0p0_s1p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.3`, `rd_class_weight_alpha=0.5`, `rd_cmr_scale=1`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | missing | — | — | — | — | — | — | — | — | — | — |
| 43 | missing | — | — | — | — | — | — | — | — | — | — |
| 44 | missing | — | — | — | — | — | — | — | — | — | — |
| mean ± std | partial | — | — | — | — | — | — | — | — | — | — |
| n | 有效 seed 数 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

说明：seed 42: log not found; seed 43: log not found; seed 44: log not found

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g021_lc0p3_a0p5_tol0p0_s2p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.3`, `rd_class_weight_alpha=0.5`, `rd_cmr_scale=2`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | missing | — | — | — | — | — | — | — | — | — | — |
| 43 | missing | — | — | — | — | — | — | — | — | — | — |
| 44 | missing | — | — | — | — | — | — | — | — | — | — |
| mean ± std | partial | — | — | — | — | — | — | — | — | — | — |
| n | 有效 seed 数 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

说明：seed 42: log not found; seed 43: log not found; seed 44: log not found

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g022_lc0p3_a0p0_tol0p0_s5p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.3`, `rd_class_weight_alpha=0`, `rd_cmr_scale=5`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | missing | — | — | — | — | — | — | — | — | — | — |
| 43 | missing | — | — | — | — | — | — | — | — | — | — |
| 44 | missing | — | — | — | — | — | — | — | — | — | — |
| mean ± std | partial | — | — | — | — | — | — | — | — | — | — |
| n | 有效 seed 数 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

说明：seed 42: log not found; seed 43: log not found; seed 44: log not found

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g023_lc0p3_a0p0_tol0p0_s1p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.3`, `rd_class_weight_alpha=0`, `rd_cmr_scale=1`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | missing | — | — | — | — | — | — | — | — | — | — |
| 43 | missing | — | — | — | — | — | — | — | — | — | — |
| 44 | missing | — | — | — | — | — | — | — | — | — | — |
| mean ± std | partial | — | — | — | — | — | — | — | — | — | — |
| n | 有效 seed 数 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

说明：seed 42: log not found; seed 43: log not found; seed 44: log not found

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g024_lc0p3_a0p0_tol0p0_s2p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.3`, `rd_class_weight_alpha=0`, `rd_cmr_scale=2`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | missing | — | — | — | — | — | — | — | — | — | — |
| 43 | missing | — | — | — | — | — | — | — | — | — | — |
| 44 | missing | — | — | — | — | — | — | — | — | — | — |
| mean ± std | partial | — | — | — | — | — | — | — | — | — | — |
| n | 有效 seed 数 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

说明：seed 42: log not found; seed 43: log not found; seed 44: log not found

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g025_lc0p3_a1p0_tol0p0_s5p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.3`, `rd_class_weight_alpha=1`, `rd_cmr_scale=5`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | missing | — | — | — | — | — | — | — | — | — | — |
| 43 | missing | — | — | — | — | — | — | — | — | — | — |
| 44 | missing | — | — | — | — | — | — | — | — | — | — |
| mean ± std | partial | — | — | — | — | — | — | — | — | — | — |
| n | 有效 seed 数 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

说明：seed 42: log not found; seed 43: log not found; seed 44: log not found

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g026_lc0p3_a1p0_tol0p0_s1p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.3`, `rd_class_weight_alpha=1`, `rd_cmr_scale=1`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | missing | — | — | — | — | — | — | — | — | — | — |
| 43 | missing | — | — | — | — | — | — | — | — | — | — |
| 44 | missing | — | — | — | — | — | — | — | — | — | — |
| mean ± std | partial | — | — | — | — | — | — | — | — | — | — |
| n | 有效 seed 数 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

说明：seed 42: log not found; seed 43: log not found; seed 44: log not found

## VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_grid_exp_g027_lc0p3_a1p0_tol0p0_s2p0_h200

参数：`rd_cmr_penalty=exp`, `lam_cmr=0.3`, `rd_class_weight_alpha=1`, `rd_cmr_scale=2`

| seed | 状态 | step_0 | step_1 | step_2 | step_3 | step_4 | step_5 | step_6 | step_7 | step_8 | step_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | missing | — | — | — | — | — | — | — | — | — | — |
| 43 | missing | — | — | — | — | — | — | — | — | — | — |
| 44 | missing | — | — | — | — | — | — | — | — | — | — |
| mean ± std | partial | — | — | — | — | — | — | — | — | — | — |
| n | 有效 seed 数 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

说明：seed 42: log not found; seed 43: log not found; seed 44: log not found
