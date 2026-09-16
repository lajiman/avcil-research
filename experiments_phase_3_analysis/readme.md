# 两个 CL 实验的多-step 分析输出说明

## 建议阅读顺序

1. `core/01_u_shape_diagnosis.csv`
2. `core/02_taskwise_accuracy_delta.csv`
3. `core/03_geometry_trajectory_by_group.csv`
4. `core/04_confusion_direction_by_group.csv`
5. `core/05_geometry_performance_correlation.csv`

## 分析问题与输出文件

| 分析问题 | 主要输出 | 重点字段 |
|---|---|---|
| U 形是否只是新 easy task 拉高平均值？ | `01_u_shape_diagnosis.csv` | `old_class_change_effect`, `new_task_composition_effect`, `old_acc`, `new_task_acc` |
| z2 的提升来自哪些 task？ | `02_taskwise_accuracy_delta.csv` | 行=`eval_step`，列=`task_k`，值=`Model B - Model A accuracy` |
| early hard classes 是否被保护？ | `03_geometry_trajectory_by_group.csv` | `aggregation_level=arrival_group`, `group_id=early`, `layer=z2_fusion` |
| z2 是否使类更紧凑？ | 同上 | `geometry_metric=intra_dispersion`，越低越好 |
| z2 是否提高边界清晰度？ | 同上 | `normalized_margin`、`knn_purity`，通常越高越好 |
| centroid 是否随增量训练漂移？ | 同上 | `centroid_drift_from_previous_analyzed_step`, `centroid_drift_from_first_reference_step`，越低越稳定 |
| early classes 是否被预测成 later classes？ | `04_confusion_direction_by_group.csv` | `aggregation_level=arrival_group`, `group_id=early`, `error_to_later_task_rate` |
| 几何改善是否与准确率改善一致？ | `05_geometry_performance_correlation.csv` | `spearman_with_delta_recall`, `spearman_with_delta_f1` |
| 哪些具体类别贡献最大？ | `detail/per_class_performance.csv` | `delta_b_minus_a_recall`, `delta_b_minus_a_f1` |
| 某个类别最近的 centroid 是谁？ | `detail/centroid_neighbors_by_step.csv` | `step`, `model`, `layer`, `neighbor_rank`, `neighbor_task_relation` |
| 单类完整几何轨迹 | `detail/class_geometry_by_step.csv` | 每个 step、class、model、layer 的全部指标 |

## 最核心的判断逻辑

- `new_task_composition_effect > 0` 且 `old_class_change_effect < 0`：U 形主要是后来的 easy classes 拉高平均值，旧类仍在遗忘。
- `old_class_change_effect` 在后半程变为正：旧类出现真实恢复或正向 backward transfer。
- Model B 的 early-group `intra_dispersion` 更低、`normalized_margin` 和 `knn_purity` 更高：支持 z2 保护 early hard-class geometry。
- Model B 的 early-group `error_to_later_task_rate` 更低：支持 z2 减少 early hard classes 被后来的 easy classes 吸引。
- geometry improvement 与 `delta recall` 的 Spearman 相关为正：支持性能提升与表示几何改善具有系统性联系。

## 推荐运行方式

默认分析两个文件夹中所有共同 step：

```bash
python -u compare_two_cl_runs_all_steps_corrected.py \
  --dataset VGGSound_hard2easy_balance \
  --modality audio-visual \
  --feature_root ../../../datasets/VGGSound \
  --meta_root ../data_hard2easy/balance \
  --num_classes 100 \
  --class_num_per_step 10 \
  --model_a_ckpt_dir xxx/save/VGGSound_h2e_wo_seed42 \
  --model_b_ckpt_dir xxx/save/VGGSound_h2e_z2_alpha2_seed42 \
  --model_a_name w_o \
  --model_b_name z2_alpha2 \
  --out_root analysis/h2e_wo_vs_z2_all_steps \
  --infer_batch_size 64 \
  --num_workers 8 \
  --topk_centroid 5 \
  --knn_k 10 \
  --cache_feature_steps 2,5,9
```

`--cache_feature_steps 2,5,9` 会保存这些 step 的 z1/z2 特征，供后续 t-SNE 使用。