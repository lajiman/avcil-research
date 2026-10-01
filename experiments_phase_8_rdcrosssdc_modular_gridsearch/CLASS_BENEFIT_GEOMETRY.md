# 逐类收益与 checkpoint 几何分析

两个独立入口及一个并行启动器。不导入训练入口，不更新模型，不覆盖 save 中的任何文件。

## g2：一次启动两部分

激活原来训练/探针使用的 Python 环境，在仓库根目录运行（交互式 CPU 节点，无需 Slurm）：

```bash
git pull --rebase
EXP=experiments_phase_8_rdcrosssdc_modular_gridsearch
FEATURE_ROOT=/groups/ytian/lujing/lujing/datasets/VGGSound
OUT="$EXP/results/class_benefit_geometry_$(date +%Y%m%d_%H%M%S)"
python -u "$EXP/run_class_benefit_geometry.py" \
  --feature-root "$FEATURE_ROOT" \
  --meta-root data2/balance \
  --output "$OUT" \
  --metrics-seeds 42 43 44 \
  --geometry-seeds 42 --steps 1 5 9 \
  --reference-per-class 20 --query-per-class 0 \
  --device cpu --threads 4 --batch-size 4
```

FEATURE_ROOT 使用此前 g2 探针记录中的路径；若数据移动，只需修改这一行。需包含 visual_features.h5 和 audio_pretrained_feature/audio_pretrained_feature_dict.npy。checkpoint 与 metrics 应位于 EXP 下的 save_supp_ckpt_v1、save_commands_cmr_hinge_tolerance_focus_3seeds。

两个子进程同时启动，CSV 部分通常先完成。只由几何进程进行模型推理；并非同时启动多个占满 CPU 的探针。保留交互式会话直至结束。CPU 几何分析仍可能耗时较长，具体受存储吞吐和 CPU 配额影响；不承诺运行时长。threads 不要超过已分配的 CPU 数量。

查看进度：另一个终端执行 `tail -f "$OUT/geometry.log"`。如果新终端没有 OUT 变量，请填入启动器打印的实际目录。退出码和日志分别保存为 status.json、metrics.log、geometry.log。

中断后，使用**原来的 OUT 路径、相同参数**再次执行上述命令，末尾加 `--resume`。已完成的 checkpoint 特征缓存会复用；被中断的单个 checkpoint 提取会重做。更改 seed、样本数、代码或输入版本需使用新的输出目录。缓存是分析产生的 NPZ，不是新训练 checkpoint。

## 单独运行

```bash
python "$EXP/analyze_class_benefits.py" --seeds 42 43 44 --output "$OUT/metrics"
python -u "$EXP/probe_class_geometry.py" \
  --feature-root "$FEATURE_ROOT" --meta-root data2/balance \
  --seeds 42 --steps 1 5 9 --device cpu --threads 4 --batch-size 4 \
  --reference-per-class 20 --query-per-class 0 --output "$OUT/geometry"
```

第一部分只需要 Python 标准库；第二部分使用原实验环境的 PyTorch、NumPy、h5py 及项目模型依赖。CPU 运行不需要 CUDA。两个入口的 --help 不加载 PyTorch。

## 默认对照与解释边界

- avcil：经典 AVCIL，CrossSDC-I/C/CMR 系数均为 0。
- hinge_A：lambda=.03，alpha=.5，tol=.01，CrossSDC-I=.1。
- hinge_B：lambda=.1，alpha=1，tol=.01，CrossSDC-I=.1。
- A/B 对 AVCIL 是整体新增约束的比较，不能归因为 CMR 单项。A 对 B 同时改变 lambda、alpha。
- 用户提供的 easy2hard 顺序原样保存在 class_difficulty_easy2hard.json，通过 category_name 关联；不是训练 class_id。
- seed42/43/44 共用同一类别顺序。跨 seed 的 SD 是描述统计，不宣称单个类别差值的显著性。

## 第一部分输出：metrics/

- report.html：简要总览和最终差值最高/最低的类。
- class_deltas.csv：每个 seed、类别、step 的 F1/precision/recall 配对差值（百分点），以及 Trust/权重参考。
- initial_delta_f1_pp：该类首次出现时的方法减 AVCIL；relative_retention_delta_f1_pp：当前差值减首次差值；step_delta_change_pp：当前差值减前一步差值（新类为空）。不直接用 max-history forget_f1 代替保持效果。
- class_summary.csv、final_class_summary.csv：按类别汇总 seed，保留平均、SD、方向计数。
- group_by_seed.csv、group_summary.csv：难度四分位、年龄、新旧类、进入批次及难度×年龄。先按 seed 计算类别均值，再汇总 seed，避免把所有行当独立重复。
- class_mapping.csv、audit.json、manifest.json：名称映射与核对记录。

Trust 来自该运行的教师 replay reference，权重来自 epoch0（选定运行使用固定 Trust-only 权重）；它们不是独立的原始类别难度。

## 第二部分输出：geometry/

默认一个 seed、三个阶段、三种方法。自动读取 step0/4/8 教师，加上 step1/5/9 学生，共 18 个模型状态。使用统一 sample_seed，固定每类 20 个 TRAIN reference 与全部 TEST query（此数据每类 50 个），跨方法和阶段保持 ID 一致。输入几何使用 100 类；每个 checkpoint 仅使用当时已见类。所有 TEST query 与 TRAIN reference 严格不重叠，因此无需 leave-one-out。

- input_geometry.csv：冻结输入 audio / 均匀池化 visual 的类内离散度、原型 margin、近邻纯度等。
- input_modal_agreement.csv：两模态类别邻域相似关系的 Spearman 一致性，不直接比较未对齐输入坐标的音视频余弦。
- input_class_pairs.csv、new_class_proximity.csv：固定输入的类间相似性，以及每一步旧类与新类的最近邻关系。
- geometry.csv、query_geometry.csv：模型 audio、visual、fusion 特征的逐类/逐样本几何。原型使用相同参考 IDs，但在各自模型空间内重新计算。
- cross_modal_geometry.csv：同一 checkpoint 内双向跨模态原型 margin，候选为所有已见类。
- retention.csv、query_retention.csv：相对本运行前一步教师的旧类 margin 下降、违反比例及尾部；候选仅旧类，temperature=.1、tol=.01。这是标准化的 TRAIN reference / TEST query 探针，**不是原 memory loss 重建**，也不乘 lambda 或 Trust。
- performance.csv、confusion.csv、confusion_changes.csv：测试 PRF、非零混淆单元、相对 AVCIL 的混淆计数变化；无混淆差异时不生成最后一个文件。old_head_only_recall 是屏蔽新类 logits 的诊断，不是正式增量测试指标。
- geometry_deltas.csv：几何差值与历史逐类 F1 收益、年龄、难度的连接表。历史 F1 与当前重测 PRF 分开保存。
- cache/：用于断点恢复的输入特征、checkpoint 特征、logits；原 checkpoint 不变。
- sample_manifest.json、manifest.json、audit.json：样本 ID、checkpoint/源码哈希、环境、模型无参数/缓冲区变动检查。大特征文件仅校验路径/大小/mtime，未计算全文件哈希。

三个阶段/一个 seed 是机制定位的起点。每个阶段 reference 相同，TEST 样本也重复使用，不能视为独立重复。不同阶段的候选类别数量不同，log-odds margin 不直接跨阶段比较。类内紧凑或双模态相似度提高不自动代表可分性提高。

原型 margin 是目标类余弦减最相似异类余弦；log-odds margin 是目标 logit 减异类 logsumexp。参考与 query 特征均 L2 归一化。近邻纯度按固定顺序打破距离平局，默认 k=5。

query-per-class=0 时会核对重新推理的 TP/FP/FN 与历史完整测试结果；差异写入 audit.json 并在报告中提示，需先排查数值或实现/数据版本差异。若采用正数子采样，明确关闭此历史一致性检查，不把子样本指标冒充历史全量结果。

传回本地时，带回整个 OUT 最方便；若只做结果分析，可先传回 CSV/JSON/HTML/log，保留体积较大的 cache/ 在 g2 供恢复和追加分析。

## 验证

```bash
python -m unittest discover -s "$EXP/tests" -p test_class_analysis.py -v
```

数学和 CSV 配对测试依赖 NumPy。真实模型 forward 的测试另需 PyTorch；缺少依赖时明确跳过。代码不会通过训练入口的 --test_only 运行，避免清理历史 metrics。
