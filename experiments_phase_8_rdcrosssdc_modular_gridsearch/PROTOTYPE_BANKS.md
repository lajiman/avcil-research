# Phase 8：A / B / C prototype bank

本文件说明 prototype 的构建与迁移；不新增 loss，不改变融合、分类器或测试推理。
训练入口仍是 `train_incremental_rd_crosssdc_modular.py`，原有模块划分保留。
新逻辑集中在 `rd_crosssdc/prototype_history.py`。

## 策略与时间点

`rd_prototype_policy` 默认 `memory`，对应原有 A。`pre_shrink` 对应 B，
`historical` 对应 C。B/C 仅用于已开启 CMR 的模式。

当前任务为 t，冻结教师为上一任务实际 best checkpoint θ*_(t−1)。定义：

- M_t：任务 t 缩减后的 replay memory，也是 Trust 的查询集合。
- H_(t−1) = D_new,(t−1) ∪ M_(t−1)：上一任务结束时仍可合法访问的训练数据。
- J_t = H_(t−1) \ M_t：这次缩减新丢弃的样本。
- E_t：在 H_(t−1) 之前已经丢弃的样本，与 H_(t−1) 不重叠。

u_i^m 表示在冻结教师下得到的单位分支特征，m ∈ {audio, visual}。
保存的是单位样本特征的 **sum** 和历史 **未再单位化的 mean**，保留集中程度。

| 策略 | 用于最终 normalize 的向量 | 构建时间 |
| --- | --- | --- |
| A | Σ_(i∈M_c,t) u_i^m | 当前任务开始，沿用原代码 |
| B | S_H,c^m = Σ_(i∈H_c,t−1) u_i^m | 上一任务结束，重载 best 后、下次缩减前 |
| C | S_H,c^m + κ_E,c,f^m μ_E,c,f^m | B 的精确统计加已迁移到同一教师空间的历史统计 |

C 的 f 是查询所属的固定 fold；一个查询对所有候选类别均使用同一 f。
B/C 的教师和 bank 在整个任务内固定。任务 0 只有原有 CE 训练，但结束后仍需导出 bank。
第一个增量任务尚无 E，C 与 B 相同。C 的历史 mass 上限设为 0，也退化为 B。

## C：任务边界的历史递推

当前任务开始已缓存 M_t 在旧教师下的特征。任务结束重载 θ*_t，
扫描 H_t = D_new,t ∪ M_t。由两个模型对同一 M_t 的输出获得 paired anchors。
不重新读取任何已丢弃样本的原始数据或预训练特征。

从上一份精确统计减去保留样本即可得到 J_t：

```text
S_J = S_H,previous − Σ_(i∈M_t) u_i,old
N_J = N_H,previous − |M_t|
mass_pre = κ_previous + N_J
mean_pre = (κ_previous μ_previous + S_J) / mass_pre
```

实际 retired unique count 单独增加 N_J；它不等于有效 mass κ。
重复扫描/重复 replay 不增加实际 count。历史质量已经衰减后，不会再用全部历史
unique count 把失去可信度的质量恢复回来。上限只作用于历史项，不截断 B 的精确 H。

对每个类别、模态和 fold f，迁移 anchors 为 M_c,t 中 **不属于 f** 的样本。
对 n 个 anchors 计算：

```text
δ_i = u_i,new − u_i,old
shift = mean_i δ_i
error = mean_i || n/(n−1) × (δ_i − shift) ||²
confidence = decay × exp(−error / error_scale²)
μ_new = mean_pre + shift
κ_new = min(mass_pre, mass_cap) × confidence
```

error 是均值位移模型的 leave-one-anchor-out 残差。anchors 少于
`rd_history_min_anchors` 时，本次该类别、模态、fold 的历史 mass 置零，使用 B。
样本充足不保证迁移正确：类均值位移假设仍要求保留样本的漂移能代表已丢弃样本。
confidence 是实验启发式，不是统计置信区间。

下一任务用的是本次保存的精确 H_t 加迁移后的 retired 历史。因此首次把 J_t
放入历史时，它已经不在下一份 H_t 中；同一个样本不会同时作为精确支持和历史质量。
J 的统计加入所有 fold。J 已退休，后续不会再次成为 replay query；不应按其 fold
额外删除它。fold 排除的是拟合迁移与可信度时使用的 **仍保留 anchors**。

## LOO、CMR 与 Trust

fold = SHA256(seed, video_id) mod F，跨任务固定。每个 fold 单独保存历史状态；
不仅当前迁移，更早的每一次历史迁移也都排除该 fold 的 anchors。
这保证原型统计层面的查询排除，不消除模型本身曾在训练中见过 query 的事实。

对 audio query i、目标类别 y_i，以 visual bank 为例：

```text
v_c,f = S_H,c^V + κ_c,f^V μ_c,f^V
p_c,f = normalize(v_c,f)
p_y,−i,f = normalize(v_y,f − u_i,teacher^V)
b_i = <q_i^A, p_y,−i,f>/τ − logsumexp_(c≠y) (<q_i^A, p_c,f>/τ)
```

B 令历史项为 0；A 使用原有缩减后 M 的统计。教师 reference 和学生 current
共用同一 bank，并都减去 **配对教师特征**，不减学生特征。CMR 的原有
hinge/direct 等 penalty、tolerance、λ 和类别加权公式不变。

Trust 查询始终只来自当前 M：

```text
R_c = (1 / |M_c|) × Σ_(i∈M_c) sigmoid(b_ref,i)
T_c = clip((R_c − 1/C_old)/(1 − 1/C_old), 0, 1)
T_shrunk,c = (|M_c| T_c + beta × mean_c T_c) / (|M_c| + beta)
```

最后一式仅在原有 beta > 0 时执行。B/C 的 `prototype_count` 是 |H_c|，
新增 `trust_query_count` 是 |M_c|，收缩使用后者。bank 改变会影响 CMR margins、
Trust，继而影响原有 weighted CrossSDC-C 和 CMR 类别权重；这些是同一机制的传导。

## 参数、文件与边界

以下 C 默认值是待验证的实验配置，不代表已经验证的优选参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| rd_history_folds | 5 | 固定跨任务 folds 数，至少 2 |
| rd_history_mass_cap | 50 | 历史有效 mass 上限，允许 0 |
| rd_history_decay | 0.9 | 每次迁移的基础折减，范围 [0, 1] |
| rd_history_error_scale | 0.25 | 迁移残差的尺度，正数 |
| rd_history_min_anchors | 2 | 每类别、fold 的最小折外 anchors 数，至少 2 |

每任务在对应 checkpoint 目录保存 `step_<t>_prototype_bank.pt` 和 JSON 摘要。
PT 包含精确 sums/counts、support/retired ID 与标签、模型和 checkpoint SHA256、
类别映射、固定 folds、各 fold 的历史 mean/mass、迁移诊断。
不持久化 H 的逐样本特征。运行时只缓存当前 M 的教师特征。
JSON 包含支持数、历史实际样本数、有效 mass、anchor 数、残差和可信度；
adaptive 模式的静态 Trust CSV 另外记录查询数。

B/C 需要额外统计存储，不能把这部分预算归入“没有额外存储”的 raw memory 对照。
历史 feature 统计约 O(F × C × D)，ID 账本约 O(累计已见训练样本数)，不保存旧样本内容。

源数据限当前新类 train 和保留 replay；验证集仅按原有机制选 best，测试集不参与 bank。
加载时核对 checkpoint、模型权重、任务、类别顺序、配置和 ID，避免 best/last 或不同
实验串用。扫描保留 Python/NumPy/Torch/CUDA RNG 与模块 train/eval 状态。
B/C 训练要求新的 experiment 输出目录，不会在已有实验内静默覆盖历史链。
`test_only` 沿用原有 classifier 推理，不需要 bank。
原有两个离线 probe 脚本仍是 A 的 memory 重建协议；现会明确拒绝 B/C checkpoint，
防止把 B/C 静默按 A 解释。B/C 的训练期 loss/gradient diagnostics 正常支持。

严格 LOO 要求每类至少 2 个精确 support；每个旧类必须有 replay query。
由于每个任务（包括最终任务）都会导出快照，CLI 要求 memory_size 至少为
2 × (num_classes − class_num_per_step)，避免训练到后期才因精确支持不足失败。
缺失/重复 ID、非有限或非单位分支特征、零范数原型等会明确报错。
极小 memory 下，历史迁移可能因 anchors 不足而经常退化为 B；不应把这种情况
误解为代码使用了全部历史。A 的原有 singleton fallback 保持不变。

## 代码对应关系

- `prototype_history.py`：B/C 数据扫描、快照核验、C 历史迁移、fold-aware LOO 和 Trust。
- `rd_method.py::compute_margin_terms`：A 保持原 margin 路径，B/C 委托 bank，后续 loss 不变。
- 训练入口：给 replay 附加原始 index；任务开始加载 bank；任务结束重载 best 导出 bank。
- `diagnostics.py`：B/C CSV 分开记录 prototype support count 和 Trust query count。
- `tests/test_prototype_history.py`：数学、历史递推、数据来源及 RNG 边界测试。
- `tests/test_prototype_training.py`：CPU 多任务训练、导出/加载和 CLI 集成测试。
