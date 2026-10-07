# Phase 9：历史 prototype bank 对照

这次新增两组实验，各使用 seeds 42/43/44，共 6 个实验。两组采用相同的历史原型方法，只改变 gate 更新规则。旧版 fresh-prototype 实验、uniform 对照、loss、融合模型和测试接口继续保留；旧命令文件不修改，新的功能默认关闭。

| 实验 | 原型 | 有效类别的 gate 更新 |
|---|---|---|
| 已有 `periodic_sample_aware` | 当前模型从参考集重新构建 | `eta=0.5*n/(n+10)` |
| 新增 `prototype_bank_smooth` | 漂移补偿后的历史统计与当前均值融合 | 同旧版 `eta=0.5*n/(n+10)` |
| 新增 `prototype_bank_direct` | 与 smooth 完全相同 | `eta=1`，直接替换 |

第一行与第二行比较历史 bank 的作用，第二行与第三行比较 bank 引入后是否仍需 gate 平滑。新 smooth 组不是旧实验的重复。prototype 层面的历史融合和 gate 平滑作用于不同对象，不能预先保证其中一层足以代替另一层，也不能保证两层都保留更好。

## 保持不变的部分

首任务默认全程等权、只用 CE。后续任务保留原 AVCIL 的 CE、按旧任务计算的 logit KD、原始实例/类别对比损失和时空注意力蒸馏，没有新增 loss。旧 gate 从上一任务的 best checkpoint 继承，新类初始化为 0.5；默认在第 40/80/120/160 轮训练、验证及保存 best 后更新，下一轮生效。任务开始不额外更新 gate，第 200 轮后不更新。

参考集仍是当前新类训练样本和选定的旧类 memory，不包含验证、测试或未来类。分类仍使用每个候选类对应的融合权重，不需要真实标签。教师沿用自己的已保存 gate；推理只需要模型和 gate，不需要重新构造原型。best 可能早于一次 gate 更新，应结合 checkpoint 的 epoch 和 gate 版本解释结果。

## 1. 保存什么、什么时候保存

每个类别在其学习任务结束、进入旧类 memory 时建立一次固定的 birth bank。使用该任务的 **最佳验证 checkpoint**，只扫描刚结束任务的训练类，不重新读取更老类别的完整训练集。这里的 birth 表示首次完成该类学习的模型，不是随机初始化时刻。

设模态分支为 `m`，对有限且非零的音视频特征对取单位特征：

```text
u_i,0^m = normalize(h_birth^m(x_i))
S_c,0^m = sum_{i in class c birth training set} u_i,0^m
N_c,0   = number of valid paired samples in that set
```

bank 保留每类 `S_c,0^m` 和 `N_c,0`，以及当前保留的 memory ID 在 birth 模型下的单位特征。类别和是未归一化的向量和；仅保存单位 prototype 会丢掉类内集中程度信息。memory 缩减后只保留剩余 ID 的 birth 特征；类别和与原始样本数保持不变。不会保存全部历史样本的逐样本特征或原始输入，也不需要长期保留完整 birth 模型。

这一做法额外保存了 memory 外历史训练样本的类别汇总，因此属于新增的持续学习状态。它不是扩大当前训练回放集：历史原始样本不再参加梯度训练或当前模型的重提取。新增内存主要为每类两份分支向量和当前 memory 的两份 birth 特征，随已见类数和 memory 数量增长。

## 2. 将历史均值补偿到当前空间

以下公式分别用于音频、视觉分支。每次 gate 更新时，用当前学生在 `eval/no_grad` 下重新提取固定参考集的特征。旧类当前有效 memory 集合记为 `A_c`，`n_c` 是去重后、当前音视频特征对均有效的样本数。只有这些 ID 全部有匹配且有效的 birth 特征对，才启用该类的历史融合；任一个 birth anchor 缺失或无效，整类回退为 fresh，保留当前有效样本，不通过删掉它来继续使用历史。

```text
current_mean_c^m = (1/n_c) * sum_{i in A_c} u_i,t^m
anchor_mean_c^m  = (1/n_c) * sum_{i in A_c} u_i,0^m
delta_c^m        = current_mean_c^m - anchor_mean_c^m
prior_c^m        = S_c,0^m / N_c,0 + delta_c^m
```

`delta` 使用同一批 ID 在两个模型状态下的差，避免把 memory 的类别内抽样差异直接当作模型漂移。每次均相对于固定 birth bank 计算，不以已经补偿过的 prior 再补偿，因此没有逐次叠加漂移或递归 EMA。

这仍是假设“保留 memory 的平均漂移能代表该类”的近似。非线性的类内变化、memory 偏差或样本过少都可能使它失准。它不是对齐整个特征空间的精确变换；视觉分支也仍包含音频引导注意力。

## 3. 历史原型与当前原型融合

```text
k_c     = min(prototype_prior_strength, max(N_c,0 - n_c, 0))
beta_c  = n_c / (n_c + k_c)
mu_c^m  = beta_c * current_mean_c^m + (1-beta_c) * prior_c^m
p_c^m   = normalize(mu_c^m)
```

默认 `prototype_prior_strength=10`。`k_c` 是历史均值的有效支持强度上限，不是历史样本的真实独立计数或贝叶斯后验；当前配置没有根据漂移质量自适应学习该强度。`N_c,0-n_c` 上限使历史 bank 在所有 birth 样本仍被保留时不额外加权；`prototype_prior_strength=0` 会退化为当前原型。

当 memory 变少，当前估计的 `beta` 下降，历史汇总所占比例上升。重复每 40 epoch 扫描相同 memory 不增加 `N_c,0` 或 `n_c`，因此不会把重复观察误算成新的独立样本。新类尚无 birth bank，仍按旧版从当前训练集重建原型。

注意，如果 birth 类别均值仅来自同一批当前 memory，漂移补偿后的 prior 与当前均值相等，没有额外信息。保存过更多样本的类别汇总才是历史 bank 相对 fresh 方法潜在的价值来源。

## 4. LOO 同时排除当前与历史贡献

计算第 `i` 个样本的可靠性时，目标类不能直接使用上述全量 prototype。对于有历史 bank 的旧类，查询 ID 同时从当前 memory、birth 汇总和漂移 anchor 中排除：

```text
current_mean_c,-i^m = (sum_{j in A_c} u_j,t^m - u_i,t^m) / (n_c - 1)
anchor_mean_c,-i^m  = (sum_{j in A_c} u_j,0^m - u_i,0^m) / (n_c - 1)
birth_mean_c,-i^m   = (S_c,0^m - u_i,0^m) / (N_c,0 - 1)
prior_c,-i^m        = birth_mean_c,-i^m + current_mean_c,-i^m - anchor_mean_c,-i^m
beta_c,-i           = (n_c - 1) / (n_c - 1 + k_c)
p_c,-i^m           = normalize(beta_c,-i * current_mean_c,-i^m
                             + (1-beta_c,-i) * prior_c,-i^m)
```

排除查询后，历史与当前的样本数差仍是 `(N_c,0-1)-(n_c-1)=N_c,0-n_c`，所以 `k_c` 不变。负类使用其全量原型；新类使用原来的 `normalize(S_c,t-u_i,t)`。两路共用有效候选类别，最少样本数、非有限/零向量和无效 LOO 的处理继续保留；当前样本的类和或 LOO 和无有效方向时，历史不能把它强行恢复成有效统计。统计无效的类不会更新 gate。

LOO 排除的是原型统计中的查询样本贡献，不能消除该样本对网络训练、checkpoint 选择及其他样本表征产生的间接影响，也不把训练参考集变成独立验证集。

## 5. 可靠性与两个 gate 对照

```text
q_i^m(c) = softmax_k(dot(u_i,t^m, p_k,-i^m) / fusion_temperature)[c]
R_c^m    = mean_{i in A_c} q_i^m(c)
g_hat_c  = R_c^A / (R_c^A + R_c^V)

smooth: eta_c = fusion_eta_max * n_c / (n_c + fusion_n_ref)
direct: eta_c = 1
g_new_c = g_old_c + eta_c * (g_hat_c - g_old_c)
```

direct 只对统计有效的类别直接替换，无效类继续保留历史 gate；“直接”不表示任意缺失统计都可以覆盖。它也适用于当前新类的有效统计。两组均保留原型层面的按样本量融合，区别只在 gate 是否再平滑。

## 参数与兼容性

| 参数 | 默认值 | 含义 |
|---|---|---|
| `fusion_prototype_mode` | `fresh` | `fresh` 使用旧流程；`history_bank` 开启历史原型 |
| `prototype_prior_strength` | 10 | 非负历史支持强度上限；0 不给历史均值权重 |
| `fusion_update_rule` | `sample_aware` | 保留 `fixed`、`sample_aware`，新增 `direct` |
| `fusion_eta_max` | 0.5 | smooth 使用；direct 有效类的实际幅度固定为 1 |
| `fusion_n_ref` | 10 | smooth 的样本量参考值，direct 不使用 |

启用 history_bank 后，从第二个任务起，训练检查点的 `prototype_bank` 字段保存类别汇总、birth 参考特征及其来源，使下一任务使用与前一最佳模型衔接的状态。首任务 checkpoint 无 bank；进入第二个任务时加载首任务 best，建立第一批 birth 统计。旧版缺少 bank 字段的 checkpoint 仍可用于原来的模型加载和测试；不能仅凭旧 checkpoint 凭空恢复已经丢弃的历史统计。该功能不增加中途恢复优化器的入口。CL history 仍仅作观测，不进入 loss 或 gate 公式。

每次 gate 更新额外输出 `save/metrics/RUN_NAME/prototype_bank/step_S_after_epoch_E.csv` 和同名 `.pt`。CSV 记录 birth/anchor 样本量、是否使用历史、有效历史强度、原型融合系数、两个模态的漂移范数与漂移离散程度，以及回退原因；`.pt` 还包含该次原型向量。`history_used` 仅表示历史强度大于零，不保证该类最终 LOO 有效；gate CSV 的 `valid` 和 `eta` 才决定是否更新。漂移离散程度用于诊断，不会自动改写历史强度。原有 gate、loss、CL history 和测试指标继续写入原来的目录。

## 两个提交任务

新文件是：

* `grid_commands/commands_prototype_bank_smooth.txt`
* `grid_commands/commands_prototype_bank_direct.txt`

每个文件恰好 3 条命令，seed 为 42/43/44，200 epochs；新名称分别为 `phase9_prototype_bank_smooth_h200_seed*` 和 `phase9_prototype_bank_direct_h200_seed*`，独立保存日志与结果。两个文件都显式使用 `history_bank`、历史强度 10、原始 AVCIL loss、CL history 每 40 epoch 记录。温度、batch size、memory size 和其余训练参数与旧动态融合实验一致。

从仓库根目录提交：

```bash
sbatch --job-name=phase9_bank_smooth experiments_phase_9/run.slurm grid_commands/commands_prototype_bank_smooth.txt
sbatch --job-name=phase9_bank_direct experiments_phase_9/run.slurm grid_commands/commands_prototype_bank_direct.txt
```

继续使用现有 Juno 脚本，每个任务申请 1 张 H200、12 CPU、180G 主机内存，在同一 GPU 上并行运行 3 个 seed。两个任务共 6 个实验；实际资源峰值和速度需要服务器验证。

如需修改 feature 路径，只生成两个新文件：

```bash
python experiments_phase_9/generate_commands.py --settings prototype_bank_smooth prototype_bank_direct --feature_root /absolute/path/to/VGGSound
```

该命令不覆盖旧三个命令文件。生成器不带 `--settings` 时仍只生成旧三组，保持过去脚本的行为。旧版方法的详细训练与数据边界说明见 [README_CLASS_FUSION_MODULAR.md](README_CLASS_FUSION_MODULAR.md)。
