# Weighted CrossSDC-C + CMR：两组参数、四个对照、三个 seed

共 24 次完整训练：2 组参数 × 4 个对照 × seeds 42、43、44。这里只准备命令，不启动训练。
当前主方案为 v2：将所有对照的 lam_cmr 从 0.03 提高到 0.1，增强 CMR 的作用。
v1 的 0.03 命令和 manifest 原样保留，供需要时作配对比较；不是要求同时运行两个版本。
v2 使用独立实验名、日志和汇总目录；四个对照的编号和 seed 顺序不变。

## 参数选择与已有证据

| 当前组名 | CMR penalty | lam_cmr | alpha | tolerance |
| --- | --- | --- | --- | --- |
| H01 | hinge | 0.1 | 0.5 | 0.01 |
| H05 | hinge | 0.1 | 0.5 | 0.05 |

以下是选取 tolerance 时参考的历史结果，lam_cmr=0.03，不是当前 v2 的结果：

| 历史配置 | tolerance | 最终准确率（%） | AIA（%） |
| --- | --- | --- | --- |
| hinge_A / lam_cmr=0.03 / alpha=0.5 | 0.01 | 40.31 ± 0.36 | 56.21 ± 0.25 |
| hinge_A / lam_cmr=0.03 / alpha=0.5 | 0.05 | 39.83 ± 0.96 | 56.15 ± 0.08 |

均为三个 seed 的均值 ± 样本标准差；AIA 为 step 0–9 准确率的平均。
来源：[汇总](../results/cmr_selected_analysis/config_summary.csv) 中的 hinge_A_t0.01、hinge_A_t0.05，
以及[各 seed 结果](../results/logs_hinge_selected/summary.csv)。

选择 H01 是因为它在已有 hinge/alpha=0.5 结果中兼顾准确率与最终结果稳定性；
H05 沿用本次讨论给出的参数，历史 AIA 跨 seed 波动较小。
两个新组都来自历史 A 系列；H05 不代表历史 alpha=1 的 B 系列。
历史 H01/H05 来源运行使用旧权重配置且关闭 CrossSDC-C；
下面的新实验取消 shrinkage/offset/裁剪、启用 weighted CrossSDC-C，并将 lam_cmr 提高到 0.1。
历史 B 系列也运行过 lam_cmr=0.1，但同时使用 alpha=1 和旧权重处理，不能当作本轮配置的直接验证。
当前 v2 的效果尚未验证，不能把历史数值当作新实验结果。

在相同模型参数、同一批数据下，0.1 相对 0.03 将 CMR 的加权标量和梯度贡献放大约 3.33 倍。
这不保证训练全过程中的贡献比值不变：模型轨迹和 hinge 激活样本会随训练改变。
对已满足 margin 保持条件的样本，hinge 梯度仍为零；增大 lam_cmr 不改变其激活阈值。
本轮不同时修改 tolerance、alpha 或其他损失系数，因此与 v1 的同组同 seed 对比只改变 CMR 的外部系数。

## 四个对照

所有对照都保留 CE、Logit KD、空间与时间注意力蒸馏。
CE/KD 在训练器增量阶段始终存在；注意力开关始终为 --attn_score_distil，空间/时间系数各为 0.5。

| 对照 | CMR | weighted CrossSDC-C | CrossSDC-I | 原始 L_i | 原始 L_c |
| --- | --- | --- | --- | --- | --- |
| c1_full | 0.1 | 0.3 | 0.1 | 0.1 | 1.0 |
| c2_no_cross_i | 0.1 | 0.3 | 0 | 0.1 | 1.0 |
| c3_cmr_wcrossc | 0.1 | 0.3 | 0 | 0 | 0 |
| c4_cross_i_only | 0.1 | 0.3 | 0.1 | 0 | 0 |

- c1 → c2：仅移除 CrossSDC-I 的优化贡献。
- c2 → c3：共同移除原始 L_i 和 L_c；该对照不能单独归因于其中某一项。
- c4 → c3：在不使用原始 L_i/L_c 时，单独移除 CrossSDC-I。
- c1 → c4：在使用 CrossSDC-I 时，共同移除原始 L_i/L_c。
- c3/c4 同时删除 --instance_contrastive、--class_contrastive，并将 lam_I/lam_C 设为 0。
- --cross_sdc 在四个对照中都保留，因为 weighted CrossSDC-C 仍需启用。
- CrossSDC-I 系数为零时，现有训练器仍会计算它，但不贡献梯度。
- step=0 始终只有 CE；以上对照在 step=1–9 生效。

这构成两个因素的完整 2×2 对照：CrossSDC-I 开/关，以及原始 L_i/L_c 共同开/关。
c4 的 only 指这两个可变因素中仅开启 CrossSDC-I；CMR、weighted CrossSDC-C、CE、KD 和注意力蒸馏仍然保留。
CrossSDC-I 在旧类回放上对齐当前模型与旧模型的跨模态实例；原始 L_i/L_c 在新样本和回放样本上对齐当前模型的两个模态。
因此 c4 在代码与目标函数上均独立成立，但会减少新类样本上的显式对比监督，其优劣需要实验检验。

结果按相同 seed 配对比较，最终准确率和 AIA 分别报告：
- 原始 L_i/L_c 开启时，CrossSDC-I 的收益：score(c1) - score(c2)。
- 原始 L_i/L_c 关闭时，CrossSDC-I 的收益：score(c4) - score(c3)。
- 交互差值：score(c1) - score(c2) - score(c4) + score(c3)。
正值表示在该指标和固定系数下，两者同时使用时呈现正交互；负值提示负交互，但不能仅凭它断言机制上的冗余或冲突。
先对每个 seed 求差，再汇总均值与样本标准差。三个 seed 用于初步判断，不将小差异直接解释为稳定机制结论。

观察 CMR 时，结合 loss_components.csv 的原始/加权值以及 epoch_summary.csv 的 weighted_cmr、激活比例和 margin deficit。
仅比较 loss 数值大小不能推断梯度主导程度；当前未开启梯度探针。
四个对照都包含 CMR，因而它们主要检验其他对比项在较强 CMR 下的作用。
若要归因于 CMR 本身，应比较相同组、对照和 seed 的 v2(0.1) 与 v1(0.03)；这检验强度效应。
检验 CMR 的有无则需要保持 weighted CrossSDC-C 和权重生成一致的零 CMR 基线。
当前 adaptive_crosssdc_cmr 校验要求 lam_cmr>0，零系数基线需要单独调整代码路径；本轮未添加该实验，也未修改训练器。

weighted CrossSDC-C 的 0.3 采用当前代码默认值，不声称是已核实的论文系数。
--rd_mode adaptive_crosssdc_cmr 使 lam_cross_sdc_c 对应加权版本。
当前实现对每个方向分别使用 w_c=0.5+0.5*T_c/mean(T)，并在一个增量任务内固定权重。
T 仍保留机会水平校正与 [0,1] 截断；这里没有更改训练代码或 reliability 定义。

## 共用设置

- num_classes=100、每步 10 类、每步 200 epochs、memory_size=500。
- train/exemplar batch=128、infer batch=32、num_workers=0。
- Adam lr=1e-3、weight_decay=1e-4、lr_decay=False。
- 原始实例/类别对比与 CrossSDC temperature=0.05；margin temperature=0.1。
- trust_offset=0、trust_shrinkage_beta=0、trust_gamma=1、rd_disable_weight_clipping。
- alpha=0.5；不使用 rd_weight_min/max 限制。
- Need eta=0；当前代码的 Need 统计不更新损失权重。
- hinge 的 rd_cmr_scale=1 不影响损失。
- --log_loss_components 记录各项原始值、加权值及总损失重建，不开启梯度探针。
- dataset 中 seed42 是固定数据/类别顺序标识；仅训练 --seed 改为 42、43、44。
- 数据路径沿用用户命令，均相对于实验目录解析：
  --feature_root ../../../datasets/VGGSound
  --meta_root ../data2/balance

## 运行

在已激活训练环境的 Linux/GPU 节点，从仓库根目录进入实验目录，再创建日志目录：

```bash
cd experiments_phase_8_rdcrosssdc_modular_gridsearch
mkdir -p logs_weighted_cmr_ablation_v2
```

每份命令清单有 12 行，按 c1/c2/c3/c4 排列，每个对照按 seed42/43/44 排列。
两份文件可直接作为 Bash 脚本顺序执行；先运行其中一组：

```bash
bash -e grid_commands/commands_weighted_cmr_ablation_v2_H01.txt
```

另一组：

```bash
bash -e grid_commands/commands_weighted_cmr_ablation_v2_H05.txt
```

仅执行某一组中的一个对照（三个 seed），例如 H01 的 c2：

```bash
sed -n '4,6p' grid_commands/commands_weighted_cmr_ablation_v2_H01.txt | bash -e
```

行 1–3 是 c1，行 4–6 是 c2，行 7–9 是 c3，行 10–12 是 c4。
仅运行 v2 的 c4（以下以 H01 为例）：

```bash
sed -n '10,12p' grid_commands/commands_weighted_cmr_ablation_v2_H01.txt | bash -e
```

完整的较强 CMR 对照要求四个 setting 均使用 v2 的 0.1；已有 v1 的 0.03 结果不能替代对应的 v2 运行。
命令不指定 GPU 编号，继承当前 CUDA_VISIBLE_DEVICES/调度器分配。
日志与模型使用 wcmr_wcrossc_ablation_v2 名称，不覆盖 v1 或既有实验；同一条命令再次运行仍会使用同一个输出位置。

## 汇总

仍在实验目录执行：

```bash
python summarize_logs.py --commands grid_commands/commands_weighted_cmr_ablation_v2_H01.txt grid_commands/commands_weighted_cmr_ablation_v2_H05.txt --log-dir logs_weighted_cmr_ablation_v2 --output-dir results/weighted_cmr_ablation_v2 --seeds 42 43 44
```

清单：[H01](commands_weighted_cmr_ablation_v2_H01.txt)、[H05](commands_weighted_cmr_ablation_v2_H05.txt)。
[逐次运行 manifest](weighted_cmr_ablation_v2_manifest.csv) 记录 24 次运行的组别、对照、seed、系数、命令行号和输出路径。

旧的低系数清单仍可使用：[v1 H01 / lam_cmr=0.03](commands_weighted_cmr_ablation_v1_H01.txt)、[v1 H05 / lam_cmr=0.03](commands_weighted_cmr_ablation_v1_H05.txt)。运行 v1 时先创建 logs_weighted_cmr_ablation_v1，并将汇总输出放到 results/weighted_cmr_ablation_v1。
