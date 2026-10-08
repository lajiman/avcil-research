# Phase 9：统一分析与 checkpoint 推理

这套工具只读 `save/`、`logs/`、原始特征和 metadata。所有新文件写入指定的 `results/` 子目录；不训练、不改 loss/gate、不覆盖原 checkpoint，也不根据测试准确率选择新权重。

## Juno：一次提交完成全部阶段

在仓库根目录、`codex/new-method-base` 分支执行：

```bash
git pull --ff-only origin codex/new-method-base
sbatch experiments_phase_9/run_analysis.slurm
```

默认使用现有 `xie` reservation、`h200` 分区、一节点两张 `nvidia_h200_nvl`、12 CPU、**总共 300G CPU 内存**、最长 48 小时。如 reservation 名称变化，在提交时用 `sbatch --reservation=实际名称 ...` 覆盖。它会自行激活现有的 `/groups/ytian/lujing/miniconda3/envs/avcil-h200`，不要求提交终端已激活环境。

这一次作业顺序执行：

1. 自动发现所有 run，读取 config/log/CSV/JSON，生成执行核验及统计表。
2. 启动两个长期推理进程，各使用 Slurm 分配的一个可见 GPU。每个进程按顺序处理自己的 checkpoint 队列，并复用一份音频字典，默认没有 DataLoader 子进程。
3. 对每个模型提取验证/测试样本特征，评估保存的 gate、等权、音频分支和视觉分支四种固定融合方式。
4. 自动比较同 seed 的 uniform 基线，生成分组统计、固定模型融合诊断、图表和报告。

默认处理每个 run 的**全部 step、best 和 last、test 和 val**。当前 15 个 run、每个 10 个 step 对应 300 个 checkpoint、600 次 split 扫描；不是 300 次重训。实际耗时取决于 HDF5 读取、CPU 和 GPU 吞吐，脚本不假定固定完成时间。

默认输出：

```text
experiments_phase_9/results/analysis_JOBID/
  report.html / report.md            # 先打开这两个之一
  pipeline_status.json              # complete / partial、失败阶段
  manifest.json                     # 配置、分析代码指纹、输入路径和当前运行标识
  offline_summary.json
  report_summary.json
  tables/                           # 所有可复用的 CSV
  figures/                          # 安装 matplotlib 时生成 PNG/PDF
  worker_logs/worker_0.log
  worker_logs/worker_1.log
  worker_0.json / worker_1.json
  cache/RUN__step_S__best/           # 逐样本预测、checkpoint 元信息及缓存状态
    test.npz / val.npz
    checkpoint.json / status.json
    history_classes.csv             # checkpoint 含 CL history 时
    history_samples.csv
```

Slurm 的 `slurm_avcil_phase9_analysis_JOBID.out/.err` 在提交目录。`pipeline_status=complete` 表示分析流程完成，**不表示被分析的训练全部通过 audit 或方法效果良好**；训练核验结论另见 `tables/audit.csv`。

## 路径与选择参数

默认读取仓库内 `experiments_phase_9/save` 和 `logs`。特征和 metadata 使用每个 run 保存的配置，相对路径按 `experiments_phase_9` 解析，忽略旧主机上的 `output_root`。如果文件位置已改变，一次提交时统一覆盖：

```bash
sbatch experiments_phase_9/run_analysis.slurm \
  --feature-root /absolute/path/to/VGGSound \
  --meta-root /absolute/path/to/data2/balance
```

也支持 `AVCIL_PROJECT_ROOT`、`AVCIL_CONDA_ROOT`、`AVCIL_CONDA_ENV`、`AVCIL_FEATURE_ROOT`、`AVCIL_META_ROOT` 环境变量。推荐显式参数，路径中有空格时加引号。

如果需要缩小范围，同一个入口仍会自动完成所有阶段。例如只分析原先三组、只推理 best：

```bash
sbatch experiments_phase_9/run_analysis.slurm \
  --runs 'phase9_uniform_cl_history_*' 'phase9_periodic_*' \
  --checkpoint-kinds best
```

`--steps 1 4 9` 选择零起始的第 1/4/9 step，即当前每步十类时的 20/50/100 类阶段。此参数仅缩小 checkpoint 推理范围，已保存的完整训练表仍会被汇总。分组和机制结论会显式记录哪些 step 有推理结果。

本地不具备 `.pt` 或 GPU 时，也可以只运行已有文件分析：

```bash
python experiments_phase_9/run_analysis.py --mode offline \
  --results-dir results/offline_analysis
```

分析脚本的相对参数统一以 `experiments_phase_9` 为基准；上例实际写入 `experiments_phase_9/results/offline_analysis`，从仓库根或 phase 9 调用都相同。`--mode offline` 完全跳过模型和预测缓存，因此不会把之前未重新验证的缓存当成本轮推理结果。

核心依赖沿用 `requirements.txt`，无需 pandas/scipy。图形是可选功能；如环境没有 matplotlib，表格及 Markdown/HTML 报告仍照常生成，报告会列出图形缺失原因。可在现有环境安装：

```bash
python -m pip install -r experiments_phase_9/requirements-analysis.txt
```

## 失败、缺文件与续跑

单个 checkpoint 失败不会阻止其他 checkpoint 执行。作业最终仍生成已完成部分的报告，并以非零状态结束；详情在 `pipeline_status.json`、worker 日志和每个 cache 的 `status.json`。缺失的 checkpoint 列在 `tables/missing_checkpoints.csv`，不会用零准确率代替。

默认缺少任何所选 checkpoint 也会使作业最终返回非零。若有意只处理当前已有模型，可以加 `--allow-missing-checkpoints`，缺失覆盖率仍会列出。`--mode offline` 不要求 checkpoint 存在。

同一结果目录可以续跑，已完整且指纹一致的推理会复用：

```bash
sbatch experiments_phase_9/run_analysis.slurm \
  --results-dir results/analysis_旧JOBID
```

缓存核对 checkpoint SHA-256、metadata SHA-256、特征路径/大小/修改时间、推理相关源码、软件版本及推理参数。特征体积很大，不逐个 checkpoint 对整个特征文件做哈希；如果更换了内容而刻意保留文件大小和时间，应加 `--force`。每轮只接纳经过本轮核验的 cache，未启动或失败 worker 遗留的旧成功状态不能冒充新结果。输入文件在扫描/推理过程中变化会被报告，应待传输完成后重跑。

不要同时向同一结果目录运行两个分析进程；`.analysis.lock` 会阻止并发。进程被强制杀死可能留下锁，确认该目录没有运行中的分析后才删除这个锁，或使用新的结果目录。正常失败或退出会释放锁。不会自动删除任何旧结果或训练文件。

## 表格的用途与口径

| 表格 | 用途 |
|---|---|
| `log_summary.csv` | 每个 log 的任务完成、错误和结尾摘要 |
| `audit.csv` | 配置、重复/缺失记录、loss 重组、gate 公式/更新时机/继承、测试/summary 一致性 |
| `run_summary.csv` / `method_summary.csv` | 各 seed 主结果及方法均值/样本标准差；完整轨迹才计算平均增量与最终指标 |
| `paired_metrics.csv` | 同 seed、共同训练协议一致的 uniform 配对差；保留每个 seed |
| `steps.csv` / `task_metrics.csv` / `new_old_metrics.csv` | 各 step、任务队列、新旧类表现与遗忘 |
| `classes.csv` / `class_deltas.csv` | 每类 recall/F1 与差值，包括下降和持平类别 |
| `gate_updates.csv` / `gate_update_summary.csv` | 原始/历史/实际 gate、eta、有效覆盖率和样本数 |
| `cl_history_summary.csv` / `cl_history_best.csv` | 固定旧 memory 上的双分支变化与配对正确性，best 与 final 不混用 |
| `subgroup_membership.csv` / `subgroup_metrics.csv` | uniform 任务开始时的 memory 可靠性差、互补余量、参考难度及类龄分组，检验测试收益 |
| `prototype_bank.csv` / `prototype_bank_summary.csv` | 历史原型使用、支持量、漂移及回退情况 |
| `inference_summary.csv` / `inference_classes.csv` / `inference_confusion.csv` | 测试集推理及融合对照的准确率、F1、旧新混淆 |
| `inference_reproduction_audit.csv` / `inference_coverage.csv` | best 测试与原记录、best/last 验证与保存指标的一致性，以及各 split 缺特征/类别的覆盖率 |
| `inference_pairs.csv` / `inference_sample_changes.csv` | 动态与等权、动态与独立 AVCIL 对照的纠正/破坏及逐样本改变 |
| `mechanism_decomposition.csv` | 完整方法差值拆为固定模型的 gate 影响和等权下的参数差异 |
| `mechanism_seed_summary.csv` / `inference_class_deltas.csv` | 分解项的 seed 汇总及重新推理得到的逐类 recall/F1/FP 差异 |
| `validation_group_membership.csv` / `validation_group_metrics.csv` / `validation_group_seed_summary.csv` | 用 uniform 验证集定义困难度、模态差异、互补余量，随后评估测试收益 |

准确率与 F1 原值为 0–1，后缀 `_pp` 为百分点。类准确率使用 `TP/support`（recall）；组准确率按 `sum(TP)/sum(support)`；F1 的 FP 来自全部候选/样本，不把分组后的类别当作独立封闭分类任务。首任务没有旧类，空组留空，不填假零。

配对依据共同训练配置、seed 和数据身份，不依据文件名字猜测；缺失或有多个匹配 uniform 时不任意挑一个。同一 seed 的重复运行不会当作独立重复。验证集决定 best，测试集不用于重新选 best/last、阈值或 gate。训练器未记录训练 commit，分析记录的是当前分析代码的 commit/源码指纹，不把它冒充原训练版本。

三 seed 只报告配对差、均值/SD及方向一致性，不把类别×step 当独立重复做显著性结论。所有预定低/中/高组都输出，分位点相同时不强行拆散同值类别，因此某些组可能为空。困难度不用同一测试集的 baseline 准确率来定义，避免与差值产生数学耦合。

memory 上的 CL history 和全已见类别 gate 原型统计采用不同候选集合；它们不会混成一条“测试遗忘”曲线。跨 run 的 reference_id 含教师来源，需比较实际样本与候选集合。模态差异和互补余量也分开：互补余量使用 `min(仅音频正确率, 仅视觉正确率)`，表示至少一个分支正确的比例相对更强单分支的额外空间。

## `.pt` 推理诊断做了什么

所有参数冻结、模型为 eval。对每份 checkpoint、每个 split 只提取一次分支特征（首批额外与原 forward 对照）。线性头保存 `z0=W(a+v)+b` 和 `d=W(a-v)`，其中 z0 就是 NPZ 的 `logits_uniform`：

```text
logits_actual  = z0 + (2*g-1)*d
logits_uniform = z0
logits_audio   = z0 + d
logits_visual  = z0 - d
```

无需保存所有 768 维隐藏特征。MLP 使用原 `class_conditional_logits` 对每个候选类进行融合，不套用线性公式。NPZ 可用 `numpy.load(path, allow_pickle=False)` 读取，包含视频 ID、标签、类别顺序、gate、四路 logits 和分支范数，线性头额外有 d。

两个分支对照保留原来的双模态特征提取，视觉仍包含音频引导注意力，不是独立训练的单模态模型。对同一 checkpoint 的即时 gate 影响不能代替完整算法对照。报告比较：

```text
DD = 动态训练模型 + 保存的 gate
D0 = 同一动态训练模型 + g=0.5
U0 = 独立 uniform 训练模型 + g=0.5
DD-U0 = (DD-D0) + (D0-U0)
```

第二项包含分支、分类头、教师继承和整个训练轨迹的差异，不是纯粹的表征因果效应。best/last 分开记录，last 仅作诊断。所有类的 gate 都会影响其对其他真实类别样本的竞争，因此同时统计 FP、纠正/破坏和旧新混淆。

`history_*.csv` 来自 checkpoint 内嵌的精确 epoch 观测；它们明确标记为旧训练 memory。完整逐 epoch 的类别历史仍读取原 CSV，不必加载每份历史 `.pt`。`prototype_bank` 只导出状态规模与来源摘要，不重新训练或修改历史原型。

当前 train/val metadata 共享部分源视频；验证定义的子组仍属于探索性分析。图表和分组只能呈现观察，不能把局部改善替代总体结果或作为无偏泛化证明。

## 验证方式

```bash
python -m pytest experiments_phase_9/tests -q
```

新增测试覆盖：安全日志解析、手算指标及负遗忘、缺失/重复/无效数据、gate 时序、linear/MLP 推理与原 forward、一致样本对齐、缓存失效与输入变化、失败任务继续、同模型分解、验证分组和两 GPU Slurm 调度参数。CPU 合成数据的统一入口测试包含实际训练产生的 checkpoint、子进程推理、报告及续跑。服务器 GPU 吞吐、300G 实际内存占用和 reservation 可用性仍需在 Juno 上执行验证。
