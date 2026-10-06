# Phase 9：类别级动态模态融合

本阶段保留原始 AVCIL 的分支、注意力及 loss，将类别差异用于融合。没有 CMR、CrossSDC-I 或 weighted CrossSDC-C，也不会构造这些 loss 的计算图。

## 与 phase 8 的模块对应

代码内使用三种中文标记，`P8` 均指 `experiments_phase_8_rdcrosssdc_modular`：

* **`[P8 原样沿用]`**：标注的函数或代码块计算代码一致；不要求注释、docstring 一致。
* **`[P8 逻辑沿用]`**：保留原有公式或流程，但抽取函数、调整接口或增加处理；旁边说明来源与差异。
* **`[P9 新增]`**：类别融合、周期更新、样本量控制、CL 历史或新的工程处理。

标记只覆盖紧随其后的函数或注明范围的代码块。一个函数可同时含有沿用与新增部分，不能据此认为整个模块与 phase 8 相同。

可先看 [exact_losses.py](class_fusion/exact_losses.py)：`CE_loss`、`cal_contrastive_loss`、`class_contrastive_loss` 原样沿用，`avcil_loss` 则从旧训练循环抽取，各 CE/KD/对比/注意力块分别标注。
另有 `setup_seed` 的可执行函数体、两个数据加载器的 `__len__`、`metrics.py` 的 `safe_div`、`compute_per_class_prf`、`save_json`、`append_csv_rows` 及注明边界的测试指标块原样沿用。

`fusion_method.py` 中原型求和、LOO 与概率转换公式和 phase 8 有联系，但已改为**当前学生、所有已见类、同模态**统计；其整体不是原 RD 方法。共享模型的来源是仓库根目录下的 `model/audio_visual_model_incremental.py`，模型注释会单独注明。

| Phase 8 | Phase 9 | 职责 |
|---|---|---|
| `train_incremental_rd_crosssdc_modular.py` | `train_incremental_fusion_modular.py` | 增量训练、验证、任务循环、调用方法模块 |
| `dataloader_ours.py` | `dataloader_ours.py` | 相同 `IcaAVELoader` / `exemplarLoader` 接口与随机回放策略 |
| `rd_crosssdc/rd_method.py` | `class_fusion/fusion_method.py` | 原型可靠性、统计样本、更新时机、更新幅度 |
| `rd_crosssdc/exact_losses.py` | `class_fusion/exact_losses.py` | 原始 AVCIL loss；移除了 CrossSDC 部分 |
| `rd_crosssdc/diagnostics.py` | `class_fusion/diagnostics.py` | 逐类别权重、epoch loss 等 CSV |
| `rd_crosssdc/metrics.py` | `class_fusion/metrics.py` | 测试准确率、类别 P/R/F1、遗忘指标 |
| 新增观察模块 | `class_fusion/cl_history.py` | 固定旧类 memory 上的教师—学生可靠性变化与旧新混淆，只记录、不改变 gate/loss |
| 训练入口内保存模型 | `class_fusion/checkpoints.py` | 模型参数和融合表一起保存、加载 |
| `model/audio_visual_model_incremental.py` | `model/audio_visual_model_incremental_class_fusion.py` | 继承原始分支与注意力，增加类别条件融合 |

phase 8 和原始模型文件不作修改。phase 9 运行时不导入 phase 8 的训练或方法模块。

默认还会记录 CL 历史信息，详细字段、公式和输出说明见 [README_CL_HISTORY.md](README_CL_HISTORY.md)。它与下面用于更新 gate 的全已见类别统计分开，使用固定旧类候选集合比较教师和学生。除类别汇总外，还保存样本级分支分数、预测、判别间隔、配对正确性、预测转换和实际融合 logits，便于以后分析不对称遗忘。可用 `--no_record_cl_history` 关闭。

## 计算定义

设第 c 类的音频占比为 `g[c]`。分类前使用未归一化的原始投影特征：

```text
h_c(x) = 2*g[c]*h_A(x) + 2*(1-g[c])*h_V(x)
z_c(x) = classifier(h_c(x))[c]
```

所有候选类别都计算一个 logit，最后统一分类，不需要样本的真实类别。`g=0.5` 精确恢复原始两分支相加的尺度。视觉分支仍使用原始音频引导的时空注意力。

统计时使用当前学生在 eval/no_grad 下的快照。训练在统计期间暂停，因此不必另外复制一个完整模型。参考集是“当前新类训练样本 + 当前旧类 memory”，不使用验证集或测试集。每个 step 固定参考视频 ID，去重后每个视频只统计一次。

```text
u_i^m = normalize(h_i^m)
S_c^m = sum_{i:y_i=c} u_i^m
mu_{k,-i}^m = normalize(S_k^m - 1[y_i=k]*u_i^m)
p_i^m(c) = softmax_k(dot(u_i^m, mu_{k,-i}^m) / tau)[c]
R_c^m = mean_{i:y_i=c} p_i^m(c)
g_hat[c] = R_c^A / (R_c^A + R_c^V)
```

两次流式扫描分别构造原型和计算 leave-one-out 概率，不在 GPU 缓存全训练集特征。两个分支在相同的有效候选类别上归一化；没有有效原型的类别不作为负类。非有限/零向量样本对不进入任一分支的统计。至少有两个有效候选类别；目标类至少有 `fusion_min_samples` 个有效样本，且其 LOO 评估有效，否则保留该类历史 gate。

更新公式：

```text
fixed:        eta[c] = fusion_eta_max
sample_aware: eta[c] = fusion_eta_max * n[c] / (n[c] + fusion_n_ref)
g_new[c] = (1-eta[c])*g_old[c] + eta[c]*g_hat[c]
```

`n[c]` 是当次参考集的不同有效样本数，不是训练期间回放次数，也不跨更新累计。小 memory 降低改写历史值的幅度；没有跨类别均值归一化、向 0.5 的统一收缩或人为上下界裁剪。`g` 是无梯度 buffer，原始 loss 的梯度仍通过加权融合传给两个分支。

这仍是“分支判别可靠性”的启发式，不是互补性或最优融合权重的证明。LOO 不能消除表征在训练/memory 数据上过拟合的偏差；历史平滑也可能延迟对真实变化的响应。

## 默认训练日程

* 首个任务全程等权，只使用 CE，沿用原始脚本行为。
* 后续 step 加载上一任务 **最佳验证检查点**；旧类继承 gate，新类初始化为 0.5。
* epoch 1–40 使用初始化权重。
* 在 epoch 40、80、120、160 **完成训练、验证和最佳模型保存后**，重新估计并平滑更新。
* 新 gate 从下一个 epoch 生效；epoch 200 后不再更新。
* KD 教师保留上一任务自己的 gate，始终冻结，不随学生更新。
* `--fusion_update_first_step` 可选开启首任务的周期更新；默认关闭。

后续任务的训练目标保持：

```text
L = L_CE + L_KD + lam_I*L_I + lam_C*L_C
    + lam*L_spatial + (1-lam)*L_temporal
```

保留 phase 8 的新旧样本 CE 切片、按旧任务计算的温度 2 logit KD、原始 L_I/L_C reduction，以及注意力蒸馏维度。默认 `lam_I=0.1, lam_C=1, lam=0.5`，三个开关默认开启。原有 `--instance_contrastive --class_contrastive --attn_score_distil` 参数仍可显式填写；分别使用 `--no_instance_contrastive`、`--no_class_contrastive`、`--no_attn_score_distil` 关闭。

## 主要参数

| 参数 | 默认 | 含义 |
|---|---|---|
| `fusion_mode` | `periodic` | `uniform` 等权 AVCIL 对照，`periodic` 周期更新 |
| `fusion_update_rule` | `sample_aware` | `fixed` 固定更新幅度；`sample_aware` 按有效样本量调整 |
| `fusion_warmup_epochs` | 40 | 第一次更新在第几轮结束后发生，必须大于 0 |
| `fusion_update_interval` | 40 | 之后每隔多少 epoch 更新 |
| `fusion_eta_max` | 0.5 | 更新幅度上限，范围 (0,1] |
| `fusion_n_ref` | 10 | 样本量调整的参考数量，只在 sample_aware 中使用 |
| `fusion_temperature` | 0.1 | 同模态原型 softmax 的温度 |
| `fusion_min_samples` | 2 | 允许更新的最少有效样本数 |
| `fusion_batch_size` | 128 | 两次统计扫描的 batch size |
| `fusion_classifier` | `linear` | `linear` 保持原始头；`mlp` 验证非线性融合 |
| `fusion_hidden_dim` | 768 | MLP 隐藏维度 |
| `fusion_chunk_size` | 16 | 非线性头同时处理的候选类别数，控制内存占用 |
| `record_cl_history` | 开启 | 记录 CL 历史；用 `--no_record_cl_history` 关闭，不改变融合策略 |
| `cl_history_interval` | 40 | 额外周期观测间隔；任务开始、gate 更新前、最佳与最终检查点也记录 |

默认数值是这次设计的起点，不表示已经通过完整 VGGSound 实验验证。若 epoch 总数不超过 warmup，不会发生更新。直接替换可用 `fixed + fusion_eta_max=1` 作为对照。

MLP 对每个候选类别构造融合特征，通过同一个共享非线性头后取对应类别 logit；不能用两路 logit 加权来代替。分支提取一次，融合后的头按候选类别分块执行，计算成本会增加。训练和测试使用相同路径。类别条件融合不提供单个共享 `joint_feature`，对应接口会明确拒绝调用。

## 数据与运行

需要原有特征文件：

```text
FEATURE_ROOT/
  visual_features.h5
  audio_pretrained_feature/audio_pretrained_feature_dict.npy
META_ROOT/
  all_classId_vid_dict.npy
  all_id_category_dict.npy
  category_encode_dict.npy
```

AVE 使用 `visual_pretrained_feature_dict.npy`，与 phase 8 相同。加载器支持整数/字符串类别键，过滤缺失的音视频对，并检查类别编码一致性。memory 的每类名额仍为 `memory_size // num_old_classes`；实际有效视频不足时不通过复制视频补齐。HDF5 按进程延迟打开，支持 Windows spawn worker。默认 `num_workers=0`。

在已有 PyTorch 环境中安装依赖，然后从 phase 9 目录运行。特征路径由运行命令所在目录解析，建议使用绝对路径；默认 metadata 和输出目录则相对于本文件定位。

```bash
cd experiments_phase_9
python -m pip install -r requirements.txt
python -u train_incremental_fusion_modular.py --feature_root /absolute/path/to/VGGSound --meta_root ../data2/balance --fusion_mode periodic --fusion_update_rule sample_aware --seed 42 --experiment_name phase9_sample_aware_h200_seed42
```

默认已经设置 `num_classes=100, class_num_per_step=10, max_epoches=200, memory_size=500`，及前述原始 loss 开关。也可以从仓库根目录执行 `python experiments_phase_9/train_incremental_fusion_modular.py ...`。

预置两组实验、每组 seeds 42/43/44，共 6 条命令，分别保存在两个文件中。命令矩阵已移除 uniform；代码仍保留该模式供已有检查点及一致性测试使用。

| 命令文件 | 更新方式 | 每次提交 |
|---|---|---|
| `grid_commands/commands_periodic_fixed.txt` | 固定 `eta=0.5` | 并行运行 seeds 42/43/44 |
| `grid_commands/commands_periodic_sample_aware.txt` | `eta[c]=0.5*n[c]/(n[c]+10)` | 并行运行 seeds 42/43/44 |

以下生成器输出 Bash 命令，供 Linux/Slurm 使用；`--output_dir` 可以指定两个文件的输出目录：

```bash
python generate_commands.py --feature_root /absolute/path/to/VGGSound
mkdir -p logs
# 激活服务器的 PyTorch 环境后，按集群需要添加 partition/account：
sbatch --job-name=phase9_fixed run.slurm grid_commands/commands_periodic_fixed.txt
sbatch --job-name=phase9_sample_aware run.slurm grid_commands/commands_periodic_sample_aware.txt
```

这是两个独立的 Slurm 作业，每个作业在同一节点申请 **1 GPU、12 CPU、300G 主机内存**。一个 `srun` 调用 `run_shared_gpu.sh`，在该 step 内并行启动三个独立训练进程，共享同一张 GPU 和 300G 内存。每个 seed 的 OMP/MKL/OpenBLAS/NumExpr 计算线程默认限制为 4；任意 seed 失败会使作业返回失败，三个训练日志仍独立保存。两个作业同时运行合计申请 **2 GPU、600G 主机内存**。

`300G` 是主机内存的总申请量，不是实际占用，也不是每个 seed 独立的 100G 硬限制。三个进程继承 Slurm 设置的同一个 `CUDA_VISIBLE_DEVICES`，不手动填写物理卡号，不分别申请独占 GPU step；单卡可见时训练器不会进入 DataParallel。参见 [Slurm GPU 管理文档](https://slurm.schedmd.com/gres.html#GPU_Management)。若集群需要特定 GPU 类型，只需在 sbatch 指定，例如：

```bash
sbatch --gres=gpu:a100:1 --job-name=phase9_fixed run.slurm grid_commands/commands_periodic_fixed.txt
```

脚本要求命令文件恰好包含三条实验命令，没有作业数组。预置命令使用相对特征路径 `../../../datasets/VGGSound`，运行前应核对或重新生成。命令显式指定 `--device cuda`，避免 GPU 不可用时静默进行 CPU 训练。`run.slurm` 从提交目录执行，不包含某台机器的 Conda/项目绝对路径；`logs` 必须在 sbatch 提交前创建。

针对单卡并发，目前采用以下内存处理，保留原 batch size、FP32、loss 和 gate 更新日程：

* 一个实验内的 train/val/test/replay 共用同一份音频特征字典，完整字典从四份降为一份；三个 seed 仍各自拥有一份，不共享模型、优化器、随机状态或 gate。没有跨运行的全局特征缓存，重新运行会读取新的文件。视觉 HDF5 仍按需读取；默认 `num_workers=0`，避免 worker 复制大字典。
* 每批前向前清空旧梯度，并在更新后释放学生/教师输出和输入引用；epoch 结束时释放 cycle 的回放缓存，再开始验证和统计。回放顺序未改为重新采样。
* 任务训练及测试结束时释放 CUDA 空闲缓存，供同卡其他进程使用；不在每个 batch 调用。该操作不释放活跃模型/张量，也不能解决三个实验本身的活跃显存需求超过整卡容量的问题，见 [PyTorch empty_cache 文档](https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.empty_cache.html)。
* 保留两次流式扫描的 gate 估计和 no_grad 教师；训练暂停后直接使用当前学生统计，避免另复制一份模型。

CUDA 运行时新增 `save/metrics/RUN_NAME/resource_summary.csv`，逐 step 记录本进程 `peak_allocated_gib`、`peak_reserved_gib` 和设备总显存。统计覆盖训练、验证、gate 与 CL 观测，不包括之后单独加载模型进行的正式测试；只反映 PyTorch 分配器，不包括其他进程及全部 CUDA context/驱动开销。整卡峰值仍需结合 `nvidia-smi`，主机内存峰值可结合 Slurm 的 `sacct`。单卡三进程的实际速度和能否容纳当前 batch 尚需服务器实测；不会自动开启 MPS、混合精度或降低 batch size。

测试已有检查点时使用与训练一致的 experiment_name、output_root、类别数/每步类别数和数据路径，再加 `--test_only`。测试只创建测试数据集，不重建 memory、不读取训练或验证 split、不重新估计 gate。

## 检查点与诊断

默认输出在 `experiments_phase_9/save/`：

```text
save/RUN_NAME/
  config.json
  step_0_best_model.pt
  step_0_last_model.pt
  ...
save/metrics/RUN_NAME/
  class_fusion/
    epoch_summary.csv
    step_1_initial_gates.csv
    step_1_after_epoch_40_gates.csv
    ...
  per_class_metrics.csv
  cl_history/
    class_history.csv
    step_1_reference.json
    step_1_reference.pt
    step_1_epoch_40_history.pt
    ...
  per_class_state.json
  step_0_test.json
  summary.json
```

逐类别 CSV 包含 `counts/scored_counts`、两分支可靠性、有效标记、`eta`、`previous_gate/raw_gate/audio_gate/visual_gate`、生效 epoch 和 gate 版本。epoch CSV 包含所有原始 loss 分项、gate 范围及该轮验证使用的 gate 版本。

检查点使用显式 state_dict，包含参数、gate buffer、版本、配置、训练 epoch、验证准确率和最近一次 gate 统计。教师和测试都从同一份检查点加载 gate。`best` 依据验证准确率选取，与 phase 8 一致；它可能位于第一次更新之前，不能假定最佳模型一定已经使用动态 gate，应查看保存的 epoch 和 gate_version。`last` 保存最终训练状态，便于诊断，但下一任务仍默认继承 `best`。

这些检查点用于任务间继承和推理，不宣称支持中途恢复优化器。phase 8 的整模型 `.pkl` 不能作为 phase 9 检查点直接加载。

启用 CL 记录时，检查点的 `cl_history` 字段还嵌入教师参考条件与该检查点对应 epoch 的学生观测。关闭记录或首任务时为 `None`，旧版 phase-9 检查点缺少此字段也能加载。CL 历史不参与推理。

为避免混合实验结果，训练拒绝覆盖已有非空同名运行目录。`test_only` 输出在独立时间戳子目录，不覆盖训练日志。

## 验证

```bash
python -m pip install -r requirements-dev.txt
# 从仓库根目录运行：
python -m pytest experiments_phase_9/tests -q
```

测试覆盖等权输出/梯度与原模型一致性、线性及非线性逐类别融合、LOO 概率、无效/稀少样本、参考集去重、历史平滑、更新日程、随机状态保护、检查点和教师隔离、原始 AVCIL loss，以及合成 HDF5 数据上的三个增量 step 与 test_only 一致性。完整数据训练仍需在具有 VGGSound 特征和 GPU 的环境运行。

## 2026-10-05 数据边界与设计核对

本次核对通用可靠性融合，不引入 CL 特化策略。训练算法没有因命令拆分而修改。

| 环节 | 当前实现与核对结论 |
|---|---|
| 训练样本 | 当前任务新类训练集 + 旧类 memory；不把旧类全部训练集或未来类样本加入训练/原型统计 |
| gate 估计 | 当前学生 eval/no_grad；唯一训练视频；同模态 LOO 原型；两路共享有效样本/候选类规则；只使用已见类 |
| 更新幅度 | fixed 固定幅度；sample-aware 按实际有效样本量减小幅度，向各类自己的历史值平滑 |
| 任务开始 | 旧类继承上一任务 best 的 gate，新类为 0.5；没有额外进行一次旧类重估。这是当前实现选择，与早期讨论的“任务开始先重估旧类”方案有区别 |
| 更新时序 | 默认在第 40/80/120/160 轮的训练、验证和 checkpoint 保存后更新，下一轮生效；首任务默认始终等权 |
| loss | 首任务只有 CE；增量任务保留 CE、task-wise KD、原始 L_i/L_c、空间/时间注意力蒸馏；没有 CMR/CrossSDC 或 gate 专用 loss |
| 推理 | 对每个候选类使用其 gate 计算 logit；forward 不接收真实标签，线性及当前 Linear/ReLU MLP 均验证过显式特征融合等价性 |
| checkpoint | 参数与 gate 一起保存/恢复；测试不重新估计 gate；验证选 best，测试结果不反馈给训练 |

针对仓库当前 `data2/balance` 的实际 metadata，train/val/test 分别为 **12,500 / 5,000 / 5,000** 个片段，100 类，类编码和 ID 映射一致。三个 split 的完整片段 ID 两两无交集。详细计数、样例与文件哈希保存在 [METADATA_AUDIT.json](METADATA_AUDIT.json)。

**不能据此宣称完全没有数据泄露风险**：按 `<11 位源视频 ID>_<起始时间>` 的命名规则归并，train 与 val 共享 **318 个源视频**，涉及各 318 个片段；test 与 train/val 的源视频交集均为 0。不同片段可能共享背景、主体或录制条件，验证集不是严格的源视频隔离划分，可能影响 checkpoint 选择。这来自已有 metadata，当前保持该划分以便和之前实验比较；若需要严格的源视频隔离，应按源视频重划 train/val，并在新划分上重跑可比基线。

从算法数据流看，未发现测试数据/标签进入参数训练或 gate 统计。验证集只用于选择 checkpoint；因此选中的参数/gate 会按常规受到验证指标影响，但不在验证样本上估计原型。特征文件可能存储全部 split/类别，加载到字典不等于参与统计；进入网络计算的训练/参考样本由当前类列表和 memory ID 限定。

新增回归检查直接替换测试集全部特征、同时改变测试标签，验证后续三个增量 step 的 best/last 模型参数、gate、验证结果、选中 epoch 和融合日志完全一致；并检查实际传给 gate 估计器的参考 ID 确实只来自当前新类训练集和已选旧类 memory。测试结果不会反向改变本实验模型。

两个方法上的限制也需要保留：LOO 去掉的是原型中的当前样本，分支表征仍经过训练数据拟合，可靠性不等同于独立测试泛化率；视觉分支保留原 AVCIL 的音频引导注意力，因此称作“视觉分支可靠性”比“完全独立的视觉模态可靠性”更准确。最佳 checkpoint 也可能早于首次 gate 更新，应结合 gate_version 解读结果。

本地没有实际 VGGSound 特征文件，无法追溯预训练特征是否接触测试数据，也没有做原始媒体近重复检测。GPU、集群实际调度和内存峰值未验证；Slurm 本地检查使用模拟 srun，验证单个 step 内的三进程并发、相同 GPU 可见性、线程数限制及失败状态汇总。CPU 上对比内存优化前后的 12 份检查点（线性/MLP × 三个 step × best/last），模型参数、gate、最佳 epoch 和验证准确率逐位一致；35 项测试通过。
