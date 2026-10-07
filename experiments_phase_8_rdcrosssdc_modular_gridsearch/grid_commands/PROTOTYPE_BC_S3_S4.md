# B/C × S3/S4：Juno 预留 GPU 实验

两份主实验文件，每份 18 次训练，共 36 次：

- `commands_prototype_B_s3_s4.txt`：B，`rd_prototype_policy=pre_shrink`。
- `commands_prototype_C_s3_s4.txt`：C，`rd_prototype_policy=historical`。
- `prototype_BC_s3_s4_manifest.csv`：逐行参数、实验名和日志路径索引。

每份均为 S3/S4 × tolerance {0.05, 0.1, 0.2} × seed {42, 43, 44}。
它们逐参数继承 `commands_weighted_cmr_ablation_v2_H05.txt` 的 c3/c4。
类别顺序仍固定使用 dataset 中的 seed42；42/43/44 是训练随机种子。

| Setting | CMR | weighted CrossSDC-C | CrossSDC-I | 原始 L_i / L_c |
| --- | --- | --- | --- | --- |
| S3 | 0.1 | 0.3 | 0 | 都关闭，系数均 0 |
| S4 | 0.1 | 0.3 | 0.1 | 都关闭，系数均 0 |

CE、logit KD、空间/时间 attention KD 保留。memory=500、200 epochs、
batch=128、infer batch=32、alpha=0.5、Trust offset/shrinkage=0、关闭权重裁剪、
Need eta=0、hinge penalty 不变。C 固定 folds=5、mass cap=50、decay=0.9、
error scale=0.25、min anchors=2；本轮不扫描这些新增参数，它们仍是待验证配置。
较大的 tolerance 会放宽 hinge 条件，应结合 active ratio 和实际 CMR 数值解释结果。

## 推荐提交：两个受限 array，各自最多占一张卡

从仓库根目录提交：

```bash
sbatch --reservation=xie --array=1-6%1 --job-name=phase8_protoB \
  experiments_phase_8_rdcrosssdc_modular_gridsearch/run_prototype_grid.slurm \
  grid_commands/commands_prototype_B_s3_s4.txt

sbatch --reservation=xie --array=1-6%1 --job-name=phase8_protoC \
  experiments_phase_8_rdcrosssdc_modular_gridsearch/run_prototype_grid.slurm \
  grid_commands/commands_prototype_C_s3_s4.txt
```

共用提交脚本沿用现有 Juno 单 `srun` 共享 GPU 的结构。每个 array element 申请
**1 张 `nvidia_h200_nvl`、12 CPU、180G 主机内存、2 天**，使用 reservation `xie`。
`%1` 表示每份 array 同时只运行一个 element；因此 B/C 合计最多两张卡、360G 内存，
符合用户提供的 375G 双 H200 节点资源。Slurm 负责选择显卡与符合 reservation 的节点；
脚本不会写死 GPU 0/1，也不会额外申请三次独占 GPU。

| Array index | tolerance | setting | 同时运行的 seeds |
| --- | --- | --- | --- |
| 1 | 0.05 | S3 | 42 / 43 / 44 |
| 2 | 0.05 | S4 | 42 / 43 / 44 |
| 3 | 0.1 | S3 | 42 / 43 / 44 |
| 4 | 0.1 | S4 | 42 / 43 / 44 |
| 5 | 0.2 | S3 | 42 / 43 / 44 |
| 6 | 0.2 | S4 | 42 / 43 / 44 |

这样每组三个 seed 都有独立的 2 天时限，不需要把六组依次压进同一个 2 天作业。
也可去掉 `--array`：每份文件成为单个常驻 GPU 作业，三个 seed 分属三个执行队列，
各队列依次完成六组；此时 2 天覆盖整份 18 次训练，必须按实际耗时调整 `--time`。
每卡三进程是配置的并发上限，真实吞吐和峰值仍需 H200 运行验证。

180G 是 **CPU/主机内存**，不是 GPU 显存。
Slurm 的 `--mem` 与 GPU 分配是不同资源；参见
[Slurm sbatch 文档](https://slurm.schedmd.com/sbatch.html)。

## 路径与环境

支持从仓库根目录或本实验目录提交。`AVCIL_PROJECT_ROOT` 可覆盖仓库路径；
不通过 `BASH_SOURCE` 推断目录，因为 sbatch 会复制脚本到 spool。
Conda 默认 `/groups/ytian/lujing/miniconda3/envs/avcil-h200`，
可用 `AVCIL_CONDA_ROOT`、`AVCIL_CONDA_ENV` 覆盖。

特征及 metadata 默认沿用旧实验相对路径，以实验目录为基准：
`../../../datasets/VGGSound`、`../data2/balance`。
可在提交前设置 `AVCIL_FEATURE_ROOT`、`AVCIL_META_ROOT` 为绝对路径，
无需修改 36 条命令。训练日志会自动创建；Slurm 调度日志写入提交目录，
不要求事先创建 logs 文件夹。

## 并发控制与内存优化

- 三个进程原样继承 Slurm 分配的同一 `CUDA_VISIBLE_DEVICES`；每个训练进程只看到一张卡，
  不会触发原 trainer 的多卡 DataParallel。
- 每进程 PyTorch intra-op、OMP、MKL、OpenBLAS 设为 4 线程，inter-op=1；
  `num_workers=0`，避免再乘出大量 DataLoader workers。
- 同一进程的 train/val/test/replay 共享一份音频特征字典，四份 split metadata 保持独立。
  进程间不共享 Python 字典。原视觉缓存生命周期保留，训练与测试视觉缓存不同时保留。
- 按当前 metadata 及 8×196×768 FP32 视觉特征估算，最后任务 train+val+replay 缓存
  约 30.06 GiB/进程，三进程约 90.18 GiB，尚需加音频、pinned batches、模型和运行时。
  这不是完整进程 RSS 实测值。
- 默认每进程 PyTorch CUDA allocator 上限为可见显卡总容量的 30%，三份合计 90%。
  这是限制而非预留，不涵盖 CUDA context/第三方库的全部分配，不能保证不会 OOM。
  上限不足会明确失败，不会自动改 batch、精度或跳过样本。
- 保持 FP32、原 batch 和 `cycle(exemplar_loader)` 的 replay 顺序；不引入 AMP 或采样变更。

`AVCIL_TORCH_THREADS`、`AVCIL_CUDA_MEMORY_FRACTION`、`AVCIL_RECORD_RESOURCES`
由 launcher 注入。直接使用旧训练命令且不设置这些环境变量时，仍是原有 8 线程、
不设置 allocator 上限、不额外记录资源。

## 日志、失败和检查

- 实验日志：`logs_prototype_BC/<experiment_name>.log`。
- 队列状态：同目录 `launcher_*.json`，记录 PID、GPU token、起止时间、退出码和各实验状态。
- 内存实测：`save/metrics/<experiment_name>/resource_usage.csv`，记录每个 task 的
  train+bank / test 两阶段 GPU allocated/reserved 峰值、主机 RSS 和主机进程生命周期 RSS 峰值。
- 原有 loss_components、epoch_summary、per_class、Trust、prototype bank 诊断继续保存。

启动前检查整个 18 条参数矩阵；array 只检查/预留本组三个实验的输出。
同一个实验已存在 checkpoint、metrics、fig 或 log 时会拒绝覆盖。
三个 seed 任一失败，当前 launcher 返回失败并终止自己启动的其他 seed；不修改其他作业。
array 的其他 element 仍由 Slurm 独立调度，某组失败不会自动撤销整个 array。
重试失败组前应先检查日志，保存/整理其现有输出或使用新的实验名；程序不伪装成断点续训。

在实验目录可以先做不启动 GPU 的检查：

```bash
python run_prototype_grid.py --commands grid_commands/commands_prototype_B_s3_s4.txt --dry-run
python run_prototype_grid.py --commands grid_commands/commands_prototype_C_s3_s4.txt --group 1 --dry-run
```

Slurm 脚本后还可追加 launcher 参数，例如 `--runs-per-gpu 2` 降低并发；
不改变任何训练数学参数。更换 `--cuda-memory-fraction` 时必须保证
`runs_per_gpu × fraction < 1`，并根据实际峰值保留 CUDA 额外开销余量。
