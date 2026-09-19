# AV-CIL · H200

## 环境安装

Linux x86_64 / H200，需已配置可用的 NVIDIA 驱动。在仓库根目录执行：

```bash
conda create -n avcil-h200 python=3.10 pip -y
conda activate avcil-h200
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

依赖已指定 PyTorch 2.6.0 / CUDA 12.4。

## 数据结构

数据可放在任意位置，按以下结构组织；`--feature_root` 指向 `VGGSound/`：

```text
datasets/VGGSound/
├── audio_pretrained_feature/
│   └── audio_pretrained_feature_dict.npy
├── VGGSound_audios/
├── VGGSound_frames/
└── visual_features.h5
```

## 配置与运行

以下文件均位于 `experiments_phase_8_rdcrosssdc_modular_gridsearch/`。

修改 `grid_commands/commands_hinge.txt` 和 `grid_commands/commands_direct.txt`：

- `--feature_root`：将所有 `"please input your path of VGGSound features"` 替换为数据目录的绝对路径，保留引号。
- `--meta_root`：默认 `../data2/balance`，使用仓库内的元数据；若另行存放，改为对应路径。

选择一套 Slurm 模板；每个数组任务均申请 **1 张 H200**，不要求独占整个节点：

| 版本 | 模板（均带 `.example` 后缀） | 每个数组任务 | hinge / direct 数组范围 |
| --- | --- | --- | --- |
| 单进程 | `run_hinge.slurm` / `run_direct.slurm` | 1 个训练进程，4 个 CPU 核 | `1-81` / `1-27` |
| 多进程 | `run_hinge_multi.slurm` / `run_direct_multi.slurm` | 最多 4 个训练进程同时运行，16 个 CPU 核 | `1-21` / `1-7` |

在所选模板中修改：

| 配置 | 填写内容 |
| --- | --- |
| `CONDA_SH` | `conda.sh` 的绝对路径，如 Conda 安装目录下的 `etc/profile.d/conda.sh` |
| `EXPERIMENT_DIR` | 本仓库 `experiments_phase_8_rdcrosssdc_modular_gridsearch/` 的绝对路径 |
| `#SBATCH -p` | 集群的 H200 分区名称 |
| `#SBATCH --gres` | 默认 `gpu:h200:1`；GPU 类型名须与集群一致，可用 `sinfo -o '%P %G'` 查看 |
| `RUNS_PER_GPU` | 仅多进程模板：每张卡同时运行的训练进程数，默认 `4`；修改后数组上限为 `ceil(命令数 / RUNS_PER_GPU)` |
| 其他 `#SBATCH` 参数 | 按需调整 CPU 核数、主机内存、时限和数组范围 |

两套模板均激活 `avcil-h200`，使用同一对 hinge/direct 命令文件。单进程版本每个数组任务执行对应行；多进程版本按连续 4 行分组并发执行，最后一组仅运行剩余命令。组内任一进程失败，该数组任务最终返回失败。

修改完成后，可将所选模板的 `.example` 后缀去掉。在实验目录中先创建日志目录，再选择一种版本提交：

```bash
cd experiments_phase_8_rdcrosssdc_modular_gridsearch
mkdir -p logs logs_hinge logs_direct
```

单进程版本：

```bash
mv run_hinge.slurm.example run_hinge.slurm
mv run_direct.slurm.example run_direct.slurm

# 按需选择 hinge / direct
sbatch --array=1-81 run_hinge.slurm
sbatch --array=1-27 run_direct.slurm
```

多进程版本：

```bash
mv run_hinge_multi.slurm.example run_hinge_multi.slurm
mv run_direct_multi.slurm.example run_direct_multi.slurm

# 按需选择 hinge / direct
sbatch --array=1-21 run_hinge_multi.slurm
sbatch --array=1-7 run_direct_multi.slurm
```

数组范围后加 `%1` 可限制该数组同时只使用一张 GPU，例如 `--array=1-21%1`；限制对每个数组分别生效。

四进程共享 GPU 显存和算力，主机内存需容纳各进程的数据副本。若多进程资源不足，可直接使用单进程版本。
同一批实验的两种版本二选一；切换前结束同一实验的旧作业，避免覆盖相同输出。默认四进程时，第 `k` 组对应命令第 `4k-3` 至 `min(4k, N)` 行（`N` 为命令总数）；可根据日志只补跑失败命令对应的单进程数组项。

## 汇总日志

在实验目录运行（仅使用 Python 标准库）：

```bash
python summarize_logs.py
```

默认根据两份命令清单读取 `logs_hinge/` 和 `logs_direct/`，按实验汇总 seeds 42/43/44 的每步 `Testing res`，
输出 `results/summary.md` 和 `results/summary.csv`。只展示变化的参数和损失类型，保留各 seed 结果，
每一步有多少有效结果就汇总多少：1 个 seed 直接记录，2–3 个 seed 计算均值及样本标准差。
报告注明每步实际 seed 数 `n`（CSV 为 `n_step_*`）；只有 1 个结果时标准差留空。准确率保持日志中的 0–1 数值。

可在训练过程中反复运行；缺失、未完成均正常输出（退出码 0），缺失值不补 0。
重复结果、参数冲突等异常会标记并排除，其他有效 seed 继续汇总，此时退出码为 1。
可用 `--commands grid_commands/commands_hinge.txt` 只汇总一份清单；日志另存时用 `--log-dir /path/to/logs`，
输出位置用 `--output-dir results/hinge` 指定。PR 只提交生成的 Markdown/CSV；原始日志已由 `.gitignore` 排除。
