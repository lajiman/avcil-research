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

在 `grid_commands/commands_hinge.txt` 和 `grid_commands/commands_direct.txt` 中，将所有 `--feature_root` 占位符替换为数据目录的绝对路径，保留引号。`--meta_root` 默认使用 `../data2/balance`，可按需修改。

在所选 Slurm 模板中修改：

- `CONDA_SH`：Conda 安装目录下 `etc/profile.d/conda.sh` 的绝对路径。
- `EXPERIMENT_DIR`：本仓库实验目录的绝对路径。
- `#SBATCH -p` / `--gres`：集群的 H200 分区及 GPU 类型；按需调整 CPU、内存和时限。

模板默认激活 `avcil-h200`。填写后去掉 `.example` 后缀，先创建日志目录，再选择以下一种版本提交：

```bash
cd experiments_phase_8_rdcrosssdc_modular_gridsearch
mkdir -p logs logs_hinge logs_direct
```

单进程：每个数组任务申请 1 张 H200，运行 1 个训练进程。

```bash
sbatch --array=1-81 run_hinge.slurm
sbatch --array=1-27 run_direct.slurm
```

多进程：每个数组任务申请 1 张 H200，同时运行最多 4 个训练进程。

```bash
sbatch --array=1-21 run_hinge_multi.slurm
sbatch --array=1-7 run_direct_multi.slurm
```

若修改 `RUNS_PER_GPU`，数组上限相应改为 `ceil(命令数 / RUNS_PER_GPU)`。多进程资源不足时可改用单进程版本。

## 汇总日志

在实验目录执行，无需等待所有实验结束：

```bash
python summarize_logs.py
```

读取 `logs_hinge/` 和 `logs_direct/`，将实验调整的参数及每步 `Testing res` 写入 `results/summary.md` 和 `results/summary.csv`。

保留各 seed 结果并注明每步有效 seed 数；1 个结果直接记录，2–3 个计算均值及样本标准差，缺失值留空。

PR 只提交汇总文件，原始日志已由 `.gitignore` 排除。
