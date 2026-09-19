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

修改 `run_hinge.slurm.example` 和 `run_direct.slurm.example`：

| 配置 | 填写内容 |
| --- | --- |
| `CONDA_SH` | `conda.sh` 的绝对路径，如 Conda 安装目录下的 `etc/profile.d/conda.sh` |
| `EXPERIMENT_DIR` | 本仓库 `experiments_phase_8_rdcrosssdc_modular_gridsearch/` 的绝对路径 |
| `#SBATCH -p` | 集群的 H200 分区名称 |
| `#SBATCH --gres` | 默认 `gpu:h200:1`；GPU 类型名须与集群一致，可用 `sinfo -o '%P %G'` 查看 |
| 其他 `#SBATCH` 参数 | 按需调整 CPU 数、时限和数组范围；hinge 默认 `1-81`，direct 默认 `1-27`，对应命令文件行数 |

两份脚本均激活 `avcil-h200`，分别读取对应的 hinge/direct 命令文件。
修改完成后，可去掉 `.example` 后缀。在实验目录中先创建日志目录，再按需提交：

```bash
cd experiments_phase_8_rdcrosssdc_modular_gridsearch
mv run_hinge.slurm.example run_hinge.slurm
mv run_direct.slurm.example run_direct.slurm
mkdir -p logs logs_hinge

# 按需选择提交
sbatch --array=1-81 run_hinge.slurm
sbatch --array=1-27 run_direct.slurm
```
