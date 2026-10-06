#!/bin/bash
set -euo pipefail

# [P9 新增] 在一个 Slurm step 内并行运行三个 seed；GPU/主机内存为三进程共享。
command_file="${1:?Pass a command file containing exactly three experiments}"
if [[ ! -f "$command_file" ]]; then
    echo "Command file not found: $command_file" >&2
    exit 1
fi
commands=()
while IFS= read -r command || [[ -n "$command" ]]; do
    command="${command%$'\r'}"
    [[ "$command" =~ ^[[:space:]]*$ || "$command" =~ ^[[:space:]]*# ]] && continue
    commands+=("$command")
done < "$command_file"
if [[ "${#commands[@]}" -ne 3 ]]; then
    echo "Expected exactly 3 experiments, found ${#commands[@]} in $command_file" >&2
    exit 1
fi

# 三个 seed 各限制为分配 CPU 数的三分之一；避免每个 BLAS/PyTorch 进程各开满 12 线程。
total_cpus="${SLURM_CPUS_PER_TASK:-12}"
if [[ ! "$total_cpus" =~ ^[0-9]+$ ]] || ((total_cpus < 3)); then
    echo "At least 3 allocated CPUs are required for three concurrent seeds" >&2
    exit 1
fi
export OMP_NUM_THREADS="$((total_cpus / 3))"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p logs
printf 'Shared GPU: CUDA_VISIBLE_DEVICES=%s; CPU threads per seed=%s\n' \
    "${CUDA_VISIBLE_DEVICES:-unset}" "$OMP_NUM_THREADS"

pids=()
trap 'for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; exit 130' INT TERM
for index in "${!commands[@]}"; do
    printf 'Launching experiment %s: %s\n' "$((index + 1))" "${commands[$index]}"
    # 不改 CUDA_VISIBLE_DEVICES；三个独立 Python 进程继承同一张已分配的 GPU。
    bash -c "${commands[$index]}" &
    pids+=("$!")
done

# 等待所有 seed；任一失败都会使整个作业失败，不用裸 wait 隐藏错误。
status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        printf 'Experiment %s completed.\n' "$((index + 1))"
    else
        code=$?
        printf 'Experiment %s failed with exit code %s.\n' "$((index + 1))" "$code" >&2
        status=1
    fi
done
exit "$status"
