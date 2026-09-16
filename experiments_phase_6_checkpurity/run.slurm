#!/bin/bash
#SBATCH --job-name=z0_noise
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/z0_noise_%j.out
#SBATCH --error=logs/z0_noise_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source /scratch/ganymede2/dal400642/miniconda3/etc/profile.d/conda.sh
conda activate avcil

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python -u analyze_z0_modality_noise.py \
  --dataset VGGSound_data2_balance_z0_noise_seed42 \
  --num_classes 100 \
  --class_num_per_step 10 \
  --batch_size 128 \
  --num_workers 0 \
  --knn_k 10 \
  --knn_chunk_size 512 \
  --bootstrap_repeats 1000 \
  --bootstrap_ci 0.95 \
  --max_samples_per_class 100 \
  --annotate_top 8 \
  --rank_top_n 10 \
  --seed 42 \
  --feature_root ../../../datasets/VGGSound \
  --meta_root ../data2/balance \
  --output_root ./save/z0_modality_noise