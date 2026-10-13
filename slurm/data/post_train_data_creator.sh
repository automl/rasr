#!/bin/bash -x
#SBATCH -p <cpu-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=150G
#SBATCH --job-name=post_train_data_creator
#SBATCH -D .
#SBATCH --time=24:00:00
#SBATCH --array=0-0
#SBATCH --output=output/post-train-out.%A_%a
#SBATCH --error=error/post-train-err.%A_%a

set -euo pipefail

CODE_DIR="${CODE_DIR:-$(pwd)}"

CONFIG_PATH="${1:-src/rasr/train_configs/data_post-train.toml}"

echo "INFO: cpus-per-task=${SLURM_CPUS_PER_TASK}"
echo "INFO: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}"
echo "INFO: SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-1}"
echo "INFO: node list=${SLURM_NODELIST}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

srun --cpu-bind=cores -n 1 -c "${SLURM_CPUS_PER_TASK}" uv run --project "${CODE_DIR}" --group data-gen python -m src.rasr.data.generation.post_train_data_creator \
    -c "${CONFIG_PATH}" \
    --task_id "${SLURM_ARRAY_TASK_ID:-0}" \
    --num_tasks "${SLURM_ARRAY_TASK_COUNT:-1}"
