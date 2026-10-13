#!/bin/bash -x
#
# SLURM pre-training template. Set the partition below and adapt GPU/CPU counts to your cluster.
#
# Usage (from the repo root; launch_scripts/launch_train.sh wraps this with run naming and resume):
#   RASR_DATA_PATH=/path/to/arrow sbatch slurm/train.sh
#   RASR_DATA_PATH=/path/to/arrow sbatch slurm/train.sh --training.steps=50000
#   CONFIG=src/rasr/train_configs/pre-train_debug.toml RASR_DATA_PATH=/path/to/arrow sbatch slurm/train.sh
#
# Environment variables:
#   RASR_DATA_PATH - Arrow dataset directory (required)
#   CONFIG         - RASR config file (default: src/rasr/train_configs/pre-train.toml)
#   CONTAINER      - Apptainer image built from titan_0.2.1.def (default: titan_0.2.1.sif)
#   CACHE_BASE     - Triton/torch/HF cache directory (default: <project>/cache)
#
#SBATCH -p <gpu-partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=rasr
#SBATCH --threads-per-core=1
#SBATCH --time=24:00:00
#SBATCH -D .
#SBATCH --output=output/rasr-out.%j
#SBATCH --error=error/rasr-err.%j

# NCCL settings for L40 GPUs
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_IB_HCA=mlx5
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_SOCKET_RETRY_CNT=3

export HF_ALLOW_CODE_EVAL="1"

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

# Calculate a unique port based on job ID to avoid conflicts
PORT_BASE=20000
PORT_OFFSET=$(($SLURM_JOB_ID % 10000))
export MASTER_PORT=$((PORT_BASE + PORT_OFFSET))

export NUM_NODES=$SLURM_JOB_NUM_NODES
export GPUS_PER_NODE=8
export NUM_GPUS_PER_NODE=8
export NUM_GPUS=$((NUM_GPUS_PER_NODE*SLURM_NNODES))

# Single-node: use localhost
if [ "$SLURM_NNODES" -eq 1 ]; then
    export MASTER_ADDR="127.0.0.1"
fi

echo "INFO: MASTER_ADDR: $MASTER_ADDR"
echo "INFO: MASTER_PORT: $MASTER_PORT"
echo "INFO: number of nodes: $SLURM_NNODES"
echo "INFO: number of gpus per node: $GPUS_PER_NODE"
echo "INFO: number of SLURM_JOB_ID: $SLURM_JOB_ID"
echo "INFO: number of SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "INFO: SLURM_PROCID  $SLURM_PROCID"

NUM_GPUS_PER_NODE=8
SLURM_CPUS_PER_TASK=64

# RASR configuration
CONFIG="${CONFIG:-src/rasr/train_configs/pre-train.toml}"
CONTAINER="${CONTAINER:-titan_0.2.1.sif}"

# Support both apptainer and singularity (apptainer is the successor to singularity)
APPTAINER_CMD=$(command -v apptainer 2>/dev/null || command -v singularity 2>/dev/null)
if [ -z "$APPTAINER_CMD" ]; then
    echo "ERROR: Neither apptainer nor singularity found in PATH"
    exit 1
fi

# Project and container configuration
PROJECT_DIR=$(pwd)                          # Assume script is run from project root
CACHE_BASE="${CACHE_BASE:-$PROJECT_DIR/cache}"

# Cache directories (derived from CACHE_BASE)
export TRITON_CACHE_DIR="$CACHE_BASE/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_BASE/triton"
export TORCHINDUCTOR_FX_GRAPH_CACHE="1"
export TORCHINDUCTOR_AUTOGRAD_CACHE="1"
export HF_DATASETS_CACHE="$CACHE_BASE/hf"
export TORCH_HOME="$CACHE_BASE/torch"

echo "Configuration loaded successfully:"
echo "  PROJECT_DIR: $PROJECT_DIR"
echo "  CONTAINER: $CONTAINER"
echo "  Cache base: $CACHE_BASE"

# Create cache directories if they don't exist
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$HF_DATASETS_CACHE"
mkdir -p "$TORCH_HOME"
mkdir -p "$TORCH_HOME/kernels"  # PyTorch JIT kernel cache (PYTORCH_KERNEL_CACHE_PATH)

# CUDA allocator: reduce fragmentation and large-alloc failures
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"

# Disable cudagraphs unless explicitly enabled (can increase memory usage)
: "${TORCH_INDUCTOR_CUDAGRAPH_DISABLE:=1}"


SRUN_ARGS="
    --nodes=$SLURM_NNODES \
    --gres=gpu:$NUM_GPUS_PER_NODE \
    --cpus-per-task=$SLURM_CPUS_PER_TASK \
    --wait=20 \
    --kill-on-bad-exit=1 \
    --label \
    --jobid $SLURM_JOBID"


APPTAINER="$APPTAINER_CMD exec \
--writable-tmpfs \
--env WANDB_API_KEY=${WANDB_API_KEY:-} \
--env WANDB_PROJECT=${WANDB_PROJECT:-} \
--env WANDB_RUN_NAME=${WANDB_RUN_NAME:-} \
${WANDB_RUN_ID:+--env WANDB_RUN_ID=$WANDB_RUN_ID} \
${WANDB_RESUME:+--env WANDB_RESUME=$WANDB_RESUME} \
--env WANDB_TEAM=${WANDB_TEAM:-} \
--env MASTER_ADDR=$MASTER_ADDR \
--env MASTER_PORT=$MASTER_PORT \
--env TORCH_HOME=$TORCH_HOME \
--env PYTORCH_KERNEL_CACHE_PATH=$TORCH_HOME/kernels \
--env TRITON_CACHE_DIR=$TRITON_CACHE_DIR \
--env TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR \
--env TORCHINDUCTOR_FX_GRAPH_CACHE=$TORCHINDUCTOR_FX_GRAPH_CACHE \
--env TORCHINDUCTOR_AUTOGRAD_CACHE=$TORCHINDUCTOR_AUTOGRAD_CACHE \
--env TORCH_INDUCTOR_CUDAGRAPH_DISABLE=$TORCH_INDUCTOR_CUDAGRAPH_DISABLE \
--env PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF \
--env LD_LIBRARY_PATH=/opt/nvidia/lib:/usr/local/cuda/lib64 \
--env LIBRARY_PATH=/opt/nvidia/lib:/usr/local/cuda/lib64 \
$( [[ -f /usr/lib64/libcuda.so.1 ]] && echo "--env TRITON_LIBCUDA_PATH=/opt/nvidia/lib/libcuda.so.1" ) \
--env PYTHONPATH=/opt/rasr:/opt/rasr/torchtitan \
--bind $PROJECT_DIR:/opt/rasr \
--bind $TRITON_CACHE_DIR:$TRITON_CACHE_DIR \
--bind $HF_DATASETS_CACHE:$HF_DATASETS_CACHE \
--bind $TORCH_HOME:$TORCH_HOME \
$( [[ -f /usr/lib64/libcuda.so.1 ]]              && echo "--bind /usr/lib64/libcuda.so.1:/opt/nvidia/lib/libcuda.so.1:ro" ) \
$( [[ -f /usr/lib64/libcuda.so ]]                && echo "--bind /usr/lib64/libcuda.so:/opt/nvidia/lib/libcuda.so:ro" ) \
$( [[ -f /usr/lib64/libnvidia-ml.so.1 ]]         && echo "--bind /usr/lib64/libnvidia-ml.so.1:/opt/nvidia/lib/libnvidia-ml.so.1:ro" ) \
$( [[ -f /usr/lib64/libnvidia-ptxjitcompiler.so.1 ]] && echo "--bind /usr/lib64/libnvidia-ptxjitcompiler.so.1:/opt/nvidia/lib/libnvidia-ptxjitcompiler.so.1:ro" ) \
--nv $PROJECT_DIR/$CONTAINER "

# Generate RASR training arguments
RASR_DATA_PATH="${RASR_DATA_PATH:?ERROR: set RASR_DATA_PATH to the Arrow dataset directory}"
echo "INFO: RASR_DATA_PATH=$RASR_DATA_PATH"

CFG_PATH="/opt/rasr/$CONFIG"
if ! $APPTAINER_CMD exec \
    --bind $PROJECT_DIR:/opt/rasr \
    $PROJECT_DIR/$CONTAINER \
    python3 -c "
import tomllib
with open('$CFG_PATH', 'rb') as f:
    cfg = tomllib.load(f)
if cfg.get('model', {}).get('name') != 'rasr':
    raise SystemExit('ERROR: CONFIG $CFG_PATH is not a RASR model config (model.name != \"rasr\")')
"; then
    echo "Configuration resolution failed. Aborting."
    exit 1
fi
echo "Configuration resolved successfully."

CLUSTER_ARGS="--job.config_file=$CFG_PATH --dataloader.data_path=$RASR_DATA_PATH --experimental.custom_import=src.rasr"

if [ "$SLURM_NNODES" -eq 1 ]; then
    # Single-node launch: avoid external rendezvous endpoint/socket issues.
    LAUNCHER="torchrun \
        --standalone \
        --nproc_per_node $NUM_GPUS_PER_NODE \
        --max_restarts 3"
else
    LAUNCHER="torchrun \
        --nnodes $SLURM_NNODES \
        --nproc_per_node $NUM_GPUS_PER_NODE \
        --node_rank $SLURM_PROCID \
        --max_restarts 3 \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        --rdzv_backend c10d \
        --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT"
fi

srun $SRUN_ARGS $APPTAINER bash -c "
    echo SLURM_LOCALID \$SLURM_LOCALID
    echo SLURM_PROCID \$SLURM_PROCID \$SLURM_NTASKS
    exec $LAUNCHER -m torchtitan.train $CLUSTER_ARGS $*
"