#!/bin/bash -x
#SBATCH -p <gpu-partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=rasr-grpo
#SBATCH --threads-per-core=1
#SBATCH --time=24:00:00
#SBATCH -D .
#SBATCH --output=output/rasr-grpo-out.%j
#SBATCH --error=error/rasr-grpo-err.%j

NUM_GPUS=4

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"

MODEL_PATH="${1:?ERROR: first arg must be model_path}"
shift
DEFAULT_CONFIG="src/rasr/train_configs/post-train.toml"
if [[ "${1:-}" != "" && "${1:-}" != -* ]]; then
    CONFIG="$1"
    shift
else
    CONFIG="$DEFAULT_CONFIG"
fi

resolve_path() {
    local p="$1"
    if [[ "$p" = /* ]]; then
        echo "$p"
    else
        echo "$PROJECT_DIR/$p"
    fi
}

MODEL_PATH_RESOLVED="$(resolve_path "$MODEL_PATH")"
CONFIG_RESOLVED="$(resolve_path "$CONFIG")"

# VENV
VENV_PATH="$PROJECT_DIR/.venv"
[[ -f "$VENV_PATH/bin/activate" ]] || { echo "ERROR: missing $VENV_PATH/bin/activate" >&2; exit 1; }
source "$VENV_PATH/bin/activate"

# WANDB
KEY="${PROJECT_DIR}/.wandb_key"
[[ -f "$KEY" ]] || { echo "ERROR: missing $KEY" >&2; exit 1; }
export WANDB_API_KEY="$(cat "$KEY")"
export WANDB_PROJECT="${WANDB_PROJECT:-rasr-grpo}"

mkdir -p "$PROJECT_DIR/output"

# LAUNCH
PORT_BASE=20000
export MASTER_PORT=$((PORT_BASE + SLURM_JOB_ID % 10000))

accelerate launch \
    --num_processes=$NUM_GPUS \
    --main_process_port=$MASTER_PORT \
    -m src.rasr.post_train.run \
    --model_path "$MODEL_PATH_RESOLVED" \
    --config "$CONFIG_RESOLVED" \
    "$@"
