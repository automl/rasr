#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_CONFIG="src/rasr/train_configs/post-train.toml"

# --resume <run-dir>: continue from the latest checkpoint in an existing post-training run
if [[ "${1:-}" == "--resume" ]]; then
    shift
    RESUME_DIR="${1:?ERROR: --resume requires a run directory}"
    shift

    [[ "$RESUME_DIR" = /* ]] || RESUME_DIR="$REPO_ROOT/$RESUME_DIR"
    [[ -d "$RESUME_DIR" ]] || { echo "ERROR: resume directory not found: $RESUME_DIR" >&2; exit 1; }

    LATEST_STEP=$(find "$RESUME_DIR" -maxdepth 1 -name "step-*" -type d | sort -V | tail -1)
    LATEST_CHECKPOINT=$(find "$RESUME_DIR" -maxdepth 1 -name "checkpoint-*" -type d | sort -V | tail -1)
    [[ -n "$LATEST_STEP" ]] || { echo "ERROR: no step-* dirs found in $RESUME_DIR" >&2; exit 1; }
    [[ -n "$LATEST_CHECKPOINT" ]] || { echo "ERROR: no checkpoint-* dirs found in $RESUME_DIR" >&2; exit 1; }

    CONFIG="$RESUME_DIR/post_train.toml"
    [[ -f "$CONFIG" ]] || { echo "ERROR: no post_train.toml found in $RESUME_DIR" >&2; exit 1; }

    WANDB_META_DIR=$(find "$RESUME_DIR/wandb" -maxdepth 1 -name "run-*" -type d 2>/dev/null | sort -V | tail -1)
    if [[ -n "$WANDB_META_DIR" ]]; then
        export WANDB_RUN_ID="${WANDB_META_DIR##*-}"
        export WANDB_RESUME="allow"
    fi

    echo "Resuming post-training run: $RESUME_DIR"
    echo "  model weights : $LATEST_STEP"
    echo "  trainer state : $LATEST_CHECKPOINT"
    echo "  W&B run ID    : ${WANDB_RUN_ID:-(not found, new run will be created)}"

    cd "$REPO_ROOT"
    mkdir -p output error
    exec sbatch slurm/post_train.sh "$LATEST_STEP" "$CONFIG" \
        --resume_from_checkpoint "$LATEST_CHECKPOINT" \
        --post_training.output_dir "$RESUME_DIR" \
        "$@"
fi

MODEL_PATH="${1:?ERROR: first arg must be model_path}"
shift

CONFIG="${CONFIG:-$DEFAULT_CONFIG}"

# Optional user-supplied label inserted into the run name (must come before any --)
SUFFIX=""
if [[ "${1:-}" != "" && "${1:-}" != -* ]]; then
    SUFFIX="${1}"
    shift
fi

RUN_NAME_ARGS=()
if [[ -n "$SUFFIX" ]]; then
    RUN_NAME_ARGS=(--name "$SUFFIX")
fi

cd "$REPO_ROOT"
mkdir -p output
mkdir -p error

exec sbatch slurm/post_train.sh "$MODEL_PATH" "$CONFIG" "${RUN_NAME_ARGS[@]}" "$@"
