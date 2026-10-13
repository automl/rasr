#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KEY="${REPO_ROOT}/.wandb_key"

[[ -f "$KEY" ]] || { echo "ERROR: missing $KEY" >&2; exit 1; }
export WANDB_API_KEY="$(cat "$KEY")"

export WANDB_PROJECT="${WANDB_PROJECT:-rasr}"
export CONFIG="${CONFIG:-src/rasr/train_configs/pre-train.toml}"

# Parse relevant fields from config
CONFIG_PATH="${REPO_ROOT}/${CONFIG}"
_cfg() { grep -m1 "^${1}\s*=" "$CONFIG_PATH" | sed 's/[^=]*=\s*//' | tr -d ' "'; }

FLAVOR=$(_cfg flavor)
LR=$(_cfg lr)
STEPS=$(_cfg steps)

# Optional user-supplied label inserted between flavor and lr (must come before any --)
SUFFIX=""
if [[ "${1:-}" != "" && "${1:-}" != -* && "${1:-}" != "--resume" && "${1:-}" != "--depends-on-timeout" ]]; then
    SUFFIX="_${1}"
    shift
fi

# Launcher options (must come before model/training overrides)
#   --resume <run-name>
#   --depends-on-timeout <job-id>
RESUME=""
DEPEND_ON_TIMEOUT_JOB=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            shift
            RESUME="${1:?ERROR: --resume requires a run name}"
            shift
            ;;
        --depends-on-timeout)
            shift
            DEPEND_ON_TIMEOUT_JOB="${1:?ERROR: --depends-on-timeout requires a Slurm job id}"
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

# Allow CLI overrides (e.g. --optimizer.lr 1e-3) to also update the run name
_arg() {
    local key="$1"; shift
    local prev=""
    for a in "$@"; do
        [[ "$prev" == "$key" ]] && { echo "$a"; return; }
        prev="$a"
    done
    echo ""
}
[[ -n "$(_arg --model.flavor      "$@")" ]] && FLAVOR=$(_arg --model.flavor      "$@")
[[ -n "$(_arg --optimizer.lr      "$@")" ]] && LR=$(_arg --optimizer.lr      "$@")
[[ -n "$(_arg --training.steps    "$@")" ]] && STEPS=$(_arg --training.steps    "$@")

DATE="$(date +%Y%m%d_%H%M%S)"

if [[ -n "$RESUME" ]]; then
    RUN_NAME="$RESUME"
    DUMP_FOLDER="experiments/runs/$RUN_NAME"
    [[ -d "$REPO_ROOT/$DUMP_FOLDER" ]] || { echo "ERROR: resume folder not found: $REPO_ROOT/$DUMP_FOLDER" >&2; exit 1; }
    # W&B writes run dirs under tb/<date>/wandb/run-<date>-<id>/ inside the dump folder
    WANDB_META_DIR=$(ls -d "$REPO_ROOT/$DUMP_FOLDER/tb/"*/wandb/run-* 2>/dev/null | head -1)
    if [[ -n "$WANDB_META_DIR" ]]; then
        export WANDB_RUN_ID="${WANDB_META_DIR##*-}"
        export WANDB_RESUME="allow"
        echo "Resuming run: $RUN_NAME (W&B id: $WANDB_RUN_ID)"
    else
        echo "WARNING: no wandb run dir found under $DUMP_FOLDER/tb/*/wandb/ — resuming without W&B resume"
        echo "Resuming run: $RUN_NAME (no W&B resume)"
    fi
    export CONFIG="$DUMP_FOLDER/config.toml"
    # Preserve data_path from the saved config so train.sh doesn't overwrite it
    # with its built-in default
    if [[ -z "${RASR_DATA_PATH:-}" ]]; then
        _saved_data_path=$(grep -m1 '^data_path\s*=' "$REPO_ROOT/$CONFIG" 2>/dev/null \
            | sed 's/[^=]*=\s*//' | tr -d ' "')
        if [[ -n "$_saved_data_path" ]]; then
            export RASR_DATA_PATH="$_saved_data_path"
            echo "Resume: using data_path from saved config: $RASR_DATA_PATH"
        fi
    fi
else
    RUN_NAME="${FLAVOR}${SUFFIX}_lr${LR}_s${STEPS}_${DATE}"
    DUMP_FOLDER="experiments/runs/$RUN_NAME"
fi

export WANDB_RUN_NAME="$RUN_NAME"

cd "$REPO_ROOT"
mkdir -p output
mkdir -p error

SBATCH_ARGS=()
if [[ -n "$DEPEND_ON_TIMEOUT_JOB" ]]; then
    SBATCH_ARGS+=("--dependency=afternotok:${DEPEND_ON_TIMEOUT_JOB}")
    echo "Submitting with dependency: afternotok:${DEPEND_ON_TIMEOUT_JOB}"
fi

exec sbatch "${SBATCH_ARGS[@]}" slurm/train.sh --job.dump_folder "$DUMP_FOLDER" "$@"
