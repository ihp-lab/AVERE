#!/usr/bin/env bash
# =============================================================================
# eval.sh — Multi-GPU parallel evaluation with task and GPU selection
#
# Usage:
#   bash eval.sh [OPTIONS]
#
# Options:
#   -g, --gpus  <ids>    Comma-separated physical GPU IDs  (default: 0,1,2,3,4,5,6,7)
#   -t, --tasks <names>  Comma-separated task names        (default: emotion_qa-emorealm)
#   -m, --model <path>   Model checkpoint path
#   -b, --batch <size>   Batch size per GPU                (default: 1)
#   -l, --logdir <dir>   Log output directory              (default: logs_multi_gpu)
#   -h, --help           Show this help message
#
# Available tasks:
#   emotion_qa-emorealm
#   emotion-dfew-audio
#   emotion-ravdess-video-audio
#
# Examples:
#   bash eval.sh --gpus 0,1,2,3
#   bash eval.sh -g 4,5 -t emotion-dfew-audio,emotion-ravdess-video-audio
#   bash eval.sh -g 0,2,4,6 -t emotion_qa-emorealm -b 2
# =============================================================================
set -uo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_GPUS="0,1,2,3,4,5,6,7"
DEFAULT_TASKS="emotion_qa-emorealm"
DEFAULT_MODEL="/wekafs/ict/achaubey/emotion_reasoning/audio_exp/Video-LLaVA/checkpoints/emotion_qa/finetune-base_videollava_7b-mm_proj_vidllava-speech_proj_qformer-mafw_ferv39k_mer2025_single_descraw_avlong-qa_mafw_ferv39k_mer2025_single-250words-modality_hallucination0.05_qa_extras_shuffled_choices"
DEFAULT_BATCH=1
DEFAULT_LOGDIR="logs_multi_gpu"

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

GPUS="$DEFAULT_GPUS"
TASKS="$DEFAULT_TASKS"
MODEL_PATH="$DEFAULT_MODEL"
BATCH_SIZE="$DEFAULT_BATCH"
LOG_DIR="$DEFAULT_LOGDIR"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--gpus)   GPUS="$2";       shift 2 ;;
        -t|--tasks)  TASKS="$2";      shift 2 ;;
        -m|--model)  MODEL_PATH="$2"; shift 2 ;;
        -b|--batch)  BATCH_SIZE="$2"; shift 2 ;;
        -l|--logdir) LOG_DIR="$2";    shift 2 ;;
        -h|--help)   usage ;;
        *) echo "[eval.sh] Unknown option: $1"; usage ;;
    esac
done

# ── Parse GPU and task lists ──────────────────────────────────────────────────
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
IFS=',' read -ra TASK_ARRAY <<< "$TASKS"
NUM_GPUS=${#GPU_ARRAY[@]}

# ── Signal handling: kill every worker on Ctrl+C or any child failure ─────────
WORKER_PIDS=()

cleanup() {
    echo ""
    echo "[eval.sh] Stopping — sending SIGTERM to all workers..."
    for pid in "${WORKER_PIDS[@]}"; do
        # Kill the entire process group of each worker
        kill -- "-$pid" 2>/dev/null || kill -- "$pid" 2>/dev/null || true
    done
    # Give them a moment, then SIGKILL stragglers
    sleep 1
    for pid in "${WORKER_PIDS[@]}"; do
        kill -9 -- "-$pid" 2>/dev/null || kill -9 -- "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "[eval.sh] All workers stopped."
    exit 1
}

trap cleanup SIGINT SIGTERM

# ── Wait for all workers; abort everything if any one fails ───────────────────
wait_all_or_abort() {
    local task="$1"
    shift
    local pids=("$@")
    local remaining=("${pids[@]}")

    while [[ ${#remaining[@]} -gt 0 ]]; do
        local still_running=()
        for pid in "${remaining[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                still_running+=("$pid")
            else
                # Process ended — collect its exit code
                wait "$pid" 2>/dev/null
                local status=$?
                if [[ $status -ne 0 ]]; then
                    echo "[eval.sh] Worker PID $pid (task: $task) exited with status $status — aborting all."
                    WORKER_PIDS=("${pids[@]}")
                    cleanup
                fi
            fi
        done
        remaining=("${still_running[@]+"${still_running[@]}"}")
        [[ ${#remaining[@]} -gt 0 ]] && sleep 2
    done
}

# ── Summary ───────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════"
echo " eval.sh — Multi-GPU Evaluation"
echo "════════════════════════════════════════"
echo " GPUs      : ${GPU_ARRAY[*]}"
echo " Num GPUs  : $NUM_GPUS"
echo " Tasks     : ${TASK_ARRAY[*]}"
echo " Batch size: $BATCH_SIZE"
echo " Log dir   : $LOG_DIR"
echo " Model     : $(basename "$MODEL_PATH")"
echo "════════════════════════════════════════"
echo ""

mkdir -p "$LOG_DIR"

# ── Main loop: one task at a time, all GPUs in parallel ───────────────────────
for TASK in "${TASK_ARRAY[@]}"; do
    echo "[eval.sh] ── Starting task: $TASK ──"
    WORKER_PIDS=()

    for idx in "${!GPU_ARRAY[@]}"; do
        GPU_ID="${GPU_ARRAY[$idx]}"
        LOG_FILE="$LOG_DIR/gpu${GPU_ID}_${TASK}.out"

        # setsid gives each worker its own process group so SIGTERM propagates cleanly
        # env is required because setsid does not interpret shell variable assignments
        setsid env CUDA_VISIBLE_DEVICES="$GPU_ID" \
        python evaluate/main.py \
            --model_path="$MODEL_PATH" \
            --task="$TASK" \
            --batch_size="$BATCH_SIZE" \
            --multi_gpu_split \
            --num_gpus="$NUM_GPUS" \
            --gpu_split_idx="$idx" \
            > "$LOG_FILE" 2>&1 &

        pid=$!
        WORKER_PIDS+=("$pid")
        echo "[eval.sh]   GPU $GPU_ID | split_idx $idx | PID $pid → $LOG_FILE"
    done

    echo ""
    wait_all_or_abort "$TASK" "${WORKER_PIDS[@]}"
    echo "[eval.sh] Task '$TASK' finished successfully."
    echo ""
done

echo "[eval.sh] All tasks completed."
