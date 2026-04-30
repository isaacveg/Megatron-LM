#!/usr/bin/env bash

set -uo pipefail

RUN_DIR="${1:?missing run dir}"

mkdir -p "$RUN_DIR"

printf '%s\n' "$$" > "$RUN_DIR/pid"
date --iso-8601=seconds > "$RUN_DIR/start_time"

{
    echo "job_id=${JOB_ID:-}"
    echo "run_id=${RUN_ID:-}"
    echo "train_script=${TRAIN_SCRIPT:-}"
    echo "run_name=${RUN_NAME:-}"
    echo "ascend_rt_visible_devices=${ASCEND_RT_VISIBLE_DEVICES:-}"
    echo "npu_visible_devices=${NPU_VISIBLE_DEVICES:-}"
    echo "npus_per_node=${NPUS_PER_NODE:-}"
    echo "master_port=${MASTER_PORT:-}"
    echo "ckpt_save_dir=${CKPT_SAVE_DIR:-}"
    echo "ckpt_load_dir=${CKPT_LOAD_DIR:-}"
    echo "train_iters=${TRAIN_ITERS:-}"
    echo "save_interval=${SAVE_INTERVAL:-}"
    echo "log_file=${LOG_FILE:-}"
} > "$RUN_DIR/env_summary"

finish() {
    local status="$1"
    printf '%s\n' "$status" > "$RUN_DIR/exit_code"
    date --iso-8601=seconds > "$RUN_DIR/end_time"
}

on_term() {
    echo "received termination signal"
    finish 143
    exit 143
}

trap on_term TERM INT

if [ -z "${TRAIN_SCRIPT:-}" ]; then
    echo "TRAIN_SCRIPT is empty"
    finish 2
    exit 2
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "missing train script: $TRAIN_SCRIPT"
    finish 2
    exit 2
fi

echo "starting training script: $TRAIN_SCRIPT"
echo "launcher log: ${LAUNCHER_LOG_FILE:-}"
echo "training log: ${LOG_FILE:-}"
echo "run dir: $RUN_DIR"

set +e
bash "$TRAIN_SCRIPT"
status=$?
set -e

echo "training script exited with status $status"
finish "$status"
exit "$status"
