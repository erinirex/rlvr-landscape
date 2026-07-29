#!/usr/bin/env bash
set -euo pipefail

# bash scripts/run_qwen_gsm8k.sh

# ===== GSM8K experiment settings =====
GPU=3
PYTHON_SCRIPT="visualize_reward_landscape.py"

MODEL_CKPT="/mnt/sj/home/yichen/ms-swift/output_qwen3_1.7b_grpo_gsm8k_train200_test100_max128_full/v0-20260724-144147/checkpoint-200"
# MODEL_CKPT="/mnt/sj/home/yichen/ms-swift/output_qwen3_1.7b_grpo_gsm8k_full/v1-20260625-011426/checkpoint-600"
# MODEL_CKPT="/mnt/sj/home/yichen/ms-swift/output_qwen3_1.7b_grpo_gsm8k_full/v1-20260625-011426/checkpoint-600"
EVAL_JSON="/mnt/sj/home/yichen/landscape_rlvr/train_data/gsm8k_swift/train_sample200.jsonl"
# EVAL_JSON="/mnt/sj/home/yichen/landscape_rlvr/train_data/gsm8k_swift/train.jsonl"

TASK="gsm8k"
RL_TYPE="grpo"
DATASET="gsm8k_sample200"
MODEL_NAME="qwen3_1.7b"
CHECKPOINT_STEP=200

NUM_SAMPLES=200
BATCH_SIZE=200
MAX_NEW_TOKENS=512
SEED=42
SCALE=0.01
ALPHA_RANGE=9
NUM_POINTS=21
NUM_DIRECTIONS=2
YMIN=0.0
YMAX=1.0
OUTPUT_DIR="figs_gsm8k_new"

RUN_NAME="${RL_TYPE}_${DATASET}_scale${SCALE}_alpha${ALPHA_RANGE}_data${NUM_SAMPLES}_ckpt${CHECKPOINT_STEP}_max_new${MAX_NEW_TOKENS}_bs${BATCH_SIZE}_pts${NUM_POINTS}_seed${SEED}_${NUM_DIRECTIONS}dirs"
LOG_DIR="logs_gsm8k"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

echo "Starting ${RUN_NAME}"
echo "Log: ${LOG_FILE}"

nohup env CUDA_VISIBLE_DEVICES="${GPU}" \
  python "${PYTHON_SCRIPT}" \
    --model-ckpt "${MODEL_CKPT}" \
    --eval-json "${EVAL_JSON}" \
    --task "${TASK}" \
    --rl-type "${RL_TYPE}" \
    --dataset "${DATASET}" \
    --model-name "${MODEL_NAME}" \
    --checkpoint-step "${CHECKPOINT_STEP}" \
    --num-samples "${NUM_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    --scale "${SCALE}" \
    --alpha-range "${ALPHA_RANGE}" \
    --num-points "${NUM_POINTS}" \
    --num-directions "${NUM_DIRECTIONS}" \
    --output-dir "${OUTPUT_DIR}" \
    --ymin "${YMIN}" \
    --ymax "${YMAX}" \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Started PID ${PID}"
echo "Follow log with: tail -f ${LOG_FILE}"