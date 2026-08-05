#!/usr/bin/env bash
set -euo pipefail

# bash scripts/run_qwen_math500.sh

# ===== Math500 experiment settings =====
GPUS="2,3,4,5"
NUM_GPUS=4
MASTER_PORT=29501
# PYTHON_SCRIPT="visualize_reward_landscape.py"
PYTHON_SCRIPT="visualize_reward_landscape_paral.py"

MODEL_CKPT="/mnt2/public_models/Qwen3-0.6B-Base"
# MODEL_CKPT="/mnt/sj/home/yichen/ms-swift/output_qwen3_1.7b_grpo_gsm8k_full/v1-20260625-011426/checkpoint-600"
# MODEL_CKPT="/mnt/sj/home/yichen/ms-swift/output_qwen3_1.7b_grpo_gsm8k_full/v1-20260625-011426/checkpoint-600"
# MODEL_CKPT="/mnt/sj/home/yichen/ms-swift/output_qwen3_1.7b_dapo_gsm8k_train200_test100_max256_full/v1-20260729-142241/checkpoint-25"

EVAL_JSON="/mnt/sj/home/yichen/landscape_rlvr/train_data/math500/test/test.jsonl"
# EVAL_JSON="/mnt/sj/home/yichen/landscape_rlvr/train_data/gsm8k_swift/train.jsonl"

TASK="math500"
RL_TYPE="base"
DATASET="math500"
MODEL_NAME="qwen3_0.6b_base"
CHECKPOINT_STEP=0

NUM_SAMPLES=500
BATCH_SIZE=64
MAX_NEW_TOKENS=512
SEED=42
SCALE=0.01
ALPHA_RANGE=9
NUM_POINTS=41
NUM_DIRECTIONS=1
YMIN=0.0
YMAX=1.0
OUTPUT_DIR="figs_math500_base"

RUN_NAME="${RL_TYPE}_${DATASET}_${MODEL_NAME}_scale${SCALE}_alpha${ALPHA_RANGE}_data${NUM_SAMPLES}_ckpt${CHECKPOINT_STEP}_max_new${MAX_NEW_TOKENS}_bs${BATCH_SIZE}_pts${NUM_POINTS}_seed${SEED}_${NUM_DIRECTIONS}dirs"
LOG_DIR="logs_math500_base"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

echo "Starting ${RUN_NAME}"
echo "Log: ${LOG_FILE}"

nohup env CUDA_VISIBLE_DEVICES="${GPUS}" \
  torchrun \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${PYTHON_SCRIPT}" \
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