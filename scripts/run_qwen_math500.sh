#!/usr/bin/env bash
set -euo pipefail

# bash scripts/run_qwen_math500.sh

# ===== Math500 experiment settings =====
GPUS="1,2,3"
NUM_GPUS=3
MASTER_PORT=29502
# PYTHON_SCRIPT="visualize_reward_landscape.py"
PYTHON_SCRIPT="visualize_reward_landscape_paral.py"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_1.7b_grpo_math500_train300_test200_max512_full/v3-20260805-223143/checkpoint-10"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_1.7b_grpo_math500_train300_test200_max512_full/v0-20260805-180507/checkpoint-225"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_base_nonthink_grpo_math500_lr_5e-6_max1024/v1-20260809-141301/checkpoint-25"  
MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_base_nonthink_grpo_math500_lr_5e-6_max1024/v1-20260809-141301/checkpoint-148"  

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_base_nonthink_grpo_math500_lr_5e-6_max1024/v2-20260809-160437/checkpoint-275"
# MODEL_CKPT="/tmp/erin/Qwen3-0.6B-Base"
# MODEL_CKPT="/mnt/sj/home/yichen/ms-swift/output_qwen3_1.7b_grpo_gsm8k_full/v1-20260625-011426/checkpoint-600"
# MODEL_CKPT="/mnt/sj/home/yichen/ms-swift/output_qwen3_1.7b_grpo_gsm8k_full/v1-20260625-011426/checkpoint-600"
# MODEL_CKPT="/mnt/sj/home/yichen/ms-swift/output_qwen3_1.7b_dapo_gsm8k_train200_test100_max256_full/v1-20260729-142241/checkpoint-25"

# EVAL_JSON="/mnt/sj/home/yichen/landscape_rlvr/train_data/math500/test/test.jsonl"
EVAL_JSON="/mnt/swordfish-pool2/erinxia/rlvr-landscape/train_data/math500/train.jsonl"

TASK="math500"
RL_TYPE="grpo"
DATASET="math500"
MODEL_NAME="qwen3_0.6b"
CHECKPOINT_STEP=148

NUM_SAMPLES=80
BATCH_SIZE=22
LR="5e-6"
MAX_NEW_TOKENS=1024
SEED=42
SCALE=0.01
ALPHA_RANGE=6

NUM_POINTS=41
GROUP_SIZE=4
TMP=1.0
TOP_P=1.0

DIR_PATH="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_base_nonthink_grpo_math500_lr_5e-6_max1024/direction_148_to_150.pt"

NUM_DIRECTIONS=1
YMIN=0.0
YMAX=1.0
OUTPUT_DIR="figs_math500_grpo_new"

RUN_NAME="${RL_TYPE}_${LR}_${DATASET}_${MODEL_NAME}_scale${SCALE}_alpha${ALPHA_RANGE}_data${NUM_SAMPLES}_group${GROUP_SIZE}_ckpt${CHECKPOINT_STEP}_max_new${MAX_NEW_TOKENS}_bs${BATCH_SIZE}_temp${TMP}_topp${TOP_P}_pts${NUM_POINTS}_seed${SEED}_${NUM_DIRECTIONS}dirs"
LOG_DIR="logs_math500_grpo_new"
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
    --group-size "${GROUP_SIZE}" \
    --temperature "${TMP}" \
    --top-p "${TOP_P}" \
    --batch-size "${BATCH_SIZE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    --scale "${SCALE}" \
    --alpha-range "${ALPHA_RANGE}" \
    --num-points "${NUM_POINTS}" \
    --direction-path "${DIR_PATH}" \
    --num-directions "${NUM_DIRECTIONS}" \
    --output-dir "${OUTPUT_DIR}" \
    --ymin "${YMIN}" \
    --ymax "${YMAX}" \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Started PID ${PID}"
echo "Follow log with: tail -f ${LOG_FILE}"