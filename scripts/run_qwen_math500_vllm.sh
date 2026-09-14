#!/usr/bin/env bash
set -euo pipefail

# bash scripts/run_qwen_math500_vllm.sh

# ============================================================
# GPU settings
# ============================================================
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
GPU="2"
PYTHON_SCRIPT="visualize_reward_landscape_vllm.py"

# ============================================================
# Model
# ============================================================

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train100_lr_5e-6_max8192/v1-20260821-105102/checkpoint-100"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_5e-6_max2048/v7-20260903-071819/checkpoint-275"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_5e-6_max2048/v7-20260903-071819/checkpoint-75"
MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_3e-6_max2048/v0-20260910-192151/checkpoint-50"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_3e-6_max2048/v0-20260910-192151/checkpoint-117"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_5e-6_max2048/v18-20260903-125040/checkpoint-49"

EVAL_JSON="/mnt/swordfish-pool2/erinxia/rlvr-landscape/train_data/math500/train.jsonl"

TASK="math500"
RL_TYPE="grpo"
DATASET="math500"
MODEL_NAME="qwen3_0.6b"
LR="3e-6"
# CHECKPOINT_STEP=275
CHECKPOINT_STEP=50
# CHECKPOINT_STEP=117


# ============================================================
# Evaluation
# ============================================================

NUM_SAMPLES=100

MAX_NEW_TOKENS=2048

SEED=42

# 每个 prompt 生成多少个 samples
GROUP_SIZE=4

TMP=0.7
TOP_P=0.8
TOP_K=20

# ============================================================
# Random directions
# ============================================================

NUM_DIRECTIONS=4

# ============================================================
# Landscape
# ============================================================

SCALE=0.01
ALPHA_RANGE=12
NUM_POINTS=21

# ============================================================
# vLLM settings
# ============================================================

GPU_MEMORY_UTILIZATION=0.5
MAX_NUM_SEQS=8
MAX_MODEL_LEN=16384

# ============================================================
# Plot
# ============================================================

YMIN=0.0
YMAX=1.1

OUTPUT_DIR="figs_math500_grpo_random_lr_${LR}_vllm"

RUN_NAME="${RL_TYPE}_${DATASET}_${MODEL_NAME}"\
"_random${NUM_DIRECTIONS}"\
"_scale${SCALE}_alpha${ALPHA_RANGE}"\
"_lr${LR}"\
"_data${NUM_SAMPLES}_group${GROUP_SIZE}"\
"_ckpt${CHECKPOINT_STEP}"\
"_max_new${MAX_NEW_TOKENS}"\
"_temp${TMP}_topp${TOP_P}"\
"_pts${NUM_POINTS}_seed${SEED}"\
"_gpu${GPU}"\
"_vllm"

LOG_DIR="logs_math500_grpo_random_lr_${LR}_vllm"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

# ============================================================
# Print configuration
# ============================================================

echo "=========================================="
echo "Starting ${RUN_NAME}"
echo "=========================================="
echo "GPU: ${GPU}"
echo "MODEL: ${MODEL_CKPT}"
echo "NUM_DIRECTIONS: ${NUM_DIRECTIONS}"
echo "NUM_SAMPLES: ${NUM_SAMPLES}"
echo "GROUP_SIZE: ${GROUP_SIZE}"
echo "MAX_NEW_TOKENS: ${MAX_NEW_TOKENS}"
echo "MAX_MODEL_LEN: ${MAX_MODEL_LEN}"
echo "MAX_NUM_SEQS: ${MAX_NUM_SEQS}"
echo "GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION}"
echo "TEMPERATURE: ${TMP}"
echo "TOP_P: ${TOP_P}"
echo "TOP_K: ${TOP_K}"
echo "SCALE: ${SCALE}"
echo "ALPHA_RANGE: ${ALPHA_RANGE}"
echo "NUM_POINTS: ${NUM_POINTS}"
echo "SEED: ${SEED}"
echo "Log: ${LOG_FILE}"
echo "=========================================="

# ============================================================
# Run
# ============================================================

nohup env \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  python "${PYTHON_SCRIPT}" \
    --model-ckpt "${MODEL_CKPT}" \
    --eval-json "${EVAL_JSON}" \
    --task "${TASK}" \
    --lr "${LR}" \
    --rl-type "${RL_TYPE}" \
    --dataset "${DATASET}" \
    --model-name "${MODEL_NAME}" \
    --checkpoint-step "${CHECKPOINT_STEP}" \
    --num-samples "${NUM_SAMPLES}" \
    --group-size "${GROUP_SIZE}" \
    --temperature "${TMP}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    --scale "${SCALE}" \
    --alpha-range "${ALPHA_RANGE}" \
    --num-points "${NUM_POINTS}" \
    --num-directions "${NUM_DIRECTIONS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --output-dir "${OUTPUT_DIR}" \
    --ymin "${YMIN}" \
    --ymax "${YMAX}" \
  > "${LOG_FILE}" 2>&1 &

PID=$!

echo "Started PID ${PID}"
echo "Log: ${LOG_FILE}"
echo
echo "Follow log with:"
echo "tail -f ${LOG_FILE}"