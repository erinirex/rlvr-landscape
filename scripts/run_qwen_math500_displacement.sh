#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export VLLM_LOGGING_LEVEL=DEBUG
export NCCL_DEBUG=INFO
export NCCL_SOCKET_FAMILY=AF_INET

# bash scripts/run_qwen_math500_displacement.sh

# ============================================================
# GPU settings
# ============================================================

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# IMPORTANT:
# Python script is configured for 2-GPU vLLM TP=2
GPU="6"

PYTHON_SCRIPT="visualize_reward_landscape_displace.py"
# PYTHON_SCRIPT="visualize_reward_landscape_displace_multigpu.py"

# ============================================================
# Model
# ============================================================

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_5e-6_max2048/v7-20260903-071819/checkpoint-75"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_5e-6_max2048/v18-20260903-125040/checkpoint-49"
MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_3e-6_max2048/v0-20260910-192151/checkpoint-119"

# DIRECTION_PATH="/mnt/swordfish-pool2/erinxia/rlvr-landscape/param_displace/displacement_49_to_50.pt"
DIRECTION_PATH="/mnt/swordfish-pool2/erinxia/rlvr-landscape/param_displace/displacement_lr_3e-6_119_to_120.pt"

EVAL_JSON="/mnt/swordfish-pool2/erinxia/rlvr-landscape/train_data/math500/train.jsonl"

MODEL_NAME="qwen3_0.6b"
CHECKPOINT_STEP=119
DIR_STEP=120

TASK="math500"

# ============================================================
# Evaluation
# ============================================================

NUM_SAMPLES=100

MAX_NEW_TOKENS=2048

SEED=42

# ============================================================
# GRPO rollout
# ============================================================

PG_NUM_PROMPTS=16
GROUP_SIZE=4

# Sampling parameters
TEMPERATURE=0.7
TOP_P=0.8
TOP_K=20

# ============================================================
# Directions
# ============================================================

NUM_DIRECTIONS=1
DIRECTION_TYPE="displacement"

# ============================================================
# Landscape
# ============================================================

SCALE=0.01
ALPHA_RANGE=20
ALPHA_LEFT=70000
ALPHA_RIGHT=70000

NUM_POINTS=21

# ============================================================
# vLLM
# ============================================================

TENSOR_PARALLEL_SIZE=1

GPU_MEMORY_UTILIZATION=0.9

MAX_MODEL_LEN=16384

DTYPE="bfloat16"

# ============================================================
# Output
# ============================================================

OUTPUT_DIR="figs_math500_grpo_displacement"

LOG_DIR="logs_math500_grpo_displacement"

RUN_NAME="${DIRECTION_TYPE}"\
"_${MODEL_NAME}"\
"_dir${NUM_DIRECTIONS}"\
"_scale${SCALE}_left${ALPHA_LEFT}_right${ALPHA_RIGHT}"\
"_prompts${PG_NUM_PROMPTS}_group${GROUP_SIZE}"\
"_ckpt${CHECKPOINT_STEP}_to${DIR_STEP}"\
"_max_new${MAX_NEW_TOKENS}"\
"_temp${TEMPERATURE}_topp${TOP_P}_topk${TOP_K}"\
"_pts${NUM_POINTS}_seed${SEED}"\
"_gpu${GPU//,/}"\
"_vllm"

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

echo "DIRECTION_TYPE: ${DIRECTION_TYPE}"
echo "NUM_DIRECTIONS: ${NUM_DIRECTIONS}"

echo "GROUP_SIZE: ${GROUP_SIZE}"

echo "MAX_NEW_TOKENS: ${MAX_NEW_TOKENS}"

echo "TENSOR_PARALLEL_SIZE: ${TENSOR_PARALLEL_SIZE}"
echo "MAX_MODEL_LEN: ${MAX_MODEL_LEN}"
echo "GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION}"
echo "DTYPE: ${DTYPE}"

echo "TEMPERATURE: ${TEMPERATURE}"
echo "TOP_P: ${TOP_P}"
echo "TOP_K: ${TOP_K}"

echo "SCALE: ${SCALE}"
echo "ALPHA_LEFT: ${ALPHA_LEFT}"
echo "ALPHA_RIGHT: ${ALPHA_RIGHT}"
echo "NUM_POINTS: ${NUM_POINTS}"

echo "SEED: ${SEED}"

echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"

echo "=========================================="

# ============================================================
# Run
# ============================================================

nohup env \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  python -u "${PYTHON_SCRIPT}" \
    --model-ckpt "${MODEL_CKPT}" \
    --eval-json "${EVAL_JSON}" \
    --model-name "${MODEL_NAME}" \
    --task "${TASK}" \
    --direction-path "${DIRECTION_PATH}" \
    --checkpoint-step "${CHECKPOINT_STEP}" \
    --dir-step "${DIR_STEP}" \
    \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --dtype "${DTYPE}" \
    \
    --num-samples "${NUM_SAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    \
    --scale "${SCALE}" \
    --alpha-left "${ALPHA_LEFT}" \
    --alpha-right "${ALPHA_RIGHT}" \
    --num-points "${NUM_POINTS}" \
    --num-directions "${NUM_DIRECTIONS}" \
    --direction-type "${DIRECTION_TYPE}" \
    \
    --pg-num-prompts "${PG_NUM_PROMPTS}" \
    --group-size "${GROUP_SIZE}" \
    \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    \
    --output-dir "${OUTPUT_DIR}" \
  > "${LOG_FILE}" 2>&1 &

PID=$!

echo "Started PID ${PID}"
echo "Log: ${LOG_FILE}"
echo
echo "Follow log with:"
echo "tail -f ${LOG_FILE}"
