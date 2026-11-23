#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export VLLM_LOGGING_LEVEL=DEBUG
export NCCL_DEBUG=INFO
export NCCL_SOCKET_FAMILY=AF_INET
export VLLM_LOGGING_LEVEL=INFO

# bash scripts/run_qwen_math500_displacement.sh
# nohup bash scripts/run_qwen_math500_displacement.sh > seed_loop_gs4.log 2>&1 &
# ============================================================
# GPU settings
# ============================================================

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# IMPORTANT:
# Python script is configured for 2-GPU vLLM TP=2
GPU="0"
TRAIN_SAMPLE=32
TRAIN_SEED=50
TRAIN_GROUP_SIZE=4
LR="4e-6"
# TRAIN_GROUP_SIZE=4
# PYTHON_SCRIPT="visualize_reward_landscape_displace.py"
# PYTHON_SCRIPT="visualize_reward_landscape_displace_no_accm_error.py"
PYTHON_SCRIPT="visualize_reward_landscape_displace_no_accm_error_norm.py"

CHECKPOINT_STEP=0
DIR_STEP=170

# PYTHON_SCRIPT="visualize_reward_landscape_displace_multigpu.py"

# ============================================================
# Model
# ============================================================

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_5e-6_max2048/v7-20260903-071819/checkpoint-75"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_5e-6_max2048/v18-20260903-125040/checkpoint-49"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train1_lr_5e-6_max2048/v0-20260916-223901/checkpoint-103"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train8_lr_5e-6_max2048/v2-20260917-200704/checkpoint-109"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train1_seed42_gs4_lr_5e-6_max2048/v0-20261001-233655/checkpoint-40"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train1_seed42_gs4_lr_5e-6_max2048/v0-20261001-233655/checkpoint-10"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train1_seed42_gs16_lr_5e-6_max2048/v0-20261002-001611/checkpoint-10"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train1_seed42_gs16_lr_5e-6_max2048/v0-20261002-001611/checkpoint-60"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train8_50_80_lr_5e-6_max2048/v1-20260925-122541/checkpoint-10"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32_seed99_gs4_lr_4e-6_max2048/v2-20261009-005817/checkpoint-${CHECKPOINT_STEP}"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32_seed99_gs16_lr_4e-6_max2048/v0-20261009-105123/checkpoint-${CHECKPOINT_STEP}"
MODEL_CKPT="/tmp/erin/Qwen3-0.6B"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32_seed99_gs16_lr_6e-6_max2048/v0-20261014-183522/checkpoint-${CHECKPOINT_STEP}"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32_seed99_gs32_lr_4e-6_max2048/v0-20261014-210623/checkpoint-${CHECKPOINT_STEP}"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32_seed99_gs4_lr_4e-6_max2048/v3-20261009-015503/checkpoint-221"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train1_seed50_gs4_lr_5e-6_max2048/v0-20261001-233843/checkpoint-20"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train1_seed50_gs16_lr_5e-6_max2048/v0-20261002-001611/checkpoint-20"

# DIRECTION_PATH="/mnt/swordfish-pool2/erinxia/rlvr-landscape/param_displace/displacement_49_to_50.pt"
# DIRECTION_PATH="/mnt/swordfish-pool2/erinxia/rlvr-landscape/param_displace/displacement_lr_5e-6_109_to_110.pt"
# DIRECTION_PATH="/mnt/swordfish-pool2/erinxia/rlvr-landscape/param_displace/displacement_train1_gs4_seed_42_lr_5e-6_40_to_80.pt"
# DIRECTION_PATH="/mnt/swordfish-pool2/erinxia/rlvr-landscape/param_displace_train1_gs4/displacement_train1_gs4_seed_42_lr_5e-6_20_to_40.pt"
# DIRECTION_PATH="/mnt/swordfish-pool2/erinxia/rlvr-landscape/param_displace_train1_gs4/displacement_train1_gs4_seed_42_lr_5e-6_20_to_60.pt"
# DIRECTION_PATH="/mnt/swordfish-pool2/erinxia/rlvr-landscape/param_displace_train1_gs4/displacement_train1_gs4_seed_42_lr_5e-6_10_to_80.pt"
# DIRECTION_PATH="/mnt/swordfish-pool2/erinxia/rlvr-landscape/param_displace_train1_gs16/displacement_train1_gs16_seed_42_lr_5e-6_10_to_80.pt"
# DIRECTION_PATH="/mnt/swordfish-pool2/erinxia/rlvr-landscape/param_displace_train1_gs16/displacement_train1_gs16_seed_50_lr_5e-6_20_to_40.pt"

# DIRECTION_PATH="/mnt/swordfish-pool2/erinxia/rlvr-landscape/param_displace_train8_gs4/displacement_train8_gs4_seed_42_lr_5e-6_10_to_50.pt"

# DIRECTION_PATH="/mnt/swordfish-pool2/erinxia/rlvr-landscape/param_displace_train32_gs4/displacement_train32_gs4_seed_99_lr_4e-6_221_to_250.pt"
# DIRECTION_PATH="/mnt/swordfish-pool2/erinxia/rlvr-landscape/param_displace_train32_gs4/displacement_train32_gs4_seed_99_lr_4e-6_120_to_221.pt"
# DIRECTION_PATH="/mnt/swordfish-pool2/erinxia/rlvr-landscape/param_displace_train32_gs${TRAIN_GROUP_SIZE}/displacement_train32_gs${TRAIN_GROUP_SIZE}_seed_99_lr_4e-6_${CHECKPOINT_STEP}_to_${DIR_STEP}.pt"

# DIRECTION_PATH="/mnt/swordfish-pool2/erinxia/rlvr-landscape/param_displace_train32_gs${TRAIN_GROUP_SIZE}_norm/direction_train32_gs${TRAIN_GROUP_SIZE}_seed_99_lr_${LR}_${CHECKPOINT_STEP}_to_${DIR_STEP}.pt"

DIRECTION_PATH="/mnt/swordfish-pool2/erinxia/rlvr-landscape/param_displace_train32_newdata_gs${TRAIN_GROUP_SIZE}_norm/direction_train32_gs${TRAIN_GROUP_SIZE}_seed_99_lr_${LR}_${CHECKPOINT_STEP}_to_${DIR_STEP}.pt"
# DIRECTION_PATH="/mnt/swordfish-pool2/erinxia/rlvr-landscape/param_displace_train1_gs16/displacement_train1_gs16_seed_42_lr_5e-6_20_to_40.pt"
# DIRECTION_PATH="/mnt/swordfish-pool2/erinxia/rlvr-landscape/param_displace_train1_gs16/displacement_train1_gs16_seed_42_lr_5e-6_60_to_80.pt"


# EVAL_JSON="/mnt/swordfish-pool2/erinxia/rlvr-landscape/train_data/math500/train_id299.jsonl"
# EVAL_JSON="/mnt/swordfish-pool2/erinxia/rlvr-landscape/train_data/math500/train_8_50_80.jsonl"
# EVAL_JSON="/mnt/swordfish-pool2/erinxia/rlvr-landscape/train_data/math500/train_32.jsonl"
EVAL_JSON="/mnt/swordfish-pool2/erinxia/rlvr-landscape/train_data/math500/train_32_new.jsonl"


# EVAL_JSON="/mnt/swordfish-pool2/erinxia/rlvr-landscape/train_data/math500/train_1.jsonl"


MODEL_NAME="qwen3_0.6b"

TASK="math500"

# ============================================================
# Evaluation
# ============================================================

NUM_SAMPLES=32

MAX_NEW_TOKENS=2048

SEEDS=(42 99 1001 1002)

# ============================================================
# GRPO rollout
# ============================================================

PG_NUM_PROMPTS=16
GROUP_SIZE=64

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

SCALE=1
ALPHA_RANGE=20
# ALPHA_LEFT=50000e-6
# ALPHA_RIGHT=50000e-6

# ALPHA_LEFT=2.5
# ALPHA_RIGHT=2.5

# ALPHA_LEFT=6
# ALPHA_RIGHT=6

# ALPHA_LEFT=4
# ALPHA_RIGHT=4

ALPHA_LEFT=3
ALPHA_RIGHT=6

# NUM_POINTS=51
NUM_POINTS=25

# NUM_POINTS=9


# ============================================================
# vLLM
# ============================================================

TENSOR_PARALLEL_SIZE=1

GPU_MEMORY_UTILIZATION=0.85

MAX_MODEL_LEN=16384

DTYPE="bfloat16"

# ============================================================
# Output
# ============================================================

OUTPUT_DIR="figs_math500_grpo_norm_displacement_train${TRAIN_SAMPLE}_lr${LR}_gs${TRAIN_GROUP_SIZE}_seed${TRAIN_SEED}"

LOG_DIR="logs_math500_grpo_norm_displacement_train${TRAIN_SAMPLE}_lr${LR}_gs${TRAIN_GROUP_SIZE}_seed${TRAIN_SEED}"

for SEED in "${SEEDS[@]}"; do

RUN_NAME="${DIRECTION_TYPE}"\
"_${MODEL_NAME}"\
"_seed${SEED}"\
"_dir${NUM_DIRECTIONS}"\
"_scale${SCALE}_left${ALPHA_LEFT}_right${ALPHA_RIGHT}"\
"_group${GROUP_SIZE}"\
"_ckpt${CHECKPOINT_STEP}_to${DIR_STEP}"\
"_max_new${MAX_NEW_TOKENS}"\
"_temp${TEMPERATURE}_topp${TOP_P}_topk${TOP_K}"\
"_pts${NUM_POINTS}_train_seed${TRAIN_SEED}"\
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
    --train-seed "${TRAIN_SEED}" \
    --train-group-size "${TRAIN_GROUP_SIZE}" \
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

  wait "$PID"
done
