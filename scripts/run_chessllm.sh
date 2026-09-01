#!/usr/bin/env bash
set -euo pipefail

# bash scripts/run_chessllm.sh

# ===== Change experiment settings here =====
GPUS="1,2"
NUM_GPUS=2
MASTER_PORT=29507
# PYTHON_SCRIPT="visualize_reward_landscape_sto.py"
PYTHON_SCRIPT="visualize_reward_landscape_paral.py"


# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/chess_llm/rl_C6p5e18_50m_alpha0.400_beta0.023/global_step_100"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/chess_llm/rl_C6p5e18_200m_alpha0.100_beta0.100/global_step_50"
MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/chess_llm/rl_C6p5e19_680m_alpha1.500_beta0.030/miles_sglang_grpo_adamw_bs2048_sgl64_cvd_mrouter_ctx16fix/global_step_20"

# MODEL_CKPT="/mnt/sj/home/yichen/landscape_rlvr/chess_llm/rl_C6p5e19_200m_alpha1.000_beta0.007/miles_sglang_grpo_adamw_bs2048_sgl64_cvd_mrouter_ctx16fix/global_step_1200"
# MODEL_CKPT="/mnt/sj/home/yichen/landscape_rlvr/chess_llm/rl_C6p5e19_200m_alpha1.000_beta0.007/miles_sglang_grpo_adamw_bs2048_sgl64_cvd_mrouter_ctx16fix/global_step_100"

# EVAL_JSON="/mnt/sj/home/yichen/landscape_rlvr/train_data/chess_data/chess-rl-data-private/chess-rl-data/train_thinking/train_v4_easy_skewed_multi_turn_1move_sample500.jsonl"
EVAL_JSON="/mnt/swordfish-pool2/erinxia/rlvr-landscape/train_data/chess_data/chess-rl-data-private/chess-rl-data/train_thinking/train_v4_easy_skewed_multi_turn_sample100.jsonl"


TASK="chess"                  # chess | gsm8k 
RL_TYPE="grpo"
DATASET="chess_single_turn"
MODEL_NAME="chessllm_680m"
CHECKPOINT_STEP=20

PG_NUM_PROMPTS=4
PG_GROUP_SIZE=4
PG_TEMPERATURE=1.0

DIRECTION=grpo-grad

NUM_SAMPLES=25
BATCH_SIZE=42
MAX_NEW_TOKENS=2048 #max_position_embeddings, defined in the pretrained model 
SEED=42
SCALE=0.01
ALPHA_RANGE=50

NUM_POINTS=5
GROUP_SIZE=4
TMP=1.0
TOP_P=1.0

NUM_DIRECTIONS=1
YMIN=0.0
YMAX=1.0
OUTPUT_DIR="figs_chessllm_paral"
# OUTPUT_DIR="figs_chessllm_stochastic"


# Make chess_rl_miles importable without hard-coded sys.path in Python.
# export PYTHONPATH="/mnt/sj/home/yichen/landscape_rlvr/train_data/chess/chess rl/miles:/mnt/sj/home/yichen/landscape_rlvr/train_data/chess/chess rl/chess-rl-miles:${PYTHONPATH:-}"

RUN_NAME="${RL_TYPE}_${DATASET}_${MODEL_NAME}_scale${SCALE}_alpha${ALPHA_RANGE}_data${NUM_SAMPLES}_group${GROUP_SIZE}_ckpt${CHECKPOINT_STEP}_max_new${MAX_NEW_TOKENS}_bs${BATCH_SIZE}_temp${TMP}_topp${TOP_P}_pts${NUM_POINTS}_seed${SEED}_${NUM_DIRECTIONS}dirs"
LOG_DIR="logs_chessllm_paral"
# LOG_DIR="logs_chessllm_stochastic"

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
    --num-directions "${NUM_DIRECTIONS}" \
    --output-dir "${OUTPUT_DIR}" \
    --ymin "${YMIN}" \
    --ymax "${YMAX}" \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Started PID ${PID}"
echo "Follow log with: tail -f ${LOG_FILE}"