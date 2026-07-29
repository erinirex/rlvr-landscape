#!/usr/bin/env bash
set -euo pipefail

# bash scripts/run_chessllm.sh

# ===== Change experiment settings here =====
GPU=4,6
PYTHON_SCRIPT="visualize_reward_landscape.py"

MODEL_CKPT="/mnt/sj/home/yichen/landscape_rlvr/chess_llm/rl_C6p5e18_50m_alpha0.400_beta0.023/global_step_2000"
EVAL_JSON="/mnt/sj/home/yichen/landscape_rlvr/train_data/chess_data/chess-rl-data-private/chess-rl-data/train_thinking/train_v4_easy_skewed_multi_turn_sample500.jsonl"

TASK="chess"                  # chess | gsm8k | auto
RL_TYPE="grpo"
DATASET="chess_single_turn"
MODEL_NAME="chessllm_50m"
CHECKPOINT_STEP=2000

NUM_SAMPLES=500
BATCH_SIZE=300
MAX_NEW_TOKENS=3072 #max_position_embeddings, defined in the pretrained model 
SEED=42
SCALE=0.01
ALPHA_RANGE=10
NUM_POINTS=21
NUM_DIRECTIONS=2
YMIN=0.0
YMAX=1.0
OUTPUT_DIR="figs_chessllm_new"

# Make chess_rl_miles importable without hard-coded sys.path in Python.
export PYTHONPATH="/mnt/sj/home/yichen/landscape_rlvr/train_data/chess/chess rl/miles:/mnt/sj/home/yichen/landscape_rlvr/train_data/chess/chess rl/chess-rl-miles:${PYTHONPATH:-}"

RUN_NAME="${RL_TYPE}_${DATASET}_scale${SCALE}_alpha${ALPHA_RANGE}_data${NUM_SAMPLES}_ckpt${CHECKPOINT_STEP}_max_new${MAX_NEW_TOKENS}_bs${BATCH_SIZE}_pts${NUM_POINTS}_seed${SEED}_${NUM_DIRECTIONS}dirs"
LOG_DIR="logs_chessllm"
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