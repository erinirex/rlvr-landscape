#!/usr/bin/env bash
set -euo pipefail

# ===== Change experiment settings here =====
GPU=3
PYTHON_SCRIPT="visualize_reward_landscape.py"

MODEL_CKPT="/mnt/sj/home/yichen/ms-swift/output_qwen3_1.7b_grpo_chess_skewed_multi_turn_full/v1-20260719-054720/checkpoint-100"
EVAL_JSON="/mnt/sj/home/yichen/landscape_rlvr/train_data/chess_data/chess-rl-data-private/chess-rl-data/train_thinking/train_v4_easy_skewed_multi_turn_swift_sample100.jsonl"

TASK="chess"                  # chess | gsm8k | auto
RL_TYPE="grpo"
DATASET="chess_single_turn"
MODEL_NAME="qwen3_1.7b"
CHECKPOINT_STEP=100

NUM_SAMPLES=25
MAX_NEW_TOKENS=128
SEED=1000
SCALE=0.01
ALPHA_RANGE=10
NUM_POINTS=15
NUM_DIRECTIONS=2
YMIN=0.0
YMAX=1.0
OUTPUT_DIR="figs_qwen_chess"

# Make chess_rl_miles importable without hard-coded sys.path in Python.
export PYTHONPATH="/mnt/sj/home/yichen/landscape_rlvr/train_data/chess/chess rl/miles:/mnt/sj/home/yichen/landscape_rlvr/train_data/chess/chess rl/chess-rl-miles:${PYTHONPATH:-}"

RUN_NAME="${RL_TYPE}_${DATASET}_scale${SCALE}_alpha${ALPHA_RANGE}_data${NUM_SAMPLES}_ckpt${CHECKPOINT_STEP}_pts${NUM_POINTS}_seed${SEED}_${NUM_DIRECTIONS}dirs"
LOG_DIR="logs"
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