#!/usr/bin/env bash
set -euo pipefail

# bash scripts/run_qwen_math500_vllm.sh

# ============================================================
# GPU settings
# ============================================================
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
GPU="0"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/visualize_reward_landscape_vllm.py"
TRAIN_SAMPLE=32
CHECKPOINT_STEP=75
LR="4e-6"
TRAIN_GS=32

# ============================================================
# Model
# ============================================================

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train100_lr_5e-6_max8192/v1-20260821-105102/checkpoint-100"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_5e-6_max2048/v7-20260903-071819/checkpoint-275"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_5e-6_max2048/v7-20260903-071819/checkpoint-75"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train8_lr_5e-6_max2048/v2-20260917-200704/checkpoint-250"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train1_lr_5e-6_max2048/v0-20260916-223901/checkpoint-50"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train1_lr_5e-6_max2048/v0-20260916-223901/checkpoint-10"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32_seed99_gs4_lr_${LR}_max2048/v2-20261009-005817/checkpoint-${CHECKPOINT_STEP}"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32_seed99_gs16_lr_${LR}_max2048/v0-20261009-105123/checkpoint-${CHECKPOINT_STEP}"

MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32_seed99_gs32_lr_${LR}_max2048/v0-20261014-210623/checkpoint-${CHECKPOINT_STEP}"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_3e-6_max2048/v0-20260910-192151/checkpoint-50"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_3e-6_max2048/v0-20260910-192151/checkpoint-117"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_5e-6_max2048/v18-20260903-125040/checkpoint-49"

# EVAL_JSON="/mnt/swordfish-pool2/erinxia/rlvr-landscape/train_data/math500/train_id299.jsonl"
# EVAL_JSON="/mnt/swordfish-pool2/erinxia/rlvr-landscape/train_data/math500/train_8.jsonl"

# EVAL_JSON="/mnt/swordfish-pool2/erinxia/rlvr-landscape/train_data/math500/train.jsonl"

EVAL_JSON="/mnt/swordfish-pool2/erinxia/rlvr-landscape/train_data/math500/train_32.jsonl"

TASK="math500"
RL_TYPE="grpo"
DATASET="math500"
MODEL_NAME="qwen3_0.6b"
# CHECKPOINT_STEP=275
# CHECKPOINT_STEP=117


# ============================================================
# Evaluation
# ============================================================

NUM_SAMPLES=32

MAX_NEW_TOKENS=2048

SEED=42

# 每个 prompt 生成多少个 samples
GROUP_SIZE=64

TMP=0.7
TOP_P=0.8
TOP_K=20

# ============================================================
# Random directions
# ============================================================

NUM_DIRECTIONS=1

# ============================================================
# Landscape
# ============================================================

SCALE=0.01
ALPHA_RANGE=10
NUM_POINTS=15

# ============================================================
# vLLM settings
# ============================================================

GPU_MEMORY_UTILIZATION=0.7
MAX_NUM_SEQS=8
MAX_MODEL_LEN=16384

# ============================================================
# Plot
# ============================================================

YMIN=0.0
YMAX=1.1

OUTPUT_DIR="figs_math500_grpo_random_lr_${LR}_vllm_train${TRAIN_SAMPLE}_gs${TRAIN_GS}"

RUN_NAME="${RL_TYPE}_${DATASET}_${MODEL_NAME}"\
"_random${NUM_DIRECTIONS}"\
"_scale${SCALE}_alpha${ALPHA_RANGE}"\
"_lr${LR}"\
"_data${NUM_SAMPLES}_group${GROUP_SIZE}"\
"_ckpt${CHECKPOINT_STEP}"\
"_train_gs${TRAIN_GS}"\
"_max_new${MAX_NEW_TOKENS}"\
"_pts${NUM_POINTS}_seed${SEED}"\
"_gpu${GPU}"\
"_vllm"

LOG_DIR="logs_math500_grpo_random_lr_${LR}_vllm_train${TRAIN_SAMPLE}_gs${TRAIN_GS}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

# Examples:
# bash run_directions.sh --num-directions 12 --gpus 0,1,2,3 --seed-start 1001
# bash run_directions.sh --groups 3 --gpus 0,1,2,3 --seed-start 1001
GPU_LIST="0,1"
TOTAL_DIRECTIONS=""
NUM_GROUPS=""
SEED_START=1001
SEEDS=()
while (( $# > 0 )); do
  case "$1" in
    --num-directions|--groups|--gpus|--seed-start)
      (( $# >= 2 )) || { echo "Missing value for $1" >&2; exit 1; }
      case "$1" in
        --num-directions) TOTAL_DIRECTIONS="$2" ;;
        --groups) NUM_GROUPS="$2" ;;
        --gpus) GPU_LIST="$2" ;;
        --seed-start) SEED_START="$2" ;;
      esac
      shift 2 ;;
    --help|-h)
      echo "Usage: bash run_directions.sh [--num-directions COUNT | --groups COUNT] [--gpus 0,1,2,3] [--seed-start 1001]"
      echo "Or: bash run_directions.sh --gpus 0,1 SEED SEED ..."
      exit 0 ;;
    --*) echo "Unknown option: $1" >&2; exit 1 ;;
    *) SEEDS+=("$1"); shift ;;
  esac
done
[[ "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "Use GPU IDs separated by commas, e.g. 0,1,2,3" >&2; exit 1; }
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
declare -A gpu_seen=()
for gpu in "${GPUS[@]}"; do
  [[ -z "${gpu_seen[$gpu]+x}" ]] || { echo "Duplicate GPU: $gpu" >&2; exit 1; }
  gpu_seen[$gpu]=1
done
if [[ -n "$TOTAL_DIRECTIONS" && -n "$NUM_GROUPS" ]]; then
  echo "Specify --num-directions or --groups, not both" >&2; exit 1
fi
if (( ${#SEEDS[@]} > 0 )); then
  [[ -z "$TOTAL_DIRECTIONS$NUM_GROUPS" ]] || { echo "Explicit seeds cannot be combined with counts" >&2; exit 1; }
else
  [[ "$SEED_START" =~ ^[1-9][0-9]*$ ]] || { echo "seed-start must be a positive integer" >&2; exit 1; }
  if [[ -n "$NUM_GROUPS" ]]; then
    [[ "$NUM_GROUPS" =~ ^[1-9][0-9]*$ ]] || { echo "groups must be positive" >&2; exit 1; }
    TOTAL_DIRECTIONS=$(( NUM_GROUPS * ${#GPUS[@]} ))
  fi
  TOTAL_DIRECTIONS="${TOTAL_DIRECTIONS:-6}"
  [[ "$TOTAL_DIRECTIONS" =~ ^[1-9][0-9]*$ ]] || { echo "num-directions must be positive" >&2; exit 1; }
  for ((i=0; i<TOTAL_DIRECTIONS; i++)); do SEEDS+=("$((SEED_START+i))"); done
fi
declare -A seen=()
for seed in "${SEEDS[@]}"; do
  [[ "$seed" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "Invalid seed: $seed" >&2; exit 1; }
  [[ -z "${seen[$seed]+x}" ]] || { echo "Duplicate seed: $seed" >&2; exit 1; }
  seen[$seed]=1
done
echo "Directions: ${#SEEDS[@]}; GPUs: ${GPUS[*]}; groups: $(( (${#SEEDS[@]}+${#GPUS[@]}-1)/${#GPUS[@]} ))"
echo "Direction seeds: ${SEEDS[*]}"
# Includes experiment settings, excludes GPU and direction seed.
ROOT="${OUTPUT_DIR}/${RUN_NAME%_gpu*}"
mkdir -p "$ROOT"
for ((start=0; start<${#SEEDS[@]}; start+=${#GPUS[@]})); do
  pids=()
  for ((slot=0; slot<${#GPUS[@]} && start+slot<${#SEEDS[@]}; slot++)); do
    ds="${SEEDS[start+slot]}"
    out="$ROOT/dirseed_${ds}"
    mkdir -p "$out"
    if [[ -f "$out/DONE" ]]; then echo "Skip completed seed $ds"; continue; fi
    (
      # Prevent two launchers from writing the same direction concurrently.
      flock -n 9 || { echo "Seed $ds already running" >&2; exit 1; }
      CUDA_VISIBLE_DEVICES="${GPUS[slot]}" python -u "$PYTHON_SCRIPT" \
        --model-ckpt "$MODEL_CKPT" --eval-json "$EVAL_JSON" \
        --task "$TASK" --lr "$LR" --rl-type "$RL_TYPE" \
        --dataset "$DATASET" --model-name "$MODEL_NAME" \
        --checkpoint-step "$CHECKPOINT_STEP" --num-samples "$NUM_SAMPLES" \
        --group-size "$GROUP_SIZE" --temperature "$TMP" --top-p "$TOP_P" \
        --top-k "$TOP_K" --max-new-tokens "$MAX_NEW_TOKENS" \
        --seed "$SEED" --direction-seed "$ds" --num-directions 1 \
        --tensor-parallel-size 1 --scale "$SCALE" --alpha-range "$ALPHA_RANGE" \
        --num-points "$NUM_POINTS" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --max-num-seqs "$MAX_NUM_SEQS" --max-model-len "$MAX_MODEL_LEN" \
        --output-dir "$out" --output-stem landscape --ymin "$YMIN" --ymax "$YMAX"
      touch "$out/DONE"
    ) 9>"$out/.lock" >"$out/run.log" 2>&1 &
    pids+=("$!")
    echo "GPU ${GPUS[slot]} -> seed $ds; log: $out/run.log"
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 )) || { echo "A direction failed; inspect run.log" >&2; exit 1; }
done
python "$SCRIPT_DIR/plot_directions.py" --root "$ROOT" --seeds "${SEEDS[@]}" --expected "${#SEEDS[@]}"
