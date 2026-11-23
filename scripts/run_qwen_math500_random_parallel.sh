#!/usr/bin/env bash
set -euo pipefail
# bash scripts/run_qwen_math500_random_parallel.sh

GPU_LIST="1,2,3"
TOTAL_DIRECTIONS="20"  # Set e.g. "10" for a fixed total; leave NUM_GROUPS empty.
NUM_GROUPS=""       # Set e.g. "3" for 3 groups; leave TOTAL_DIRECTIONS empty.
SEED_START=1001
SEEDS=()             # Leave empty to generate seeds automatically.

TRAIN_SAMPLE=32
CHECKPOINT_STEP=78
LR="2.5e-6"
TRAIN_GS=32
TRAIN_PROMPT=32
TRAIN_SEED=50

RL_TYPE="grpo"
DATASET="math500"
MODEL_NAME="qwen3_0.6b"


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
# Landscape
# ============================================================

SCALE=60
ALPHA_RANGE=1
NUM_POINTS=11


OUTPUT_DIR="landscape_math500_grpo_random_newdata_train32_global_norm"

RUN_NAME="scale${SCALE}_alpha${ALPHA_RANGE}"\
"_train_gs${TRAIN_GS}"\
"_train_seed${TRAIN_SEED}"\
"_train_prompt${TRAIN_PROMPT}"\
"_lr${LR}"\
"_data${NUM_SAMPLES}_group${GROUP_SIZE}"\
"_ckpt${CHECKPOINT_STEP}"\
"_max_new${MAX_NEW_TOKENS}"\
"_pts${NUM_POINTS}_seed${SEED}"

ROOT="${OUTPUT_DIR}/${RUN_NAME}"

if [[ "${LANDSCAPE_BACKGROUND:-0}" != "1" ]]; then
  mkdir -p "${ROOT}/logs"
  LOG_FILE="${ROOT}/logs/directions_$(date +%Y%m%d_%H%M%S)_$$.log"

  nohup env LANDSCAPE_BACKGROUND=1 bash "$0" "$@" \
    > "$LOG_FILE" 2>&1 < /dev/null &

  PID=$!
  echo "已后台启动，PID: $PID"
  echo "总日志: $LOG_FILE"
  echo "查看日志: tail -f \"$LOG_FILE\""
  exit 0
fi


# ============================================================
# GPU settings
# ============================================================
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="visualize_reward_landscape_random_global_norm_parallel.py"

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
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32_seed99_gs4_lr_${LR}_max2048/v3-20261009-015503/checkpoint-${CHECKPOINT_STEP}"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32_seed99_gs4_lr_${LR}_max2048/v5-20261029-014847/checkpoint-${CHECKPOINT_STEP}"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32_seed99_gs4_lr_${LR}_max2048/v6-20261029-192306/checkpoint-${CHECKPOINT_STEP}"


# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32_seed99_gs16_lr_${LR}_max2048/v0-20261009-105123/checkpoint-${CHECKPOINT_STEP}"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32_seed99_gs16_lr_2e-6_max2048/v0-20261013-002811/checkpoint-${CHECKPOINT_STEP}"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32_seed99_gs16_lr_6e-6_max2048/v0-20261014-183522/checkpoint-${CHECKPOINT_STEP}"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32_seed99_gs32_lr_${LR}_max2048/v0-20261014-210623/checkpoint-${CHECKPOINT_STEP}"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32_seed42_gs16_lr_4e-6_max2048/v0-20261106-014427/checkpoint-${CHECKPOINT_STEP}"


# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32_seed42_gs16_lr_6e-6_max2048/v1-20261106-031949/checkpoint-${CHECKPOINT_STEP}"


# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs32_lr_4e-6_max2048/v1-20261105-235107/checkpoint-${CHECKPOINT_STEP}"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs4_lr_4e-6_max2048/v1-20261105-201300/checkpoint-${CHECKPOINT_STEP}"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed99_gs16_lr_4e-6_max2048/v2-20261112-155726/checkpoint-${CHECKPOINT_STEP}"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs16_lr_4e-6_max2048/v1-20261114-182621/checkpoint-${CHECKPOINT_STEP}"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs32_lr_4e-6_max2048/v2-20261114-143227/checkpoint-${CHECKPOINT_STEP}"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs4_prompt8_lr_5e-6_max2048/v1-20261114-180927/checkpoint-${CHECKPOINT_STEP}"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs16_lr_3e-6_max2048/v0-20261118-231555/checkpoint-${CHECKPOINT_STEP}"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs32_lr_3e-6_max2048/v0-20261118-165927/checkpoint-${CHECKPOINT_STEP}"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs4_prompt8_lr_3.5e-6_max2048/v0-20261119-212529/checkpoint-${CHECKPOINT_STEP}"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs32_lr_2e-6_max2048/v0-20261122-002714/checkpoint-${CHECKPOINT_STEP}"

MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs32_lr_2.5e-6_max2048/v0-20261121-171458/checkpoint-${CHECKPOINT_STEP}"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs16_lr_3e-6_max2048/v0-20261118-231555/checkpoint-${CHECKPOINT_STEP}"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs32_lr_3e-6_max2048/v0-20261118-165927/checkpoint-${CHECKPOINT_STEP}"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs16_lr_2e-6_max2048/v0-20261112-202701/checkpoint-${CHECKPOINT_STEP}"


# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32_seed99_gs4_prompt32_lr_4e-6_max2048/v0-20261106-112208/checkpoint-${CHECKPOINT_STEP}"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32_seed42_gs16_lr_2e-6_max2048/v1-20261106-031734/checkpoint-${CHECKPOINT_STEP}"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32_seed42_gs32_lr_4e-6_max2048/v0-20261106-013215/checkpoint-${CHECKPOINT_STEP}"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32_seed42_gs4_lr_4e-6_max2048/v0-20261102-230504/checkpoint-${CHECKPOINT_STEP}"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_3e-6_max2048/v0-20260910-192151/checkpoint-50"
# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_3e-6_max2048/v0-20260910-192151/checkpoint-117"

# MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_5e-6_max2048/v18-20260903-125040/checkpoint-49"

# EVAL_JSON="/mnt/swordfish-pool2/erinxia/rlvr-landscape/train_data/math500/train_id299.jsonl"
# EVAL_JSON="/mnt/swordfish-pool2/erinxia/rlvr-landscape/train_data/math500/train_8.jsonl"

# EVAL_JSON="/mnt/swordfish-pool2/erinxia/rlvr-landscape/train_data/math500/train.jsonl"

EVAL_JSON="/mnt/swordfish-pool2/erinxia/rlvr-landscape/train_data/math500/train_32_new.jsonl"

TASK="math500"

# CHECKPOINT_STEP=275
# CHECKPOINT_STEP=117


# ============================================================
# vLLM settings
# ============================================================

GPU_MEMORY_UTILIZATION=0.85
MAX_MODEL_LEN=16384

# ============================================================
# Plot
# ============================================================

YMIN=0.0
YMAX=1.1


# Examples:
# bash run_directions.sh --num-directions 12 --gpus 0,1,2,3 --seed-start 1001
# bash run_directions.sh --groups 3 --gpus 0,1,2,3 --seed-start 1001
# Edit these settings, then run: bash run_directions.sh

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

run_direction() {
  local ds=$1
  local gpu=$2
  local out="$ROOT/dirseed_${ds}"

  mkdir -p "$out" || return 1

  (
    flock -n 9 || { echo "Seed $ds already running" >&2; exit 1; }
    if [[ -f "$out/DONE" ]]; then
      echo "Skip completed seed $ds"
      exit 0
    fi

    echo "GPU $gpu -> seed $ds; log: $out/run.log"
    CUDA_VISIBLE_DEVICES="$gpu" python -u "$PYTHON_SCRIPT" \
      --model-ckpt "$MODEL_CKPT" --eval-json "$EVAL_JSON" \
      --train-gs "$TRAIN_GS" --train-seed "$TRAIN_SEED" \
      --task "$TASK" --lr "$LR" --rl-type "$RL_TYPE" \
      --dataset "$DATASET" --model-name "$MODEL_NAME" \
      --checkpoint-step "$CHECKPOINT_STEP" --num-samples "$NUM_SAMPLES" \
      --group-size "$GROUP_SIZE" --temperature "$TMP" --top-p "$TOP_P" \
      --top-k "$TOP_K" --max-new-tokens "$MAX_NEW_TOKENS" \
      --seed "$SEED" --direction-seed "$ds" --num-directions 1 \
      --tensor-parallel-size 1 --scale "$SCALE" --alpha-range "$ALPHA_RANGE" \
      --num-points "$NUM_POINTS" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --output-dir "$out" --output-stem landscape --ymin "$YMIN" --ymax "$YMAX" \
      >"$out/run.log" 2>&1 || { echo "Seed $ds failed: $out/run.log" >&2; exit 1; }

    touch "$out/DONE"
  ) 9>"$out/.lock"
}

# xargs 启动的子 shell 需要用到这些变量和函数
export -f run_direction
export GPU_LIST ROOT PYTHON_SCRIPT MODEL_CKPT EVAL_JSON TRAIN_GS TRAIN_SEED
export TASK LR RL_TYPE DATASET MODEL_NAME CHECKPOINT_STEP NUM_SAMPLES
export GROUP_SIZE TMP TOP_P TOP_K MAX_NEW_TOKENS SEED SCALE ALPHA_RANGE
export NUM_POINTS GPU_MEMORY_UTILIZATION MAX_MODEL_LEN YMIN YMAX

printf '%s\n' "${SEEDS[@]}" |
  xargs -n 1 -P "${#GPUS[@]}" --process-slot-var=SLOT bash -c '
    IFS=, read -r -a gpu_ids <<< "$GPU_LIST"
    run_direction "$1" "${gpu_ids[$SLOT]}"
  ' _
python "$SCRIPT_DIR/plot_directions.py" --root "$ROOT"
