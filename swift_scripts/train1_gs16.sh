#!/usr/bin/env bash
# bash swift_scripts/train1_gs16.sh

set -euo pipefail

SEED=42
SWIFT_DIR="/mnt/sj/home/yichen/ms-swift"
RUN_DIR="${RUN_DIR:-$(pwd)}"
SWANLAB_API_KEY="aXbj9cLiFZXRzTiSUeHba"
LOG_FILE="${RUN_DIR}/grpo_qwen3_0.6b_nonthink_math500_train1_lr_5e-6_max2048_125stp_gs8.log"
OUTPUT_DIR="${RUN_DIR}/output_qwen3_0.6b_nonthink_grpo_math500_train1_lr_5e-6_max2048"

if [[ -z "${SWANLAB_API_KEY:-}" ]]; then
  echo "ERROR: SWANLAB_API_KEY is not set." >&2
  exit 1
fi

cd "${SWIFT_DIR}"

nohup env \
CUDA_VISIBLE_DEVICES=0 \
SWANLAB_API_KEY="${SWANLAB_API_KEY}" \
torchrun --nproc_per_node=1 --master_port=29501 \
-m swift.cli.rlhf \
--seed "${SEED}" \
--save_strategy steps \
--save_steps 100 \
--external_plugins "${SWIFT_DIR}/custom_save.py" "${SWIFT_DIR}/sample_pass_rate.py" \
--load_best_model_at_end false \
--rlhf_type grpo \
--loss_type grpo \
--deepspeed zero3 \
--model /tmp/erin/Qwen3-0.6B \
--remove_unused_columns false \
--steps_per_generation 4 \
--num_iterations 1 \
--max_steps 125 \
--log_completions true \
--log_entropy true \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 4 \
--temperature 0.7 \
--top_k 20 \
--top_p 0.8 \
--lr_scheduler_type constant_with_warmup \
--warmup_ratio 0.05 \
--learning_rate 5e-6 \
--eval_steps 10 \
--eval_strategy steps \
--save_only_model false \
--enable_thinking false \
--reward_func math500_acc \
--max_completion_length 2048 \
--max_new_tokens 2048 \
--logging_steps 1 \
--num_generations 8 \
--num_generations_eval 2 \
--tuner_type full \
--report_to swanlab \
--use_vllm true \
--vllm_mode colocate \
--vllm_gpu_memory_utilization 0.8 \
--dataset /mnt/swordfish-pool2/erinxia/rlvr-landscape/train_data/math500/train_1.jsonl \
--val_dataset /mnt/swordfish-pool2/erinxia/rlvr-landscape/train_data/math500/test.jsonl \
--output_dir "${OUTPUT_DIR}" \
> "${LOG_FILE}" 2>&1 &

echo "Started GRPO training."
echo "PID: $!"
echo "Log: ${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}"