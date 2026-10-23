#!/usr/bin/env bash
# bash swift_scripts/train32_gs4.sh
# seed_list=[42, 50, 7, 10, 99, 100, 200, 300, 400, 500]

set -euo pipefail

SEED=99
GS=4
RUN_DIR="${RUN_DIR:-$(pwd)}"
SWIFT_DIR="/mnt/swordfish-pool2/erinxia/ms-swift"

SWANLAB_API_KEY="aXbj9cLiFZXRzTiSUeHba"
LOG_FILE="${RUN_DIR}/grpo_qwen3_0.6b_nonthink_math500_train32_seed${SEED}_lr_4e-6_max2048_250stp_gs${GS}.log"
OUTPUT_DIR="${RUN_DIR}/output_qwen3_0.6b_nonthink_grpo_math500_train32_seed${SEED}_gs${GS}_lr_4e-6_max2048"

if [[ -z "${SWANLAB_API_KEY:-}" ]]; then
  echo "ERROR: SWANLAB_API_KEY is not set." >&2
  exit 1
fi

cd "${SWIFT_DIR}"

nohup env \
CUDA_VISIBLE_DEVICES=0,5,6,7 \
SWANLAB_API_KEY="${SWANLAB_API_KEY}" \
SWANLAB_GROUP="qwen3_0.6b/train32_gs${GS}" \
torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29502 \
-m swift.cli.rlhf \
--seed "${SEED}" \
--callbacks custom_save \
--save_strategy steps \
--external_plugins "${SWIFT_DIR}/custom_save.py" \
--load_best_model_at_end false \
--rlhf_type grpo \
--loss_type grpo \
--deepspeed zero3 \
--model /tmp/erin/Qwen3-0.6B \
--remove_unused_columns false \
--num_iterations 1 \
--max_steps 250 \
--log_completions true \
--log_entropy true \
--swanlab_project math500-grpo-run30 \
--swanlab_exp_name "train32_seed${SEED}_gs${GS}_lr4e-6_max2048" \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--temperature 0.7 \
--top_k 20 \
--top_p 0.8 \
--lr_scheduler_type constant_with_warmup \
--warmup_ratio 0.05 \
--learning_rate 4e-6 \
--eval_steps 50 \
--eval_strategy steps \
--save_only_model false \
--enable_thinking false \
--reward_func math500_acc \
--max_completion_length 2048 \
--max_new_tokens 2048 \
--logging_steps 1 \
--num_generations 4 \
--num_generations_eval 2 \
--tuner_type full \
--report_to swanlab \
--use_vllm true \
--vllm_mode colocate \
--vllm_gpu_memory_utilization 0.5 \
--dataset /mnt/swordfish-pool2/erinxia/rlvr-landscape/train_data/math500/train_32.jsonl \
--val_dataset /mnt/swordfish-pool2/erinxia/rlvr-landscape/train_data/math500/test.jsonl \
--output_dir "${OUTPUT_DIR}" \
> "${LOG_FILE}" 2>&1 &

echo "Started GRPO training."
echo "PID: $!"
echo "Log: ${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}"