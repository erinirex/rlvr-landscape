# env CUDA_VISIBLE_DEVICES=0 nohup python sgd_direction.py > sgd_gs32_stp110_train1.log 2>&1 &
import argparse
import gc
import json
import math
import os
import random
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from swift.grading.grader import grade_answer

from vllm import LLM, SamplingParams

def cleanup_cuda():

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

def create_vllm(args):

    print()
    print("=" * 80)
    print("Initializing vLLM")
    print("=" * 80)

    print(
        f"Model:              {args.model_ckpt}"
    )

    print(
        f"Tensor parallel:    {args.tensor_parallel_size}"
    )

    print(
        f"GPU memory util:    {args.gpu_memory_utilization}"
    )

    print(
        f"Max model length:   {args.max_model_len}"
    )

    llm = LLM(
        model=args.model_ckpt,

        # ====================================================
        # SINGLE GPU
        # ====================================================

        tensor_parallel_size=1,

        trust_remote_code=True,

        dtype=args.dtype,

        gpu_memory_utilization=(
            args.gpu_memory_utilization
        ),

        max_model_len=args.max_model_len,

        # Required because we mutate model parameters.
        enforce_eager=True,

        enable_prefix_caching=False,

        seed=args.seed,
    )

    return llm


DEFAULT_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within "
    r"\boxed{}."
)


def extract_text_content(
    content: Any,
) -> str:

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        return "\n".join(
            str(item.get("text", ""))
            for item in content
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
            )
        ).strip()

    return str(content)


def extract_prompt_from_messages(
    messages: list[dict[str, Any]],
) -> str:
    """
    Extract raw user problem only.

    IMPORTANT:
    DEFAULT_SUFFIX is NOT added here.
    """

    prompt_parts = []

    for message in messages:

        if not isinstance(message, dict):
            continue

        if message.get("role") != "user":
            continue

        text = extract_text_content(
            message.get(
                "content",
                "",
            )
        )

        if text:
            prompt_parts.append(text)

    return "\n".join(
        prompt_parts
    ).strip()

def load_math500_example(
    example: dict[str, Any],
) -> dict[str, Any]:

    if "messages" not in example:

        raise KeyError(
            "Math500 example is missing 'messages'"
        )

    if "solution" not in example:

        raise KeyError(
            "Math500 example is missing 'solution'"
        )

    # Raw math problem, without suffix.
    prompt = extract_prompt_from_messages(
        example["messages"]
    )

    answer = str(
        example["solution"]
    ).strip()

    if not prompt:

        raise ValueError(
            "Math500 example has no user prompt"
        )

    if not answer:

        raise ValueError(
            "Math500 example has an empty solution"
        )

    return {
        "prompt": prompt,
        "answer": answer,
        "metadata": example.get(
            "metadata",
            {},
        ),
        "extra_info": example.get(
            "extra_info",
            {},
        ),
    }


def build_prompt_text(
    tokenizer,
    example,
):
    """
    EXACTLY equivalent to format_prompt()
    in eval_math500_displacement.py.
    """

    content = (
        example["prompt"]
        + DEFAULT_SUFFIX
    )

    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": content,
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def extract_answer_tag(
    text: str,
) -> tuple[str, bool]:
    """Extract the final <answer>...</answer> block."""
    matches = re.findall(
        r"<answer>(.*?)</answer>",
        str(text),
        flags=re.IGNORECASE | re.DOTALL,
    )

    if matches:
        return matches[-1].strip(), True

    return str(text).strip(), False


def extract_braced_command(
    text: str,
    command: str,
) -> list[str]:

    results: list[str] = []

    pattern = re.compile(
        rf"\\{re.escape(command)}\s*\{{"
    )

    for match in pattern.finditer(text):

        start = match.end()

        depth = 1

        index = start

        while index < len(text) and depth:

            if text[index] == "{":
                depth += 1

            elif text[index] == "}":
                depth -= 1

            index += 1

        if depth == 0:

            results.append(
                text[
                    start:index - 1
                ].strip()
            )

    return results


def extract_math500_answer(
    text: str,
) -> tuple[str, str]:
    """
    Extraction priority:

    1. \\boxed{...}
    2. <answer>...</answer>
    3. Final Answer: ...
    4. Answer: ...

    If none succeeds:
        ("", "no_answer_marker")
    """

    text = str(text)

    # --------------------------------------------------------
    # 1. \boxed{...}
    # --------------------------------------------------------

    boxed = extract_braced_command(
        text,
        "boxed",
    )

    if boxed:
        answer = boxed[-1].strip()

        if answer:
            return answer, "boxed"

    # --------------------------------------------------------
    # 2. <answer>...</answer>
    # --------------------------------------------------------

    tags = re.findall(
        r"<answer>(.*?)</answer>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if tags:
        answer = tags[-1].strip()

        if answer:
            return answer, "answer_tag"

    # --------------------------------------------------------
    # 3. Final Answer: ...
    # --------------------------------------------------------

    matches = list(
        re.finditer(
            r"Final\s+Answer\s*:?\s*",
            text,
            flags=re.IGNORECASE,
        )
    )

    if matches:
        remainder = text[
            matches[-1].end():
        ].strip()

        for line in remainder.splitlines():
            line = line.strip()

            if line:
                return line, "final_answer"

    # --------------------------------------------------------
    # 4. Answer: ...
    # --------------------------------------------------------

    matches = list(
        re.finditer(
            r"Answer\s*:?\s*",
            text,
            flags=re.IGNORECASE,
        )
    )

    if matches:
        remainder = text[
            matches[-1].end():
        ].strip()

        for line in remainder.splitlines():
            line = line.strip()

            if line:
                return line, "answer"

    return "", "no_answer_marker"


def extract_math500_gold(
    answer: Any,
) -> str:
    """
    EXACTLY aligned with eval_math500_displacement.py:

        <answer>...</answer>
        -> \\boxed{...}
        -> raw text
    """

    if answer is None:
        return ""

    if isinstance(answer, dict):

        if "answer" in answer:
            answer = answer["answer"]

        elif "solution" in answer:
            answer = answer["solution"]

    text = str(answer).strip()

    # --------------------------------------------------------
    # 1. <answer>...</answer>
    # --------------------------------------------------------

    tags = re.findall(
        r"<answer>(.*?)</answer>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if tags:
        return tags[-1].strip()

    # --------------------------------------------------------
    # 2. \boxed{...}
    # --------------------------------------------------------

    boxed = extract_braced_command(
        text,
        "boxed",
    )

    if boxed:
        return boxed[-1].strip()

    # --------------------------------------------------------
    # 3. raw text
    # --------------------------------------------------------

    return text


def math500_reward_func(
    completion: str,
    answer: Any,
    **_: Any,
) -> float:
    """
    EXACTLY aligned with eval_math500_displacement.py.

    Empty / unextractable prediction -> reward 0.

    Otherwise:

        grade_answer(
            given_answer=prediction,
            ground_truth=gold,
        )
    """

    prediction, method = (
        extract_math500_answer(
            completion
        )
    )

    gold = extract_math500_gold(
        answer
    )

    # Same behavior as displacement evaluator:
    # no answer marker -> reward = 0
    if not prediction:
        return 0.0

    if not gold:
        return 0.0

    try:

        reward = float(
            grade_answer(
                given_answer=prediction,
                ground_truth=gold,
            )
        )

        return reward

    except Exception as error:

        print(
            "[math500_reward_func] "
            f"prediction={prediction!r}, "
            f"gold={gold!r}, "
            f"method={method}, "
            f"error={type(error).__name__}: "
            f"{error}",
            flush=True,
        )

        return 0.0

def vllm_grpo_rollout(
    llm,
    tokenizer,
    data,
    args,
    direction_index,
):

    rng = random.Random(
        args.seed
        + 1000
        + direction_index
    )

    num_prompts = min(
        args.pg_num_prompts,
        len(data),
    )

    prompt_indices = rng.sample(
        range(len(data)),
        num_prompts,
    )

    selected = [
        data[i]
        for i in prompt_indices
    ]

    prompt_texts = [
        build_prompt_text(
            tokenizer,
            ex,
        )
        for ex in selected
    ]

    sampling_params = SamplingParams(
        n=args.group_size,

        temperature=args.temperature,

        top_p=args.top_p,

        top_k=args.top_k,

        max_tokens=args.max_new_tokens,

        seed=(
            args.seed
            + 1000
            + direction_index
        ),

        skip_special_tokens=True,
    )

    print()
    print("=" * 80)

    print(
        f"GRPO rollout for direction "
        f"{direction_index}"
    )

    print(
        f"Prompts: {num_prompts}"
    )

    print(
        f"Group size: "
        f"{args.group_size}"
    )

    print(
        f"Total completions: "
        f"{num_prompts * args.group_size}"
    )

    print("=" * 80)

    outputs = llm.generate(
        prompt_texts,
        sampling_params,
        use_tqdm=True,
    )

    rollout = []

    for group_id, (
        example,
        prompt_text,
        request_output,
    ) in enumerate(
        zip(
            selected,
            prompt_texts,
            outputs,
        )
    ):

        for completion_output in (
            request_output.outputs
        ):

            completion_text = (
                completion_output.text
            )

            completion_token_ids = list(
                completion_output.token_ids
            )

            reward = math500_reward_func(
                completion_text,
                example["answer"],
            )

            # Already rendered chat template.
            #
            # We use the HF tokenization for
            # the backward pass.
            prompt_token_ids = (
                tokenizer.encode(
                    prompt_text,
                    add_special_tokens=True,
                )
            )

            rollout.append(
                {
                    "group_id":
                        group_id,

                    "prompt":
                        prompt_text,

                    "prompt_token_ids":
                        prompt_token_ids,

                    "completion_token_ids":
                        completion_token_ids,

                    "completion_text":
                        completion_text,

                    "reward":
                        float(reward),
                }
            )

    rewards = [
        x["reward"]
        for x in rollout
    ]

    print(
        "Rollout mean reward:",
        np.mean(rewards),
    )

    return rollout


# ============================================================
# GRPO advantages
# ============================================================

def compute_group_advantages(
    rollout,
    group_size,
):

    rewards_by_group = {}

    for item in rollout:

        gid = item["group_id"]

        if gid not in rewards_by_group:
            rewards_by_group[gid] = []

        rewards_by_group[gid].append(
            float(item["reward"])
        )

    for gid, rewards in (
        rewards_by_group.items()
    ):

        if len(rewards) != group_size:

            raise RuntimeError(
                f"Group {gid} has "
                f"{len(rewards)} rewards, "
                f"expected {group_size}"
            )

        rewards_tensor = torch.tensor(
            rewards,
            dtype=torch.float32,
        )

        mean = rewards_tensor.mean()

        std = rewards_tensor.std(
            unbiased=False
        )

        for item in rollout:

            if item["group_id"] != gid:
                continue

            reward = torch.tensor(
                float(item["reward"]),
                dtype=torch.float32,
            )

            advantage = (
                reward - mean
            ) / (
                std + 1e-4
            )

            item["advantage"] = float(
                advantage.item()
            )


# ============================================================
# Parameter state
# ============================================================

def get_target_state(
    model,
):

    state = {}

    for name, param in (
        model.named_parameters()
    ):

        if param.requires_grad:

            state[name] = (
                param.detach()
                .cpu()
                .float()
                .clone()
            )

    return state


def global_norm(
    state,
):

    total = 0.0

    for tensor in state.values():

        total += (
            tensor.float()
            .pow(2)
            .sum()
            .item()
        )

    return math.sqrt(total)


def scale_to_state_norm(
    direction,
    target_norm,
):

    current_norm = global_norm(
        direction
    )

    if current_norm == 0:

        raise RuntimeError(
            "Direction has zero norm."
        )

    scale = (
        target_norm
        / current_norm
    )

    return {
        name:
            tensor * scale
        for name, tensor
        in direction.items()
    }


# ============================================================
# HF log probabilities
# ============================================================

def compute_response_log_probs(
    model,
    prompt_token_ids,
    completion_token_ids,
):

    input_ids = (
        prompt_token_ids
        +
        completion_token_ids
    )

    device = next(
        model.parameters()
    ).device

    input_tensor = torch.tensor(
        input_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    attention_mask = torch.ones_like(
        input_tensor
    )

    outputs = model(
        input_ids=input_tensor,
        attention_mask=attention_mask,
    )

    logits = outputs.logits

    shift_logits = logits[:, :-1, :]

    shift_labels = input_tensor[:, 1:]

    log_probs = torch.log_softmax(
        shift_logits,
        dim=-1,
    )

    token_log_probs = (
        log_probs
        .gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        )
        .squeeze(-1)
    )

    prompt_len = len(
        prompt_token_ids
    )

    completion_len = len(
        completion_token_ids
    )

    if completion_len == 0:

        return token_log_probs.new_empty(
            (0,)
        )

    start = prompt_len - 1

    end = (
        start
        + completion_len
    )

    response_log_probs = (
        token_log_probs[
            0,
            start:end
        ]
    )

    return response_log_probs


# ============================================================
# GRPO gradient
# ============================================================
def compute_grpo_gradient(
    model,
    rollout,
    args,
):
    model.zero_grad(set_to_none=True)

    print()
    print("=" * 80)
    print("Computing GRPO gradient")
    print("=" * 80)

    # Count total completion tokens.
    total_tokens = sum(
        len(item["completion_token_ids"])
        for item in rollout
    )
    num_completions = len(rollout)

    if total_tokens == 0:
        raise RuntimeError("No valid completion tokens.")

    print("Total completion tokens:", total_tokens)

    total_loss_value = 0.0

    for item in tqdm(
        rollout,
        desc="GRPO backward",
    ):
        if len(item["completion_token_ids"]) == 0:
            continue

        advantage = float(item["advantage"])

        if not math.isfinite(advantage):
            raise RuntimeError(
                f"Non-finite advantage: {advantage}"
            )

        log_probs = compute_response_log_probs(
            model,
            item["prompt_token_ids"],
            item["completion_token_ids"],
        )

        if log_probs.numel() == 0:
            continue

        # GRPO policy-gradient surrogate:
        # average over tokens within each completion,
        # then average over all completions.

        # loss = -sum(advantage * token_log_prob) / total_tokens

        loss_i = -(advantage * log_probs).sum() / total_tokens
        # loss_i = -advantage * log_probs.mean() / num_completions

        if not torch.isfinite(loss_i).item():
            raise RuntimeError(
                "Non-finite GRPO loss encountered."
            )

        loss_i.backward()

        total_loss_value += loss_i.detach().float().item()

        del log_probs
        del loss_i

    print("GRPO loss:", total_loss_value)

    # Raw SGD direction: d = -gradient.
    # No normalization and no scaling to parameter norm.
    direction = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.grad is None:
            direction[name] = torch.zeros_like(
                param.detach(),
                dtype=torch.float32,
                device="cpu",
            )
        else:
            direction[name] = (
                param.grad
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .neg()
            )

    # Diagnostic only; this does not modify the direction.
    grad_norm = global_norm(direction)

    print("Raw SGD direction norm:", grad_norm)

    model.zero_grad(set_to_none=True)

    if not math.isfinite(grad_norm):
        raise RuntimeError(
            f"GRPO gradient norm is non-finite: {grad_norm}"
        )

    if grad_norm == 0.0:
        raise RuntimeError("GRPO gradient is zero.")

    return direction

# ============================================================
# Save direction
# ============================================================

def save_direction(
    direction,
    path,
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print("Saving direction")
    print("=" * 80)

    print(
        "Path:",
        path,
    )

    print(
        "Direction norm:",
        global_norm(direction),
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Store CPU float32.
    #
    # The worker will read this file itself.
    #
    # We DO NOT pass this dictionary through apply_model().
    # --------------------------------------------------------

    cpu_direction = {}

    for name, tensor in direction.items():

        cpu_direction[name] = (
            tensor.detach()
            .cpu()
            .float()
            .contiguous()
        )

    torch.save(
        cpu_direction,
        path,
    )

    file_size_gb = (
        path.stat().st_size
        / (1024 ** 3)
    )

    print(
        f"Direction file size: "
        f"{file_size_gb:.3f} GB"
    )

    print(
        "Direction saved."
    )


def load_hf_model(
    args,
):

    print()
    print("=" * 80)
    print(
        "Loading HuggingFace model "
        "for GRPO backward"
    )
    print("=" * 80)

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_ckpt,

            torch_dtype=(
                torch.bfloat16
                if args.dtype == "bfloat16"
                else torch.float16
            ),

            trust_remote_code=True,

            # Single GPU
            device_map="cuda",
        )
    )

    # --------------------------------------------------------
    # Important for memory-efficient backward
    # --------------------------------------------------------

    model.gradient_checkpointing_enable()

    model.config.use_cache = False

    model.train()

    return model


def load_eval_data(
    path: str,
    limit: int,
    task: str,
) -> list[dict[str, Any]]:

    if limit <= 0:
        raise ValueError(
            f"limit must be positive, got {limit}"
        )

    data: list[dict[str, Any]] = []

    with open(
        path,
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            if not line.strip():
                continue

            try:

                example = json.loads(line)

                if task == "math500":

                    parsed_example = (
                        load_math500_example(
                            example
                        )
                    )

                else:
                    raise ValueError(
                        f"Unsupported task: {task}"
                    )

            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as error:

                raise ValueError(
                    f"Failed to parse {task} "
                    f"example at {path}:{line_number}: "
                    f"{error}"
                ) from error

            data.append(
                parsed_example
            )

            if len(data) >= limit:
                break

    if not data:
        raise ValueError(
            f"No evaluation examples loaded from {path}"
        )

    return data

 

def main():
    # ========================================================
    # 固定配置：直接在这里修改
    # ========================================================
    args = SimpleNamespace(
        # 模型：替换为你的实际 checkpoint 路径
        # model_ckpt="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_5e-6_max2048/v7-20260903-071819/checkpoint-75",
        # model_ckpt="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_5e-6_max2048/v7-20260903-071819/checkpoint-275",
        # model_ckpt="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_3e-6_max2048/v0-20260910-192151/checkpoint-120",
        # model_ckpt="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train8_lr_5e-6_max2048/v2-20260917-200704/checkpoint-50",
        model_ckpt="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train1_lr_5e-6_max2048/v0-20260916-223901/checkpoint-110",
        model_name="qwen3_0.6b",

        # 仅用于输出文件命名，不负责选择 checkpoint
        checkpoint_step=110,

        dtype="bfloat16",

        # 数据：替换为你的实际数据路径
        eval_json="/mnt/swordfish-pool2/erinxia/rlvr-landscape/train_data/math500/train_id299.jsonl",
        task="math500",
        num_samples=1,
        pg_num_prompts=1,

        # vLLM
        tensor_parallel_size=1,
        gpu_memory_utilization=0.25,
        max_model_len=16384,

        # GRPO rollout
        num_directions=1,
        group_size=32,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_new_tokens=2048,
        seed=42,

        # 保存目录
        direction_dir="./sgd_directions_train1",
    )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.tensor_parallel_size != 1:
        raise ValueError("This script requires tensor_parallel_size=1.")

    if args.num_directions < 1:
        raise ValueError("num_directions must be >= 1.")

    if args.group_size < 2:
        raise ValueError("group_size must be >= 2.")

    if args.num_samples < 1 or args.pg_num_prompts < 1:
        raise ValueError("num_samples and pg_num_prompts must be >= 1.")

    print("=" * 80)
    print("Configuration")
    print("=" * 80)

    for name, value in vars(args).items():
        print(f"{name}: {value}")

    # ========================================================
    # Tokenizer 和数据
    # ========================================================
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_ckpt,
        trust_remote_code=True,
    )

    data = load_eval_data(
        args.eval_json,
        args.num_samples,
        args.task,
    )

    if len(data) == 0:
        raise RuntimeError("No examples loaded.")

    print(f"\nLoaded {len(data)} examples.")

    direction_dir = Path(args.direction_dir)
    direction_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # 1. 使用原始 checkpoint 生成 rollout 并计算 advantage
    # ========================================================
    grpo_rollouts = []
    llm = create_vllm(args)

    try:
        for direction_idx in range(args.num_directions):
            rollout = vllm_grpo_rollout(
                llm,
                tokenizer,
                data,
                args,
                direction_idx,
            )

            compute_group_advantages(
                rollout,
                args.group_size,
            )

            active_groups = {
                item["group_id"]
                for item in rollout
                if item["advantage"] != 0.0
                and len(item["completion_token_ids"]) > 0
            }

            print(
                f"Direction {direction_idx}: "
                f"{len(active_groups)} groups with usable "
                "nonzero advantages."
            )

            if not active_groups:
                raise RuntimeError(
                    f"Direction {direction_idx} has no usable "
                    "nonzero advantages. Increase pg_num_prompts "
                    "or group_size and try again."
                )

            grpo_rollouts.append(rollout)

    finally:
        print("\nDestroying vLLM after rollout...")
        del llm
        cleanup_cuda()

    # ========================================================
    # 2. 计算并保存原始 SGD 方向：direction = -gradient
    #    不归一化，不更新 checkpoint 参数
    # ========================================================
    hf_model = load_hf_model(args)
    direction_paths = []

    try:
        # 关闭 dropout，但仍然允许计算梯度
        hf_model.eval()

        step_name = (
            f"step{args.checkpoint_step}"
            if args.checkpoint_step is not None
            else "checkpoint"
        )

        for direction_idx, rollout in enumerate(grpo_rollouts):
            print()
            print("=" * 80)
            print(
                f"Computing raw SGD direction "
                f"{direction_idx + 1}/{args.num_directions}"
            )
            print("=" * 80)

            # 使用前面提供的无归一化版本
            direction = compute_grpo_gradient(
                hf_model,
                rollout,
                args,
            )

            direction_path = direction_dir / (
                f"{args.model_name}_"
                f"{step_name}_"
                f"sgd_raw_direction_{direction_idx}_gs{args.group_size}.pt"
            )

            try:
                save_direction(
                    direction,
                    direction_path,
                )
            finally:
                del direction

            direction_paths.append(direction_path.resolve())
            cleanup_cuda()

    finally:
        del hf_model
        cleanup_cuda()

    print()
    print("=" * 80)
    print("Saved raw SGD direction files:")

    for path in direction_paths:
        print(f"  {path}")

    print("=" * 80)


if __name__ == "__main__":
    main()