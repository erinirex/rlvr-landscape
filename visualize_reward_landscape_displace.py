#!/usr/bin/env python3

import argparse
import gc
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from swift.grading.grader import grade_answer

from vllm import LLM, SamplingParams


# ============================================================
# Math / answer extraction
# ============================================================
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

 

# ============================================================
# vLLM
# ============================================================

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

        seed=args.seed,
    )

    return llm

@torch.inference_mode()
def vllm_eval_reward(
    llm,
    tokenizer,
    data,
    args,
):
    """Evaluate vLLM and return reward statistics plus per-generation rows."""
    selected = data
    prompts = [
        build_prompt_text(tokenizer, ex)
        for ex in selected
    ]

    # sampling_params = SamplingParams(
    #     n=args.group_size,
    #     temperature=(
    #         args.temperature
    #         # if args.group_size > 1
    #         # else 0.0
    #     ),
    #     top_p=args.top_p,
    #     top_k=args.top_k,
    #     max_tokens=args.max_new_tokens,
    #     seed=args.seed,
    #     skip_special_tokens=True,
    # )

    sampling_params = SamplingParams(
        n=args.group_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        stop=[
            "<|im_end|>",
            "<|endoftext|>",
        ],
    )

    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=True,
    )

    if len(outputs) != len(selected):
        raise RuntimeError(
            f"Expected {len(selected)} outputs, got {len(outputs)}"
        )

    reward_matrix = np.zeros(
        (len(selected), args.group_size),
        dtype=np.float64,
    )
    generation_records = []

    for index, (example, prompt, request_output) in enumerate(
        zip(selected, prompts, outputs)
    ):
        if len(request_output.outputs) != args.group_size:
            raise RuntimeError(
                f"Example {index}: expected {args.group_size} completions, "
                f"got {len(request_output.outputs)}"
            )

        sample_rewards = []

        for sample_idx, completion_output in enumerate(
            request_output.outputs
        ):
            completion = completion_output.text
            token_ids = list(completion_output.token_ids)
            finish_reason = completion_output.finish_reason

            reward = float(
                math500_reward_func(
                    completion,
                    example["answer"],
                )
            )

            reward_matrix[index, sample_idx] = reward
            sample_rewards.append(reward)

            generation_records.append(
                {
                    "prompt_index": index,
                    "group_index": sample_idx,
                    "group_sample": sample_idx + 1,
                    "completion_tokens": len(token_ids),
                    "raw_completion_tokens": len(token_ids),
                    "completion_characters": len(completion),
                    "ended_with_eos": finish_reason == "stop",
                    "likely_truncated": finish_reason == "length",
                    "finish_reason": finish_reason,
                    "reward": reward,
                }
            )

            if index < 3:
                prediction_text, extraction_method = (
                    extract_math500_answer(completion)
                )
                target_text = extract_math500_gold(
                    example["answer"]
                )

                print("\n" + "#" * 100)
                print(
                    f"[DEBUG EVAL EXAMPLE {index} / "
                    f"SAMPLE {sample_idx + 1}/{args.group_size}]"
                )
                print("#" * 100)

                print("\nPROMPT:\n" + "-" * 80)
                print(prompt)
                print("-" * 80)

                print(
                    "\nPROMPT TOKEN COUNT:",
                    len(
                        tokenizer.encode(
                            prompt,
                            add_special_tokens=True,
                        )
                    ),
                )

                print("\nGENERATED COMPLETION:\n" + "-" * 80)
                print(completion)
                print("-" * 80)

                print("\nCOMPLETION TOKEN COUNT:", len(token_ids))
                print("FINISH REASON:", finish_reason)
                print("EXTRACTION METHOD:", extraction_method)
                print("PREDICTION:", repr(prediction_text))
                print("GOLD:", repr(target_text))
                print("REWARD:", reward)
                print("#" * 100)

        if index < 3:
            print(f"[EXAMPLE {index} SUMMARY]")
            print(f"Rewards:       {sample_rewards}")
            print(f"Mean reward:   {np.mean(sample_rewards):.6f}")
            print(
                f"Pass@{args.group_size}:       "
                f"{int(any(x > 0.5 for x in sample_rewards))}"
            )

    flat_rewards = reward_matrix.reshape(-1)
    if flat_rewards.size == 0:
        raise RuntimeError("No rewards were computed.")

    # y value: mean reward over all M * N generations.
    mean_reward = float(reward_matrix.mean())

    # Per-prompt mean reward. Its mean is exactly equal to mean_reward
    # because every prompt has the same number of generations.
    prompt_mean_rewards = reward_matrix.mean(axis=1)

    if len(prompt_mean_rewards) >= 2:
        prompt_mean_std = float(
            prompt_mean_rewards.std(ddof=1)
        )
        prompt_mean_se = float(
            prompt_mean_std / np.sqrt(len(prompt_mean_rewards))
        )
    else:
        prompt_mean_std = 0.0
        prompt_mean_se = 0.0

    # Diagnostics based on generation slots.
    generation_means = reward_matrix.mean(axis=0)

    if args.group_size >= 2:
        generation_mean_std = float(
            generation_means.std(ddof=1)
        )
        std_error_generation_mean = float(
            generation_mean_std / np.sqrt(args.group_size)
        )

        prompt_variances = reward_matrix.var(
            axis=1,
            ddof=1,
        )
        prompt_within_group_rms_std = float(
            np.sqrt(prompt_variances.mean())
        )
    else:
        generation_mean_std = 0.0
        std_error_generation_mean = 0.0
        prompt_within_group_rms_std = 0.0

    # Naive generation-level dispersion and SE.
    if flat_rewards.size >= 2:
        reward_std = float(
            flat_rewards.std(ddof=1)
        )
        reward_se = float(
            reward_std / np.sqrt(flat_rewards.size)
        )
    else:
        reward_std = 0.0
        reward_se = 0.0

    generation_correct_mask = flat_rewards > 0.5
    num_correct_generations = int(
        generation_correct_mask.sum()
    )
    generation_accuracy = float(
        generation_correct_mask.mean()
    )

    pass_at_n = float(
        (reward_matrix > 0.5).any(axis=1).mean()
    )

    print("\n" + "=" * 100)
    print("EVALUATION RESULT")
    print("=" * 100)
    print(f"Examples:              {len(selected)}")
    print(f"Generations/example:   {args.group_size}")
    print(f"Total generations:     {flat_rewards.size}")
    print(f"Correct generations:   {num_correct_generations}")
    print(f"Generation accuracy:   {generation_accuracy:.6f}")
    print(f"Pass@{args.group_size}:                {pass_at_n:.6f}")
    print(f"Mean reward:           {mean_reward:.6f}")
    # print(f"Prompt-mean std:       {prompt_mean_std:.6f}")
    # print(f"Prompt-mean SE:        {prompt_mean_se:.6f}")
    print(f"Generation-mean std:   {generation_mean_std:.6f}")
    print(
        f"Generation-mean SE:    "
        f"{std_error_generation_mean:.6f}"
    )
    # print(
    #     f"Prompt within-group RMS std: "
    #     f"{prompt_within_group_rms_std:.6f}"
    # )
    # print(f"Reward std:            {reward_std:.6f}")
    # print(f"Reward SE:             {reward_se:.6f}")
    print("=" * 100)

    return (
        mean_reward,
        prompt_mean_std,
        prompt_mean_se,
        generation_mean_std,
        std_error_generation_mean,
        prompt_within_group_rms_std,
        reward_std,
        reward_se,
        generation_records,
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


# ============================================================
# Load HF model
# ============================================================

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


# ============================================================
# Random direction
# ============================================================

def random_direction_like(
    theta_state,
    seed,
):

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        seed
    )

    direction = {}

    for name, tensor in (
        theta_state.items()
    ):

        direction[name] = torch.randn(
            tensor.shape,
            generator=generator,
            dtype=torch.float32,
        )

    norm = global_norm(
        direction
    )

    direction = {
        name:
            tensor / norm
        for name, tensor
        in direction.items()
    }

    direction = (
        scale_to_state_norm(
            direction,
            global_norm(theta_state),
        )
    )

    return direction


# ============================================================
# CUDA cleanup
# ============================================================

def cleanup_cuda():

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# vLLM direction loading
# ============================================================

def install_direction_on_vllm(
    llm,
    direction_path,
):
    """
    IMPORTANT:

    The driver does NOT send the direction tensor
    through apply_model().

    Instead, apply_model() only sends the path.

    Inside the vLLM worker:

        torch.load(direction_path)

    loads the direction from disk into CPU memory.

    This avoids serializing a potentially ~2.4 GB
    direction through the vLLM RPC mechanism.
    """

    direction_path = str(
        Path(direction_path).resolve()
    )

    print()
    print("=" * 80)
    print(
        "Installing direction from file"
    )
    print("=" * 80)

    print(
        "Direction file:",
        direction_path,
    )

    if not os.path.exists(
        direction_path
    ):

        raise FileNotFoundError(
            f"Direction file does not exist: "
            f"{direction_path}"
        )

    # --------------------------------------------------------
    # Only a string is captured by this closure.
    # --------------------------------------------------------

    def worker_install(
        model,
    ):

        print(
            "[vLLM worker] "
            f"Loading direction from "
            f"{direction_path}",
            flush=True,
        )

        direction = torch.load(
            direction_path,
            map_location="cpu",
            weights_only=True,
        )

        if not isinstance(
            direction,
            dict,
        ):

            raise RuntimeError(
                "Direction file must contain "
                "a dictionary."
            )

        # ----------------------------------------------------
        # Store CPU direction in worker.
        #
        # Single GPU means only one worker.
        # ----------------------------------------------------

        model._reward_landscape_direction = (
            direction
        )

        model._reward_landscape_direction_path = (
            direction_path
        )

        model._reward_landscape_coefficient = (
            0.0
        )

        return len(
            direction
        )

    results = llm.apply_model(
        worker_install
    )

    print(
        "Direction loaded by worker.",
        results,
    )


# ============================================================
# Set vLLM coefficient
# ============================================================

def set_vllm_coefficient(
    llm,
    coefficient,
):
    """
    Current model:

        theta =
            theta_ckpt
            + coefficient * direction

    We update incrementally:

        delta =
            new_coefficient
            - old_coefficient

        theta <- theta + delta * direction
    """

    coefficient = float(
        coefficient
    )

    def worker_set_coefficient(
        model,
    ):

        old = getattr(
            model,
            "_reward_landscape_coefficient",
            0.0,
        )

        delta = (
            coefficient
            - old
        )

        if abs(delta) > 0:

            direction = getattr(
                model,
                "_reward_landscape_direction",
                None,
            )

            if direction is None:

                raise RuntimeError(
                    "Direction has not been "
                    "loaded into this vLLM worker."
                )

            with torch.no_grad():

                matched = 0

                for name, param in (
                    model.named_parameters()
                ):

                    if name not in direction:
                        continue

                    d = direction[name]

                    if tuple(
                        d.shape
                    ) != tuple(
                        param.shape
                    ):

                        raise RuntimeError(
                            "Shape mismatch for "
                            f"{name}: "
                            f"direction={d.shape}, "
                            f"parameter={param.shape}"
                        )

                    # ------------------------------------------------
                    # CPU -> GPU only during this update.
                    #
                    # d itself remains stored on CPU.
                    # ------------------------------------------------

                    d_device = d.to(
                        device=param.device,
                        dtype=param.dtype,
                    )

                    param.add_(
                        d_device,
                        alpha=delta,
                    )

                    matched += 1

                    del d_device

                if matched == 0:

                    raise RuntimeError(
                        "No parameters matched "
                        "between direction file "
                        "and vLLM model."
                    )

            # --------------------------------------------------------
            # Optional garbage collection on worker.
            # --------------------------------------------------------

            del direction

        model._reward_landscape_coefficient = (
            coefficient
        )

        return coefficient

    results = llm.apply_model(
        worker_set_coefficient
    )

    return results


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    # ========================================================
    # Model
    # ========================================================

    parser.add_argument(
        "--model-ckpt",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--dir-step",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--pg-num-prompts",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--task",
        choices=(
            "chess",
            "gsm8k",
            "math500",
        ),
        default="math500",
    )

    parser.add_argument(
        "--direction-path",
        type=str,
        default=None,
        help=(
            "Existing SGD direction .pt file. "
            "If provided, skip GRPO gradient computation and load it directly."
        ),
    )

    parser.add_argument(
        "--eval-json",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen3_0.6b",
    )

    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
    )

    # ========================================================
    # vLLM
    # ========================================================

    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=[
            "bfloat16",
            "float16",
        ],
    )

    # ========================================================
    # Reward landscape
    # ========================================================

    parser.add_argument(
        "--num-samples",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=(
            "Kept for compatibility. "
            "vLLM automatically batches prompts."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--alpha-range",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--alpha-left",
        type=float,
        default=10.0,
        help="Absolute alpha range on the left side. "
            "The actual left endpoint is -alpha-left.",
    )

    parser.add_argument(
        "--alpha-right",
        type=float,
        default=10.0,
        help="Absolute alpha range on the right side. "
            "The actual right endpoint is +alpha-right.",
    )

    parser.add_argument(
        "--num-points",
        type=int,
        default=15,
    )

    # ========================================================
    # IMPORTANT:
    #
    # One SGD rollout -> one SGD direction.
    # ========================================================

    parser.add_argument(
        "--num-directions",
        type=int,
        default=1,
    )

    # ========================================================
    # Direction
    # ========================================================

    parser.add_argument(
        "--direction-type",
        type=str,
        choices=[
            "random",
            "grpo-grad",
            "displacement",
        ],
        default="grpo-grad",
    )

    # ========================================================
    # GRPO
    # ========================================================

    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    # ========================================================
    # Output
    # ========================================================

    parser.add_argument(
        "--output-dir",
        type=str,
        default="figs_qwen_math500_sgd",
    )

    parser.add_argument(
        "--direction-dir",
        type=str,
        default=None,
        help=(
            "Directory for saved SGD direction files. "
            "Defaults to <output-dir>/directions."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Environment
    # ========================================================

    os.environ.setdefault(
        "TOKENIZERS_PARALLELISM",
        "false",
    )

    print("=" * 80)
    print("Configuration")
    print("=" * 80)

    for k, v in vars(args).items():

        print(
            f"{k}: {v}"
        )

    # ========================================================
    # Seed
    # ========================================================

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    # ========================================================
    # Check single GPU
    # ========================================================

    if args.tensor_parallel_size != 1:

        raise ValueError(
            "This version is configured "
            "for SINGLE-GPU vLLM. "
            "Use --tensor-parallel-size 1."
        )

    # ========================================================
    # Tokenizer
    # ========================================================

    tokenizer = (
        AutoTokenizer.from_pretrained(
            args.model_ckpt,
            trust_remote_code=True,
        )
    )

    # ========================================================
    # Data
    # ========================================================

    task = (
        args.task
        .lower()
        .strip()
    )

    data = load_eval_data(
        args.eval_json,
        args.num_samples,
        task,
    )

    print()
    print(
        f"Loaded {len(data)} "
        f"evaluation examples."
    )

    # ========================================================
    # Output directories
    # ========================================================

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.direction_dir is None:

        direction_dir = (
            output_dir
            / "directions"
        )

    else:

        direction_dir = Path(
            args.direction_dir
        )

    direction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # PHASE 1
    #
    # vLLM SINGLE GPU
    #
    # Generate GRPO rollout
    # ========================================================


    # ========================================================
    # PHASE 2
    #
    # HF model
    #
    # Compute SGD / GRPO direction
    # ========================================================

    directions = []
    direction_paths = []

    if args.direction_type in ("displacement"):

        # ========================================================
        # Existing direction: directly use it
        # ========================================================

        if args.direction_path is not None:

            direction_path = Path(
                args.direction_path
            )

            if not direction_path.exists():
                raise FileNotFoundError(
                    f"SGD or displacement direction file does not exist: "
                    f"{direction_path}"
                )

            print()
            print("=" * 80)
            print("Using existing displacement direction")
            print("=" * 80)
            print(f"Direction path: {direction_path}")

            direction_paths.append(
                str(direction_path.resolve())
            )

        # ========================================================
        # No existing direction: compute from GRPO gradient
        # ========================================================

        else:

            print("requires --direction-path for displacement direction type.")



    # ========================================================
    # Random direction
    # ========================================================

    else:

        # Random direction requires checkpoint parameter shapes.
        hf_model = load_hf_model(args)

        theta_state = get_target_state(
            hf_model
        )

        for direction_idx in range(
            args.num_directions
        ):

            direction = random_direction_like(
                theta_state,
                args.seed + direction_idx,
            )

            step_name = (
                f"step{args.checkpoint_step}"
                if args.checkpoint_step is not None
                else "checkpoint"
            )

            direction_path = (
                direction_dir
                / (
                    f"{args.model_name}_"
                    f"{step_name}_"
                    f"random_direction_"
                    f"{direction_idx}.pt"
                )
            )

            save_direction(
                direction,
                direction_path,
            )

            direction_paths.append(
                str(direction_path.resolve())
            )

            del direction

            cleanup_cuda()

        del theta_state
        del hf_model

        cleanup_cuda()

    # ========================================================
    # IMPORTANT:
    #
    # We do NOT keep the direction tensors in memory.
    #
    # directions = []
    #
    # Everything will be loaded by vLLM worker from disk.
    # ========================================================

    directions = []

    print()
    print(
        "Saved direction files:"
    )

    for path in direction_paths:

        print(
            "  ",
            path
        )

    # ========================================================
    # PHASE 3
    #
    # Reload SAME checkpoint
    #
    # SINGLE GPU vLLM
    # ========================================================

    llm = create_vllm(
        args
    )

    # ========================================================
    # Alpha grid
    # ========================================================

    # alpha_values = np.linspace(
    #     -args.alpha_range,
    #     args.alpha_range,
    #     args.num_points,
    # )

    alpha_values = np.linspace(
        -args.alpha_left,
        args.alpha_right,
        args.num_points,
    )

    alpha_values = np.unique(np.append(alpha_values, [0.0, 1.0]))

    # ========================================================
    # Results
    # ========================================================

    all_results = []

    # ========================================================
    # PHASE 4
    #
    # Direction file -> worker
    #
    # Parameter perturbation
    # + vLLM evaluation
    # ========================================================

    try:

        for direction_idx, direction_path in enumerate(
            direction_paths
        ):

            print()
            print("=" * 80)

            print(
                f"DIRECTION "
                f"{direction_idx + 1}/"
                f"{len(direction_paths)}"
            )

            print("=" * 80)

            print(
                "Direction file:",
                direction_path,
            )

            # ------------------------------------------------
            # Worker reads direction from disk.
            # ------------------------------------------------

            install_direction_on_vllm(
                llm,
                direction_path,
            )

            direction_results = []

            # ------------------------------------------------
            # Sweep alpha
            # ------------------------------------------------
            all_generation_records = []
            for alpha in alpha_values:

                coefficient = (
                    float(alpha)
                    * args.scale
                )

                print()
                print(
                    "-" * 70
                )

                print(
                    f"direction="
                    f"{direction_idx}"
                )

                print(
                    f"alpha="
                    f"{alpha:.6f}"
                )

                print(
                    f"coefficient="
                    f"{coefficient:.8f}"
                )

                # ------------------------------------------------
                # theta =
                #
                # theta_ckpt
                # +
                # coefficient * direction
                # ------------------------------------------------

                set_vllm_coefficient(
                    llm,
                    coefficient,
                )

                # ------------------------------------------------
                # Evaluate
                # ------------------------------------------------
                

                (
                    mean_reward,
                    prompt_mean_std,
                    prompt_mean_se,
                    generation_mean_std,
                    std_error_generation_mean,
                    prompt_within_group_rms_std,
                    reward_std,
                    reward_se,
                    generation_records,
                ) = vllm_eval_reward(llm, tokenizer, data, args)

                row = {
                    "direction": direction_idx,
                    "direction_path": direction_path,
                    "alpha": float(alpha),
                    "coefficient": float(coefficient),
                    "mean_reward": mean_reward,
                    "prompt_mean_std": prompt_mean_std,
                    "prompt_mean_se": prompt_mean_se,
                    "generation_mean_std": generation_mean_std,
                    "std_error_generation_mean": std_error_generation_mean,
                    "prompt_within_group_rms_std": prompt_within_group_rms_std,
                    "reward_std": reward_std,
                    "reward_se": reward_se,
                }
                direction_results.append(row)
                all_results.append(row)

                for record in generation_records:
                    all_generation_records.append(
                        {
                            "direction": direction_idx,
                            "direction_path": direction_path,
                            "alpha": float(alpha),
                            "coefficient": float(coefficient),
                            **record,
                        }
                    )

                print(f"Mean reward = {mean_reward:.6f}")
                # print(f"Prompt-mean std = {prompt_mean_std:.6f}")
                # print(f"Prompt-mean SE = {prompt_mean_se:.6f}")
                print(f"Generation-mean std = {generation_mean_std:.6f}")
                print(f"Generation-mean SE = {std_error_generation_mean:.6f}")
                # print(
                #     f"Prompt within-group RMS std = "
                #     f"{prompt_within_group_rms_std:.6f}"
                # )
                # print(f"Reward std = {reward_std:.6f}")
                # print(f"Reward SE = {reward_se:.6f}", flush=True)

            # ------------------------------------------------
            # Restore checkpoint
            # ------------------------------------------------

            print()
            print(
                "Restoring checkpoint "
                "parameters by setting "
                "coefficient to 0..."
            )

            set_vllm_coefficient(
                llm,
                0.0,
            )

            print(
                "Direction finished."
            )

    finally:

        print(
            "Destroying final "
            "vLLM instance..."
        )

        del llm

        cleanup_cuda()

    # ========================================================
    # Save CSV
    # ========================================================

    df = pd.DataFrame(all_results)
    generation_df = pd.DataFrame(all_generation_records)

    step_name = (
        f"stp{args.checkpoint_step}"
        if args.checkpoint_step is not None
        else "checkpoint"
    )
    stem = f"{args.model_name}_{step_name}_to{args.dir_step}_{args.direction_type}_grp{args.group_size}_tmp{args.temperature}_scale{args.scale:.2f}_alpha{args.alpha_left}_{args.alpha_right}_npts{args.num_points}_seed{args.seed}"

    csv_path = output_dir / f"{stem}.csv"
    generation_csv_path = output_dir / f"completion_lengths_{stem}.csv"
    df.to_csv(csv_path, index=False)

    if not generation_df.empty:
        generation_df = generation_df.sort_values(
            ["direction", "alpha", "prompt_index", "group_index"]
        )
    generation_df.to_csv(generation_csv_path, index=False)

    def plot_landscape(relative: bool, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))

        for direction_idx, subset in df.groupby("direction"):
            subset = subset.sort_values("coefficient")

            x = subset["coefficient"].to_numpy(dtype=float)
            y = subset["mean_reward"].to_numpy(dtype=float)
            se = subset["std_error_generation_mean"].to_numpy(dtype=float)

            if relative:
                # 使用该 direction 在 alpha=0 时实际测得的 reward。
                zero_rows = subset.loc[subset["alpha"] == 0.0]

                if len(zero_rows) != 1:
                    raise ValueError(
                        f"Direction {direction_idx}: 需要唯一的 alpha=0 结果"
                    )

                baseline = float(zero_rows["mean_reward"].iloc[0])

                if not np.isfinite(baseline) or baseline == 0.0:
                    raise ValueError(
                        f"Direction {direction_idx}: "
                        f"0 点 reward={baseline}，无法归一化"
                    )

                y = y / baseline
                se = se / abs(baseline)

            line = ax.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                label=f"Direction {direction_idx}",
            )[0]

            lower = y - se
            upper = y + se

            # 原始 reward 保留 [0, 1] 裁剪；
            # relative reward 可以超过 1，不能按 [0, 1] 裁剪。
            if not relative:
                lower = np.clip(lower, 0.0, 1.0)
                upper = np.clip(upper, 0.0, 1.0)

            ax.fill_between(
                x,
                lower,
                upper,
                alpha=0.2,
                color=line.get_color(),
            )

        ax.axvline(0.0, linestyle="--", linewidth=1, color="black")

        if relative:
            ax.axhline(1.0, linestyle=":", linewidth=1, color="gray")
            ax.set_ylabel("Relative Reward: R(coefficient) / R(0)")
            ax.set_title("Math500 Relative Reward Landscape")
        else:
            ax.set_ylabel("Mean Math500 Reward")
            ax.set_title("Math500 Absolute Reward Landscape")

        ax.set_ylim(0.0, 1.5)
        ax.set_xlabel("Perturbation coefficient")
        ax.minorticks_on()
        ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.8)
        ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=300)
        plt.close(fig)

    plot_landscape(
        relative=False,
        filename=f"{stem}_absolute_reward_generation_mean_se.png",
    )

    plot_landscape(
        relative=True,
        filename=f"{stem}_relative_reward_generation_mean_se.png",
    )

    # plot_landscape(
    #     "prompt_mean_se",
    #     f"{stem}_prompt_mean_se.png",
    #     "prompt-level mean SE",
    # )

    # # Prompt-level 标准差。
    # plot_landscape(
    #     "prompt_mean_std",
    #     f"{stem}_prompt_mean_std.png",
    #     "prompt-level mean std",
    # )


    # 不同 generation slot 的平均 reward 波动。
    # plot_landscape(
    #     "generation_mean_std",
    #     f"{stem}_generation_mean_std.png",
    #     "generation-mean std",
    # )

    # # 基于 generation slot 的标准误。
    # plot_landscape(
    #     "std_error_generation_mean",
    #     f"{stem}_generation_mean_se.png",
    #     "generation-mean SE",
    # )

    # # 同一道题内部，不同生成之间的典型波动。
    # plot_landscape(
    #     "prompt_within_group_rms_std",
    #     f"{stem}_prompt_rms.png",
    #     "within-prompt generation RMS std",
    # )

    # # 所有 generation reward 的总体标准差。
    # plot_landscape(
    #     "reward_std",
    #     f"{stem}_reward_std.png",
    #     "reward std",
    # )

    # # 假设所有 generation 独立时的 naive SE。
    # plot_landscape(
    #     "reward_se",
    #     f"{stem}_reward_se.png",
    #     "naive reward SE",
    # )

    print(f"Saved results to: {csv_path}")
    print(f"Saved generation records to: {generation_csv_path}")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":

    main()