#!/usr/bin/env python3
# env CUDA_VISIBLE_DEVICE=0,2,3 nohup python eval_math500.py > eval_math500.log 2>&1 &

"""Evaluate a Qwen3 model on Math500 with vLLM and the PRM800K grader."""

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from swift.grading.grader import grade_answer


# ============================================================
# Configuration
# ============================================================

DEFAULT_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within "
    r"\boxed{}."
)

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

# MODEL_NAME = "Qwen3-0.6B-Base"
# MODEL_NAME = "Qwen3-1.7B-Base"
MODEL_NAME = "Qwen3-1.7B"
# MODEL_NAME = "qwen3_max1024_lr_2e-6_grpo_stp225"

# MODEL_NAME = "qwen3_max1024_lr_8e-6_grpo_stp350"

MODEL_PATH = f"/tmp/erin/{MODEL_NAME}"

# MODEL_PATH = (
#     "/mnt/swordfish-pool2/erinxia/ms-swift/"
#     "output_qwen3_0.6b_base_nonthink_grpo_math500_lr_2e-6_max1024/"
#     "v0-20260809-175321/checkpoint-225"
# )

# MODEL_PATH = (
#     "/mnt/swordfish-pool2/erinxia/ms-swift/"
#     "output_qwen3_0.6b_base_nonthink_grpo_math500_lr_8e-6_max1024/"
#     "v1-20260809-232426/checkpoint-350"
# )


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

# DATA_PATH = (
#     "/mnt/swordfish-pool2/erinxia/rlvr-landscape/"
#     "train_data/math500/train.jsonl"
# )

# If you want the standard 500-example Math500 test set,
# replace DATA_PATH with:
#
DATA_PATH = (
    "/mnt/swordfish-pool2/erinxia/rlvr-landscape/"
    "train_data/math500/test/test_500.jsonl"
)


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

# OUTPUT_PATH = (
#     "/mnt/swordfish-pool2/erinxia/rlvr-landscape/"
#     "math500_eval/"
#     f"{MODEL_NAME.lower().replace('-', '_')}"
#     "_math500_train300_predictions.jsonl"
# )

OUTPUT_PATH = (
    "/mnt/swordfish-pool2/erinxia/rlvr-landscape/"
    "math500_eval/"
    f"{MODEL_NAME.lower().replace('-', '_')}"
    "_math500_predictions.jsonl"
)

SUMMARY_PATH = OUTPUT_PATH.replace(
    "_predictions.jsonl",
    "_summary.json",
)


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

LIMIT: int | None = None

SEED = 42

# Qwen3 non-thinking recommended sampling configuration.
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20

# Maximum generation length.
MAX_NEW_TOKENS = 32768

# Qwen3 context length.
MAX_MODEL_LEN = 32768

# vLLM GPU memory usage.
GPU_MEMORY_UTILIZATION = 0.9

DTYPE = "bfloat16"

PROMPT_SUFFIX = DEFAULT_SUFFIX

# Use Qwen3 chat template with thinking explicitly disabled.
USE_CHAT_TEMPLATE = True


# ============================================================
# Text extraction helpers
# ============================================================

def extract_text_content(content: Any) -> str:
    """Extract text from either a string or multimodal message content."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict)
            and item.get("type") == "text"
        ).strip()

    return str(content)


def extract_problem(record: dict[str, Any]) -> str:
    """Extract the problem statement from a dataset record."""

    if "problem" in record:
        return str(record["problem"]).strip()

    if "prompt" in record and isinstance(record["prompt"], str):
        return record["prompt"].strip()

    if "messages" in record:
        parts = [
            extract_text_content(message.get("content", ""))
            for message in record["messages"]
            if (
                isinstance(message, dict)
                and message.get("role") == "user"
            )
        ]

        return "\n".join(
            part for part in parts if part
        ).strip()

    raise KeyError(
        "record has no problem, prompt, or user message"
    )


def extract_braced_command(
    text: str,
    command: str,
) -> list[str]:
    """Extract contents of commands such as \\boxed{...}."""

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
                text[start:index - 1].strip()
            )

    return results


def extract_prediction(
    text: str,
) -> tuple[str, str]:
    """
    Extract the final answer from model output.

    Priority:
      1. <answer>...</answer>
      2. \\boxed{...}
      3. Final Answer: ...
      4. Answer: ...
    """

    # --------------------------------------------------------
    # <answer>...</answer>
    # --------------------------------------------------------

    tags = re.findall(
        r"<answer>(.*?)</answer>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if tags:
        return tags[-1].strip(), "answer_tag"

    # --------------------------------------------------------
    # \boxed{...}
    # --------------------------------------------------------

    boxed = extract_braced_command(
        text,
        "boxed",
    )

    if boxed:
        return boxed[-1], "boxed"

    # --------------------------------------------------------
    # Final Answer / Answer
    # --------------------------------------------------------

    matches = list(
        re.finditer(
            r"(?:Final\s+Answer|Answer)\s*:?\s*",
            text,
            flags=re.IGNORECASE,
        )
    )

    if matches:
        remainder = text[
            matches[-1].end():
        ].strip()

        for line in remainder.splitlines():
            if line.strip():
                return (
                    line.strip(),
                    "final_answer",
                )

    return "", "no_answer_marker"


def extract_gold(
    record: dict[str, Any],
) -> str:
    """Extract the ground-truth answer."""

    value = record.get(
        "answer",
        record.get("solution"),
    )

    if value is None:
        raise KeyError(
            "record has neither answer nor solution"
        )

    text = str(value).strip()

    tags = re.findall(
        r"<answer>(.*?)</answer>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if tags:
        return tags[-1].strip()

    boxed = extract_braced_command(
        text,
        "boxed",
    )

    return boxed[-1] if boxed else text


# ============================================================
# Dataset loading
# ============================================================

def load_records(
    path: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Load JSONL or JSON dataset."""

    data_path = Path(path)

    # --------------------------------------------------------
    # JSONL
    # --------------------------------------------------------

    if data_path.suffix.lower() == ".jsonl":
        with data_path.open(
            encoding="utf-8"
        ) as file:
            records = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    else:
        with data_path.open(
            encoding="utf-8"
        ) as file:
            loaded = json.load(file)

        if isinstance(loaded, dict):
            loaded = loaded.get(
                "test",
                loaded.get(
                    "data",
                    loaded,
                ),
            )

        if not isinstance(loaded, list):
            raise TypeError(
                "JSON input must contain a list of examples"
            )

        records = loaded

    if limit is not None:
        records = records[:limit]

    return records


# ============================================================
# Prompt formatting
# ============================================================

def format_prompt(
    tokenizer: Any,
    problem: str,
    suffix: str,
    use_chat_template: bool,
) -> str:
    """
    Format a Qwen3 non-thinking prompt.

    enable_thinking=False explicitly disables <think> mode.
    """

    content = problem + suffix

    if use_chat_template:
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

    return content


# ============================================================
# Main
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print("=" * 70)
    print("Math500 Evaluation")
    print("=" * 70)

    print(f"Model:              {MODEL_PATH}")
    print(f"Dataset:            {DATA_PATH}")
    print(f"Output:             {OUTPUT_PATH}")

    print()
    print("Backend:")
    print("  vLLM")

    print()
    print("Qwen3 mode:")
    print("  enable_thinking:   False")

    print()
    print("Sampling:")
    print(f"  do_sample:         True")
    print(f"  temperature:       {TEMPERATURE}")
    print(f"  top_p:             {TOP_P}")
    print(f"  top_k:             {TOP_K}")
    print(f"  max_new_tokens:    {MAX_NEW_TOKENS}")

    print()
    print("vLLM:")
    print(f"  dtype:             {DTYPE}")
    print(f"  max_model_len:     {MAX_MODEL_LEN}")
    print(
        f"  gpu_memory_util:   "
        f"{GPU_MEMORY_UTILIZATION}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------------
    # Load model with vLLM
    # --------------------------------------------------------

    llm = LLM(
        model=MODEL_PATH,
        dtype=DTYPE,
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        seed=SEED,
    )

    # --------------------------------------------------------
    # Sampling parameters
    # --------------------------------------------------------

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_NEW_TOKENS,
        seed=SEED,
        stop=[
            "<|im_end|>",
            "<|endoftext|>",
        ],
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    records = load_records(
        DATA_PATH,
        LIMIT,
    )

    print()
    print(f"Number of examples: {len(records)}")

    # --------------------------------------------------------
    # Prepare output
    # --------------------------------------------------------

    output_path = Path(OUTPUT_PATH)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    correct = 0
    scored = 0

    # --------------------------------------------------------
    # Prepare prompts
    # --------------------------------------------------------

    problems = [
        extract_problem(record)
        for record in records
    ]

    golds = [
        extract_gold(record)
        for record in records
    ]

    prompts = [
        format_prompt(
            tokenizer,
            problem,
            PROMPT_SUFFIX,
            USE_CHAT_TEMPLATE,
        )
        for problem in problems
    ]

    # --------------------------------------------------------
    # Generate
    #
    # vLLM automatically handles batching.
    # --------------------------------------------------------

    print()
    print("Starting generation...")

    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=True,
    )

    print("Generation finished.")

    # --------------------------------------------------------
    # Grade
    # --------------------------------------------------------

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for index, (
            problem,
            gold,
            output,
        ) in enumerate(
            zip(
                problems,
                golds,
                outputs,
            )
        ):

            # vLLM returns one completion because
            # n=1 by default.
            completion = output.outputs[0].text

            prediction, method = extract_prediction(
                completion
            )

            # ------------------------------------------------
            # PRM800K-style answer grading
            # ------------------------------------------------

            try:
                reward = float(
                    bool(prediction)
                    and grade_answer(
                        given_answer=prediction,
                        ground_truth=gold,
                    )
                )

                error = None

            except Exception as exc:

                reward = 0.0

                error = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

            correct += int(reward)
            scored += 1

            # ------------------------------------------------
            # Save prediction
            # ------------------------------------------------

            result = {
                "index": index,
                "problem": problem,
                "gold": gold,
                "completion": completion,
                "prediction": prediction,
                "extraction_method": method,
                "reward": reward,
            }

            if error is not None:
                result["grader_error"] = error

            output_file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if scored % 10 == 0:
                accuracy = correct / scored

                tqdm.write(
                    f"scored={scored} "
                    f"correct={correct} "
                    f"accuracy={accuracy:.4%}"
                )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    accuracy = (
        correct / scored
        if scored
        else float("nan")
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = Path(SUMMARY_PATH)

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "model_name": MODEL_NAME,
        "model_path": MODEL_PATH,
        "data_path": DATA_PATH,
        "predictions_path": str(
            output_path.resolve()
        ),

        "num_examples": scored,
        "num_correct": correct,

        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,

        # ----------------------------------------------------
        # Qwen3 non-thinking configuration
        # ----------------------------------------------------

        "enable_thinking": False,

        "do_sample": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,

        "max_new_tokens": MAX_NEW_TOKENS,

        # ----------------------------------------------------
        # vLLM configuration
        # ----------------------------------------------------

        "backend": "vllm",
        "dtype": DTYPE,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": (
            GPU_MEMORY_UTILIZATION
        ),

        "seed": SEED,

        # ----------------------------------------------------
        # Grader
        # ----------------------------------------------------

        "grader": (
            "swift.grading.grader.grade_answer"
        ),
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as summary_file:

        json.dump(
            summary,
            summary_file,
            ensure_ascii=False,
            indent=2,
        )

        summary_file.write("\n")

    # --------------------------------------------------------
    # Print final result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Evaluation Finished")
    print("=" * 70)

    print(f"Model:       {MODEL_PATH}")
    print(f"Examples:    {scored}")
    print(f"Correct:     {correct}")

    print(
        f"Accuracy:    "
        f"{accuracy:.6f} "
        f"({accuracy:.2%})"
    )

    print(
        f"Predictions: "
        f"{output_path.resolve()}"
    )

    print(
        f"Summary:     "
        f"{summary_path.resolve()}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()