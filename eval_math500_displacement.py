#!/usr/bin/env python3
# Run:
# CUDA_VISIBLE_DEVICES=7 nohup python eval_math500_displacement.py > eval_math500_train_qwen3_0.6b_step50_max2048_temp0.log 2>&1 &

"""Evaluate Qwen3 models on Math500 with vLLM and the PRM800K grader."""

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

# DEFAULT_SUFFIX = (
#     ""
# )


# ============================================================
# Model
# ============================================================

# Choose one model.

# MODEL_NAME = "Qwen3-0.6B-Base"
# MODEL_NAME = "Qwen3-1.7B-Base"
# MODEL_NAME = "Qwen3-1.7B"

MAX_NEW_TOKENS = 2048
# MAX_NEW_TOKENS = 8192
STEP_I = 50
# TEMPERATURE = 0.7
TEMPERATURE = 0.0
MODEL_PATH = "/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_5e-6_max2048/v18-20260903-125040/checkpoint-50"
# MODEL_PATH = "/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_5e-6_max2048/v15-20260903-122903/checkpoint-3"
# ============================================================
# Dataset
# ============================================================

DATA_PATH = (
    "/mnt/swordfish-pool2/erinxia/rlvr-landscape/"
    "train_data/math500/train.jsonl"
)


# ============================================================
# Output
# ============================================================

OUTPUT_PATH = (
    "/mnt/swordfish-pool2/erinxia/rlvr-landscape/"
    "math500_eval/"
    f"step{STEP_I}_max{MAX_NEW_TOKENS}"
    f"_temp{TEMPERATURE}"
    "_math500_train_predictions.jsonl"
)

SUMMARY_PATH = OUTPUT_PATH.replace(
    "_predictions.jsonl",
    "_summary.json",
)


# ============================================================
# General evaluation settings
# ============================================================

LIMIT: int | None = None

SEED = 42

# --------------------------------------------------------
# Qwen3 Instruct model
#
# Official non-thinking sampling configuration.
# --------------------------------------------------------

ENABLE_THINKING = False

USE_CHAT_TEMPLATE = True




TOP_P = 0.8

TOP_K = 20

MAX_NEW_TOKENS = 32768

STOP = [
    "<|im_end|>",
    "<|endoftext|>",
]


# ============================================================
# vLLM settings
# ============================================================

MAX_MODEL_LEN = 32768

GPU_MEMORY_UTILIZATION = 0.8

DTYPE = "bfloat16"

# Change this according to the number of GPUs.
#
# For:
#   CUDA_VISIBLE_DEVICES=0
# use 1.
#
# For:
#   CUDA_VISIBLE_DEVICES=0,1,2
# use 3.

TENSOR_PARALLEL_SIZE = 1


# ============================================================
# Text extraction helpers
# ============================================================

def extract_text_content(content: Any) -> str:

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


def extract_problem(
    record: dict[str, Any],
) -> str:

    if "problem" in record:
        return str(record["problem"]).strip()

    if (
        "prompt" in record
        and isinstance(record["prompt"], str)
    ):
        return record["prompt"].strip()

    if "messages" in record:

        parts = [
            extract_text_content(
                message.get("content", "")
            )
            for message in record["messages"]
            if (
                isinstance(message, dict)
                and message.get("role") == "user"
            )
        ]

        return "\n".join(
            part
            for part in parts
            if part
        ).strip()

    raise KeyError(
        "record has no problem, prompt, or user message"
    )


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


# ============================================================
# Prediction extraction
# ============================================================

def extract_prediction(
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

    # --------------------------------------------------------
    # No answer
    # --------------------------------------------------------

    return "", "no_answer_marker"


# ============================================================
# Gold answer
# ============================================================

def extract_gold(
    record: dict[str, Any],
) -> str:

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

    if boxed:
        return boxed[-1]

    return text


# ============================================================
# Dataset loading
# ============================================================

def load_records(
    path: str,
    limit: int | None,
) -> list[dict[str, Any]]:

    data_path = Path(path)

    if data_path.suffix.lower() == ".jsonl":

        with data_path.open(
            encoding="utf-8"
        ) as file:

            records = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

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

        if not isinstance(
            loaded,
            list,
        ):

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
) -> str:

    content = (
        problem
        + DEFAULT_SUFFIX
    )


    # --------------------------------------------------------
    # Instruct model
    # --------------------------------------------------------

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


# ============================================================
# Main
# ============================================================

def main() -> None:

    # ========================================================
    # Seeds
    # ========================================================

    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


    # ========================================================
    # Print configuration
    # ========================================================

    print("=" * 70)

    print("Math500 Evaluation")

    print("=" * 70)

    print(
        f"Model:              {MODEL_PATH}"
    )

    print(
        f"Dataset:            {DATA_PATH}"
    )

    print(
        f"Output:             {OUTPUT_PATH}"
    )

    print()

    print("Model type:")

    print("  Instruct model")

    print()

    print("Prompt:")

    print(
        f"  chat_template:     "
        f"{USE_CHAT_TEMPLATE}"
    )

    print(
        f"  enable_thinking:   "
        f"{ENABLE_THINKING}"
    )

    print()

    print("Sampling:")

    print(
        f"  temperature:       "
        f"{TEMPERATURE}"
    )

    print(
        f"  top_p:             "
        f"{TOP_P}"
    )

    print(
        f"  top_k:             "
        f"{TOP_K}"
    )

    print(
        f"  max_new_tokens:    "
        f"{MAX_NEW_TOKENS}"
    )

    print()

    print("vLLM:")

    print(
        f"  dtype:             "
        f"{DTYPE}"
    )

    print(
        f"  max_model_len:     "
        f"{MAX_MODEL_LEN}"
    )

    print(
        f"  gpu_memory_util:   "
        f"{GPU_MEMORY_UTILIZATION}"
    )

    print(
        f"  tensor_parallel:   "
        f"{TENSOR_PARALLEL_SIZE}"
    )

    print("=" * 70)


    # ========================================================
    # Tokenizer
    # ========================================================

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


    # ========================================================
    # vLLM
    # ========================================================

    llm = LLM(
        model=MODEL_PATH,
        dtype=DTYPE,
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        seed=SEED,
    )


    # ========================================================
    # Sampling parameters
    # ========================================================

    sampling_kwargs = {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
    }

    if STOP is not None:
        sampling_kwargs["stop"] = STOP

    sampling_params = SamplingParams(
        **sampling_kwargs
    )


    # ========================================================
    # Load dataset
    # ========================================================

    records = load_records(
        DATA_PATH,
        LIMIT,
    )

    print()

    print(
        f"Number of examples: "
        f"{len(records)}"
    )


    # ========================================================
    # Prepare output
    # ========================================================

    output_path = Path(
        OUTPUT_PATH
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # Prepare metrics
    # ========================================================

    correct = 0

    scored = 0

    num_truncated = 0

    num_length_finished = 0


    # ========================================================
    # Prepare prompts
    # ========================================================

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
        )
        for problem in problems
    ]


    # ========================================================
    # Generate
    # ========================================================

    print()

    print(
        "Starting generation..."
    )

    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=True,
    )

    print(
        "Generation finished."
    )


    # ========================================================
    # Grade
    # ========================================================

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

            completion_output = (
                output.outputs[0]
            )

            completion = (
                completion_output.text
            )

            finish_reason = getattr(
                completion_output,
                "finish_reason",
                None,
            )

            if finish_reason == "length":

                num_length_finished += 1


            # =================================================
            # Extract answer
            # =================================================

            prediction, method = (
                extract_prediction(
                    completion
                )
            )


            # =================================================
            # Truncation
            # =================================================

            truncated = not bool(
                prediction
            )

            if truncated:

                num_truncated += 1

                reward = 0.0

                error = None

            else:

                try:

                    reward = float(
                        grade_answer(
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


            # =================================================
            # Metrics
            # =================================================

            correct += int(reward)

            scored += 1


            # =================================================
            # Save result
            # =================================================

            result = {

                "index": index,

                "problem": problem,

                "gold": gold,

                "completion": completion,

                "prediction": prediction,

                "extraction_method": method,

                "reward": reward,

                "truncated": truncated,

                "finish_reason": finish_reason,
            }

            if error is not None:

                result[
                    "grader_error"
                ] = error


            output_file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )


            # =================================================
            # Progress
            # =================================================

            if scored % 10 == 0:

                accuracy = (
                    correct / scored
                )

                truncation_rate = (
                    num_truncated
                    / scored
                )

                length_rate = (
                    num_length_finished
                    / scored
                )

                tqdm.write(
                    f"scored={scored} "
                    f"correct={correct} "
                    f"accuracy={accuracy:.4%} "
                    f"truncated={num_truncated} "
                    f"truncation_rate="
                    f"{truncation_rate:.4%} "
                    f"length_finish="
                    f"{num_length_finished} "
                    f"length_rate="
                    f"{length_rate:.4%}"
                )


    # ========================================================
    # Final metrics
    # ========================================================

    accuracy = (
        correct / scored
        if scored
        else float("nan")
    )

    truncation_rate = (
        num_truncated / scored
        if scored
        else float("nan")
    )

    length_finish_rate = (
        num_length_finished / scored
        if scored
        else float("nan")
    )


    # ========================================================
    # Summary
    # ========================================================

    summary_path = Path(
        SUMMARY_PATH
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        "step": STEP_I,

        "model_path": MODEL_PATH,

        "model_type": (
            "instruct"
        ),

        "data_path": DATA_PATH,

        "predictions_path": str(
            output_path.resolve()
        ),


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        "num_examples": scored,

        "num_correct": correct,

        "accuracy": accuracy,

        "accuracy_percent": (
            accuracy * 100
        ),

        "num_truncated": (
            num_truncated
        ),

        "truncation_rate": (
            truncation_rate
        ),

        "truncation_rate_percent": (
            truncation_rate * 100
        ),

        "num_length_finished": (
            num_length_finished
        ),

        "length_finish_rate": (
            length_finish_rate
        ),

        "length_finish_rate_percent": (
            length_finish_rate * 100
        ),


        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        "use_chat_template": (
            USE_CHAT_TEMPLATE
        ),

        "enable_thinking": (
            ENABLE_THINKING
        ),


        # ----------------------------------------------------
        # Sampling
        # ----------------------------------------------------

        "temperature": TEMPERATURE,

        "top_p": TOP_P,

        "top_k": TOP_K,

        "max_new_tokens": (
            MAX_NEW_TOKENS
        ),

        "stop": STOP,


        # ----------------------------------------------------
        # vLLM
        # ----------------------------------------------------

        "backend": "vllm",

        "dtype": DTYPE,

        "max_model_len": (
            MAX_MODEL_LEN
        ),

        "gpu_memory_utilization": (
            GPU_MEMORY_UTILIZATION
        ),

        "tensor_parallel_size": (
            TENSOR_PARALLEL_SIZE
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


    # ========================================================
    # Final output
    # ========================================================

    print()

    print("=" * 70)

    print(
        "Evaluation Finished"
    )

    print("=" * 70)

    print(
        f"Model:               "
        f"{MODEL_PATH}"
    )

    print(
        f"Examples:            "
        f"{scored}"
    )

    print(
        f"Correct:             "
        f"{correct}"
    )

    print(
        f"Accuracy:            "
        f"{accuracy:.6f} "
        f"({accuracy:.2%})"
    )

    print(
        f"Truncated:           "
        f"{num_truncated}/{scored}"
    )

    print(
        f"Truncation rate:     "
        f"{truncation_rate:.6f} "
        f"({truncation_rate:.2%})"
    )

    print(
        f"Length finished:     "
        f"{num_length_finished}/{scored}"
    )

    print(
        f"Length finish rate:  "
        f"{length_finish_rate:.6f} "
        f"({length_finish_rate:.2%})"
    )

    print(
        f"Predictions:         "
        f"{output_path.resolve()}"
    )

    print(
        f"Summary:             "
        f"{summary_path.resolve()}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()