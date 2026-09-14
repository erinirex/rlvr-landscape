#!/usr/bin/env python3
"""
Evaluate a model's reward landscape along random parameter-space directions
using vLLM.

Key features:

1. No HuggingFace model object is loaded.
2. No direction file is required.
3. Random directions are generated deterministically inside vLLM workers.
4. The random direction is NOT stored permanently on GPU.
5. The same random direction is regenerated for every alpha point.
6. Sequential perturbation is used:

       theta <- theta + (alpha_new - alpha_old) * scale * d

7. group_size is implemented with SamplingParams(n=group_size).
8. vLLM automatically batches all prompts.
"""

import argparse
import ast
import hashlib
import json
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from swift.grading.grader import grade_answer

from vllm import LLM, SamplingParams


RewardFn = Callable[..., float]


# ============================================================
# Arguments
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate reward landscape along random "
            "parameter-space directions using vLLM."
        )
    )

    parser.add_argument(
        "--model-ckpt",
        required=True,
        help="Checkpoint path.",
    )

    parser.add_argument(
        "--eval-json",
        required=True,
        help="Evaluation JSONL path.",
    )

    parser.add_argument(
        "--task",
        choices=("chess", "gsm8k", "math500"),
        default="chess",
    )

    parser.add_argument(
        "--rl-type",
        default="dapo",
    )

    parser.add_argument(
        "--lr",
        type=float, 
        default="5e-6"
    )

    parser.add_argument(
        "--dataset",
        default="chess_single_turn",
    )

    parser.add_argument(
        "--model-name",
        default="qwen3_1.7b",
    )

    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=25,
        help="Number of evaluation prompts.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help=(
            "Number of sampled completions per prompt. "
            "Implemented by SamplingParams(n=group_size)."
        ),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Generation seed and base random-direction seed.",
    )

    parser.add_argument(
        "--direction-seed",
        type=int,
        default=None,
        help=(
            "Seed for random parameter-space directions. "
            "Defaults to --seed."
        ),
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=0.01,
        help="Global perturbation scale.",
    )

    parser.add_argument(
        "--alpha-range",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--num-points",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--num-directions",
        type=int,
        default=2,
        help="Number of independent random directions.",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used by one vLLM instance.",
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--output-dir",
        default="figs_qwen_chess_vllm",
    )

    parser.add_argument(
        "--output-stem",
        default=None,
    )

    parser.add_argument(
        "--ymin",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--ymax",
        type=float,
        default=1.0,
    )

    return parser.parse_args()


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


# ============================================================
# Data loading
# ============================================================

def load_gsm8k_example(
    example: dict[str, Any],
) -> dict[str, Any]:

    if "messages" not in example:
        raise KeyError(
            "GSM8K example is missing 'messages'"
        )

    if "solution" not in example:
        raise KeyError(
            "GSM8K example is missing 'solution'"
        )

    prompt = extract_prompt_from_messages(
        example["messages"]
    )

    if not prompt:
        raise ValueError(
            "GSM8K example has no user prompt"
        )

    return {
        "prompt": prompt,
        "answer": str(
            example["solution"]
        ).strip(),
        "metadata": example.get(
            "metadata",
            {},
        ),
        "extra_info": example.get(
            "extra_info",
            {},
        ),
    }



def load_chess_example(
    example: dict[str, Any],
) -> dict[str, Any]:

    if "prompt" not in example:
        raise KeyError(
            "Chess example is missing 'prompt'"
        )

    reward_model = example.get(
        "reward_model",
        {},
    )

    if "ground_truth" not in reward_model:
        raise KeyError(
            "Chess example is missing "
            "'reward_model.ground_truth'"
        )

    extra_info = example.get(
        "extra_info",
        {},
    )

    metadata = dict(
        example.get(
            "metadata",
            {},
        )
    )

    metadata.setdefault(
        "data_source",
        example.get("data_source"),
    )

    metadata.setdefault(
        "ability",
        example.get("ability"),
    )

    metadata.setdefault(
        "difficulty",
        example.get("difficulty"),
    )

    for key in (
        "FEN",
        "Moves",
        "PuzzleId",
        "Rating",
        "Themes",
        "env_replies",
        "first_move_san",
        "first_move_uci",
        "second_move_uci",
        "original_FEN",
    ):

        if key in extra_info:
            metadata.setdefault(
                key,
                extra_info[key],
            )

    return {
        "prompt": str(
            example["prompt"]
        ).strip(),

        "answer": str(
            reward_model["ground_truth"]
        ).strip(),

        "metadata": metadata,
        "extra_info": extra_info,
    }


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

                if task == "gsm8k":

                    parsed_example = (
                        load_gsm8k_example(
                            example
                        )
                    )

                elif task == "math500":

                    parsed_example = (
                        load_math500_example(
                            example
                        )
                    )

                else:

                    parsed_example = (
                        load_chess_example(
                            example
                        )
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
# GSM8K
# ============================================================

NUMBER_PATTERN = (
    r"[-+]?"
    r"(?:"
    r"\d{1,3}(?:,\d{3})+"
    r"|"
    r"\d+(?:\.\d*)?"
    r"|"
    r"\.\d+"
    r")"
    r"(?:[eE][-+]?\d+)?"
)


def normalize_number(
    text: str,
) -> str:

    return (
        str(text)
        .replace(",", "")
        .replace("$", "")
        .strip()
        .rstrip(".")
    )


def parse_decimal(
    text: str,
) -> Decimal | None:

    cleaned = normalize_number(text)

    if not cleaned:
        return None

    try:

        return Decimal(cleaned)

    except InvalidOperation:

        return None


def extract_last_number(
    text: str,
) -> str:

    matches = list(
        re.finditer(
            NUMBER_PATTERN,
            text,
        )
    )

    if not matches:
        return ""

    return normalize_number(
        matches[-1].group(0)
    )


def extract_final_answer(
    text: str,
) -> str:

    boxed_matches = re.findall(
        r"\\boxed\s*\{\s*([^{}]+?)\s*\}",
        text,
    )

    if boxed_matches:

        extracted = extract_last_number(
            boxed_matches[-1]
        )

        if extracted:
            return extracted

    final_answer_match = re.search(
        r"Final\s+Answer\s*:?\s*(.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if final_answer_match:

        extracted = extract_last_number(
            final_answer_match.group(1)
        )

        if extracted:
            return extracted

    return ""


def extract_ground_truth(
    answer: str,
) -> str:

    answer = str(answer).strip()

    if "####" in answer:

        final_part = answer.rsplit(
            "####",
            1,
        )[-1]

        extracted = extract_last_number(
            final_part
        )

        if extracted:
            return extracted

    if re.fullmatch(
        NUMBER_PATTERN,
        answer,
    ):
        return normalize_number(answer)

    return extract_final_answer(answer)


def gsm8k_reward_func(
    completion: str,
    answer: str,
    **_: Any,
) -> float:

    prediction_text = extract_final_answer(
        completion
    )

    target_text = extract_ground_truth(
        answer
    )

    prediction = parse_decimal(
        prediction_text
    )

    target = parse_decimal(
        target_text
    )

    if prediction is None or target is None:
        return 0.0

    return float(
        prediction == target
    )




# ============================================================
# Chess
# ============================================================

CHESS_SINGLE_TURN_REWARD_MODEL_TYPE = (
    os.environ.get(
        "REWARD_MODEL_TYPE",
        "RULE_BASED",
    ).upper()
)


def get_fen(
    metadata: dict[str, Any],
    extra_info: dict[str, Any],
) -> str:

    for source in (
        metadata,
        extra_info,
    ):

        fen = (
            source.get("FEN")
            or source.get("fen")
        )

        if fen:
            return str(fen)

    return ""


def lan_to_uci(
    lan: str,
    side_to_move: str = "white",
) -> str:

    lan = lan.rstrip(
        "+#"
    ).strip()

    if lan == "O-O":

        if side_to_move == "white":
            return "e1g1"

        return "e8g8"

    if lan == "O-O-O":

        if side_to_move == "white":
            return "e1c1"

        return "e8c8"

    match = re.match(
        r"^([PNBRQK])"
        r"([a-h][1-8])"
        r"(x)?"
        r"([a-h][1-8])"
        r"(=([QRBN]))?$",
        lan,
    )

    if not match:
        raise ValueError(
            f"Invalid LAN format: {lan}"
        )

    (
        _piece,
        from_square,
        _capture,
        to_square,
        _promo_group,
        promo,
    ) = match.groups()

    uci = (
        from_square
        + to_square
    )

    if promo:
        uci += promo.lower()

    return uci


def is_complete_move(
    text: str,
) -> bool:

    if not text:
        return False

    move = text.rstrip(
        "+#"
    )

    if move in (
        "O-O",
        "O-O-O",
    ):
        return True

    return bool(
        re.match(
            r"^[PNBRQK]"
            r"[a-h][1-8]"
            r"(x)?"
            r"[a-h][1-8]"
            r"(=[QRBN])?$",
            move,
        )
    )


def extract_first_move(
    text: str,
) -> str | None:

    tokens = text.strip().split()

    for token in tokens:

        if re.match(
            r"^\d+\.{1,3}$",
            token,
        ):
            continue

        if is_complete_move(token):
            return token

    return None


def extract_move_after_thinking(
    text: str,
) -> tuple[str | None, bool]:

    text = text.strip()

    follows_format = (
        text.count("</T>") == 1
    )

    if not follows_format:
        return None, False

    text_after_thinking = text[
        text.find("</T>")
        + len("</T>"):
    ].strip()

    if not text_after_thinking:
        return None, True

    return (
        extract_first_move(
            text_after_thinking
        ),
        True,
    )


def parse_chess_ground_truth(
    answer: Any,
) -> str:

    ground_truth = answer

    if isinstance(
        ground_truth,
        str,
    ):

        try:

            ground_truth = json.loads(
                ground_truth
            )

        except json.JSONDecodeError:

            try:

                ground_truth = ast.literal_eval(
                    ground_truth
                )

            except (
                ValueError,
                SyntaxError,
            ):
                pass

    if (
        isinstance(
            ground_truth,
            list,
        )
        and ground_truth
    ):
        return str(
            ground_truth[0]
        ).strip()

    return str(
        ground_truth
    ).strip()


def chess_single_turn_move_to_uci(
    move_text: str,
) -> str:

    if not move_text:
        return ""

    try:

        return lan_to_uci(
            move_text
        )

    except ValueError:

        return ""


def extract_chess_single_turn_move(
    completion: str,
) -> tuple[str, bool]:

    move, follows_format = (
        extract_move_after_thinking(
            completion
        )
    )

    if (
        move is None
        and not follows_format
    ):
        move = extract_first_move(
            completion
        )

    return (
        move.strip()
        if move
        else "",
        follows_format,
    )


def chess_single_turn_reward_func(
    completion: str,
    answer: str,
    metadata: dict[str, Any] | None = None,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> float:

    raw_move, follows_format = (
        extract_chess_single_turn_move(
            completion
        )
    )

    prediction = (
        chess_single_turn_move_to_uci(
            raw_move
        )
    )

    target = parse_chess_ground_truth(
        answer
    )

    score = float(
        bool(prediction)
        and prediction == target
    )

    if (
        CHESS_SINGLE_TURN_REWARD_MODEL_TYPE
        == "RULE_FORMAT_BASED"
        and not follows_format
    ):
        return 0.0

    return score


# ============================================================
# Reward function
# ============================================================

def select_reward_func(
    task: str,
) -> RewardFn:

    if task == "chess":
        return chess_single_turn_reward_func

    if task == "math500":
        return math500_reward_func

    return gsm8k_reward_func


# ============================================================
# Random direction
# ============================================================

def stable_seed(
    base_seed: int,
    direction_index: int,
    parameter_name: str,
) -> int:
    """
    Generate a deterministic seed from:

        base_seed
        direction_index
        parameter_name

    This guarantees that every time the same parameter is visited,
    the same random direction is generated.

    Therefore we don't need to store the direction.
    """

    key = (
        f"{base_seed}:"
        f"{direction_index}:"
        f"{parameter_name}"
    )

    digest = hashlib.sha256(
        key.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    ) % (2**63 - 1)


def make_random_delta_perturb_function(
    direction_seed: int,
    direction_index: int,
    delta_coefficient: float,
):
    """
    Modify current vLLM model:

        theta <- theta + delta_coefficient * d

    where d is a deterministic random direction.

    IMPORTANT:

    We do NOT store d.

    Instead, every time this function is called, the same random
    direction is regenerated from:

        direction_seed
        direction_index
        parameter_name

    Therefore:

        alpha_1 -> generate d -> theta + delta_1 d
        alpha_2 -> generate SAME d -> theta + delta_2 d

    The direction is not kept in GPU memory.
    """

    def perturb_model(
        model: torch.nn.Module,
    ) -> int:

        updated = 0

        with torch.no_grad():

            for name, parameter in (
                model.named_parameters()
            ):

                # ------------------------------------------------
                # Skip parameters that should not be perturbed.
                #
                # Usually all trainable parameters are desired.
                # ------------------------------------------------

                if not parameter.requires_grad:
                    continue

                seed = stable_seed(
                    direction_seed,
                    direction_index,
                    name,
                )

                # ------------------------------------------------
                # Generate deterministic random direction directly
                # on the parameter's device.
                # ------------------------------------------------

                generator = torch.Generator(
                    device=parameter.device
                )

                generator.manual_seed(
                    seed
                )

                d = torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    device=parameter.device,
                    generator=generator,
                )

                # ------------------------------------------------
                # Normalize each tensor so that its RMS is ~1.
                #
                # This prevents large parameter matrices from having
                # an arbitrary scale simply because of their shape.
                # ------------------------------------------------

                rms = torch.sqrt(
                    torch.mean(
                        d.float().square()
                    )
                    + 1e-12
                )

                d = d / rms.to(
                    dtype=d.dtype
                )

                parameter.add_(
                    d,
                    alpha=delta_coefficient,
                )

                updated += 1

        return updated

    return perturb_model


# ============================================================
# Prompt formatting
# ============================================================

def build_prompts(
    tokenizer: Any,
    data: list[dict[str, Any]],
    task: str,
) -> list[str]:

    prompts = []

    for example in data:

        if task == "chess":

            prompts.append(
                example["prompt"]
            )

        else:

            prompt = (
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": example[
                                "prompt"
                            ],
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

            prompts.append(prompt)

    return prompts


# ============================================================
# vLLM evaluation
# ============================================================

def eval_mean_reward_vllm(
    llm: LLM,
    tokenizer: Any,
    data: list[dict[str, Any]],
    prompts: list[str],
    reward_fn: RewardFn,
    max_new_tokens: int,
    group_size: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    task: str,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    list[dict[str, Any]],
]:

    sampling_params = SamplingParams(
        n=group_size,
        temperature=(
            temperature
            if group_size > 1
            else 0.0
        ),
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        seed=seed,
    )

    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=True,
    )

    local_generation_records = []

    reward_matrix = np.zeros(
        (
            len(data),
            group_size,
        ),
        dtype=np.float64,
    )

    for prompt_index, (
        example,
        request_output,
    ) in enumerate(
        zip(
            data,
            outputs,
        )
    ):

        completions = (
            request_output.outputs
        )

        if len(completions) != group_size:

            raise RuntimeError(
                f"Prompt {prompt_index}: "
                f"expected {group_size} outputs, "
                f"got {len(completions)}"
            )

        for group_index, completion_output in enumerate(
            completions
        ):

            completion = (
                completion_output.text
            )

            reward = reward_fn(
                completion,
                example["answer"],
                metadata=example.get(
                    "metadata"
                ),
                extra_info=example.get(
                    "extra_info"
                ),
            )

            reward_value = float(
                reward
            )

            reward_matrix[
                prompt_index,
                group_index,
            ] = reward_value

            token_ids = (
                completion_output.token_ids
            )

            completion_token_length = (
                len(token_ids)
            )

            ended_with_eos = (
                completion_output.finish_reason
                == "stop"
            )

            likely_truncated = (
                completion_output.finish_reason
                == "length"
            )

            local_generation_records.append(
                {
                    "prompt_index":
                        prompt_index,
                    "group_index":
                        group_index,
                    "group_sample":
                        group_index + 1,
                    "completion_tokens":
                        completion_token_length,
                    "raw_completion_tokens":
                        completion_token_length,
                    "completion_characters":
                        len(completion),
                    "ended_with_eos":
                        ended_with_eos,
                    "likely_truncated":
                        likely_truncated,
                    "reward":
                        reward_value,
                }
            )

            if prompt_index < 3:

                print(
                    "\n"
                    + "=" * 100
                )

                print(
                    f"Prompt index: "
                    f"{prompt_index}"
                )

                print(
                    f"Group sample: "
                    f"{group_index + 1}/"
                    f"{group_size}"
                )

                print(
                    "Generated tokens: "
                    f"{completion_token_length}"
                )

                print(
                    "Finish reason: "
                    f"{completion_output.finish_reason}"
                )

                if task == "chess":

                    (
                        raw_move,
                        follows_format,
                    ) = (
                        extract_chess_single_turn_move(
                            completion
                        )
                    )

                    predicted_uci = (
                        chess_single_turn_move_to_uci(
                            raw_move
                        )
                    )

                    target_move = (
                        parse_chess_ground_truth(
                            example["answer"]
                        )
                    )

                    print(
                        "Follows <T></T>: "
                        f"{follows_format}"
                    )

                    print(
                        "Raw move: "
                        f"{raw_move!r}"
                    )

                    print(
                        "Predicted UCI: "
                        f"{predicted_uci!r}"
                    )

                    print(
                        "Target: "
                        f"{target_move!r}"
                    )

                elif task == "gsm8k":

                    prediction = (
                        extract_final_answer(
                            completion
                        )
                    )

                    target = (
                        extract_ground_truth(
                            example["answer"]
                        )
                    )

                    print(
                        "Prediction: "
                        f"{prediction!r}"
                    )

                    print(
                        "Target: "
                        f"{target!r}"
                    )

                elif task == "math500":

                    (
                        prediction_text,
                        extraction_method,
                    ) = (
                        extract_math500_answer(
                            completion
                        )
                    )

                    target_text = (
                        extract_math500_gold(
                            example["answer"]
                        )
                    )

                    print(
                        "Extraction method: "
                        f"{extraction_method}"
                    )

                    print(
                        "Prediction: "
                        f"{prediction_text!r}"
                    )

                    print(
                        "Target: "
                        f"{target_text!r}"
                    )

                print(
                    f"Reward: {reward_value}"
                )

                print(
                    "FULL GENERATION:"
                )

                print(completion)

                print(
                    "=" * 100,
                    flush=True,
                )

    # ========================================================
    # Statistics
    # ========================================================

    mean_reward = (
        reward_matrix.mean()
    )

    generation_means = (
        reward_matrix.mean(
            axis=0
        )
    )

    if group_size >= 2:

        generation_mean_std = (
            generation_means.std(
                ddof=1
            )
        )

        std_error_generation = (
            generation_mean_std
            / np.sqrt(group_size)
        )

    else:

        generation_mean_std = 0.0
        std_error_generation = 0.0

    if group_size >= 2:

        prompt_variances = (
            reward_matrix.var(
                axis=1,
                ddof=1,
            )
        )

        std_error_prompt_rms = float(
            np.sqrt(
                prompt_variances.mean()
            )
        )

    else:

        std_error_prompt_rms = 0.0

    flat_rewards = (
        reward_matrix.reshape(-1)
    )

    if len(flat_rewards) >= 2:

        std = float(
            flat_rewards.std(
                ddof=1
            )
        )

        se = (
            std
            / np.sqrt(
                len(flat_rewards)
            )
        )

    else:

        std = 0.0
        se = 0.0

    return (
        float(mean_reward),
        float(generation_mean_std),
        float(std_error_generation),
        float(std_error_prompt_rms),
        float(std),
        float(se),
        local_generation_records,
    )


# ============================================================
# Output
# ============================================================

def infer_checkpoint_step(
    model_ckpt: str,
) -> str:

    match = re.search(
        r"(?:checkpoint-|global_step_)(\d+)",
        model_ckpt,
    )

    return (
        match.group(1)
        if match
        else "unknown"
    )


def build_output_stem(
    args: argparse.Namespace,
) -> str:

    if args.output_stem:
        return args.output_stem

    checkpoint_step = (
        str(args.checkpoint_step)
        if args.checkpoint_step is not None
        else infer_checkpoint_step(
            args.model_ckpt
        )
    )

    return (
        f"{args.rl_type}_{args.dataset}_"
        f"reward_random_"
        f"scale{args.scale}_"
        f"alpha_range{args.alpha_range}_"
        f"num{args.num_samples}_"
        f"group{args.group_size}_"
        f"ckpt{checkpoint_step}_"
        f"model_{args.model_name}_"
        f"max_new{args.max_new_tokens}_"
        f"temp{args.temperature}_"
        f"topp{args.top_p}_"
        f"topk{args.top_k}_"
        f"seed{args.seed}_"
        f"dirseed{args.direction_seed}_"
        f"pts{args.num_points}_"
        f"{args.num_directions}randomdirs_"
        f"lr{args.lr}_"
        f"vllm"
    )


def save_results(
    df: pd.DataFrame,
    generation_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = build_output_stem(
        args
    )

    csv_path = (
        output_dir
        / f"{stem}.csv"
    )

    generation_csv_path = (
        output_dir
        / f"completion_lengths_{stem}.csv"
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    generation_df.to_csv(
        generation_csv_path,
        index=False,
    )

    def plot_landscape(
        std_column: str,
        filename: str,
        title_suffix: str,
        relative: bool = False,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))

        for direction_name, subset in df.groupby("direction"):
            subset = subset.sort_values("perturbation_coefficient")

            x = subset["perturbation_coefficient"].to_numpy(dtype=float)
            y = subset["reward"].to_numpy(dtype=float)
            error = subset[std_column].to_numpy(dtype=float)

            if relative:
                zero_rows = subset.loc[
                    subset["perturbation_coefficient"] == 0.0
                ]

                if len(zero_rows) != 1:
                    raise ValueError(
                        f"Direction {direction_name}: "
                        "需要唯一的 perturbation_coefficient=0 结果"
                    )

                baseline = float(zero_rows["reward"].iloc[0])

                if not np.isfinite(baseline) or baseline == 0.0:
                    raise ValueError(
                        f"Direction {direction_name}: "
                        f"0 点 reward={baseline}，无法归一化"
                    )

                y = y / baseline
                error = error / abs(baseline)

            line = ax.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                label=direction_name,
            )[0]

            lower = y - error
            upper = y + error

            # Absolute reward 的误差带裁剪到 [0, 1]。
            # Relative reward 可以超过 1，因此不裁剪。
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
            ax.set_ylabel("Relative reward: R(coefficient) / R(0)")
            # 自动缩放，避免截断归一化后的曲线和误差带。
        else:
            ax.set_ylabel("Mean reward")
            ax.set_ylim(args.ymin, args.ymax)

        mode = "relative" if relative else "absolute"
        ax.set_title(
            f"{args.rl_type.upper()} "
            f"{args.dataset} "
            f"step {args.checkpoint_step} "
            f"{mode} reward landscape "
            f"({title_suffix})"
        )

        ax.set_xlabel("Perturbation coefficient")
        ax.minorticks_on()
        ax.grid(
            which="major",
            linestyle="--",
            linewidth=0.5,
            alpha=0.8,
        )
        ax.grid(
            which="minor",
            linestyle=":",
            linewidth=0.3,
            alpha=0.3,
        )

        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=300)
        plt.close(fig)


    plot_landscape(
        std_column="std_error_generation_mean",
        filename=f"{stem}_absolute_reward_generation_mean_se.png",
        title_suffix="generation-mean SE",
        relative=False,
    )

    plot_landscape(
        std_column="std_error_generation_mean",
        filename=f"{stem}_relative_reward_generation_mean_se.png",
        title_suffix="generation-mean SE",
        relative=True,
    )

    # plot_landscape(
    #     "std",
    #     f"{stem}_reward_std.png",
    #     "reward std",
    # )

    # plot_landscape(
    #     "se",
    #     f"{stem}_reward_se.png",
    #     "reward SE",
    # )

    # plot_landscape(
    #     "std_error_prompt_rms",
    #     f"{stem}_prompt_rms.png",
    #     "prompt-level RMS std",
    # )

    print(
        f"\nSaved results to {output_dir}",
        flush=True,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    if args.direction_seed is None:
        args.direction_seed = args.seed

    if args.num_points < 1:
        raise ValueError(
            "--num-points must be positive"
        )

    if args.num_directions < 1:
        raise ValueError(
            "--num-directions must be positive"
        )

    if args.num_samples < 1:
        raise ValueError(
            "--num-samples must be positive"
        )

    if args.group_size < 1:
        raise ValueError(
            "--group-size must be positive"
        )

    # --------------------------------------------------------
    # Alpha grid
    # --------------------------------------------------------

    if args.num_points == 1:

        grid = np.array(
            [0.0],
            dtype=float,
        )

    else:

        grid = np.linspace(
            -args.alpha_range,
            args.alpha_range,
            args.num_points,
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "Initializing vLLM..."
    )

    print(
        f"Model: {args.model_ckpt}"
    )

    print(
        f"Tensor parallel size: "
        f"{args.tensor_parallel_size}"
    )

    print(
        f"Group size: {args.group_size}"
    )

    print(
        f"Num prompts: {args.num_samples}"
    )

    print(
        f"Max new tokens: "
        f"{args.max_new_tokens}"
    )

    print(
        f"Random direction seed: "
        f"{args.direction_seed}"
    )

    print(
        f"Number of random directions: "
        f"{args.num_directions}"
    )

    print(
        "=" * 80,
        flush=True,
    )

    # --------------------------------------------------------
    # vLLM
    # --------------------------------------------------------

    llm = LLM(
        model=args.model_ckpt,
        tokenizer=args.model_ckpt,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=(
            args.tensor_parallel_size
        ),
        gpu_memory_utilization=(
            args.gpu_memory_utilization
        ),
        seed=args.seed,
        max_model_len=args.max_model_len,
    )

    tokenizer = llm.get_tokenizer()

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    task = args.task.lower().strip()

    data = load_eval_data(
        args.eval_json,
        args.num_samples,
        task,
    )

    reward_fn = select_reward_func(
        task
    )

    prompts = build_prompts(
        tokenizer,
        data,
        task,
    )

    print(
        f"Loaded {len(data)} prompts.",
        flush=True,
    )

    print(
        f"Reward function: "
        f"{reward_fn.__name__}",
        flush=True,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    rows = []

    generation_rows = []

    # --------------------------------------------------------
    # Each direction is independent.
    #
    # At the beginning of every direction, the model must be
    # back at theta_0.
    #
    # The safest way with vLLM is to recreate the LLM instance
    # for each direction.
    #
    # This is slower, but avoids needing a full theta snapshot
    # and avoids accidentally accumulating direction changes.
    # --------------------------------------------------------

    for direction_index in range(
        1,
        args.num_directions + 1,
    ):

        direction_name = (
            f"random_direction_{direction_index}"
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"Direction {direction_index}/"
            f"{args.num_directions}"
        )

        print(
            f"Direction name: "
            f"{direction_name}"
        )

        print(
            f"Direction seed: "
            f"{args.direction_seed}"
        )

        print(
            "=" * 80,
            flush=True,
        )

        # ----------------------------------------------------
        # Reinitialize model for every direction.
        #
        # This guarantees:
        #
        #     theta = theta_0
        #
        # at alpha = -alpha_range.
        #
        # More importantly, different random directions do not
        # accumulate on top of each other.
        # ----------------------------------------------------

        if direction_index > 1:

            print(
                "\nReinitializing vLLM "
                "for the next random direction...",
                flush=True,
            )

            del llm

            llm = LLM(
                model=args.model_ckpt,
                tokenizer=args.model_ckpt,
                dtype="bfloat16",
                trust_remote_code=True,
                tensor_parallel_size=(
                    args.tensor_parallel_size
                ),
                gpu_memory_utilization=(
                    args.gpu_memory_utilization
                ),
                seed=args.seed,
                max_model_len=args.max_model_len,
            )

            tokenizer = llm.get_tokenizer()

        # ----------------------------------------------------
        # Important:
        #
        # We need to start at alpha=0.
        #
        # Then sequentially move:
        #
        # theta <- theta + delta * d
        #
        # where d is deterministic random direction.
        # ----------------------------------------------------

        previous_coefficient = 0.0

        for alpha in grid:

            coefficient = (
                args.scale
                * float(alpha)
            )

            delta = (
                coefficient
                - previous_coefficient
            )

            print(
                "\n"
                + "-" * 80
            )

            print(
                f"Direction: "
                f"{direction_name}"
            )

            print(
                f"alpha = {alpha:.6f}"
            )

            print(
                f"coefficient = "
                f"{coefficient:.8f}"
            )

            print(
                f"delta = "
                f"{delta:.8f}"
            )

            print(
                "-" * 80,
                flush=True,
            )

            # ------------------------------------------------
            # Modify vLLM model directly.
            #
            # The direction is generated deterministically
            # inside every worker.
            #
            # No direction file.
            # No CPU direction copy.
            # No persistent GPU direction copy.
            # ------------------------------------------------

            if abs(delta) > 0:

                perturb_fn = (
                    make_random_delta_perturb_function(
                        direction_seed=args.direction_seed,
                        direction_index=direction_index,
                        delta_coefficient=delta,
                    )
                )

                result = llm.apply_model(
                    perturb_fn
                )

                print(
                    "Updated parameter tensors "
                    f"per worker: {result}",
                    flush=True,
                )

            # ------------------------------------------------
            # Generate
            # ------------------------------------------------

            (
                reward,
                generation_mean_std,
                std_error_generation_mean,
                std_error_prompt_rms,
                std,
                se,
                generation_records,
            ) = eval_mean_reward_vllm(
                llm=llm,
                tokenizer=tokenizer,
                data=data,
                prompts=prompts,
                reward_fn=reward_fn,
                max_new_tokens=args.max_new_tokens,
                group_size=args.group_size,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed,
                task=task,
            )

            rows.append(
                {
                    "direction":
                        direction_name,

                    "direction_index":
                        direction_index,

                    "direction_seed":
                        args.direction_seed,

                    "alpha":
                        float(alpha),

                    "perturbation_coefficient":
                        coefficient,

                    "reward":
                        reward,

                    "generation_mean_std":
                        generation_mean_std,

                    "std_error_generation_mean":
                        std_error_generation_mean,

                    "std":
                        std,

                    "se":
                        se,

                    "std_error_prompt_rms":
                        std_error_prompt_rms,
                }
            )

            for record in generation_records:

                generation_rows.append(
                    {
                        "direction":
                            direction_name,

                        "direction_index":
                            direction_index,

                        "direction_seed":
                            args.direction_seed,

                        "alpha":
                            float(alpha),

                        "perturbation_coefficient":
                            coefficient,

                        **record,
                    }
                )

            print(
                "\nRESULT:"
            )

            print(
                f"direction={direction_name}, "
                f"alpha={alpha:.4f}, "
                f"coefficient={coefficient:.6f}, "
                f"reward={reward:.6f}"
            )

            print(
                f"generation_mean_std="
                f"{generation_mean_std:.6f}"
            )

            print(
                f"generation_mean_SE="
                f"{std_error_generation_mean:.6f}"
            )

            print(
                f"prompt_RMS="
                f"{std_error_prompt_rms:.6f}"
            )

            print(
                f"reward_std="
                f"{std:.6f}"
            )

            print(
                f"reward_SE="
                f"{se:.6f}",
                flush=True,
            )

            previous_coefficient = (
                coefficient
            )

    print(
        "\nExperiment finished.",
        flush=True,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if rows:

        result_df = pd.DataFrame(
            rows
        )

        generation_df = pd.DataFrame(
            generation_rows
        )

        if not generation_df.empty:

            generation_df = generation_df.sort_values(
                [
                    "direction_index",
                    "alpha",
                    "prompt_index",
                    "group_index",
                ]
            )

        save_results(
            result_df,
            generation_df,
            args,
        )

    print(
        "Done.",
        flush=True,
    )


if __name__ == "__main__":
    main()