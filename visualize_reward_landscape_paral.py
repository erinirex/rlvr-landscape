#!/usr/bin/env python3
"""Evaluate a model's mean reward along random parameter-space directions."""

import argparse
import ast
import json
import os
from swift.grading.grader import grade_answer
import re
from pathlib import Path
from typing import Any, Callable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from decimal import Decimal, InvalidOperation

RewardFn = Callable[..., float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a 1D reward landscape along random model directions."
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Number of sampled generations per prompt",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature when group-size > 1",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling value when group-size > 1",
    )
    parser.add_argument("--model-ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--eval-json", required=True, help="Evaluation JSONL path")
    parser.add_argument(
        "--task",
        choices=("chess", "gsm8k", "math500"),
        default="chess",
        help="Reward function to use",
    )
    parser.add_argument("--rl-type", default="dapo")
    parser.add_argument("--dataset", default="chess_single_turn")
    parser.add_argument("--model-name", default="qwen3_1.7b")
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument("--alpha-range", type=float, default=10.0)
    parser.add_argument("--num-points", type=int, default=15)
    parser.add_argument("--num-directions", type=int, default=2)
    parser.add_argument("--output-dir", default="figs_qwen_chess")
    parser.add_argument("--output-stem", default=None)
    parser.add_argument("--ymin", type=float, default=0.0)
    parser.add_argument("--ymax", type=float, default=1.0)
    return parser.parse_args()


def extract_latex_command_argument(
    text: str,
    command: str,
) -> list[str]:
    """Extract arguments from commands such as \\boxed{...}.

    Supports nested braces, unlike a regular expression.
    """
    results: list[str] = []
    pattern = re.compile(
        rf"\\{re.escape(command)}\s*\{{"
    )

    for match in pattern.finditer(text):
        start = match.end()
        depth = 1
        index = start

        while index < len(text) and depth > 0:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1

            index += 1

        if depth == 0:
            results.append(
                text[start : index - 1].strip()
            )

    return results


def extract_prompt_from_messages(
    messages: list[dict[str, Any]],
) -> str:
    prompt_parts: list[str] = []

    for message in messages:
        if message.get("role") != "user":
            continue

        content = message.get("content", "")

        if isinstance(content, str):
            prompt_parts.append(content)
            continue

        if isinstance(content, list):
            prompt_parts.extend(
                str(item["text"])
                for item in content
                if isinstance(item, dict)
                and item.get("type") == "text"
                and "text" in item
            )

    return "\n".join(prompt_parts).strip()

def extract_answer_tag(
    text: str,
) -> tuple[str, bool]:
    """Extract the last <answer>...</answer> block."""
    matches = re.findall(
        r"<answer>(.*?)</answer>",
        str(text),
        flags=re.IGNORECASE | re.DOTALL,
    )

    if matches:
        return matches[-1].strip(), True

    return str(text).strip(), False


def extract_math500_answer(
    text: str,
) -> tuple[str, str]:
    """Extract a Math500 answer and its extraction method."""
    text = str(text).strip()

    tagged_answer, has_answer_tag = extract_answer_tag(
        text
    )

    if has_answer_tag:
        return tagged_answer, "answer_tag"

    boxed_answers = extract_latex_command_argument(
        text,
        "boxed",
    )

    if boxed_answers:
        return boxed_answers[-1].strip(), "boxed"

    final_answer_matches = list(
        re.finditer(
            r"(?:Final\s+Answer|Answer)\s*:?\s*",
            text,
            flags=re.IGNORECASE,
        )
    )

    if final_answer_matches:
        answer_start = final_answer_matches[-1].end()
        answer_text = text[answer_start:].strip()

        for line in answer_text.splitlines():
            cleaned_line = line.strip()

            if cleaned_line:
                return cleaned_line, "final_answer"

    # Do not send the entire reasoning trace to grade_answer.
    return "", "no_answer_marker"


def extract_math500_gold(answer: Any) -> str:
    """Normalize a Math500 ground-truth answer."""
    if answer is None:
        return ""

    if isinstance(answer, dict):
        if "answer" in answer:
            answer = answer["answer"]
        elif "solution" in answer:
            answer = answer["solution"]

    text = str(answer).strip()

    tagged_answer, has_answer_tag = extract_answer_tag(
        text
    )

    if has_answer_tag:
        return tagged_answer

    # This also supports a full worked solution whose final
    # answer is contained in \boxed{...}.
    boxed_answers = extract_latex_command_argument(
        text,
        "boxed",
    )

    if boxed_answers:
        return boxed_answers[-1].strip()

    # Math500's answer/solution field may already be a plain answer.
    return text

def load_gsm8k_example(
    example: dict[str, Any],
) -> dict[str, Any]:
    if "messages" not in example:
        raise KeyError("GSM8K example is missing 'messages'")

    if "solution" not in example:
        raise KeyError("GSM8K example is missing 'solution'")

    prompt = extract_prompt_from_messages(
        example["messages"]
    )

    if not prompt:
        raise ValueError("GSM8K example has no user prompt")

    return {
        "prompt": prompt,
        "answer": str(example["solution"]).strip(),
        "metadata": example.get("metadata", {}),
        "extra_info": example.get("extra_info", {}),
    }

def load_math500_example(
    example: dict[str, Any],
) -> dict[str, Any]:
    """Parse one Math500 JSONL example.

    Expected format:
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "..."
                    }
                ]
            }
        ],
        "solution": "\\frac{8}{15}"
    }
    """
    if "messages" not in example:
        raise KeyError("Math500 example is missing 'messages'")

    if "solution" not in example:
        raise KeyError("Math500 example is missing 'solution'")

    prompt = extract_prompt_from_messages(
        example["messages"]
    )
    answer = str(example["solution"]).strip()

    if not prompt:
        raise ValueError("Math500 example has no user prompt")

    if not answer:
        raise ValueError("Math500 example has an empty solution")

    return {
        "prompt": prompt,
        "answer": answer,
        "metadata": example.get("metadata", {}),
        "extra_info": example.get("extra_info", {}),
    }

def load_chess_example(
    example: dict[str, Any],
) -> dict[str, Any]:
    if "prompt" not in example:
        raise KeyError("Chess example is missing 'prompt'")

    reward_model = example.get("reward_model", {})

    if "ground_truth" not in reward_model:
        raise KeyError(
            "Chess example is missing "
            "'reward_model.ground_truth'"
        )

    extra_info = example.get("extra_info", {})

    metadata = dict(example.get("metadata", {}))

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
            metadata.setdefault(key, extra_info[key])

    return {
        "prompt": str(example["prompt"]).strip(),
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

    task = task.lower().strip()

    if task not in {"gsm8k", "math500", "chess"}:
        raise ValueError(
            f"Unsupported task: {task!r}. "
            "Expected 'gsm8k', 'math500', or 'chess'."
        )

    data: list[dict[str, Any]] = []

    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                example = json.loads(line)

                if task == "gsm8k":
                    parsed_example = load_gsm8k_example(
                        example
                    )
                elif task == "math500":
                    parsed_example = load_math500_example(
                        example
                    )
                else:
                    parsed_example = load_chess_example(
                        example
                    )

            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                raise ValueError(
                    f"Failed to parse {task} example "
                    f"at {path}:{line_number}: {error}"
                ) from error

            data.append(parsed_example)

            if len(data) >= limit:
                break

    if not data:
        raise ValueError(
            f"No evaluation examples loaded from {path}"
        )

    return data



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


def normalize_number(text: str) -> str:
    """Basic textual cleanup for an extracted number."""
    return (
        str(text)
        .replace(",", "")
        .replace("$", "")
        .strip()
        .rstrip(".")
    )


def parse_decimal(text: str) -> Decimal | None:
    cleaned = normalize_number(text)

    if not cleaned:
        return None

    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def extract_last_number(text: str) -> str:
    """Extract the last valid numeric value from text."""
    matches = list(re.finditer(NUMBER_PATTERN, text))

    if not matches:
        return ""

    return normalize_number(matches[-1].group(0))


def extract_final_answer(text: str) -> str:
    boxed_matches = re.findall(
        r"\\boxed\s*\{\s*([^{}]+?)\s*\}",
        text,
    )

    if boxed_matches:
        extracted = extract_last_number(boxed_matches[-1])

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


def extract_ground_truth(answer: str) -> str:
    answer = str(answer).strip()

    if "####" in answer:
        final_part = answer.rsplit("####", 1)[-1]
        extracted = extract_last_number(final_part)

        if extracted:
            return extracted

    if re.fullmatch(NUMBER_PATTERN, answer):
        return normalize_number(answer)

    return extract_final_answer(answer)


def gsm8k_reward_func(
    completion: str,
    answer: str,
    **_: Any,
) -> float:
    prediction_text = extract_final_answer(completion)
    target_text = extract_ground_truth(answer)

    prediction = parse_decimal(prediction_text)
    target = parse_decimal(target_text)

    if prediction is None or target is None:
        return 0.0

    return float(prediction == target)


def math500_reward_func(
    completion: str,
    answer: str,
    **_: Any,
) -> float:
    prediction_text, _ = extract_math500_answer(
        completion
    )

    target_text = extract_math500_gold(
        answer
    )

    if not prediction_text or not target_text:
        return 0.0

    try:
        return float(
            grade_answer(
                given_answer=prediction_text,
                ground_truth=target_text,
            )
        )

    except Exception as error:
        print(
            "[math500_reward_func] Failed to score: "
            f"prediction={prediction_text!r}, "
            f"target={target_text!r}, "
            f"error={type(error).__name__}: {error}",
            flush=True,
        )

        return 0.0
    

def get_fen(metadata: dict[str, Any], extra_info: dict[str, Any]) -> str:
    for source in (metadata, extra_info):
        fen = source.get("FEN") or source.get("fen")
        if fen:
            return str(fen)
    return ""


# Single-turn chess reward: mirrors the training-time reward in
# pre2post-chess/rl/verl/reward_function.py (single-turn variant, i.e. only the
# first move after </T> is scored). The multiturn variant that splits on
# <call_env> is deliberately NOT used here.
CHESS_SINGLE_TURN_REWARD_MODEL_TYPE = os.environ.get(
    "REWARD_MODEL_TYPE", "RULE_BASED"
).upper()


def lan_to_uci(lan: str, side_to_move: str = "white") -> str:
    """Convert custom LAN move (e.g. "Pd2d4", "Pd4xe5", "Pe7e8=Q", "O-O") to UCI.

    Raises:
        ValueError if the LAN string is not in the expected format.
    """
    lan = lan.rstrip("+#").strip()

    if lan == "O-O":
        if side_to_move == "white":
            return "e1g1"
        if side_to_move == "black":
            return "e8g8"
        raise ValueError("Invalid side_to_move for castling")

    if lan == "O-O-O":
        if side_to_move == "white":
            return "e1c1"
        if side_to_move == "black":
            return "e8c8"
        raise ValueError("Invalid side_to_move for castling")

    match = re.match(
        r"^([PNBRQK])([a-h][1-8])(x)?([a-h][1-8])(=([QRBN]))?$", lan
    )
    if not match:
        raise ValueError(f"Invalid LAN format: {lan}")

    _piece, from_square, _capture, to_square, _promo_group, promo = match.groups()

    uci = from_square + to_square
    if promo:
        uci += promo.lower()  # UCI uses lowercase for promotion (q/r/b/n)

    return uci


def is_complete_move(text: str) -> bool:
    """Whether text is a complete move in the custom LAN format."""
    if not text:
        return False

    move = text.rstrip("+#")

    if move in ("O-O", "O-O-O"):
        return True

    return bool(
        re.match(r"^[PNBRQK][a-h][1-8](x)?[a-h][1-8](=[QRBN])?$", move)
    )


def extract_first_move(text: str) -> str | None:
    """Return the first complete move in text, skipping move numbers."""
    tokens = text.strip().split()

    for token in tokens:
        if re.match(r"^\d+\.{1,3}$", token):
            continue
        if is_complete_move(token):
            return token

    return None


def extract_move_after_thinking(text: str) -> tuple[str | None, bool]:
    """Extract the first move after </T>.

    Strict mode: the format only counts as followed when exactly one </T> is
    present; otherwise returns (None, False).
    """
    text = text.strip()

    follows_format = text.count("</T>") == 1

    if not follows_format:
        return None, False

    text_after_thinking = text[
        text.find("</T>") + len("</T>") :
    ].strip()

    if not text_after_thinking:
        return None, follows_format

    return extract_first_move(text_after_thinking), follows_format


def parse_chess_ground_truth(answer: Any) -> str:
    """Normalize the ground truth into a single target UCI move (first move)."""
    ground_truth = answer

    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            try:
                ground_truth = ast.literal_eval(ground_truth)
            except (ValueError, SyntaxError):
                pass

    if isinstance(ground_truth, list) and ground_truth:
        return str(ground_truth[0]).strip()

    return str(ground_truth).strip()


def check_move_legality(fen: str, uci_move: str) -> float:
    """1.0 if uci_move is legal on the board described by fen, else 0.0."""
    if not fen or not uci_move:
        return 0.0
    try:
        import chess

        board = chess.Board(fen)
        move = chess.Move.from_uci(uci_move)
        return 1.0 if move in board.legal_moves else 0.0
    except Exception:
        return 0.0


def chess_single_turn_move_to_uci(move_text: str) -> str:
    """Convert an extracted LAN move to UCI, returning "" on failure."""
    if not move_text:
        return ""
    try:
        return lan_to_uci(move_text)
    except ValueError:
        return ""


def extract_chess_single_turn_move(completion: str) -> tuple[str, bool]:
    """Extract the scored move from a single-turn completion.

    Returns:
        (raw_move, follows_format) where raw_move is "" when nothing parsed.
    """
    move, follows_format = extract_move_after_thinking(completion)

    if move is None and not follows_format:
        move = extract_first_move(completion)

    return (move.strip() if move else ""), follows_format


def chess_single_turn_reward_func(
    completion: str,
    answer: str,
    metadata: dict[str, Any] | None = None,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> float:
    raw_move, follows_format = extract_chess_single_turn_move(completion)
    prediction = chess_single_turn_move_to_uci(raw_move)
    target = parse_chess_ground_truth(answer)

    score = float(bool(prediction) and prediction == target)

    if CHESS_SINGLE_TURN_REWARD_MODEL_TYPE == "RULE_FORMAT_BASED" and not follows_format:
        return 0.0

    return score


def resolve_task(task: str, checkpoint_path: str) -> str:
    """Resolve --task auto into a concrete task."""
    task = task.lower().strip()

    if task != "auto":
        return task

    checkpoint_lower = checkpoint_path.lower()

    if "chess" in checkpoint_lower:
        return "chess"

    if "math500" in checkpoint_lower:
        return "math500"

    return "gsm8k"


def select_reward_func(
    task: str,
    checkpoint_path: str,
) -> RewardFn:
    resolved_task = resolve_task(
        task,
        checkpoint_path,
    )

    if resolved_task == "chess":
        return chess_single_turn_reward_func

    if resolved_task == "math500":
        return math500_reward_func

    return gsm8k_reward_func


def encode_batch_left_padded(
    tokenizer: Any,
    texts: list[str],
    device: Any,
) -> dict[str, torch.Tensor]:
    """Tokenize texts and left-pad them for decoder-only generation.

    The chess checkpoint ships a custom tokenizer whose __call__ always appends
    padding on the right and ignores tokenizer.padding_side, which makes
    model.generate() warn about right-padding and continue from pad tokens.
    Padding here keeps batching correct regardless of the tokenizer.
    """
    sequences = [
        list(tokenizer.encode(text, add_special_tokens=True)) for text in texts
    ]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    width = max(len(ids) for ids in sequences)

    input_ids = [
        [pad_id] * (width - len(ids)) + ids for ids in sequences
    ]
    attention_mask = [
        [0] * (width - len(ids)) + [1] * len(ids) for ids in sequences
    ]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(
            attention_mask, dtype=torch.long, device=device
        ),
    }


def get_target_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_target_state(
    model: torch.nn.Module, state: dict[str, torch.Tensor]
) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in state:
                parameter.copy_(state[name])


def load_perturbed_state(
    model: torch.nn.Module,
    theta: dict[str, torch.Tensor],
    direction: dict[str, torch.Tensor],
    coefficient: float,
) -> None:
    """Set model parameters to theta + coefficient * direction in place."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in theta:
                parameter.copy_(theta[name])
                parameter.add_(
                    direction[name],
                    alpha=coefficient,
                )


def random_direction_like(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    direction = {
        name: torch.randn_like(value)
        for name, value in state.items()
    }

    direction_squared_norm = sum(
        torch.sum(value.float() ** 2).item()
        for value in direction.values()
    )

    state_squared_norm = sum(
        torch.sum(value.float() ** 2).item()
        for value in state.values()
    )

    direction_norm = direction_squared_norm**0.5
    state_norm = state_squared_norm**0.5

    scale = state_norm / (direction_norm + 1e-12)

    return {
        name: value * scale
        for name, value in direction.items()
    }


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1

def setup_distributed() -> tuple[int, int, torch.device]:
    """Initialize one inference process per GPU."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
            )

        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    return local_rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def eval_mean_reward(
    model: torch.nn.Module,
    tokenizer: Any,
    data: list[dict[str, Any]],
    reward_fn: RewardFn,
    max_new_tokens: int,
    task: str,
    batch_size: int = 8,
    group_size: int = 4,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[
    float,
    float,
    float,
    list[dict[str, Any]],
]:
    """Evaluate mean reward and two different standard-error estimates.

    For each prompt, generate ``group_size`` completions.

    Reward matrix:

        scores.shape == (num_prompts, group_size)

    Mean reward:

        mean_reward = scores.mean()

    Method 1 -- generation-level standard error:

        generation_means = scores.mean(dim=0)

        std_error_generation =
            generation_means.std(unbiased=True)
            / sqrt(group_size)

    This is the original method.

    Method 2 -- prompt-level standard deviation RMS:

        prompt_stds = scores.std(dim=1, unbiased=True)

        std_error_prompt_rms =
            prompt_stds.square().mean().sqrt()

    This is equivalent to:

        scores.std(dim=1).square().mean().sqrt()

    Importantly, Method 2 is computed using sufficient statistics
    (sum of rewards and sum of squared rewards) and therefore does
    NOT require gathering the entire reward matrix across ranks.

    Returns:
        (
            global_mean_reward,
            std_error_generation,
            std_error_prompt_rms,
            global_generation_records,
        )
    """

    if group_size < 1:
        raise ValueError(
            f"group_size must be positive, got {group_size}"
        )

    if group_size > 1 and temperature <= 0:
        raise ValueError(
            "temperature must be positive when group_size > 1"
        )

    rank = get_rank()
    world_size = get_world_size()

    # ------------------------------------------------------------------
    # Assign prompts to ranks.
    # ------------------------------------------------------------------

    local_indexed_data = list(
        enumerate(data)
    )[rank::world_size]

    local_data = [
        example
        for _, example in local_indexed_data
    ]

    local_data_indices = [
        data_index
        for data_index, _ in local_indexed_data
    ]

    # ------------------------------------------------------------------
    # We still keep generation records because rank 0 needs to save
    # completion-level information to CSV.
    #
    # We DO NOT gather local_prompt_reward_rows anymore.
    # This avoids all_gather_object() and its CUDA memory overhead.
    # ------------------------------------------------------------------

    local_generation_records: list[
        dict[str, Any]
    ] = []

    # ------------------------------------------------------------------
    # Sufficient statistics for Method 2.
    #
    # For every prompt i:
    #
    #   sum_i  = sum_j scores[i, j]
    #   sqsum_i = sum_j scores[i, j]^2
    #
    # Then:
    #
    #   sample_variance_i =
    #       (sqsum_i - sum_i^2 / G) / (G - 1)
    #
    # We only need the sum of sample_variance_i over prompts.
    # ------------------------------------------------------------------

    local_prompt_variance_sum = 0.0
    local_prompt_count = 0

    # ------------------------------------------------------------------
    # Method 1 statistics.
    #
    # We need the reward sum for each generation position:
    #
    #   generation_reward_sums[j]
    #
    # and the total number of prompts.
    # ------------------------------------------------------------------

    local_generation_reward_sums = torch.zeros(
        group_size,
        dtype=torch.float64,
        device=model.device,
    )

    debug_prompt_count = 0

    # ------------------------------------------------------------------
    # Generation loop.
    # ------------------------------------------------------------------

    for start in range(
        0,
        len(local_data),
        batch_size,
    ):
        prompt_batch = local_data[
            start : start + batch_size
        ]

        # --------------------------------------------------------------
        # Build prompts.
        # --------------------------------------------------------------

        if task == "chess":
            prompt_texts = [
                example["prompt"]
                for example in prompt_batch
            ]

        else:
            prompt_texts = [
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": example["prompt"],
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for example in prompt_batch
            ]

        # --------------------------------------------------------------
        # Repeat each prompt group_size times.
        #
        # Example group_size=4:
        #
        #   prompt_0 x4
        #   prompt_1 x4
        #   prompt_2 x4
        # --------------------------------------------------------------

        expanded_texts: list[str] = []
        expanded_examples: list[
            dict[str, Any]
        ] = []
        expanded_group_indices: list[int] = []

        for prompt_index, (
            text,
            example,
        ) in enumerate(
            zip(
                prompt_texts,
                prompt_batch,
            )
        ):
            for group_index in range(
                group_size
            ):
                expanded_texts.append(text)
                expanded_examples.append(example)
                expanded_group_indices.append(
                    group_index
                )

        # --------------------------------------------------------------
        # Tokenize.
        # --------------------------------------------------------------

        inputs = encode_batch_left_padded(
            tokenizer,
            expanded_texts,
            model.device,
        )

        # --------------------------------------------------------------
        # Generation.
        # --------------------------------------------------------------

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }

        if group_size > 1:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )
        else:
            generation_kwargs["do_sample"] = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
        )

        input_width = inputs[
            "input_ids"
        ].shape[1]

        # --------------------------------------------------------------
        # Reward matrix for this batch:
        #
        #   (num_prompts_in_batch, group_size)
        #
        # This is small and lives only for the current batch.
        # --------------------------------------------------------------

        batch_reward_scores = torch.zeros(
            (
                len(prompt_batch),
                group_size,
            ),
            dtype=torch.float64,
        )

        # --------------------------------------------------------------
        # Decode and score every generation.
        # --------------------------------------------------------------

        for expanded_index, example in enumerate(
            expanded_examples
        ):
            prompt_index = (
                expanded_index // group_size
            )

            group_index = (
                expanded_group_indices[
                    expanded_index
                ]
            )

            raw_generated_ids = outputs[
                expanded_index,
                input_width:,
            ]

            # ----------------------------------------------------------
            # Remove EOS from decoded completion.
            # ----------------------------------------------------------

            eos_positions = (
                raw_generated_ids
                == tokenizer.eos_token_id
            ).nonzero(
                as_tuple=True
            )[0]

            if len(eos_positions) > 0:
                eos_position = int(
                    eos_positions[0].item()
                )

                generated_ids = (
                    raw_generated_ids[
                        :eos_position
                    ]
                )

                ended_with_eos = True

            else:
                generated_ids = (
                    raw_generated_ids
                )

                ended_with_eos = False

            completion = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )

            # ----------------------------------------------------------
            # Global prompt index.
            # ----------------------------------------------------------

            global_prompt_index = (
                local_data_indices[
                    start + prompt_index
                ]
            )

            # ----------------------------------------------------------
            # Completion statistics.
            # ----------------------------------------------------------

            completion_token_length = int(
                generated_ids.numel()
            )

            raw_completion_token_length = int(
                raw_generated_ids.numel()
            )

            likely_truncated = (
                not ended_with_eos
                and raw_completion_token_length
                >= max_new_tokens
            )

            # ----------------------------------------------------------
            # Reward.
            # ----------------------------------------------------------

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

            reward_value = float(reward)

            # ----------------------------------------------------------
            # Store reward.
            # ----------------------------------------------------------

            batch_reward_scores[
                prompt_index,
                group_index,
            ] = reward_value

            # ----------------------------------------------------------
            # Save generation-level record.
            # ----------------------------------------------------------

            local_generation_records.append(
                {
                    "prompt_index": (
                        global_prompt_index
                    ),
                    "group_index": (
                        group_index
                    ),
                    "group_sample": (
                        group_index + 1
                    ),
                    "completion_tokens": (
                        completion_token_length
                    ),
                    "raw_completion_tokens": (
                        raw_completion_token_length
                    ),
                    "completion_characters": (
                        len(completion)
                    ),
                    "ended_with_eos": (
                        ended_with_eos
                    ),
                    "likely_truncated": (
                        likely_truncated
                    ),
                    "reward": reward_value,
                }
            )

            # ----------------------------------------------------------
            # Debug output.
            # ----------------------------------------------------------

            if (
                rank == 0
                and debug_prompt_count
                + prompt_index
                < 3
            ):
                print(
                    "\n" + "=" * 100
                )

                print(
                    "Prompt index: "
                    f"{debug_prompt_count + prompt_index}"
                )

                print(
                    "Group sample: "
                    f"{group_index + 1}/{group_size}"
                )

                print(
                    "Generated tokens: "
                    f"{len(generated_ids)}"
                )

                print(
                    "Ended with EOS: "
                    f"{ended_with_eos}"
                )

                print(
                    "Likely truncated: "
                    f"{likely_truncated}"
                )

                # ------------------------------------------------------
                # Task-specific debug information.
                # ------------------------------------------------------

                if task == "chess":
                    metadata = (
                        example.get(
                            "metadata"
                        )
                        or {}
                    )

                    extra_info = (
                        example.get(
                            "extra_info"
                        )
                        or {}
                    )

                    fen = get_fen(
                        metadata,
                        extra_info,
                    )

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
                        "Contains </T>: "
                        f"{'</T>' in completion}"
                    )

                    print(
                        "Follows <T></T> format: "
                        f"{follows_format}"
                    )

                    print(
                        "Raw extracted move: "
                        f"{raw_move!r}"
                    )

                    print(
                        "Predicted UCI: "
                        f"{predicted_uci!r}"
                    )

                    print(
                        "Target move: "
                        f"{target_move!r}"
                    )

                    print(
                        "First move legality: "
                        f"{check_move_legality(fen, predicted_uci)}"
                    )

                    print(
                        f"FEN: {fen}"
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
                        "Extracted answer: "
                        f"{prediction!r}"
                    )

                    print(
                        "Ground truth: "
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

                    try:
                        grader_result = (
                            bool(
                                prediction_text
                            )
                            and bool(
                                target_text
                            )
                            and grade_answer(
                                given_answer=(
                                    prediction_text
                                ),
                                ground_truth=(
                                    target_text
                                ),
                            )
                        )

                    except Exception as error:
                        grader_result = False

                        print(
                            "Grader error: "
                            f"{type(error).__name__}: "
                            f"{error}"
                        )

                    print(
                        "Extraction method: "
                        f"{extraction_method}"
                    )

                    print(
                        "Extracted answer: "
                        f"{prediction_text!r}"
                    )

                    print(
                        "Ground truth: "
                        f"{target_text!r}"
                    )

                    print(
                        "PRM800K grader result: "
                        f"{grader_result}"
                    )

                print(
                    f"Reward: {reward_value}"
                )

                print(
                    "-" * 100
                )

                print(
                    "FULL GENERATION:"
                )

                print(completion)

                print(
                    "=" * 100,
                    flush=True,
                )

        # --------------------------------------------------------------
        # Accumulate Method 1 statistics.
        #
        # Sum reward for each generation position.
        #
        # shape:
        #   (group_size,)
        # --------------------------------------------------------------

        batch_generation_reward_sums = (
            batch_reward_scores.sum(
                dim=0
            )
        )

        local_generation_reward_sums += (
            batch_generation_reward_sums.to(
                device=model.device
            )
        )

        # --------------------------------------------------------------
        # Accumulate Method 2 statistics.
        #
        # For every prompt:
        #
        #   mean_i = sum_i / G
        #
        #   sample_var_i =
        #       (sum(x^2) - sum(x)^2 / G)
        #       / (G - 1)
        #
        # Then:
        #
        #   Method 2 =
        #       sqrt(mean_i(sample_var_i))
        #
        # We only accumulate the sum of sample variances.
        # --------------------------------------------------------------

        if group_size >= 2:
            batch_prompt_reward_sums = (
                batch_reward_scores.sum(
                    dim=1
                )
            )

            batch_prompt_reward_squared_sums = (
                batch_reward_scores.square().sum(
                    dim=1
                )
            )

            batch_prompt_variances = (
                batch_prompt_reward_squared_sums
                - (
                    batch_prompt_reward_sums.square()
                    / group_size
                )
            ) / (group_size - 1)

            # Numerical safety.
            batch_prompt_variances = (
                batch_prompt_variances.clamp(
                    min=0.0
                )
            )

            local_prompt_variance_sum += (
                batch_prompt_variances.sum().item()
            )

        local_prompt_count += len(
            prompt_batch
        )

        debug_prompt_count += len(
            prompt_batch
        )

        # --------------------------------------------------------------
        # Explicitly release temporary generation tensors before the
        # next batch. This is particularly useful for large models.
        # --------------------------------------------------------------

        del outputs
        del inputs
        del batch_reward_scores

    # ==================================================================
    # Distributed reduction.
    # ==================================================================

    local_prompt_count_tensor = torch.tensor(
        float(local_prompt_count),
        dtype=torch.float64,
        device=model.device,
    )

    local_prompt_variance_sum_tensor = (
        torch.tensor(
            float(
                local_prompt_variance_sum
            ),
            dtype=torch.float64,
            device=model.device,
        )
    )

    # --------------------------------------------------------------
    # Combine all statistics into one tiny tensor.
    #
    # First group_size entries:
    #     generation reward sums
    #
    # Entry group_size:
    #     prompt count
    #
    # Entry group_size + 1:
    #     sum of prompt-level sample variances
    # --------------------------------------------------------------

    stats = torch.cat(
        [
            local_generation_reward_sums,
            local_prompt_count_tensor.unsqueeze(
                0
            ),
            local_prompt_variance_sum_tensor.unsqueeze(
                0
            ),
        ]
    )

    if is_distributed():
        dist.all_reduce(
            stats,
            op=dist.ReduceOp.SUM,
        )

    # ==================================================================
    # Recover global statistics.
    # ==================================================================

    global_generation_reward_sums = (
        stats[:group_size]
    )

    global_prompt_count = stats[
        group_size
    ].item()

    global_prompt_variance_sum = stats[
        group_size + 1
    ].item()

    if global_prompt_count == 0:
        mean_reward = float("nan")
        std_error_generation = float("nan")
        std_error_prompt_rms = float("nan")

    else:
        # ==============================================================
        # Mean reward.
        # ==============================================================

        generation_means = (
            global_generation_reward_sums
            / global_prompt_count
        )

        mean_reward = (
            generation_means.mean().item()
        )

        # ==============================================================
        # Method 1:
        #
        # Original generation-level standard error.
        #
        # generation_means.shape == (group_size,)
        #
        # std_error =
        #     std(generation_means) / sqrt(group_size)
        # ==============================================================

        if group_size < 2:
            std_error_generation = 0.0

        else:
            std_error_generation = (
                generation_means.std(
                    unbiased=True
                )
                / np.sqrt(group_size)
            ).item()

        # ==============================================================
        # Method 2:
        #
        # scores.shape:
        #
        #     (num_prompts, group_size)
        #
        # For each prompt:
        #
        #     prompt_std_i =
        #         scores[i].std(unbiased=True)
        #
        # Then:
        #
        #     sqrt(
        #         mean(prompt_std_i^2)
        #     )
        #
        # We computed the numerator using sufficient statistics,
        # so no reward matrix needs to be gathered across GPUs.
        # ==============================================================

        if group_size < 2:
            std_error_prompt_rms = 0.0

        else:
            mean_prompt_variance = (
                global_prompt_variance_sum
                / global_prompt_count
            )

            # Numerical safety.
            mean_prompt_variance = max(
                0.0,
                mean_prompt_variance,
            )

            std_error_prompt_rms = (
                mean_prompt_variance ** 0.5
            )

    # ==================================================================
    # Gather generation records.
    #
    # This is still needed for the completion CSV.
    #
    # Unlike reward scores, these records contain strings and therefore
    # cannot be efficiently reduced with a normal all_reduce.
    #
    # If this itself causes memory problems for very large generation
    # outputs, it can also be moved to CPU/Gloo, but for your current
    # completion-length records this should be small.
    # ==================================================================

    # if is_distributed():
    #     gathered_generation_records: list[
    #         list[dict[str, Any]] | None
    #     ] = [
    #         None
    #         for _ in range(world_size)
    #     ]

    #     dist.all_gather_object(
    #         gathered_generation_records,
    #         local_generation_records,
    #     )

    #     if rank == 0:
    #         global_generation_records = [
    #             record
    #             for rank_records
    #             in gathered_generation_records
    #             if rank_records is not None
    #             for record in rank_records
    #         ]

    #         global_generation_records.sort(
    #             key=lambda record: (
    #                 record[
    #                     "prompt_index"
    #                 ],
    #                 record[
    #                     "group_index"
    #                 ],
    #             )
    #         )

    #     else:
    #         global_generation_records = []

    # else:
    #     global_generation_records = (
    #         local_generation_records
    #     )

    #     global_generation_records.sort(
    #         key=lambda record: (
    #             record[
    #                 "prompt_index"
    #             ],
    #             record[
    #                 "group_index"
    #             ],
    #         )
    #     )

    return (
        mean_reward,
        std_error_generation,
        std_error_prompt_rms,
        local_generation_records,
    )



def infer_checkpoint_step(model_ckpt: str) -> str:
    match = re.search(r"(?:checkpoint-|global_step_)(\d+)", model_ckpt)
    return match.group(1) if match else "unknown"


def build_output_stem(args: argparse.Namespace) -> str:
    if args.output_stem:
        return args.output_stem
    checkpoint_step = (
        str(args.checkpoint_step)
        if args.checkpoint_step is not None
        else infer_checkpoint_step(args.model_ckpt)
    )
    return (
        f"{args.rl_type}_{args.dataset}_reward_line_"
        f"scale{args.scale}_alpha_range{args.alpha_range}_"
        f"num{args.num_samples}_group{args.group_size}_"
        f"ckpt{checkpoint_step}_"
        f"model_{args.model_name}_"
        f"max_new{args.max_new_tokens}_"
        f"bs{args.batch_size}_"
        f"temp{args.temperature}_"
        f"topp{args.top_p}_"
        f"seed{args.seed}_pts{args.num_points}_"
        f"{args.num_directions}dirs"
    )


def save_results(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path]:

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = build_output_stem(args)

    csv_path = (
        output_dir / f"{stem}.csv"
    )

    generation_mean_png_path = (
        output_dir
        / f"{stem}_generation_mean.png"
    )

    prompt_rms_png_path = (
        output_dir
        / f"{stem}_prompt_rms.png"
    )

    # ============================================================
    # Save CSV
    # ============================================================

    df.to_csv(
        csv_path,
        index=False,
    )

    # ============================================================
    # Generic plotting function
    # ============================================================

    def plot_landscape(
        std_column: str,
        png_path: Path,
        title_suffix: str,
    ) -> None:

        plt.figure(
            figsize=(8, 5)
        )

        for (
            direction_name,
            subset,
        ) in df.groupby(
            "direction"
        ):

            subset = subset.sort_values(
                "perturbation_coefficient"
            )

            x = subset[
                "perturbation_coefficient"
            ].to_numpy()

            mean_reward = subset[
                "reward"
            ].to_numpy()

            std_error = subset[
                std_column
            ].to_numpy()

            line = plt.plot(
                x,
                mean_reward,
                marker="o",
                linewidth=2,
                label=direction_name,
            )[0]

            lower_bound = np.clip(
                mean_reward - std_error,
                0.0,
                1.0,
            )

            upper_bound = np.clip(
                mean_reward + std_error,
                0.0,
                1.0,
            )

            plt.fill_between(
                x,
                lower_bound,
                upper_bound,
                alpha=0.2,
                color=line.get_color(),
            )

        plt.xlabel(
            "Perturbation coefficient"
        )

        plt.ylabel(
            "Mean reward"
        )

        plt.title(
            f"{args.rl_type.upper()} "
            f"{args.dataset} "
            f"step {args.checkpoint_step} "
            f"reward landscape "
            f"({title_suffix})"
        )

        plt.ylim(
            args.ymin,
            args.ymax,
        )

        plt.minorticks_on()

        plt.grid(
            which="major",
            linestyle="--",
            linewidth=0.5,
            alpha=0.8,
        )

        plt.grid(
            which="minor",
            linestyle=":",
            linewidth=0.3,
            alpha=0.3,
        )

        plt.legend()
        plt.tight_layout()

        plt.savefig(
            png_path,
            dpi=300,
        )

        plt.close()

    # ============================================================
    # Figure 1
    #
    # Method 1:
    # std(generation_means) / sqrt(group_size)
    # ============================================================

    plot_landscape(
        std_column=(
            "std_error_generation_mean"
        ),
        png_path=(
            generation_mean_png_path
        ),
        title_suffix=(
            "generation-mean SE"
        ),
    )

    # ============================================================
    # Figure 2
    #
    # Method 2:
    # sqrt(mean(prompt_std^2))
    # ============================================================

    plot_landscape(
        std_column=(
            "std_error_prompt_rms"
        ),
        png_path=(
            prompt_rms_png_path
        ),
        title_suffix=(
            "prompt-level RMS std"
        ),
    )

    return (
        csv_path,
        generation_mean_png_path,
        prompt_rms_png_path,
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def verify_direction_across_ranks(
    direction: dict[str, torch.Tensor],
) -> None:
    if not is_distributed():
        return

    local_checksum = sum(
        value.float().sum()
        for value in direction.values()
    )

    min_checksum = local_checksum.clone()
    max_checksum = local_checksum.clone()

    dist.all_reduce(
        min_checksum,
        op=dist.ReduceOp.MIN,
    )
    dist.all_reduce(
        max_checksum,
        op=dist.ReduceOp.MAX,
    )

    if not torch.allclose(
        min_checksum,
        max_checksum,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise RuntimeError(
            "Random directions differ across distributed ranks"
        )


def main() -> None:
    args = parse_args()
    if args.num_points < 1 or args.num_directions < 1 or args.num_samples < 1:
        raise ValueError("num-points, num-directions, and num-samples must be positive")

    local_rank, world_size, device = setup_distributed()
    rank = get_rank()

    set_seed(args.seed)

    if args.num_points == 1:
        grid = np.array([0.0], dtype=float)
    else:
        grid = np.linspace(
            -args.alpha_range,
            args.alpha_range,
            args.num_points,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_ckpt,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 每个进程在对应 GPU 上加载一份完整模型。
    # 不要使用 device_map="auto"。
    model = AutoModelForCausalLM.from_pretrained(
        args.model_ckpt,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        # attn_implementation="flash_attention_2",
    ).to(device)


    model.eval()

    # 模型加载完成后重新设置，保证所有 rank 产生相同方向。
    set_seed(args.seed)

    if rank == 0:
        print(
            f"Distributed inference with {world_size} process(es)",
            flush=True,
        )

    resolved_task = resolve_task(
        args.task,
        args.model_ckpt,
    )

    data = load_eval_data(
        args.eval_json,
        args.num_samples,
        resolved_task,
    )

    reward_fn = select_reward_func(
        resolved_task,
        args.model_ckpt,
    )

    if rank == 0:
        print(f"Resolved task: {resolved_task}")
        print(f"Using reward function: {reward_fn.__name__}")

    theta = get_target_state(model)
    print(f"Perturbing {len(theta)} tensors")
    # directions = [random_direction_like(theta) for _ in range(args.num_directions)]


    rows: list[dict[str, float | str]] = []
    generation_rows: list[dict[str, Any]] = []

    try:
        for direction_index in range(
            1,
            args.num_directions + 1,
        ):
            direction = random_direction_like(theta)
            direction_name = f"direction_{direction_index}"

            for alpha in tqdm(
                grid,
                desc=direction_name,
                disable=(rank != 0),
            ):
                coefficient = args.scale * float(alpha)

                load_perturbed_state(
                    model=model,
                    theta=theta,
                    direction=direction,
                    coefficient=coefficient,
                )

                (
                    reward,
                    std_error_generation_mean,
                    std_error_prompt_rms,
                    generation_records,
                ) = eval_mean_reward(
                    model=model,
                    tokenizer=tokenizer,
                    data=data,
                    reward_fn=reward_fn,
                    max_new_tokens=args.max_new_tokens,
                    task=resolved_task,
                    batch_size=args.batch_size,
                    group_size=args.group_size,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                if rank == 0:
                    rows.append(
                        {
                            "direction": direction_name,
                            "alpha": float(alpha),
                            "perturbation_coefficient": coefficient,
                            "reward": reward,
                            "std_error_generation_mean": (
                                std_error_generation_mean
                            ),
                            "std_error_prompt_rms": (
                                std_error_prompt_rms
                            ),
                        }
                    )
                    for record in generation_records:
                        generation_rows.append(
                            {
                                "direction": direction_name,
                                "alpha": float(alpha),
                                "perturbation_coefficient": coefficient,
                                **record,
                            }
                        )

                    print(
                        f"direction={direction_name}, "
                        f"alpha={alpha:.4f}, "
                        f"coefficient={coefficient:.6f}, "
                        f"reward={reward:.4f}, "
                        f"std_generation_mean="
                        f"{std_error_generation_mean:.6f}, "
                        f"std_prompt_rms="
                        f"{std_error_prompt_rms:.6f}",
                        flush=True,
                    )

            del direction
    finally:
        load_target_state(model, theta)

    if is_distributed():
        dist.barrier()

    if rank == 0:
        (
            csv_path,
            generation_mean_png_path,
            prompt_rms_png_path,
        ) = save_results(
            pd.DataFrame(rows),
            args,
        )

        generation_csv_path = (
            Path(args.output_dir)
            / (
                f"completion_lengths_{build_output_stem(args)}"
                ".csv"
            )
        )

        generation_df = pd.DataFrame(
            generation_rows
        ).sort_values(
            [
                "direction",
                "alpha",
                "prompt_index",
                "group_index",
            ]
        )

        generation_df.to_csv(
            generation_csv_path,
            index=False,
        )
        print(
            f"Saved {csv_path}, "
            f"{generation_mean_png_path}, "
            f"{prompt_rms_png_path}, "
            f"and {generation_csv_path}",
            flush=True,
        )

    cleanup_distributed()


if __name__ == "__main__":
    main()