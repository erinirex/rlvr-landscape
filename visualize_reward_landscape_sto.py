#!/usr/bin/env python3
"""Evaluate a model's mean reward along random parameter-space directions."""

import argparse
import ast
import json
import os
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
    parser.add_argument("--model-ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--eval-json", required=True, help="Evaluation JSONL path")
    parser.add_argument(
        "--task",
        choices=("auto", "chess", "gsm8k"),
        default="auto",
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
    parser.add_argument(
        "--direction-type",
        choices=("random", "grpo-grad"),
        default="random",
        help=(
            "random: isotropic Gaussian direction. "
            "grpo-grad: d = -grad L_GRPO(theta) from a minibatch, using the "
            "actual GRPO training loss (std-normalized advantages, dual-clip "
            "policy loss, low_var_kl) and a single loss.backward()."
        ),
    )
    parser.add_argument(
        "--pg-num-prompts",
        type=int,
        default=16,
        help="Prompts in the minibatch used for the -grad L_GRPO estimate",
    )
    parser.add_argument(
        "--pg-group-size",
        type=int,
        default=8,
        help="Completions sampled per prompt (GRPO group)",
    )
    parser.add_argument(
        "--pg-temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for the rollout (matches training rollout)",
    )
    parser.add_argument("--output-dir", default="figs_qwen_chess")
    parser.add_argument("--output-stem", default=None)
    parser.add_argument("--ymin", type=float, default=0.0)
    parser.add_argument("--ymax", type=float, default=1.0)
    return parser.parse_args()


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

    if task not in {"gsm8k", "chess"}:
        raise ValueError(
            f"Unsupported task: {task!r}. "
            "Expected 'gsm8k' or 'chess'."
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


from decimal import Decimal, InvalidOperation
import re


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


def select_reward_func(task: str, checkpoint_path: str) -> RewardFn:
    if task == "auto":
        task = "chess" if "chess" in checkpoint_path.lower() else "gsm8k"
    return (
        chess_single_turn_reward_func if task == "chess" else gsm8k_reward_func
    )


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


def global_norm(state: dict[str, torch.Tensor]) -> float:
    squared = sum(
        torch.sum(value.float() ** 2).item() for value in state.values()
    )
    return squared**0.5


def scale_to_state_norm(
    direction: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Rescale direction so ||direction|| == ||state||.

    Both direction types share this convention so that --scale and
    --alpha-range mean the same thing regardless of how the direction was
    obtained.
    """
    scale = global_norm(state) / (global_norm(direction) + 1e-12)
    return {name: value * scale for name, value in direction.items()}


def random_direction_like(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    direction = {
        name: torch.randn_like(value)
        for name, value in state.items()
    }

    return scale_to_state_norm(direction, state)


def setup_distributed() -> tuple[int, int, int, torch.device]:
    """Initialize one process per GPU when launched with torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Distributed NCCL execution requires CUDA."
            )

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        device = torch.device(
            f"cuda:{local_rank}"
        )
    else:
        rank = 0
        world_size = 1
        local_rank = 0

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1



@torch.no_grad()
def eval_mean_reward(
    model: torch.nn.Module,
    tokenizer: Any,
    data: list[dict[str, Any]],
    reward_fn: RewardFn,
    max_new_tokens: int,
    task: str,
    batch_size: int = 8,
) -> float:
    rank = get_rank()
    world_size = get_world_size()

    local_data = data[rank::world_size]

    local_reward_sum = 0.0
    local_reward_count = 0

    for start in range(0, len(local_data), batch_size):
        batch = local_data[start : start + batch_size]

        if task == "chess":
            texts = [
                example["prompt"]
                for example in batch
            ]
        else:
            texts = [
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
                for example in batch
            ]

        inputs = encode_batch_left_padded(tokenizer, texts, model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        input_width = inputs["input_ids"].shape[1]

        for i, example in enumerate(batch):
            raw_generated_ids = outputs[i, input_width:]

            eos_positions = (
                raw_generated_ids == tokenizer.eos_token_id
            ).nonzero(as_tuple=True)[0]

            if len(eos_positions) > 0:
                eos_position = int(eos_positions[0].item())
                generated_ids = raw_generated_ids[:eos_position]
                ended_with_eos = True
            else:
                generated_ids = raw_generated_ids
                ended_with_eos = False

            completion = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )

            reward = reward_fn(
                completion,
                example["answer"],
                metadata=example.get("metadata"),
                extra_info=example.get("extra_info"),
            )

            if rank == 0 and local_reward_count < 3:
                likely_truncated = (
                    not ended_with_eos
                    and len(raw_generated_ids) >= max_new_tokens
                )

                print("\n" + "=" * 100)
                print(f"Example index: {local_reward_count}")
                print(f"Generated tokens: {len(generated_ids)}")
                print(f"Ended with EOS: {ended_with_eos}")
                print(f"Likely truncated: {likely_truncated}")

                if task == "chess":
                    metadata = example.get("metadata") or {}
                    extra_info = example.get("extra_info") or {}

                    fen = get_fen(metadata, extra_info)
                    raw_move, follows_format = extract_chess_single_turn_move(
                        completion
                    )
                    predicted_uci = chess_single_turn_move_to_uci(raw_move)
                    target_move = parse_chess_ground_truth(example["answer"])

                    print(f"Contains </T>: {'</T>' in completion}")
                    print(f"Follows <T></T> format: {follows_format}")
                    print(f"Raw extracted move: {raw_move!r}")
                    print(f"Predicted UCI: {predicted_uci!r}")
                    print(f"Target move: {target_move!r}")
                    print(f"Ground truth: {example['answer']!r}")
                    print(
                        "First move legality: "
                        f"{check_move_legality(fen, predicted_uci)}"
                    )
                    print(f"FEN: {fen}")

                elif task == "gsm8k":
                    prediction = extract_final_answer(completion)
                    target = extract_ground_truth(example["answer"])

                    contains_final = bool(
                        re.search(r"Final\s+Answer", completion, re.IGNORECASE)
                    )
                    print(f"Contains final answer: {contains_final}")
                    contains_boxed = bool(
                        re.search(r"\\boxed\s*\{", completion)
                    )
                    print(f"Contains boxed answer: {contains_boxed}")
                    print(f"Extracted answer: {prediction!r}")
                    print(f"Ground truth: {target!r}")

                print(f"Reward: {reward}")
                print("-" * 100)
                print("FULL GENERATION:")
                print(completion)
                print("=" * 100, flush=True)

            local_reward_sum += float(reward)
            local_reward_count += 1

    stats = torch.tensor(
        [
            local_reward_sum,
            float(local_reward_count),
        ],
        dtype=torch.float64,
        device=model.device,
    )

    if is_distributed():
        dist.all_reduce(
            stats,
            op=dist.ReduceOp.SUM,
        )

    global_reward_sum = stats[0].item()
    global_reward_count = int(stats[1].item())

    if global_reward_count == 0:
        return float("nan")

    return global_reward_sum / global_reward_count


def build_prompt_text(tokenizer: Any, example: dict[str, Any], task: str) -> str:
    """Render one example's prompt exactly as eval_mean_reward does."""
    if task == "chess":
        return example["prompt"]

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": example["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# GRPO hyperparameters, matching the training invocation in
# pre2post-chess/rl/verl/8_gpu_bash/run_multi_turn.sh and the defaults in
# verl/trainer/config/ppo_trainer.yaml. These are the settings under which the
# checkpoint was actually trained.
GRPO_CLIP_RATIO_LOW = 0.2
GRPO_CLIP_RATIO_HIGH = 0.2
GRPO_CLIP_RATIO_C = 3.0
GRPO_LOSS_AGG_MODE = "token-mean"
GRPO_NORM_ADV_BY_STD = True   # norm_adv_by_std_in_grpo=True
GRPO_USE_KL_LOSS = True       # actor.use_kl_loss=True
GRPO_KL_LOSS_COEF = 0.001     # KL_LOSS_COEF
GRPO_KL_LOSS_TYPE = "low_var_kl"
GRPO_ADV_EPSILON = 1e-6


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Port of verl.utils.torch_functional.masked_mean (axis=None)."""
    return (values * mask).sum() / (mask.sum() + 1e-8)


def grpo_outcome_advantages(
    scores: torch.Tensor,
    group_index: list[int],
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute group-relative, optionally std-normalized advantages."""
    id2indices: dict[int, list[int]] = {}

    for index, group_id in enumerate(group_index):
        id2indices.setdefault(group_id, []).append(index)

    normalized = torch.zeros_like(scores)

    for group_id, indices in id2indices.items():
        index_tensor = torch.tensor(
            indices,
            dtype=torch.long,
            device=scores.device,
        )

        group_scores = scores[index_tensor]

        group_mean = group_scores.mean()

        if len(indices) > 1:
            group_std = group_scores.std(unbiased=True)
        else:
            group_std = torch.ones(
                (),
                dtype=scores.dtype,
                device=scores.device,
            )

        if GRPO_NORM_ADV_BY_STD:
            group_advantages = (
                group_scores - group_mean
            ) / (group_std + GRPO_ADV_EPSILON)
        else:
            group_advantages = group_scores - group_mean

        normalized[index_tensor] = group_advantages

    return normalized.unsqueeze(-1) * response_mask


def grpo_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Port of core_algos.compute_policy_loss (dual-clip PPO, token-mean)."""
    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - GRPO_CLIP_RATIO_LOW, 1 + GRPO_CLIP_RATIO_HIGH
    )
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

    pg_losses3 = -advantages * GRPO_CLIP_RATIO_C
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    return masked_mean(pg_losses, response_mask)


def grpo_kl_loss(
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Port of core_algos.kl_penalty(low_var_kl) + token-mean agg_loss."""
    kl = torch.clamp(ref_log_prob - log_prob, min=-20, max=20)
    ratio = torch.exp(kl)
    kld = torch.clamp(ratio - kl - 1, min=-10, max=10)
    return masked_mean(kld, response_mask)


def sample_rollout_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    reward_fn: RewardFn,
    task: str,
    max_new_tokens: int,
    group_size: int,
    temperature: float,
) -> tuple[
    list[list[int]], list[list[int]], list[float], list[int], float, int
]:
    """Sample group_size completions per prompt at the unperturbed theta.

    Returns (prompt_ids_per_seq, completion_ids_per_seq, rewards, group_index,
    reward_total, reward_count).
    """
    prompt_ids_per_seq: list[list[int]] = []
    completion_ids_per_seq: list[list[int]] = []
    rewards: list[float] = []
    group_index: list[int] = []

    reward_total = 0.0
    reward_count = 0

    for gid, example in enumerate(tqdm(prompts, desc="grpo-rollout")):
        prompt_text = build_prompt_text(tokenizer, example, task)
        encoded = encode_batch_left_padded(tokenizer, [prompt_text], model.device)
        prompt_ids = encoded["input_ids"][0].tolist()

        with torch.no_grad():
            sampled = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=0,
                top_p=1.0,
                num_return_sequences=group_size,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        input_width = encoded["input_ids"].shape[1]

        for row in range(sampled.shape[0]):
            generated = sampled[row, input_width:]

            eos_positions = (
                generated == tokenizer.eos_token_id
            ).nonzero(as_tuple=True)[0]

            if len(eos_positions) > 0:
                # Keep the EOS token itself: stopping is part of the action.
                generated = generated[: int(eos_positions[0].item()) + 1]

            completion_ids = generated.tolist()

            if not completion_ids:
                continue

            text = tokenizer.decode(
                generated,
                skip_special_tokens=True,
            )

            reward = float(
                reward_fn(
                    text,
                    example["answer"],
                    metadata=example.get("metadata"),
                    extra_info=example.get("extra_info"),
                )
            )

            prompt_ids_per_seq.append(prompt_ids)
            completion_ids_per_seq.append(completion_ids)
            rewards.append(reward)
            group_index.append(gid)

            reward_total += reward
            reward_count += 1

    return (
        prompt_ids_per_seq,
        completion_ids_per_seq,
        rewards,
        group_index,
        reward_total,
        reward_count,
    )


def compute_batch_log_probs(
    model: torch.nn.Module,
    prompt_ids_per_seq: list[list[int]],
    completion_ids_per_seq: list[list[int]],
    pad_id: int,
    temperature: float,
    device: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched per-token log pi(token | prefix) with the graph kept alive.

    Mirrors verl's actor forward: logits are divided by the rollout temperature
    before log-softmax, and only completion tokens are scored.

    Returns (log_prob, response_mask), each shape (bs, max_len - 1) in the
    shifted "next-token" frame where column t scores full[t + 1].
    """
    fulls = [
        prompt + completion
        for prompt, completion in zip(
            prompt_ids_per_seq, completion_ids_per_seq, strict=True
        )
    ]
    max_len = max(len(ids) for ids in fulls)

    input_ids = torch.full(
        (len(fulls), max_len), pad_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros(
        (len(fulls), max_len), dtype=torch.long, device=device
    )
    # Right-pad: forward log-prob is position-invariant to right padding as long
    # as attention_mask masks it out; generate()'s left-padding requirement does
    # not apply here.
    for row, ids in enumerate(fulls):
        input_ids[row, : len(ids)] = torch.tensor(
            ids, dtype=torch.long, device=device
        )
        attention_mask[row, : len(ids)] = 1

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :].float() / temperature   # predicts tokens 1..L-1
    targets = input_ids[:, 1:]                          # (bs, L-1)

    log_probs_all = torch.log_softmax(logits, dim=-1)
    log_prob = log_probs_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # response_mask[b, t] == 1 iff full index t+1 is a completion token of b.
    response_mask = torch.zeros_like(log_prob)
    for row, (prompt, completion) in enumerate(
        zip(prompt_ids_per_seq, completion_ids_per_seq, strict=True)
    ):
        start = len(prompt) - 1  # column predicting the first completion token
        response_mask[row, start : start + len(completion)] = 1.0

    return log_prob, response_mask


def grpo_gradient_direction(
    model: torch.nn.Module,
    tokenizer: Any,
    data: list[dict[str, Any]],
    reward_fn: RewardFn,
    state: dict[str, torch.Tensor],
    task: str,
    max_new_tokens: int,
    group_size: int,
    num_prompts: int,
    temperature: float,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """d = -grad L_GRPO(theta) from one minibatch, via a single loss.backward().

    Faithfully ports the training objective from verl/trainer/ppo/core_algos.py:
    GRPO std-normalized group advantages, dual-clip policy loss with token-mean
    aggregation, and (since use_kl_loss=True) the low_var_kl penalty. The
    rollout is sampled at the unperturbed theta.

    Note: with a single checkpoint there is no stale policy and no separate
    reference model, so old_log_prob == log_prob (ratio == 1, clipping inactive)
    and ref_log_prob == log_prob (KL == 0 with zero gradient). At theta the loss
    therefore reduces to -masked_mean(advantage * log_prob); the full formula is
    computed anyway for fidelity to training.

    Returns (direction, stats) where direction is rescaled to ||theta||.
    """
    actual_num_prompts = min(
        num_prompts,
        len(data),
    )

    prompt_indices = torch.randperm(
        len(data)
    )[:actual_num_prompts].tolist()

    prompts = [
        data[index]
        for index in prompt_indices
    ]

    (
        prompt_ids_per_seq,
        completion_ids_per_seq,
        rewards,
        group_index,
        reward_total,
        reward_count,
    ) = sample_rollout_batch(
        model,
        tokenizer,
        prompts,
        reward_fn,
        task,
        max_new_tokens,
        group_size,
        temperature,
    )

    if reward_count == 0:
        raise ValueError(
            "GRPO direction is undefined: no non-empty completions were "
            "sampled. Try increasing --max-new-tokens or --pg-num-prompts."
        )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    log_prob, response_mask = compute_batch_log_probs(
        model,
        prompt_ids_per_seq,
        completion_ids_per_seq,
        pad_id,
        temperature,
        model.device,
    )

    scores = torch.tensor(rewards, dtype=torch.float32, device=model.device)
    advantages = grpo_outcome_advantages(scores, group_index, response_mask).to(
        log_prob.dtype
    )

    old_log_prob = log_prob.detach()

    pg_loss = grpo_policy_loss(
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
    )

    # No true reference model is loaded here, so do not add a fake
    # zero-gradient KL term.
    policy_loss = pg_loss
    # if GRPO_USE_KL_LOSS:
    #     ref_log_prob = log_prob.detach()
    #     kl_loss = grpo_kl_loss(log_prob, ref_log_prob, response_mask)
    #     policy_loss = policy_loss + kl_loss * GRPO_KL_LOSS_COEF

    model.zero_grad(set_to_none=True)
    policy_loss.backward()

    # d = -grad L, exactly the training-time gradient direction.
    direction = {
        name: -parameter.grad.detach().clone().float()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)

    # Any perturbed parameter that received no gradient gets a zero direction,
    # so the downstream theta + coeff * direction sweep never misses a key.
    for name, value in state.items():
        if name not in direction:
            direction[name] = torch.zeros_like(value, dtype=torch.float32)

    gradient_norm = global_norm(direction)
    if gradient_norm < 1e-12:
        raise ValueError(
            "GRPO direction is undefined: the gradient norm is ~0. Every "
            "sampled group likely had identical rewards (zero advantage). Try "
            "increasing --pg-group-size, --pg-num-prompts, or --pg-temperature."
        )

    direction = {
        name: value.to(dtype=state[name].dtype)
        for name, value in direction.items()
        if name in state
    }

    stats = {
        "gradient_norm": gradient_norm,
        "loss": float(policy_loss.detach().item()),
        "num_groups": float(len(prompts)),
        "mean_sampled_reward": (
            reward_total / reward_count if reward_count else float("nan")
        ),
        "num_sampled_completions": float(reward_count),
    }

    return scale_to_state_norm(direction, state), stats


def build_directions(
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: Any,
    data: list[dict[str, Any]],
    reward_fn: RewardFn,
    theta: dict[str, torch.Tensor],
) -> list[tuple[str, dict[str, torch.Tensor]]]:
    """Build the named directions to sweep, per --direction-type."""
    if args.direction_type == "random":
        return [
            (f"random_{index}", random_direction_like(theta))
            for index in range(1, args.num_directions + 1)
        ]

    directions: list[tuple[str, dict[str, torch.Tensor]]] = []

    for index in range(1, args.num_directions + 1):
        # Each call resamples completions, so repeated calls give independent
        # stochastic-gradient directions.
        # direction, stats = grpo_gradient_direction(
        #     model,
        #     tokenizer,
        #     data,
        #     reward_fn,
        #     theta,
        #     args.task,
        #     args.max_new_tokens,
        #     args.pg_group_size,
        #     args.pg_num_prompts,
        #     args.pg_temperature,
        # )
        direction = None
        stats = None

        for attempt in range(1, 3 + 1):
            try:
                direction, stats = grpo_gradient_direction(
                    model,
                    tokenizer,
                    data,
                    reward_fn,
                    theta,
                    args.task,
                    args.max_new_tokens,
                    args.pg_group_size,
                    args.pg_num_prompts,
                    args.pg_temperature,
                )
                break
            except ValueError as error:
                if "gradient norm is ~0" not in str(error):
                    raise

                print(
                    f"grpo direction {index}: zero-gradient rollout "
                    f"on attempt {attempt}/3; resampling.",
                    flush=True,
                )

        if direction is None or stats is None:
            raise ValueError(
                f"Failed to obtain a non-zero GRPO direction after "
                f"{3} rollout attempts."
            )
        print(
            f"grpo direction {index}: grad_norm={stats['gradient_norm']:.6g}, "
            f"loss={stats['loss']:.6g}, groups={int(stats['num_groups'])}, "
            f"mean_sampled_reward={stats['mean_sampled_reward']:.4f}, "
            f"completions={int(stats['num_sampled_completions'])}",
            flush=True,
        )
        directions.append((f"grpo_grad_{index}", direction))

    return directions


def broadcast_direction(
    direction: dict[str, torch.Tensor] | None,
    state: dict[str, torch.Tensor],
    src: int = 0,
) -> dict[str, torch.Tensor]:
    """Broadcast one parameter-space direction from src to every rank.

    All ranks must receive exactly the same direction. Otherwise they would
    evaluate different perturbed models and the reduced reward would be invalid.
    """
    if not is_distributed():
        if direction is None:
            raise ValueError(
                "direction cannot be None in non-distributed mode"
            )
        return direction

    rank = get_rank()
    synchronized: dict[str, torch.Tensor] = {}

    for name, reference_value in state.items():
        if rank == src:
            if direction is None or name not in direction:
                tensor = torch.zeros_like(
                    reference_value
                )
            else:
                tensor = direction[name].to(
                    device=reference_value.device,
                    dtype=reference_value.dtype,
                ).contiguous()
        else:
            tensor = torch.empty_like(
                reference_value
            )

        dist.broadcast(
            tensor,
            src=src,
        )

        synchronized[name] = tensor

    return synchronized

def build_and_broadcast_directions(
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: Any,
    data: list[dict[str, Any]],
    reward_fn: RewardFn,
    theta: dict[str, torch.Tensor],
) -> list[
    tuple[str, dict[str, torch.Tensor]]
]:
    """Build directions on rank 0 and broadcast them to every rank."""
    rank = get_rank()

    if rank == 0:
        local_directions = build_directions(
            args,
            model,
            tokenizer,
            data,
            reward_fn,
            theta,
        )

        direction_names = [
            name
            for name, _ in local_directions
        ]
    else:
        local_directions = []
        direction_names = []

    if is_distributed():
        names_object: list[Any] = [
            direction_names
            if rank == 0
            else None
        ]

        dist.broadcast_object_list(
            names_object,
            src=0,
        )

        direction_names = names_object[0]

    synchronized_directions: list[
        tuple[str, dict[str, torch.Tensor]]
    ] = []

    for index, direction_name in enumerate(
        direction_names
    ):
        if rank == 0:
            local_direction = local_directions[index][1]
        else:
            local_direction = None

        synchronized_direction = broadcast_direction(
            local_direction,
            theta,
            src=0,
        )

        synchronized_directions.append(
            (
                direction_name,
                synchronized_direction,
            )
        )

    if is_distributed():
        dist.barrier()

    return synchronized_directions

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
    stem = (
        f"{args.rl_type}_{args.dataset}_reward_line_"
        f"dir_{args.direction_type.replace('-', '')}_"
        f"scale{args.scale}_alpha_range{args.alpha_range}_"
        f"num{args.num_samples}_ckpt{checkpoint_step}_"
        f"model_{args.model_name}_max_new{args.max_new_tokens}_bs{args.batch_size}_"
        f"seed{args.seed}_pts{args.num_points}_{args.num_directions}dirs"
    )

    if args.direction_type == "grpo-grad":
        stem += (
            f"_pg{args.pg_num_prompts}x{args.pg_group_size}"
            f"_temp{args.pg_temperature}"
        )

    return stem


def save_results(df: pd.DataFrame, args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = build_output_stem(args)
    csv_path = output_dir / f"{stem}.csv"
    png_path = output_dir / f"{stem}.png"
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5))
    for direction_name, subset in df.groupby("direction"):
        subset = subset.sort_values("perturbation_coefficient")
        plt.plot(
            subset["perturbation_coefficient"],
            subset["reward"],
            marker="o",
            linewidth=2,
            label=direction_name,
        )
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.xlabel("Perturbation coefficient")
    plt.ylabel("Mean reward")

    direction_label = (
        "d = -grad L_GRPO"
        if args.direction_type == "grpo-grad"
        else "d ~ N(0, I)"
    )
    plt.title(
        f"{args.rl_type.upper()} {args.dataset} step {args.checkpoint_step} "
        f"reward landscape\n{direction_label}"
    )
    plt.ylim(args.ymin, args.ymax)
    plt.minorticks_on()
    plt.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.8)
    plt.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
    return csv_path, png_path


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()

    rank, world_size, local_rank, device = (
        setup_distributed()
    )

    try:
        if (
            args.num_points < 1
            or args.num_directions < 1
            or args.num_samples < 1
        ):
            raise ValueError(
                "num-points, num-directions, and "
                "num-samples must be positive"
            )

        if args.direction_type == "grpo-grad":
            if args.pg_num_prompts < 1:
                raise ValueError(
                    "pg-num-prompts must be positive"
                )

            if args.pg_group_size < 2:
                raise ValueError(
                    "pg-group-size must be >= 2 so "
                    "GRPO group normalization can "
                    "produce non-zero advantages"
                )

            if args.pg_temperature <= 0:
                raise ValueError(
                    "pg-temperature must be positive"
                )


        # Each rank uses the same seed. Directions are only created on rank 0
        # and then broadcast, so this remains deterministic.
        set_seed(args.seed)

        grid = np.linspace(
            -args.alpha_range,
            args.alpha_range,
            args.num_points,
        )

        if rank == 0:
            print(
                f"Distributed mode: "
                f"{is_distributed()}",
                flush=True,
            )
            print(
                f"World size: {world_size}",
                flush=True,
            )
            print(
                f"Task: {args.task}",
                flush=True,
            )
            print(
                f"Per-GPU evaluation batch size: "
                f"{args.batch_size}",
                flush=True,
            )
            print(
                f"Approximate global evaluation batch size: "
                f"{args.batch_size * world_size}",
                flush=True,
            )

        print(
            f"[rank {rank}] local_rank={local_rank}, "
            f"device={device}",
            flush=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_ckpt,
            trust_remote_code=True,
        )
        tokenizer.padding_side = "left"

        if tokenizer.pad_token is None:
            tokenizer.pad_token = (
                tokenizer.eos_token
            )

        # Important:
        # Do not use device_map="auto" under torchrun.
        # Every process loads one complete model on its own GPU.
        model = AutoModelForCausalLM.from_pretrained(
            args.model_ckpt,
            torch_dtype=(
                torch.bfloat16
                if device.type == "cuda"
                else torch.float32
            ),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        model.to(device)
        model.eval()

        print(
            f"[rank {rank}] model loaded on "
            f"{next(model.parameters()).device}",
            flush=True,
        )

        # Every rank loads the same data list, then eval_mean_reward selects:
        # data[rank::world_size]
        data = load_eval_data(
            args.eval_json,
            args.num_samples,
            args.task,
        )

        reward_fn = select_reward_func(
            args.task,
            args.model_ckpt,
        )

        if rank == 0:
            print(
                f"Using reward function: "
                f"{reward_fn.__name__}",
                flush=True,
            )

        local_example_count = len(
            data[rank::world_size]
        )

        print(
            f"[rank {rank}] assigned "
            f"{local_example_count} evaluation examples",
            flush=True,
        )

        theta = get_target_state(model)

        if rank == 0:
            print(
                f"Perturbing {len(theta)} tensors",
                flush=True,
            )
            print(
                f"Direction type: "
                f"{args.direction_type}",
                flush=True,
            )

        # Only rank 0 constructs random/GRPO directions.
        # The identical tensors are broadcast to every GPU.
        directions = build_and_broadcast_directions(
            args,
            model,
            tokenizer,
            data,
            reward_fn,
            theta,
        )

        rows: list[
            dict[str, float | str]
        ] = []

        try:
            for direction_name, direction in directions:
                alpha_iterator = tqdm(
                    grid,
                    desc=direction_name,
                    disable=(rank != 0),
                )

                for alpha in alpha_iterator:
                    coefficient = (
                        args.scale * float(alpha)
                    )

                    perturbed_state = {
                        name: (
                            theta[name]
                            + coefficient
                            * direction[name]
                        )
                        for name in theta
                    }

                    load_target_state(
                        model,
                        perturbed_state,
                    )

                    reward = eval_mean_reward(
                        model,
                        tokenizer,
                        data,
                        reward_fn,
                        args.max_new_tokens,
                        args.task,
                        args.batch_size,
                    )

                    # all_reduce inside eval_mean_reward returns the same
                    # global reward to every rank. Only rank 0 stores it.
                    if rank == 0:
                        rows.append(
                            {
                                "direction": direction_name,
                                "alpha": float(alpha),
                                "perturbation_coefficient": (
                                    coefficient
                                ),
                                "reward": reward,
                            }
                        )

                        print(
                            f"direction={direction_name}, "
                            f"alpha={alpha:.4f}, "
                            f"coefficient="
                            f"{coefficient:.6f}, "
                            f"reward={reward:.4f}",
                            flush=True,
                        )

                    if is_distributed():
                        dist.barrier()

        finally:
            load_target_state(
                model,
                theta,
            )

        if rank == 0:
            result_df = pd.DataFrame(rows)

            csv_path, png_path = save_results(
                result_df,
                args,
            )

            print(
                f"Saved {csv_path} and {png_path}",
                flush=True,
            )

        if is_distributed():
            dist.barrier()

    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()