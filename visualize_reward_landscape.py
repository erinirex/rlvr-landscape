#!/usr/bin/env python3
"""Evaluate a model's mean reward along random parameter-space directions."""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Callable
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


sys.path.append(
    "/mnt/sj/home/yichen/landscape_rlvr/"
    "train_data/chess/chess rl/miles"
)
sys.path.append(
    "/mnt/sj/home/yichen/landscape_rlvr/"
    "train_data/chess/chess rl/chess-rl-miles"
)

from chess_rl_miles.moves import (
    extract_first_move,
    extract_move_after_thinking,
    parse_ground_truth,
    safe_move_to_uci,
)

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


def normalize_number(text: str) -> str:
    return text.replace(",", "").strip()


def extract_final_answer(text: str) -> str:
    boxed_matches = re.findall(
        r"\\boxed\s*\{\s*([^{}]+?)\s*\}",
        text,
    )
    if boxed_matches:
        return normalize_number(boxed_matches[-1])

    final_answer_match = re.search(
        r"Final\s+Answer\s*:?\s*(.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if final_answer_match:
        final_section = final_answer_match.group(1)
        numbers = re.findall(
            r"-?\d+(?:,\d{3})*(?:\.\d+)?",
            final_section,
        )
        if numbers:
            return normalize_number(numbers[-1])

    return ""


def extract_ground_truth(answer: str) -> str:
    answer = str(answer).strip()

    if re.fullmatch(r"-?\d+(?:,\d{3})*(?:\.\d+)?", answer):
        return normalize_number(answer)

    if "####" in answer:
        final_part = answer.rsplit("####", 1)[-1]
        numbers = re.findall(
            r"-?\d+(?:,\d{3})*(?:\.\d+)?",
            final_part,
        )
        if numbers:
            return normalize_number(numbers[-1])

    extracted = extract_final_answer(answer)
    if extracted:
        return extracted

    return ""


def gsm8k_reward_func(
    completion: str,
    answer: str,
    **_: Any,
) -> float:
    prediction = extract_final_answer(completion)
    target = extract_ground_truth(answer)

    return float(
        bool(prediction)
        and bool(target)
        and prediction == target
    )


def get_fen(metadata: dict[str, Any], extra_info: dict[str, Any]) -> str:
    for source in (metadata, extra_info):
        fen = source.get("FEN") or source.get("fen")
        if fen:
            return str(fen)
    return ""


def extract_raw_move(text: str) -> str:
    move, follows_format = extract_move_after_thinking(
        text, strict_single_close=True
    )
    if move is None and not follows_format:
        move = extract_first_move(text)
    if move is None and "</T>" in text:
        suffix = text.split("</T>", 1)[1].strip()
        move = suffix.split()[0] if suffix else None
    return str(move).strip("`*_.,;: ") if move else ""


def move_to_uci(move_text: str, fen: str = "") -> str:
    if not move_text:
        return ""

    coordinate_move = re.search(
        r"\b[a-h][1-8][a-h][1-8][qrbn]?\b", move_text.lower()
    )
    if coordinate_move:
        return coordinate_move.group(0)

    try:
        uci = safe_move_to_uci(move_text)
        if uci:
            return uci.lower()
    except Exception:
        pass

    if fen:
        try:
            import chess

            return chess.Board(fen).parse_san(move_text).uci().lower()
        except Exception:
            pass
    return ""


def chess_reward_func(
    completion: str,
    answer: str,
    metadata: dict[str, Any] | None = None,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> float:
    metadata = metadata or {}
    extra_info = extra_info or {}
    target_moves = {
        str(move).strip().lower()
        for move in parse_ground_truth(answer, metadata)
        if str(move).strip()
    }
    prediction = move_to_uci(
        extract_raw_move(completion), get_fen(metadata, extra_info)
    )
    return float(bool(prediction) and prediction in target_moves)


def select_reward_func(task: str, checkpoint_path: str) -> RewardFn:
    if task == "auto":
        task = "chess" if "chess" in checkpoint_path.lower() else "gsm8k"
    return chess_reward_func if task == "chess" else gsm8k_reward_func


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

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

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

            if rank == 0 and local_reward_count < 2:
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
                    raw_move = extract_raw_move(completion)
                    predicted_uci = move_to_uci(raw_move, fen)

                    target_moves = {
                        str(move).strip().lower()
                        for move in parse_ground_truth(
                            example["answer"],
                            metadata,
                        )
                        if str(move).strip()
                    }

                    print(f"Contains </T>: {'</T>' in completion}")
                    print(f"Raw extracted move: {raw_move!r}")
                    print(f"Predicted UCI: {predicted_uci!r}")
                    print(f"Target moves: {sorted(target_moves)}")
                    print(f"Ground truth: {example['answer']!r}")
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
        f"num{args.num_samples}_ckpt{checkpoint_step}_"
        f"model_{args.model_name}_max_new{args.max_new_tokens}_bs{args.batch_size}_"
        f"seed{args.seed}_pts{args.num_points}_{args.num_directions}dirs"
    )


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
    plt.xlabel("Perturbation coefficient")
    plt.ylabel("Mean reward")
    plt.title(
        f"{args.rl_type.upper()} {args.dataset} reward landscape"
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
    if args.num_points < 1 or args.num_directions < 1 or args.num_samples < 1:
        raise ValueError("num-points, num-directions, and num-samples must be positive")

    set_seed(args.seed)
    grid = np.linspace(-args.alpha_range, args.alpha_range, args.num_points)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_ckpt, trust_remote_code=True
    )
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_ckpt,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    data = load_eval_data(args.eval_json, args.num_samples, args.task)
    reward_fn = select_reward_func(args.task, args.model_ckpt)
    print(f"Using reward function: {reward_fn.__name__}")

    theta = get_target_state(model)
    print(f"Perturbing {len(theta)} tensors")
    directions = [random_direction_like(theta) for _ in range(args.num_directions)]
    rows: list[dict[str, float | str]] = []

    try:
        for direction_index, direction in enumerate(directions, start=1):
            direction_name = f"direction_{direction_index}"
            for alpha in tqdm(grid, desc=direction_name):
                coefficient = args.scale * float(alpha)
                load_target_state(
                    model,
                    {
                        name: theta[name] + coefficient * direction[name]
                        for name in theta
                    },
                )
                reward = eval_mean_reward(
                    model,
                    tokenizer,
                    data,
                    reward_fn,
                    args.max_new_tokens,
                    args.task,
                    args.batch_size
                )
                rows.append(
                    {
                        "direction": direction_name,
                        "alpha": float(alpha),
                        "perturbation_coefficient": coefficient,
                        "reward": reward,
                    }
                )
                print(
                    f"direction={direction_name}, alpha={alpha:.4f}, "
                    f"coefficient={coefficient:.6f}, reward={reward:.4f}",
                    flush=True,
                )
    finally:
        load_target_state(model, theta)

    csv_path, png_path = save_results(pd.DataFrame(rows), args)
    print(f"Saved {csv_path} and {png_path}")


if __name__ == "__main__":
    main()