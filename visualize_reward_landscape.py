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
        f"{args.rl_type.upper()} {args.dataset} step {args.checkpoint_step} reward landscape"
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