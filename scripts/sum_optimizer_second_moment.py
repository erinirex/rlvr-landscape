#!/usr/bin/env python3
import argparse
import gc
import json
import math
from pathlib import Path

import torch


# python scripts/sum_optimizer_second_moment.py \
#   --checkpoint-dir "/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs4_prompt8_lr_3.5e-6_max2048/v0-20261119-212529/checkpoint-52/global_step52" \
#   --output second_moment_gs4_lr3.5e-6_step52.json

# python scripts/sum_optimizer_second_moment.py \
#   --checkpoint-dir "/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs16_lr_3e-6_max2048/v0-20261118-231555/checkpoint-52/global_step52" \
#   --output second_moment_gs16_lr3e-6_step52.json


# python scripts/sum_optimizer_second_moment.py \
#   --checkpoint-dir "/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs32_lr_3e-6_max2048/v0-20261118-165927/checkpoint-54/global_step54" \
#   --output second_moment_gs32_lr3e-6_step54.json

# python scripts/sum_optimizer_second_moment.py \
#   --checkpoint-dir "/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs4_prompt8_lr_3.5e-6_max2048/v0-20261119-212529/checkpoint-94/global_step94" \
#   --output second_moment_gs4_lr3.5e-6_step94.json

# python scripts/sum_optimizer_second_moment.py \
#   --checkpoint-dir "/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs16_lr_3e-6_max2048/v0-20261118-231555/checkpoint-94/global_step94" \
#   --output second_moment_gs16_lr3e-6_step94.json

# python scripts/sum_optimizer_second_moment.py \
#   --checkpoint-dir "/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs32_lr_3e-6_max2048/v0-20261118-165927/checkpoint-94/global_step94" \
#   --output second_moment_gs32_lr3e-6_step94.json

# python scripts/sum_optimizer_second_moment.py \
#   --checkpoint-dir "/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs4_prompt8_lr_3.5e-6_max2048/v0-20261119-212529/checkpoint-150/global_step150" \
#   --output second_moment_gs4_lr3.5e-6_step150.json

# python scripts/sum_optimizer_second_moment.py \
#   --checkpoint-dir "/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs16_lr_3e-6_max2048/v0-20261118-231555/checkpoint-150/global_step150" \
#   --output second_moment_gs16_lr3e-6_step150.json

# python scripts/sum_optimizer_second_moment.py \
#   --checkpoint-dir "/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs32_lr_3e-6_max2048/v0-20261118-165927/checkpoint-150/global_step150" \
#   --output second_moment_gs32_lr3e-6_step150.json



def as_number(value):
    if torch.is_tensor(value):
        return value.item()
    return value


def find_adam_state_dict(obj):
    """Locate the nested optimizer dict with state and param_groups."""
    if not isinstance(obj, dict):
        raise ValueError("Invalid optimizer checkpoint structure")

    if "state" in obj and "param_groups" in obj:
        return obj

    nested = obj.get("optimizer_state_dict")
    if isinstance(nested, dict):
        return find_adam_state_dict(nested)

    raise ValueError(
        "Cannot find optimizer state/param_groups. "
        f"Available keys: {list(obj.keys())}"
    )


def sum_sqrt_tensor(tensor, chunk_size=1_000_000):
    """Use float64 accumulation without converting the whole tensor at once."""
    flat = tensor.detach().reshape(-1)
    partials = []

    for start in range(0, flat.numel(), chunk_size):
        chunk = flat[start:start + chunk_size].to(
            device="cpu", dtype=torch.float64
        )

        if not torch.isfinite(chunk).all().item():
            raise ValueError("exp_avg_sq contains NaN or Inf")

        if (chunk < 0).any().item():
            raise ValueError("exp_avg_sq contains negative values")

        partials.append(chunk.sqrt().sum().item())

    return math.fsum(partials)


def process_shard(path):
    # Only load your own trusted training checkpoints.
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    wrapper = checkpoint.get("optimizer_state_dict", checkpoint)
    optimizer = find_adam_state_dict(wrapper)

    raw_parts = []
    corrected_parts = []
    steps = set()
    beta2_values = set()
    used_ids = set()
    num_elements = 0
    num_buffers = 0
    num_missing_states = 0

    for group in optimizer["param_groups"]:
        if "betas" not in group:
            raise ValueError("Missing Adam betas in param_groups")

        beta2 = float(group["betas"][1])
        if not 0 <= beta2 < 1:
            raise ValueError(f"Invalid beta2: {beta2}")

        for param_id in group["params"]:
            if param_id in used_ids:
                raise ValueError(f"Duplicate optimizer parameter ID: {param_id}")
            used_ids.add(param_id)

            state = optimizer["state"].get(param_id)
            if state is None:
                state = optimizer["state"].get(str(param_id))

            if not state or "exp_avg_sq" not in state:
                num_missing_states += 1
                continue

            if "step" not in state:
                raise ValueError(f"Missing step for parameter ID {param_id}")

            step = float(as_number(state["step"]))
            if not math.isfinite(step) or step <= 0:
                raise ValueError(f"Invalid optimizer step: {step}")

            v = state["exp_avg_sq"]
            raw = sum_sqrt_tensor(v)

            # More stable than 1 - beta2**step when beta2 is close to 1.
            correction = (
                1.0 if beta2 == 0
                else -math.expm1(step * math.log(beta2))
            )

            raw_parts.append(raw)
            corrected_parts.append(raw / math.sqrt(correction))
            num_elements += v.numel()
            num_buffers += 1
            steps.add(step)
            beta2_values.add(beta2)

    if num_buffers == 0:
        raise ValueError(f"No exp_avg_sq found in {path}")

    partition_count = wrapper.get("partition_count")
    if isinstance(partition_count, (list, tuple)):
        partition_count = [
            int(as_number(x)) for x in partition_count
        ]
    elif partition_count is not None:
        partition_count = int(as_number(partition_count))

    result = {
        "file": path.name,
        "zero_stage": wrapper.get("zero_stage"),
        "partition_count": partition_count,
        "sum_sqrt_v": math.fsum(raw_parts),
        "sum_sqrt_v_bias_corrected": math.fsum(corrected_parts),
        "num_stored_elements": num_elements,
        "num_second_moment_buffers": num_buffers,
        "num_missing_states": num_missing_states,
        "optimizer_steps": sorted(steps),
        "beta2_values": sorted(beta2_values),
    }

    del optimizer, wrapper, checkpoint
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="The global_step52 directory containing optimizer shards",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path",
    )
    args = parser.parse_args()

    root = args.checkpoint_dir.resolve()
    files = sorted(root.glob("*_optim_states.pt"))

    if not files:
        raise FileNotFoundError(f"No optimizer shards found in {root}")

    # This script targets the single-model-parallel-rank layout shown.
    if any("_mp_rank_00_optim_states.pt" not in f.name for f in files):
        raise ValueError(
            "Multiple or unexpected model-parallel ranks detected; "
            "this layout needs separate handling."
        )

    shards = []
    for path in files:
        print(f"Reading {path.name}", flush=True)
        shard = process_shard(path)

        count = shard["partition_count"]
        counts = count if isinstance(count, list) else [count]
        for value in counts:
            if value is not None and value != len(files):
                raise ValueError(
                    f"Expected {value} partition files, found {len(files)}"
                )

        shards.append(shard)
        print(
            f"  raw={shard['sum_sqrt_v']:.12g}, "
            f"corrected={shard['sum_sqrt_v_bias_corrected']:.12g}, "
            f"steps={shard['optimizer_steps']}",
            flush=True,
        )

    # All shards should describe the same checkpoint state.
    if any(
        shard["optimizer_steps"] != shards[0]["optimizer_steps"]
        or shard["beta2_values"] != shards[0]["beta2_values"]
        for shard in shards[1:]
    ):
        raise ValueError("Optimizer steps or beta2 differ between shards")

    result = {
        "checkpoint_dir": str(root),
        "definition": "sum over stored optimizer elements of sqrt(exp_avg_sq)",
        "num_shards": len(shards),
        "sum_sqrt_v": math.fsum(s["sum_sqrt_v"] for s in shards),
        "sum_sqrt_v_bias_corrected": math.fsum(
            s["sum_sqrt_v_bias_corrected"] for s in shards
        ),
        "num_stored_elements": sum(
            s["num_stored_elements"] for s in shards
        ),
        "num_missing_states": sum(
            s["num_missing_states"] for s in shards
        ),
        "optimizer_steps": shards[0]["optimizer_steps"],
        "beta2_values": shards[0]["beta2_values"],
        "shards": shards,
    }

    print("\n" + json.dumps(result, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()