#!/usr/bin/env python3
# 将多个 mean_direction.csv 画在同一张图上
# python scripts/plot_mean_direction_compare.py \
#   --csv <dir1>/mean_direction.csv <dir2>/mean_direction.csv \
#   --labels gs4_ckpt170 gs32_ckpt40 --out compare.png


# python scripts/plot_mean_direction_compare.py \
#   --csv landscape_math500_grpo_random_newdata_train32_global_norm/scale60_alpha1_train_gs4_train_seed50_train_prompt32_lr4e-6_data32_group64_ckpt170_max_new2048_pts15_seed42/mean_direction.csv landscape_math500_grpo_random_newdata_train32_global_norm/scale60_alpha1_train_gs16_train_seed99_train_prompt32_lr4e-6_data32_group64_ckpt46_max_new2048_pts15_seed42/mean_direction.csv landscape_math500_grpo_random_newdata_train32_global_norm/scale60_alpha1_train_gs32_train_seed50_train_prompt32_lr4e-6_data32_group64_ckpt40_max_new2048_pts15_seed42/mean_direction.csv \
#   --labels gs4_ckpt170 gs16_ckpt46 gs32_ckpt40 \
#   --out mean_direction_compare_gs4_gs16_gs32.png


# python scripts/plot_mean_direction_compare.py \
#   --csv landscape_math500_grpo_random_newdata_train32_global_norm/scale60_alpha1_train_gs4_train_seed50_train_prompt32_lr5e-6_data32_group64_ckpt74_max_new2048_pts15_seed42/mean_direction.csv landscape_math500_grpo_random_newdata_train32_global_norm/scale60_alpha1_train_gs16_train_seed50_train_prompt32_lr4e-6_data32_group64_ckpt74_max_new2048_pts15_seed42/mean_direction.csv landscape_math500_grpo_random_newdata_train32_global_norm/scale60_alpha1_train_gs32_train_seed50_train_prompt32_lr4e-6_data32_group64_ckpt74_max_new2048_pts15_seed42/mean_direction.csv \
#   --labels gs4_ckpt74 gs16_ckpt74 gs32_ckpt74 \
#   --out /mnt/swordfish-pool2/erinxia/rlvr-landscape/landscape_math500_grpo_random_newdata_train32_global_norm/mean_direction_compare_gs4_gs16_gs32_step74.png



import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=Path, nargs="+", required=True)
parser.add_argument("--labels", nargs="+", default=None)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

if args.labels and len(args.labels) != len(args.csv):
    raise ValueError("--labels 数量需与 --csv 一致")


def default_label(path):
    name = path.parent.name
    gs = re.search(r"_gs(\d+)_", name)
    ckpt = re.search(r"_ckpt(\d+)_", name)
    parts = []
    if gs:
        parts.append(f"gs{gs.group(1)}")
    if ckpt:
        parts.append(f"ckpt{ckpt.group(1)}")
    return ", ".join(parts) or name


fig, ax = plt.subplots(figsize=(8, 5))

for i, path in enumerate(args.csv):
    summary = pd.read_csv(path).sort_values("perturbation_coefficient")
    x = summary["perturbation_coefficient"].to_numpy(float)
    y = summary["mean_reward"].to_numpy(float)
    std = summary["direction_std"].to_numpy(float)
    label = args.labels[i] if args.labels else default_label(path)

    line, = ax.plot(x, y, "o-", linewidth=2, label=f"{label} (mean)")
    ax.fill_between(
        x,
        np.clip(y - std, 0, 1),
        np.clip(y + std, 0, 1),
        color=line.get_color(),
        alpha=0.2,
        label=f"{label} (±1 SD)",
    )

ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Perturbation coefficient")
ax.set_ylabel("Mean reward")
ax.set_title("Reward landscape averaged across directions")
ax.set_ylim(0, 1.1)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=300)
plt.close(fig)
print(f"保存到：{args.out}")
