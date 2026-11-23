"""Overlay displacement landscapes from several eval seeds (same checkpoint/direction).

Reads the per-generation completion_lengths_*.csv files and recomputes, per seed and
coefficient, the same statistics as visualize_reward_landscape_displace_no_accm_error_norm.py:
mean over the [prompt x group] matrix, with SE = std(generation-slot means, ddof=1) / sqrt(G).
"""

# python scripts/plot_displacement_seeds.py --seeds 42 99 1001 1002 --fig_dir /mnt/swordfish-pool2/erinxia/rlvr-landscape/figs_math500_grpo_norm_displacement_train32_lr4e-6_gs4_seed99 

# python scripts/plot_displacement_seeds.py --seeds 42 99 1001 1002 --fig_dir /mnt/swordfish-pool2/erinxia/rlvr-landscape/figs_math500_grpo_norm_displacement_train32_lr4e-6_gs32_seed50
# python scripts/plot_displacement_seeds.py --seeds 42 99 1001 1002 --fig_dir /mnt/swordfish-pool2/erinxia/rlvr-landscape/figs_math500_grpo_norm_displacement_train32_lr4e-6_gs4_seed50

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = Path(
    "/mnt/swordfish-pool2/erinxia/rlvr-landscape/"
    "figs_math500_grpo_norm_displacement_train32_lr4e-6_gs32_seed99"
)

STEM = (
    "qwen3_0.6b_stp0_to40_train_seed_50_displacement_grp64_tmp0.7"
    "_scale1.00_alpha3.0_6.0_npts25_seed{seed}"
)


def summarize(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    for coef, sub in df.groupby("coefficient"):
        mat = sub.pivot(index="prompt_index", columns="group_index", values=value_col).to_numpy(float)
        gen_means = mat.mean(axis=0)
        g = mat.shape[1]
        se = gen_means.std(ddof=1) / np.sqrt(g) if g >= 2 else 0.0
        rows.append({"coefficient": coef, "mean": mat.mean(), "se": se})
    return pd.DataFrame(rows).sort_values("coefficient")


def plot(per_seed, ylabel, title, out_path, relative=False, clip01=False, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    for seed, s in per_seed.items():
        x = s["coefficient"].to_numpy(float)
        y = s["mean"].to_numpy(float)
        se = s["se"].to_numpy(float)
        if relative:
            base = float(s.loc[s["coefficient"] == 0.0, "mean"].iloc[0])
            y, se = y / base, se / abs(base)
        line = ax.plot(x, y, marker="o", linewidth=2, label=f"seed {seed}")[0]
        lo, hi = y - se, y + se
        if clip01:
            lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
        ax.fill_between(x, lo, hi, alpha=0.2, color=line.get_color())

    ax.axvline(0.0, linestyle="--", linewidth=1, color="black")
    if relative:
        ax.axhline(1.0, linestyle=":", linewidth=1, color="gray")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Perturbation coefficient")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.minorticks_on()
    ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.8)
    ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.3)
    ax.legend(title="Eval seed")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fig_dir", type=Path, default=FIG_DIR)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 99, 1001, 1002])
    args = p.parse_args()

    dfs = {
        s: pd.read_csv(args.fig_dir / f"completion_lengths_{STEM.format(seed=s)}.csv")
        for s in args.seeds
    }
    reward = {s: summarize(d, "reward") for s, d in dfs.items()}
    length = {s: summarize(d, "completion_tokens") for s, d in dfs.items()}

    tag = STEM.format(seed="s" + "_".join(map(str, args.seeds)))
    head = "Math500 stp0 to 40 train seed 50 train gs 32"
    plot(reward, "Mean Math500 Reward", f"{head}\nAbsolute Reward Landscape across eval seeds",
         args.fig_dir / f"{tag}_absolute_reward_generation_mean_se.png", clip01=True, ylim=(0.0, 1.5))
    # plot(reward, "Relative Reward: R(coefficient) / R(0)", f"{head}\nRelative Reward Landscape across eval seeds",
    #      args.fig_dir / f"{tag}_relative_reward_generation_mean_se.png", relative=True, ylim=(0.0, 1.5))
    # plot(length, "Mean completion length (tokens)", f"{head}\nCompletion Length across eval seeds",
    #      args.fig_dir / f"{tag}_completion_length_generation_mean_se.png")


if __name__ == "__main__":
    main()
