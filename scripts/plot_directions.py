#!/usr/bin/env python3
"""Plot completed per-direction CSV files without loading vLLM or torch."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--expected", type=int)
p.add_argument("--seeds", type=int, nargs="+", help="Plot only these direction seeds")
a = p.parse_args()
frames = []
reference_grid = None
reference_config = None
for f in sorted(a.root.glob("dirseed_*/landscape.csv")):
    folder_seed = int(f.parent.name.removeprefix("dirseed_"))
    if a.seeds is not None and folder_seed not in a.seeds:
        continue
    if not (f.parent / "DONE").exists():
        continue
    config = json.loads((f.parent / "config.json").read_text(encoding="utf-8"))
    if reference_config is None:
        reference_config = config
    elif config != reference_config:
        raise ValueError(f"Different experiment configuration: {f}")
    d = pd.read_csv(f).sort_values("perturbation_coefficient")
    seed = int(f.parent.name.removeprefix("dirseed_"))
    if d.empty or d.direction_seed.nunique() != 1 or int(d.direction_seed.iloc[0]) != seed:
        raise ValueError(f"Invalid seed data: {f}")
    if d.perturbation_coefficient.duplicated().any():
        raise ValueError(f"Duplicate points: {f}")
    grid = d.perturbation_coefficient.to_numpy(float)
    if reference_grid is None:
        reference_grid = grid
    elif len(grid) != len(reference_grid) or not np.allclose(grid, reference_grid):
        raise ValueError(f"Different coefficient grid: {f}")
    if not np.isfinite(d[["reward", "std_error_generation_mean"]].to_numpy(float)).all():
        raise ValueError(f"Non-finite result: {f}")
    d["direction"] = f"seed_{seed}"
    frames.append(d)
if not frames:
    raise SystemExit("No completed directions")
if a.expected is not None and len(frames) != a.expected:
    raise SystemExit(f"Expected {a.expected} directions, found {len(frames)}; set --expected to total completed directions")
df = pd.concat(frames, ignore_index=True)
df.to_csv(a.root / "all_directions.csv", index=False)
for relative in (False, True):
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, d in df.groupby("direction", sort=False):
        x = d.perturbation_coefficient.to_numpy(float)
        y = d.reward.to_numpy(float)
        e = d.std_error_generation_mean.to_numpy(float)
        if relative:
            zeros = np.flatnonzero(np.isclose(x, 0, rtol=0, atol=1e-12))
            if len(zeros) != 1 or y[zeros[0]] == 0:
                raise ValueError(f"{name}: relative plot requires one nonzero-reward baseline")
            baseline = y[zeros[0]]
            y, e = y / baseline, e / abs(baseline)
        line, = ax.plot(x, y, "o-", label=name)
        lo, hi = y-e, y+e
        if not relative:
            lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
        ax.fill_between(x, lo, hi, color=line.get_color(), alpha=.12)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Perturbation coefficient (scale * alpha)")
    ax.set_ylabel("R(coefficient) / R(0)" if relative else "Mean reward")
    if not relative:
        ax.set_ylim(0, 1.1)
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.root / ("all_relative.png" if relative else "all_absolute.png"), dpi=300)
    plt.close(fig)
print(f"Merged {len(frames)} directions into {a.root}")
