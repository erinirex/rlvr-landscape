import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# python scripts/plot_train_reward_curve.py


# 修改为你实际的实验目录
# runs = {
#     "lr=2e-6": "/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs4_prompt8_lr_2e-6_max2048/v0-20261120-202511",
#     "lr=2.5e-6": "/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs4_prompt8_lr_2.5e-6_max2048/v0-20261120-233923",
#     "lr=3e-6": "/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs4_prompt8_lr_3e-6_max2048/v0-20261119-212603",
#     "lr=3.5e-6": "/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs4_prompt8_lr_3.5e-6_max2048/v0-20261119-212529",
#     "lr=4e-6": "/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs4_prompt8_lr_4e-6_max2048/v0-20261121-123430",
#     "lr=4.7e-6": "/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs4_prompt8_lr_4.7e-6_max2048/v0-20261118-175143",
# }

gs=32
fig_name="train_reward_comparison_gs32"

runs = {
    "lr=2e-6": "/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs32_lr_2e-6_max2048/v0-20261122-002714",
    "lr=2.5e-6": "/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs32_lr_2.5e-6_max2048/v0-20261121-171458",
    "lr=3e-6": "/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs32_lr_3e-6_max2048/v0-20261118-165927",
    "lr=4e-6": "/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train32new_seed50_gs32_lr_4e-6_max2048/v2-20261114-143227",
}

fig, ax = plt.subplots(figsize=(9, 5))
plotted = 0

for label, run_dir in runs.items():
    # 汇总各 checkpoint 的日志；同一个 step 只保留一个值
    points = {}
    files = list(Path(run_dir).rglob("trainer_state.json"))

    # 后写入的文件覆盖相同步数的旧记录
    files.sort(key=lambda p: p.stat().st_mtime)

    for path in files:
        with path.open(encoding="utf-8") as f:
            state = json.load(f)

        for row in state.get("log_history", []):
            step = row.get("step")
            reward = row.get("eval_reward", row.get("eval/reward"))

            if step is not None and reward is not None:
                points[int(step)] = float(reward)

    if not points:
        print(f"[跳过] {label}: 没找到 eval reward，检查路径和指标名")
        continue

    steps = sorted(points)
    rewards = [points[s] for s in steps]

    ax.plot(steps, rewards, label=label, linewidth=1.6)
    plotted += 1
    print(f"{label}: {len(steps)} 个评估点，最后一步 {steps[-1]}")

if plotted == 0:
    raise RuntimeError("没有读到评估数据，请检查 trainer_state.json 的内容")

ax.set_xlabel("Training step")
ax.set_ylabel("Train reward")
ax.set_title(f"GS={gs}: Train reward across learning rates")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fname=f"{fig_name}.png"
fig.savefig(fname, dpi=300)
# fig.savefig("train_reward_comparison.pdf")
print(f"已保存 {fname}")