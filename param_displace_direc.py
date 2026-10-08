# python param_displace_direc.py

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM

# CKPT_I = "/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_5e-6_max2048/v7-20260903-071819/checkpoint-75"
# CKPT_J = "/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_5e-6_max2048/v7-20260903-071819/checkpoint-100"

# CKPT_I = "/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_3e-6_max2048/v0-20260910-192151/checkpoint-119"
# CKPT_J = "/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train300_lr_3e-6_max2048/v0-20260910-192151/checkpoint-120"

# CKPT_I = "/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train1_lr_5e-6_max2048/v0-20260916-223901/checkpoint-103"
# CKPT_J = "/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train1_lr_5e-6_max2048/v0-20260916-223901/checkpoint-104"

# CKPT_I = "/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train8_lr_5e-6_max2048/v2-20260917-200704/checkpoint-109"
# CKPT_J = "/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train8_lr_5e-6_max2048/v2-20260917-200704/checkpoint-110"

TRAIN_SEED = 42
# CKPT_I = "/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train1_seed50_gs4_lr_5e-6_max2048/v0-20261001-233843/checkpoint-20"
# CKPT_J = "/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train1_seed50_gs4_lr_5e-6_max2048/v0-20261001-233843/checkpoint-40"

CKPT_I = "/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train8_50_80_lr_5e-6_max2048/v1-20260925-122541/checkpoint-10"
CKPT_J = "/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train8_50_80_lr_5e-6_max2048/v1-20260925-122541/checkpoint-50"

# CKPT_I = "/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train1_seed50_gs4_lr_5e-6_max2048/v0-20261002-001611/checkpoint-20"
# CKPT_J = "/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train1_seed50_gs4_lr_5e-6_max2048/v0-20261002-001611/checkpoint-40"

# CKPT_I="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train1_seed42_gs4_lr_5e-6_max2048/v0-20261001-233655/checkpoint-10"
# CKPT_J="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train1_seed42_gs4_lr_5e-6_max2048/v0-20261001-233655/checkpoint-80"

# CKPT_I="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train1_seed42_gs16_lr_5e-6_max2048/v0-20261002-001611/checkpoint-10"
# CKPT_J="/mnt/swordfish-pool2/erinxia/rlvr-landscape/output_qwen3_0.6b_nonthink_grpo_math500_train1_seed42_gs16_lr_5e-6_max2048/v0-20261002-001611/checkpoint-80"
# STEP_I = 75
# STEP_J = 100

# STEP_I = 109
# STEP_J = 110

# STEP_I = 40
# STEP_J = 80

# STEP_I = 20
# STEP_J = 40

STEP_I = 10
STEP_J = 50

OUTPUT = (
    f"/mnt/swordfish-pool2/erinxia/rlvr-landscape/param_displace_train8_gs4/"
    f"displacement_train8_gs4_seed_{TRAIN_SEED}_lr_5e-6_{STEP_I}_to_{STEP_J}.pt"
)

if not Path(OUTPUT).parent.exists():
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

def load_state_dict(checkpoint: str) -> dict[str, torch.Tensor]:
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    # 用 state_dict 而不是 named_parameters：
    # 它也会包含模型 buffer；对浮点 tensor 做位移计算。
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if tensor.is_floating_point()
    }

    del model
    return state


print(f"Loading checkpoint {STEP_I}...")
state_i = load_state_dict(CKPT_I)

print(f"Loading checkpoint {STEP_J}...")
state_j = load_state_dict(CKPT_J)

if state_i.keys() != state_j.keys():
    missing_in_j = state_i.keys() - state_j.keys()
    missing_in_i = state_j.keys() - state_i.keys()
    raise ValueError(
        f"Checkpoint structures differ.\n"
        f"Only in checkpoint {STEP_I}: {missing_in_j}\n"
        f"Only in checkpoint {STEP_J}: {missing_in_i}"
    )

displacement = {}

for name in state_i:
    if state_i[name].shape != state_j[name].shape:
        raise ValueError(
            f"Shape mismatch for {name}: "
            f"{state_i[name].shape} vs {state_j[name].shape}"
        )

    # 不做 Normalize；直接保留真实参数差。
    displacement[name] = state_j[name] - state_i[name]

# 验证：checkpoint i + displacement 是否等于 checkpoint j
max_error = max(
    (state_i[name] + displacement[name] - state_j[name]).abs().max().item()
    for name in displacement
)
print(f"Maximum reconstruction error: {max_error:.8e}")

torch.save(displacement, OUTPUT)
print(f"Saved displacement to {OUTPUT}")