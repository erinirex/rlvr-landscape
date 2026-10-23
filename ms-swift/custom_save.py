from swift.callbacks import TrainerCallback, callbacks_map


class CustomSaveCallback(TrainerCallback):
    def __init__(self):
        self.save_steps = {
            5, 10, 20, 30, 50, 51, 75, 80, 90, 100,
            110, 120, 140, 150, 169, 180, 190, 200, 220, 250
        }
        self.first_eq_one_step = None
        self.first_gt_099_step = None

    def on_step_end(self, args, state, control, **kwargs):
        control.should_save = state.global_step in self.save_steps
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control

        reward = logs.get("reward")
        if reward is None:
            reward = logs.get("train/reward")
        if reward is None:
            return control

        reward = float(reward)
        step = state.global_step
        reasons = []

        if self.first_eq_one_step is None and reward == 1.0:
            self.first_eq_one_step = step
            reasons.append("首次 reward == 1")

        if self.first_gt_099_step is None and reward > 0.99:
            self.first_gt_099_step = step
            reasons.append("首次 reward > 0.99")

        if reasons:
            control.should_save = True

            if state.is_world_process_zero:
                print(
                    f"[CustomSaveCallback] {', '.join(reasons)}，"
                    f"reward={reward}，保存 checkpoint-{step}"
                )

        return control


callbacks_map["custom_save"] = CustomSaveCallback