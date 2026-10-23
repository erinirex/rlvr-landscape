"""Data-parallel GRPO per-question pass-rate logger (single/multiple GPUs).

Load with --external_plugins ./custom_save.py ./sample_pass_rate.py
Requires one synchronous binary reward, num_iterations=1, and
steps_per_generation == gradient_accumulation_steps. Original reward is unchanged.
All ranks collect LOCAL rewards; at optimizer-step end all ranks gather, and
only global rank 0 writes. A sample_id identifies a QUESTION, not a completion.
Sequence/tensor-parallel training and dynamic/async rollouts are not supported.
"""
import inspect
import json
import uuid
from functools import wraps
from pathlib import Path
import torch.distributed as dist

from transformers import TrainerCallback
from swift.rlhf_trainers import GRPOTrainer


class SamplePassRate(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.pending = {}
        self.source_step = None
        self.error = None
        self.run_id = uuid.uuid4().hex
        self.path = Path(trainer.args.output_dir) / 'train_sample_pass_rate.jsonl'

    def collect(self, rewards, columns):
        # Defer logger validation failures until every rank reaches the collective.
        # Raising on one rank during reward computation can strand the others.
        if not self.trainer.model.training or self.error is not None:
            return
        try:
            self._collect(rewards, columns)
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'

    def _collect(self, rewards, columns):
        trainer = self.trainer
        if not trainer.model.training:
            return
        before = int(trainer.state.global_step)
        if self.source_step is not None and self.source_step != before:
            raise RuntimeError('Pass-rate buffer crossed an optimizer step.')
        self.source_step = before
        n = len(rewards)
        ids = columns.get('unique_id')
        if ids is None:
            ids = columns.get('id')
        if ids is None or isinstance(ids, (str, bytes)) or len(ids) != n:
            raise RuntimeError(
                'Missing per-completion unique_id/id. Use --remove_unused_columns false '
                'and retain those columns in your dataset preprocessor.')
        local_ids = columns.get('id')
        if local_ids is not None and (isinstance(local_ids, (str, bytes)) or len(local_ids) != n):
            raise RuntimeError('The id column is not aligned with rewards.')
        for i, reward in enumerate(rewards):
            reward = float(reward)
            if reward not in (0.0, 1.0):
                raise RuntimeError(f'Expected binary accuracy reward; got {reward!r}.')
            sample_id = ids[i]
            if sample_id is None:
                raise RuntimeError('A sample has a null ID.')
            key = json.dumps(sample_id, ensure_ascii=False)
            entry = self.pending.setdefault(key, {
                'sample_id': sample_id,
                'id': local_ids[i] if local_ids is not None else None,
                'rewards': [],
            })
            entry['rewards'].append(int(reward))

    def on_step_end(self, args, state, control, **kwargs):
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        payload = {
            'step': int(state.global_step), 'source_step': self.source_step,
            'entries': list(self.pending.values()), 'error': self.error,
        }
        if distributed:
            payloads = [None] * world_size
            dist.all_gather_object(payloads, payload)
        else:
            payloads = [payload]
        # Every rank validates the same data before rank 0 performs any I/O.
        merged = {}
        for source_rank, item in enumerate(payloads):
            if item['error']:
                raise RuntimeError(f"Pass-rate logger rank {source_rank}: {item['error']}")
            if item['step'] != int(state.global_step):
                raise RuntimeError('Ranks disagree on global_step.')
            if item['entries'] and item['source_step'] != int(state.global_step) - 1:
                raise RuntimeError('Unexpected global_step timing; refusing to mislabel samples.')
            for entry in item['entries']:
                key = json.dumps(entry['sample_id'], ensure_ascii=False)
                target = merged.setdefault(key, {
                    'sample_id': entry['sample_id'], 'id': entry['id'], 'rewards': [],
                })
                if target['id'] != entry['id']:
                    raise RuntimeError('Conflicting id values for the same unique_id across ranks.')
                target['rewards'].extend(entry['rewards'])
        if not merged:
            raise RuntimeError('No training rewards captured on any rank for this optimizer step.')
        rows = []
        completion_count = sum(len(entry['rewards']) for entry in merged.values())
        expected_count = args.per_device_train_batch_size * world_size * args.steps_per_generation
        if completion_count != expected_count:
            raise RuntimeError(
                f'Captured {completion_count} rewards; expected {expected_count}. '
                'Possible duplicate/global rewards, padding, or unsupported rollout configuration.')
        for entry in merged.values():
            rewards = entry['rewards']
            if len(rewards) % args.num_generations:
                raise RuntimeError('Incomplete generation group for a question; check dataset IDs.')
            rows.append({
                'run_id': self.run_id,
                'step': int(state.global_step),
                'policy_step': int(state.global_step) - 1,
                'world_size': world_size,
                'global_num_questions': len(merged),
                'global_num_completions': completion_count,
                **entry,
                'num_generations': len(rewards),
                'num_correct': sum(rewards),
                'pass_rate': sum(rewards) / len(rewards),
            })
        write_error = [None]
        if rank == 0:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open('a', encoding='utf-8') as f:
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False) + '\n')
            except Exception as exc:
                write_error[0] = f'{type(exc).__name__}: {exc}'
        if distributed:
            dist.broadcast_object_list(write_error, src=0)
        if write_error[0]:
            raise RuntimeError(f'Cannot write pass-rate log: {write_error[0]}')
        self.pending.clear()
        self.source_step = None
        self.error = None
        return control


def install():
    if getattr(GRPOTrainer, '_sample_pass_rate_installed', False):
        return
    original_init = GRPOTrainer.__init__

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        cfg = self.args
        if self.accelerator.num_processes > 1 and not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError('Multi-process logging requires initialized torch.distributed.')
        if any((getattr(cfg, name, 1) or 1) != 1 for name in (
            'sequence_parallel_size', 'tensor_parallel_size'
        )):
            raise RuntimeError('This plugin supports data-parallel training only (including ZeRO).')
        if getattr(cfg, 'num_iterations', None) != 1:
            raise RuntimeError('Use --num_iterations 1 for per-step pass-rate logging.')
        if getattr(cfg, 'steps_per_generation', None) != cfg.gradient_accumulation_steps:
            raise RuntimeError('Set --steps_per_generation equal to --gradient_accumulation_steps.')
        if any(getattr(cfg, name, False) for name in (
            'async_generate', 'dynamic_sample', 'dynamic_num_samples', 'use_gym_env'
        )):
            raise RuntimeError('Async generation/dynamic sampling/gym are not supported by this plugin.')
        if len(self.reward_funcs) != 1:
            raise RuntimeError('Keep only your binary --reward_funcs math500_acc.')
        original_reward = self.reward_funcs[0]
        if inspect.iscoroutinefunction(original_reward) or inspect.iscoroutinefunction(
            getattr(original_reward, '__call__', None)
        ):
            raise RuntimeError('An asynchronous reward is not supported.')
        recorder = SamplePassRate(self)

        def logged_reward(completions, **columns):
            rewards = original_reward(completions, **columns)
            if len(rewards) != len(completions):
                if self.model.training:
                    recorder.error = 'Reward count does not match completion count.'
            else:
                recorder.collect(rewards, columns)
            return rewards

        logged_reward.__name__ = getattr(original_reward, '__name__', type(original_reward).__name__)
        funcs = list(self.reward_funcs)
        funcs[0] = logged_reward
        self.reward_funcs = funcs
        self.add_callback(recorder)
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f'[sample_pass_rate] Writing to {recorder.path}', flush=True)

    GRPOTrainer.__init__ = patched_init
    GRPOTrainer._sample_pass_rate_installed = True


install()
