"""
Unified training loop utilities: gradient accumulation, mixed-precision handling,
checkpoint save/load, and model wrapping.
"""
import os
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from typing import Optional, Dict, Any

from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint


class Accumulator:
    """Unified gradient accumulation with optional GradScaler and scheduler.

    Handles: loss scaling, backward, gradient clipping, optimizer stepping,
    scheduler stepping, and gradient draining. Works correctly with both
    bfloat16 (scaler disabled) and float16 (scaler enabled).

    Usage:
        acc = Accumulator(model, optimizer, scaler, accumulation_steps=8, grad_clip=1.0, scheduler=scheduler)
        for step, batch in enumerate(loader):
            loss = compute_loss(batch)
            acc.backward(loss)
            if step % log_interval == 0:
                current_loss = acc.loss_value
        acc.finalize()  # drain remaining gradients at epoch end
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        accumulation_steps: int = 1,
        grad_clip: float = 1.0,
        scheduler: Optional[LRScheduler] = None,
        max_norm: Optional[float] = None,  # For PPO's per-model clips
    ):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.accumulation_steps = accumulation_steps
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        self._counter = 0
        self._loss_sum = 0.0
        self._last_loss = 0.0
        self.use_scaler = scaler is not None and (getattr(scaler, '_enabled', True))

    def backward(self, loss: torch.Tensor):
        """Backward pass with gradient scaling and periodic stepping."""
        self._last_loss = loss.item()
        scaled = loss / self.accumulation_steps

        if self.use_scaler:
            self.scaler.scale(scaled).backward()
        else:
            scaled.backward()

        self._counter += 1

        if self._counter % self.accumulation_steps == 0:
            self._step()

    def _step(self):
        """Internal: unscale, clip, step optimizer, step scheduler, zero grad."""
        if self.use_scaler:
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.optimizer.zero_grad(set_to_none=True)

    def _step_multi_model(self, models_and_clips):
        """Step for multi-model setups (e.g., PPO with actor + critic).

        Args:
            models_and_clips: List of (model, grad_clip) tuples.
        """
        if self.use_scaler:
            for model, clip in models_and_clips:
                self.scaler.unscale_(self.optimizer)
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            for model, clip_val in models_and_clips:
                if clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.optimizer.zero_grad(set_to_none=True)

    def finalize(self):
        """Drain any remaining accumulated gradients at epoch/loop end."""
        if self._counter % self.accumulation_steps != 0:
            self._step()

    @property
    def loss_value(self) -> float:
        return self._last_loss * self.accumulation_steps


class MultiOptimizerAccumulator:
    """Accumulator for multi-optimizer setups (PPO actor + critic).

    Manages two sets of model/optimizer/scheduler with a single unified step trigger.
    """

    def __init__(
        self,
        models: list,
        optimizers: list,
        accumulation_steps: int = 1,
        grad_clip: float = 1.0,
        schedulers: Optional[list] = None,
    ):
        self.models = models
        self.optimizers = optimizers
        self.accumulation_steps = accumulation_steps
        self.grad_clip = grad_clip
        self.schedulers = schedulers or [None] * len(optimizers)
        self._counter = 0

    def backward(self, loss: torch.Tensor):
        loss = loss / self.accumulation_steps
        loss.backward()
        self._counter += 1
        if self._counter % self.accumulation_steps == 0:
            self._step()

    def _step(self):
        for model, optimizer, scheduler in zip(self.models, self.optimizers, self.schedulers):
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    def finalize(self):
        if self._counter % self.accumulation_steps != 0:
            self._step()


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    iters_per_epoch: int,
    epochs: int,
    accumulation_steps: int = 1,
    learning_rate: float = 1e-5,
    extra_factor: int = 1,
) -> CosineAnnealingLR:
    """Factory: create a CosineAnnealingLR with correct T_max.

    Args:
        extra_factor: Multiplier for extra iterations per step (e.g., PPO minibatch loops).
    """
    total_steps = (iters_per_epoch * epochs * extra_factor) // accumulation_steps
    total_steps = max(total_steps, 1)
    return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=learning_rate / 10)


def wrap_model(
    model: nn.Module,
    use_compile: bool = False,
    local_rank: int = 0,
    rollout_engine=None,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """Unified model wrapping: torch.compile -> DDP -> rollout update.

    Args:
        model: The raw model.
        use_compile: Enable torch.compile.
        local_rank: Local rank for DDP.
        rollout_engine: Optional rollout engine to update after wrapping.
        find_unused_parameters: DDP find_unused_parameters flag.
    """
    if use_compile:
        try:
            model = torch.compile(model)
            Logger("torch.compile enabled")
        except Exception as e:
            Logger(f"torch.compile failed ({e}), continuing without compile")

    if dist.is_initialized():
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=find_unused_parameters,
        )

    if rollout_engine is not None:
        rollout_engine.update_policy(model)

    return model


def save_model_weights(
    model: nn.Module,
    save_dir: str,
    name: str,
    hidden_size: int,
    use_moe: bool = False,
):
    """Save model weights (half precision) to disk."""
    moe_suffix = "_moe" if use_moe else ""
    ckp = f"{save_dir}/{name}_{hidden_size}{moe_suffix}.pth"
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    raw = getattr(raw, "_orig_mod", raw)
    state_dict = {k: v.half().cpu() for k, v in raw.state_dict().items()}
    torch.save(state_dict, ckp)
    return ckp


def get_raw_model(model: nn.Module) -> nn.Module:
    """Unwrap DDP and torch.compile wrappers."""
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(raw, "_orig_mod", raw)
