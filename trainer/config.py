"""
Training configuration system with YAML defaults + CLI override.

Loads built-in defaults from ``config_defaults.yaml``, optionally merges a
user-supplied YAML file, then overlays CLI arguments.  Returns a frozen
``TrainingConfig`` dataclass.

Usage::

    from trainer.config import load_config
    config = load_config("full_sft", sys.argv[1:])
    # config.learning_rate, config.batch_size, ...
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field, fields, MISSING
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

# ---------------------------------------------------------------------------
# Lazy YAML import (only needed when a user YAML is supplied)
# ---------------------------------------------------------------------------
try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DEFAULTS_YAML = Path(__file__).resolve().parent / "config_defaults.yaml"


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """All MiniMind training hyper-parameters in one place."""

    # task identity
    task: str = "full_sft"

    # model
    hidden_size: int = 768
    num_hidden_layers: int = 8
    use_moe: bool = False
    max_seq_len: int = 768
    max_gen_len: int = 1024
    max_total_len: int = 2500

    # data
    data_path: List[str] = field(default_factory=lambda: ["../dataset/sft_t2t_mini.jsonl"])
    from_weight: str = "full_sft"
    save_weight: str = "full_sft"

    # training
    batch_size: int = 32
    learning_rate: float = 1e-5
    epochs: int = 3
    accumulation_steps: int = 1
    grad_clip: float = 1.0
    dtype: str = "bfloat16"
    num_workers: int = 8

    # scheduler
    scheduler_eta_min_factor: float = 0.1

    # logging & saving
    log_interval: int = 100
    save_interval: int = 100
    save_dir: str = "../out"
    checkpoint_dir: str = "../checkpoints"

    # device
    device: str = ""

    # resume
    from_resume: bool = False
    use_compile: bool = False
    debug_mode: bool = False
    debug_interval: int = 20

    # wandb / swanlab
    use_wandb: bool = False
    wandb_project: str = "MiniMind"

    # ----------- RL-specific -----------
    num_generations: int = 6
    beta: float = 0.1
    loss_type: str = "cispo"
    epsilon: float = 0.2
    epsilon_high: float = 5.0
    thinking_ratio: float = 0.9

    # rollout engine
    rollout_engine: str = "torch"
    sglang_base_url: str = "http://localhost:8998"
    sglang_model_path: str = "../model"
    sglang_shared_path: str = "./sglang_ckpt"

    # reward model
    reward_model_path: str = "../../internlm2-1_8b-reward"

    # ----------- distillation-specific -----------
    student_hidden_size: int = 768
    student_num_layers: int = 8
    teacher_hidden_size: int = 768
    teacher_num_layers: int = 8
    student_use_moe: bool = False
    teacher_use_moe: bool = True
    from_student_weight: str = "full_sft"
    from_teacher_weight: str = "full_sft"
    alpha: float = 0.5
    temperature: float = 1.5

    # ----------- lora-specific -----------
    lora_name: str = "lora_medical"

    # ----------- checkpoint -----------
    checkpoint_max_keep: int = 3
    checkpoint_format: str = "torch"

    def __post_init__(self) -> None:
        pass

    def auto_device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


# ==========================================================================
# Helpers
# ==========================================================================

def _load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required for YAML config support.  pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dict_to_args(d: Dict[str, Any]) -> argparse.Namespace:
    """Recursively convert a flat-or-nested dict into an argparse.Namespace."""
    ns = argparse.Namespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, _dict_to_args(v))
        else:
            setattr(ns, k, v)
    return ns


def _merge_namespace(base: argparse.Namespace, override: argparse.Namespace) -> argparse.Namespace:
    """Override keys in *base* with non-None values from *override*."""
    result = argparse.Namespace(**vars(base))
    for k, v in vars(override).items():
        if v is not None and not (isinstance(v, list) and v == []):
            setattr(result, k, v)
    return result


def _namespace_to_dataclass(ns: argparse.Namespace, dc: Type[TrainingConfig]) -> TrainingConfig:
    """Map an argparse.Namespace to a TrainingConfig dataclass instance.

    Only passes keys that exist as fields on the dataclass (ignoring extras).
    """
    field_names = {f.name for f in fields(dc)}
    kwargs = {k: v for k, v in vars(ns).items() if k in field_names and v is not None}
    return dc(**kwargs)


# ==========================================================================
# Main API
# ==========================================================================

def load_config(
    defaults: str = "full_sft",
    cli_args: Optional[List[str]] = None,
    user_yaml: Optional[str] = None,
) -> TrainingConfig:
    """Load training config from defaults + optional YAML + CLI args.

    Parameters
    ----------
    defaults:
        Task key in ``config_defaults.yaml`` (e.g. ``"pretrain"``,
        ``"full_sft"``, ``"dpo"``, …).
    cli_args:
        CLI argument list, typically ``sys.argv[1:]``.  ``None`` means
        no CLI overrides.
    user_yaml:
        Path to a user-supplied YAML config file that overrides the
        built-in defaults.

    Returns
    -------
    TrainingConfig
    """
    # 1. built-in defaults from YAML
    builtin = _load_yaml(_DEFAULTS_YAML)
    task_defaults = builtin.get(defaults, {})
    ns = _dict_to_args(task_defaults)

    # 2. user YAML overlay (optional)
    if user_yaml:
        user_ns = _dict_to_args(_load_yaml(user_yaml))
        ns = _merge_namespace(ns, user_ns)

    # 3. CLI args overlay
    if cli_args:
        parser = _build_arg_parser(defaults)
        cli_ns, _ = parser.parse_known_args(cli_args)
        ns = _merge_namespace(ns, cli_ns)

    return _namespace_to_dataclass(ns, TrainingConfig)


# ==========================================================================
# Shared argparse builder (keeps backward compat with existing scripts)
# ==========================================================================

def _build_arg_parser(task: str) -> argparse.ArgumentParser:
    """Build an argparse.ArgumentParser matching the flags used across all
    training scripts.  ``task`` selects a specific preset."""
    p = argparse.ArgumentParser(description=f"MiniMind {task} training")

    # model
    p.add_argument("--hidden_size", type=int, default=None)
    p.add_argument("--num_hidden_layers", type=int, default=None)
    p.add_argument("--use_moe", type=int, choices=[0, 1], default=None)
    p.add_argument("--max_seq_len", type=int, default=None)
    p.add_argument("--max_gen_len", type=int, default=None)
    p.add_argument("--max_total_len", type=int, default=None)

    # data
    p.add_argument("--data_path", type=str, nargs="+", default=None)
    p.add_argument("--from_weight", type=str, default=None)
    p.add_argument("--save_weight", type=str, default=None)

    # training
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--accumulation_steps", type=int, default=None)
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--dtype", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=None)

    # logging & saving
    p.add_argument("--log_interval", type=int, default=None)
    p.add_argument("--save_interval", type=int, default=None)
    p.add_argument("--save_dir", type=str, default=None)

    # device
    p.add_argument("--device", type=str, default=None)

    # resume & compile
    p.add_argument("--from_resume", type=int, choices=[0, 1], default=None)
    p.add_argument("--use_compile", type=int, choices=[0, 1], default=None)
    p.add_argument("--debug_mode", action="store_true", default=None)
    p.add_argument("--debug_interval", type=int, default=None)

    # wandb
    p.add_argument("--use_wandb", action="store_true", default=None)
    p.add_argument("--wandb_project", type=str, default=None)

    # RL
    p.add_argument("--num_generations", type=int, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--loss_type", type=str, default=None)
    p.add_argument("--epsilon", type=float, default=None)
    p.add_argument("--epsilon_high", type=float, default=None)
    p.add_argument("--thinking_ratio", type=float, default=None)
    p.add_argument("--rollout_engine", type=str, default=None)
    p.add_argument("--sglang_base_url", type=str, default=None)
    p.add_argument("--sglang_model_path", type=str, default=None)
    p.add_argument("--sglang_shared_path", type=str, default=None)
    p.add_argument("--reward_model_path", type=str, default=None)

    # distillation
    p.add_argument("--student_hidden_size", type=int, default=None)
    p.add_argument("--student_num_layers", type=int, default=None)
    p.add_argument("--teacher_hidden_size", type=int, default=None)
    p.add_argument("--teacher_num_layers", type=int, default=None)
    p.add_argument("--student_use_moe", type=int, choices=[0, 1], default=None)
    p.add_argument("--teacher_use_moe", type=int, choices=[0, 1], default=None)
    p.add_argument("--from_student_weight", type=str, default=None)
    p.add_argument("--from_teacher_weight", type=str, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)

    # lora
    p.add_argument("--lora_name", type=str, default=None)

    # user YAML
    p.add_argument("--config", type=str, default=None,
                   help="Path to a YAML config file (overrides built-in defaults)")

    return p


# ==========================================================================
# Convenience — direct config access
# ==========================================================================

def create_arg_parser(task: str) -> argparse.ArgumentParser:
    """Return a standalone argparse.ArgumentParser for *task*.

    This is the public wrapper used by training scripts that want to keep
    their existing ``parser.parse_args()`` pattern while also supporting
    ``--config my.yaml``.
    """
    return _build_arg_parser(task)
