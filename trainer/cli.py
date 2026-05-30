"""
CLI entry points for MiniMind training and inference scripts.

Each function launches the corresponding module via ``runpy`` so that
existing ``if __name__ == "__main__"`` blocks execute normally.
"""

from __future__ import annotations

import runpy
import sys
from typing import Optional


def _run(module: str, args: Optional[list] = None) -> None:
    """Run *module* as ``__main__``, forwarding *argv*."""
    if args is not None:
        saved = sys.argv[:]
        sys.argv[1:] = args
    else:
        saved = None

    try:
        runpy.run_module(module, run_name="__main__")
    finally:
        if saved is not None:
            sys.argv[:] = saved


# ── Training ──────────────────────────────────────────────────────────

def pretrain(args: Optional[list] = None) -> None:
    """Pre-training from scratch."""
    _run("trainer.train_pretrain", args)


def full_sft(args: Optional[list] = None) -> None:
    """Full-parameter supervised fine-tuning."""
    _run("trainer.train_full_sft", args)


def lora(args: Optional[list] = None) -> None:
    """LoRA parameter-efficient fine-tuning."""
    _run("trainer.train_lora", args)


def distillation(args: Optional[list] = None) -> None:
    """Knowledge distillation (teacher → student)."""
    _run("trainer.train_distillation", args)


def dpo(args: Optional[list] = None) -> None:
    """Direct Preference Optimization."""
    _run("trainer.train_dpo", args)


def grpo(args: Optional[list] = None) -> None:
    """Group Relative Policy Optimization."""
    _run("trainer.train_grpo", args)


def ppo(args: Optional[list] = None) -> None:
    """Proximal Policy Optimization."""
    _run("trainer.train_ppo", args)


def agent(args: Optional[list] = None) -> None:
    """Agent reinforcement learning."""
    _run("trainer.train_agent", args)


# ── Inference / Eval ─────────────────────────────────────────────────

def eval_llm(args: Optional[list] = None) -> None:
    """LLM evaluation."""
    _run("eval_llm", args)


def serve(args: Optional[list] = None) -> None:
    """OpenAI-compatible API server."""
    _run("scripts.serve_openai_api", args)


def chat(args: Optional[list] = None) -> None:
    """Interactive chat API."""
    _run("scripts.chat_api", args)


def auto_config(args: Optional[list] = None) -> None:
    """Auto-configuration tool."""
    _run("scripts.auto_config", args)
