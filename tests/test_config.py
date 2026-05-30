"""
Sanity tests for ``trainer.config`` — YAML defaults + CLI merging.
All tests run on CPU; no GPU required.
"""

import sys
import pytest
from trainer.config import load_config, TrainingConfig


class TestLoadConfig:
    def test_full_sft_defaults(self):
        """Default full_sft config loads with expected values."""
        cfg = load_config("full_sft", [])
        assert cfg.task == "full_sft"
        assert cfg.hidden_size == 768
        assert cfg.num_hidden_layers == 8
        assert cfg.use_moe is False
        assert cfg.epochs == 3
        assert cfg.batch_size == 32
        assert cfg.learning_rate == 1.0e-5
        assert isinstance(cfg, TrainingConfig)

    def test_pretrain_defaults(self):
        cfg = load_config("pretrain", [])
        assert cfg.task == "pretrain"
        assert cfg.batch_size == 64
        assert cfg.learning_rate == 5.0e-4

    def test_dpo_defaults(self):
        cfg = load_config("dpo", [])
        assert cfg.task == "dpo"
        assert cfg.beta == 0.15
        assert cfg.batch_size == 4

    def test_cli_override(self):
        """CLI args override YAML defaults."""
        cfg = load_config("full_sft", ["--epochs", "10", "--batch_size", "64"])
        assert cfg.epochs == 10
        assert cfg.batch_size == 64
        # unchanged
        assert cfg.hidden_size == 768

    def test_device_auto_detect(self):
        """Device is auto-detected when empty."""
        cfg = load_config("full_sft", [])
        assert cfg.device in ("cuda:0", "cpu", "mps")

    def test_unknown_task_falls_through(self):
        """Unknown task key yields dataclass defaults (no crash)."""
        cfg = load_config("nonexistent", [])
        assert isinstance(cfg, TrainingConfig)
        # falls back to dataclass field defaults
        assert cfg.task == "full_sft"

    def test_rl_config(self):
        cfg = load_config("grpo", [])
        assert cfg.task == "grpo"
        assert cfg.num_generations == 6
        assert cfg.loss_type == "cispo"
        assert cfg.epsilon == 0.2
