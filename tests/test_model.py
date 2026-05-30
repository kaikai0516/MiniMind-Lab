"""
Sanity tests for MiniMind model instantiation and parameter counting.
All tests run on CPU; no GPU required.
"""

import torch
import pytest
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


class TestMiniMindConfig:
    def test_default_config(self):
        cfg = MiniMindConfig()
        assert cfg.hidden_size == 768
        assert cfg.num_hidden_layers == 8
        assert cfg.use_moe is False
        assert cfg.vocab_size == 6400
        assert cfg.num_attention_heads == 8
        assert cfg.num_key_value_heads == 4

    def test_custom_config(self):
        cfg = MiniMindConfig(hidden_size=512, num_hidden_layers=4, use_moe=True)
        assert cfg.hidden_size == 512
        assert cfg.num_hidden_layers == 4
        assert cfg.use_moe is True


class TestMiniMindModel:
    @pytest.fixture
    def config(self):
        return MiniMindConfig(hidden_size=256, num_hidden_layers=2, use_moe=False)

    def test_model_creation(self, config):
        model = MiniMindForCausalLM(config)
        assert model is not None
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    def test_forward_pass(self, config):
        model = MiniMindForCausalLM(config)
        model.eval()
        batch, seq = 1, 64
        input_ids = torch.randint(0, config.vocab_size, (batch, seq))
        labels = input_ids.clone()
        with torch.no_grad():
            out = model(input_ids, labels=labels)
        assert out.loss is not None
        assert out.logits.shape == (batch, seq, config.vocab_size)

    def test_moe_model(self):
        cfg = MiniMindConfig(hidden_size=256, num_hidden_layers=2, use_moe=True,
                             num_experts=4, num_experts_per_tok=1)
        model = MiniMindForCausalLM(cfg)
        model.eval()
        input_ids = torch.randint(0, cfg.vocab_size, (1, 64))
        with torch.no_grad():
            out = model(input_ids, labels=input_ids)
        # MOE model should produce aux_loss
        assert out.aux_loss is not None
