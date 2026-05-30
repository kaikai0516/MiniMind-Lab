# MiniMind Makefile
# Usage: make <target> [ARGS="--epochs 5 --batch_size 32"]

.DEFAULT_GOAL := help
ARGS ?=

PYTHON := python

# ── Installation ────────────────────────────────────────────────────
.PHONY: install install-dev install-all

install:
	pip install -e .

install-dev:
	pip install -e ".[train]"

install-all:
	pip install -e ".[all]"

# ── Training ────────────────────────────────────────────────────────
.PHONY: pretrain full-sft lora distillation dpo grpo ppo agent

pretrain:
	$(PYTHON) trainer/train_pretrain.py $(ARGS)

full-sft:
	$(PYTHON) trainer/train_full_sft.py $(ARGS)

lora:
	$(PYTHON) trainer/train_lora.py $(ARGS)

distillation:
	$(PYTHON) trainer/train_distillation.py $(ARGS)

dpo:
	$(PYTHON) trainer/train_dpo.py $(ARGS)

grpo:
	$(PYTHON) trainer/train_grpo.py $(ARGS)

ppo:
	$(PYTHON) trainer/train_ppo.py $(ARGS)

agent:
	$(PYTHON) trainer/train_agent.py $(ARGS)

# ── Inference & Serving ─────────────────────────────────────────────
.PHONY: eval serve chat auto-config

eval:
	$(PYTHON) eval_llm.py $(ARGS)

serve:
	$(PYTHON) scripts/serve_openai_api.py $(ARGS)

chat:
	$(PYTHON) scripts/chat_api.py $(ARGS)

auto-config:
	$(PYTHON) scripts/auto_config.py $(ARGS)

# ── Quality ─────────────────────────────────────────────────────────
.PHONY: lint test clean

lint:
	ruff check . --config pyproject.toml

format:
	ruff format . --config pyproject.toml

test:
	pytest tests/ -v $(ARGS)

clean:
	rm -rf __pycache__/ .pytest_cache/ .ruff_cache/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# ── Docker ──────────────────────────────────────────────────────────
.PHONY: docker-build docker-run

docker-build:
	docker build -t minimind:dev --target dev .

docker-run:
	docker run --gpus all -p 8000:8000 -v $(PWD)/out:/app/out minimind:dev

# ── Help ────────────────────────────────────────────────────────────
.PHONY: help

help:
	@echo "MiniMind Makefile"
	@echo ""
	@echo "Install:"
	@echo "  make install       pip install -e ."
	@echo "  make install-dev   pip install -e '.[train]'"
	@echo "  make install-all   pip install -e '.[all]'"
	@echo ""
	@echo "Training (pass args via ARGS=\"...\"):"
	@echo "  make pretrain      Pre-training"
	@echo "  make full-sft      Full-parameter SFT"
	@echo "  make lora          LoRA fine-tuning"
	@echo "  make distillation  Knowledge distillation"
	@echo "  make dpo           Direct Preference Optimization"
	@echo "  make grpo          Group Relative Policy Optimization"
	@echo "  make ppo           Proximal Policy Optimization"
	@echo "  make agent         Agent RL training"
	@echo ""
	@echo "Inference:"
	@echo "  make eval          LLM evaluation"
	@echo "  make serve         OpenAI-compatible API server"
	@echo "  make chat          Interactive chat"
	@echo "  make auto-config   Auto-configuration tool"
	@echo ""
	@echo "Quality:"
	@echo "  make lint          Run ruff linter"
	@echo "  make format        Run ruff formatter"
	@echo "  make test          Run pytest"
	@echo "  make clean         Remove artifacts"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run Docker container"
	@echo ""
	@echo "Example:"
	@echo "  make full-sft ARGS=\"--epochs 5 --batch_size 64\""
