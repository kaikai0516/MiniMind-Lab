# MiniMind-Lab

<div align="center">

**从零训练微型大语言模型 — 企业级实验平台**

[![GitHub stars](https://img.shields.io/github/stars/kaikai0516/MiniMind-Lab)](https://github.com/kaikai0516/MiniMind-Lab)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green)](https://developer.nvidia.com/cuda-downloads)
[![ROCm](https://img.shields.io/badge/ROCm-6.2-red)](https://www.amd.com/en/products/software/rocm.html)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

**NVIDIA CUDA · AMD ROCm · CPU | 8 种训练模式 | Jupyter 可视化 | Docker 一键部署**

</div>

---

## 快速开始

```bash
# 1. 安装
git clone https://github.com/kaikai0516/MiniMind-Lab
cd MiniMind-Lab
pip install -e ".[train]"

# 2. 配置 → 打开 config.yaml 查看/修改所有参数

# 3. 下载数据
modelscope download --model gongjy/minimind_dataset \
    pretrain_t2t_mini.jsonl sft_t2t_mini.jsonl --local_dir ./dataset

# 4. 开始训练
minimind-pretrain --epochs 3
minimind-full-sft --epochs 3

# 5. 测试模型
minimind-eval --weight full_sft
```

---

## 项目结构

| 文档 | 内容 |
|------|------|
| **[📊 数据准备与智能配置](docs/1_数据准备与智能配置.md)** | DataLoaderHub · DataProcessor · auto_config 智能分析 |
| **[🏋️ 训练流程](docs/2_训练流程.md)** | 8 种训练模式 · 多卡 DDP · 断点续训 · RL 详解 |
| **[🏗️ 企业级基础设施](docs/3_企业级基础设施.md)** | 结构化日志 · YAML 配置 · Checkpoint · 异常 · 测试 |
| **[🚀 部署指南](docs/4_部署指南.md)** | NVIDIA/AMD/CPU · Docker · pip · Makefile |
| **[📖 API 参考](docs/5_API参考.md)** | 完整 API 文档 |

| 入口 | 用途 |
|------|------|
| **[`config.yaml`](config.yaml)** | 唯一配置文件 — 所有参数带中文注释 |
| **[`notebooks/`](notebooks/)** | Jupyter 训练可视化 |

---

## 训练 Pipeline

```
数据准备 → 智能分析 → 预训练 → SFT → (蒸馏) → (LoRA) → (DPO) → (GRPO/PPO) → (Agent)
  ↓           ↓          ↓        ↓         ↓         ↓         ↓          ↓          ↓
 [1_xxx.md] [auto_config]  pretrain  full_sft distillation lora     dpo      grpo/ppo   agent
```

每一步的详细说明见 **[训练流程文档](docs/2_训练流程.md)**。

---

## 一键命令

```bash
# 训练
make pretrain ARGS="--epochs 3"       make dpo ARGS="--epochs 1"
make full-sft ARGS="--epochs 5"       make grpo ARGS="--num_generations 6"
make lora ARGS="--epochs 10"          make agent ARGS="--epochs 1"

# 推理与监控
make eval                             make serve
make chat                             make auto-config

# 质量
make lint                             make test
make clean                            make format
```

---

## 特性

- **9 种新文件格式**支持 — JSONL / JSON / CSV / Parquet / TXT 自动识别
- **8 种训练模式** — Pretrain → SFT → LoRA → Distillation → DPO → GRPO → PPO → Agent
- **3 种 GPU 后端** — NVIDIA CUDA / AMD ROCm / CPU 自动检测
- **YAML 配置驱动** — 根目录 `config.yaml` 一键修改，CLI 可覆盖
- **Jupyter 可视化** — 实时训练曲线、数据分布、GPU 监控
- **Docker 部署** — 多阶段构建，开箱即用
- **safetensors** — 新一代检查点格式，更安全更快
- **结构化日志** — Rich 彩色终端 + 文件持久化

---

## 致谢

基于 [MiniMind](https://github.com/jingyaogong/minimind)，在原有基础上进行了企业级工程化改造。感谢原作者的开源精神。
