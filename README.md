# MiniMind-Lab

<div align="center">

**从零训练微型大语言模型 — 企业级实验平台**

[![GitHub stars](https://img.shields.io/github/stars/kaikai0516/MiniMind-Lab)](https://github.com/kaikai0516/MiniMind-Lab)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green)](https://developer.nvidia.com/cuda-downloads)
[![ROCm](https://img.shields.io/badge/ROCm-6.2-red)](https://www.amd.com/en/products/software/rocm.html)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

**NVIDIA CUDA · AMD ROCm · CPU | 8 种训练模式 | Jupyter 实时可视化 | Docker 一键部署 | YAML 驱动配置**

</div>

---

## 项目简介

MiniMind-Lab 是一个**从零训练微型大语言模型的完整实验平台**。核心模型仅 64M 参数，2 小时内可在消费级 GPU 上完成预训练，成本约 3 元人民币。项目在 [MiniMind](https://github.com/jingyaogong/minimind) 基础上进行了深度企业级工程化改造，新增了结构化日志系统、YAML 配置驱动、Jupyter 训练可视化、Docker 容器化部署、自动化测试体系等工业级基础设施。

### 核心能力矩阵

| 维度 | 能力 |
|------|------|
| **模型架构** | Decoder-Only Transformer · RMSNorm · RoPE · GQA · SwiGLU · MoE |
| **训练模式** | Pretrain → SFT → Knowledge Distillation → LoRA → DPO → GRPO → PPO → Agent RL |
| **数据格式** | JSONL / JSON / CSV / Parquet / TXT 自动识别 + 列别名映射 + SimHash 去重 |
| **GPU 后端** | NVIDIA CUDA 12.x · AMD ROCm 6.x · CPU 自动检测降级 |
| **训练加速** | 梯度累积 · 混合精度 (bfloat16/float16) · torch.compile · DDP 多卡 |
| **RL 算法** | DPO · GRPO · CISPO · PPO · Agentic RL (工具调用) · GAE 优势估计 |
| **推理部署** | CLI · Streamlit WebUI · OpenAI 兼容 API · Ollama · vLLM · SGLang |
| **工程设施** | YAML 配置 · Rich 日志 · safetensors · 原子写入 · pre-commit · pytest · Docker |
| **可视化** | Jupyter Notebook 实时监控 · SwanLab/Wandb 云端追踪 · metrics CSV 导出 |
| **智能工具** | auto_config — AI 自动分析硬件+数据 → 推荐训练参数 |

---

## 15 秒快速开始

```bash
# 1. 克隆安装
git clone https://github.com/kaikai0516/MiniMind-Lab && cd MiniMind-Lab
pip install -e ".[train]"

# 2. 下载数据
modelscope download --model gongjy/minimind_dataset \
    pretrain_t2t_mini.jsonl sft_t2t_mini.jsonl --local_dir ./dataset

# 3. 预训练 (~2h on RTX 3090 / RX 7900 XTX)
minimind-pretrain --epochs 2 --batch_size 32 --data_path dataset/pretrain_t2t_mini.jsonl

# 4. SFT 微调
minimind-full-sft --epochs 2 --batch_size 16 --data_path dataset/sft_t2t_mini.jsonl

# 5. 对话测试
minimind-eval --weight full_sft
```

---

## 文档导航

| 文档 | 内容 | 适合 |
|------|------|------|
| **[📊 数据准备与智能配置](docs/1_数据准备与智能配置.md)** | 数据集下载 · DataLoaderHub 多格式加载 · DataProcessor 清洗管线 · auto_config AI 分析工具 | 训练开始前必读 |
| **[🏋️ 训练流程](docs/2_训练流程.md)** | 8 种训练模式详解 (Pretrain/SFT/蒸馏/LoRA/DPO/GRPO/PPO/Agent) · 多卡 DDP · 断点续训 · RL 算法原理 · 模型推理 | 训练时查阅 |
| **[🏗️ 企业级基础设施](docs/3_企业级基础设施.md)** | 结构化日志 · YAML 配置系统 · Checkpoint 管理器 · 异常体系 · CLI 入口点 · pytest 测试 · pre-commit | 开发/调试时查阅 |
| **[🚀 部署指南](docs/4_部署指南.md)** | NVIDIA/AMD/CPU 环境安装 · Docker 多阶段构建 · Makefile 操作 · 网络/显存故障排查 | 环境搭建时查阅 |
| **[📖 API 参考](docs/5_API参考.md)** | DataLoaderHub · DataProcessor · Accumulator · TrainingConfig · LMCheckpointer · MetricsWriter · 所有模块完整签名 | 二次开发时查阅 |

---

## 训练管线全景

```
┌────────────┐    ┌──────────────┐    ┌──────────┐    ┌───────────┐    ┌────────┐
│ 📊 数据准备 │ → │ 🤖 智能分析   │ → │ 🔤 预训练 │ → │ 📝 SFT   │ → │ 🏗️ 蒸馏 │
│ DataLoader │    │ auto_config  │    │ pretrain │    │ full_sft │    │ distill │
│ DataProc   │    │              │    │ [必须]    │    │ [必须]    │    │ [可选]   │
└────────────┘    └──────────────┘    └──────────┘    └───────────┘    └────────┘
                                                                           │
                              ┌────────────────────────────────────────────┘
                              ↓
┌────────┐    ┌───────┐    ┌────────┐    ┌────────┐    ┌─────────┐
│ 🎯 DPO │ ← │ 🎮GRPO│ ← │ 🎮PPO │ ← │ 🤖Agent│ ← │ 🔧 LoRA │
│ align  │    │ group │    │ clip   │    │ tool   │    │ adapt   │
│ [可选]  │    │ [可选] │    │ [可选]  │    │ [可选]  │    │ [可选]   │
└────────┘    └───────┘    └────────┘    └────────┘    └─────────┘
```

每一步的完整参数、原理、调参建议见 **[训练流程文档](docs/2_训练流程.md)**。

---

## 一键命令 (Makefile)

```bash
# 训练
make pretrain ARGS="--epochs 3 --batch_size 64"
make full-sft ARGS="--epochs 5 --batch_size 32"
make lora    ARGS="--lora_name medical --epochs 10"
make dpo     ARGS="--beta 0.15 --epochs 1"
make grpo    ARGS="--num_generations 6"
make ppo     ARGS="--num_generations 6"
make agent   ARGS="--thinking_ratio 0.1"
make distillation ARGS="--alpha 0.5 --temperature 1.5"

# 推理
make eval                      make serve                 make chat

# 智能配置
make auto-config ARGS="--task full_sft --data_path dataset/sft_t2t_mini.jsonl"

# 质量
make lint                      make test                  make clean
make format                    make docker-build          make docker-run
```

---

## 配置系统

所有训练参数集中在根目录 **[`config.yaml`](config.yaml)** 中，带完整中文注释。配置优先级：

```
config.yaml 内置默认 → 用户自定义 YAML → CLI 参数 (--epochs 10)
```

```bash
# 方式1: 直接修改 config.yaml → 直接运行
python trainer/train_full_sft.py

# 方式2: CLI 快速覆盖
python trainer/train_full_sft.py --epochs 10 --batch_size 64

# 方式3: 自定义 YAML + CLI
python trainer/train_full_sft.py --config my_experiment.yaml --epochs 5
```

---

## Jupyter 训练监控

```bash
# 1. 正常启动训练 (自动写入 metrics.csv)
python trainer/train_full_sft.py --epochs 5

# 2. 新终端启动 Jupyter
jupyter notebook notebooks/training_monitor.ipynb

# 3. 运行所有 Cell → 实时绘制 loss/lr 曲线
```

---

## Docker 部署

```bash
docker build -t minimind-lab:dev --target dev .
docker run --gpus all -p 8000:8000 -v $(pwd)/out:/app/out minimind-lab:dev
```

---

## 项目结构

```
MiniMind-Lab/
├── config.yaml                  ← 🔧 唯一配置入口 (修改这里)
├── README.md                    ← 📖 本文件
├── docs/                        ← 📚 详细文档 (5个文件)
│   ├── 1_数据准备与智能配置.md
│   ├── 2_训练流程.md
│   ├── 3_企业级基础设施.md
│   ├── 4_部署指南.md
│   └── 5_API参考.md
├── notebooks/                   ← 📊 Jupyter 可视化
├── trainer/                     ← 🏋️ 训练核心
│   ├── train_pretrain.py        ← 8 个训练脚本
│   ├── train_full_sft.py        ← (pretrain/full_sft/lora/
│   ├── train_lora.py            ←  distillation/dpo/grpo/
│   ├── train_distillation.py    ←  ppo/agent)
│   ├── train_dpo.py
│   ├── train_grpo.py
│   ├── train_ppo.py
│   ├── train_agent.py
│   ├── logger.py                ← 结构化日志
│   ├── config.py                ← 配置系统
│   ├── checkpoint.py            ← Checkpoint 管理
│   ├── exceptions.py            ← 异常体系
│   ├── metrics.py               ← CSV 指标输出
│   ├── cli.py                   ← CLI 入口点
│   ├── training_loop.py         ← 统一训练循环
│   └── trainer_utils.py         ← 工具函数
├── dataset/                     ← 数据加载与处理
├── model/                       ← 模型定义
├── scripts/                     ← 推理/服务/配置工具
├── tests/                       ← pytest 测试
├── pyproject.toml               ← 打包配置
├── Makefile                     ← 快捷命令
├── Dockerfile                   ← 容器构建
└── requirements.txt             ← 依赖清单
```

---

## 特色亮点

### 1. DataLoaderHub — 任意格式即插即用
```python
# JSONL + CSV + Parquet 混合加载，自动列名映射
ds = hub.load_multi(["sft_*.jsonl", "extra.csv", "wiki.parquet"], dataset_type="sft")
```

### 2. Accumulator — 一行替代 8 行
```python
acc = Accumulator(model, optimizer, scaler, accumulation_steps=8, grad_clip=1.0)
acc.backward(loss)  # 自动处理 loss 缩放、梯度裁剪、optimizer.step()、scheduler.step()
```

### 3. auto_config — AI 帮你调参
```bash
python scripts/auto_config.py --task full_sft --data_path dataset/sft.jsonl --api_key sk-xxx
# → 自动探测 GPU 显存 + 分析数据分布 + 调用 LLM = 输出最优训练参数
```

### 4. LMCheckpointer — safetensors + 自动轮转
```python
ckp = LMCheckpointer("../checkpoints", max_keep=3, fmt="safetensors")
ckp.save(model, optimizer, epoch=0, step=100)
# → 原子写入、元数据 JSON、自动清理旧 checkpoint
```

---

## 致谢

基于 [MiniMind](https://github.com/jingyaogong/minimind) 原项目，感谢原作者的开源精神。本仓库在原有基础上进行了企业级工程化改造，新增了完整的工程基础设施、文档体系和可视化工具。
