# MiniMind 增强版

<div align="center">

基于 [MiniMind](https://github.com/jingyaogong/minimind) 项目的三大创新增强：多数据集支持、训练机制统一、智能配置工具。

**兼容 NVIDIA CUDA / AMD ROCm / CPU 三种后端**



</div>

---

## 目录

1. [部署流程](#一部署流程)
   - [环境准备](#11-环境准备)
   - [NVIDIA GPU 环境](#12-nvidia-gpu-环境)
   - [AMD GPU 环境 (ROCm)](#13-amd-gpu-环境-ROCm)
   - [CPU 环境](#14-cpu-环境)
   - [数据下载](#15-数据下载)
2. [使用方法](#二使用方法)
   - [标准训练 Pipeline](#21-标准训练-pipeline)
   - [多数据集训练](#22-多数据集训练)
   - [RL 训练](#23-rl-训练)
   - [模型推理](#24-模型推理)
   - [多卡训练 (DDP)](#25-多卡训练-ddp)
3. [新功能详解](#三新功能详解)
   - [DataLoaderHub — 多格式数据加载](#31-dataloaderhub--多格式数据加载)
   - [DataProcessor — 数据处理工具集](#32-dataprocessor--数据处理工具集)
   - [Accumulator — 统一梯度累积](#33-accumulator--统一梯度累积)
   - [智能训练配置工具 (auto_config.py)](#34-智能训练配置工具-auto_configpy)
4. [文件结构](#四文件结构)
5. [API 参考](#五-api-参考)

---

## 一、部署流程

### 1.1 环境准备

**系统要求**:
- Python 3.10+
- 至少 4GB RAM (训练推荐 16GB+)
- (可选) CUDA 12.x 或 ROCm 6.x GPU
- (可选) 8GB+ 显存 GPU (训练推荐 12GB+)

**克隆项目**:

```bash
git clone --depth 1 https://github.com/jingyaogong/minimind
cd minimind
```

**安装基础依赖**:

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

> `requirements.txt` 中 torch 相关行已注释，需要根据你的硬件单独安装。见下方说明。

---

### 1.2 NVIDIA GPU 环境

适用显卡: GeForce RTX / GTX 系列, Tesla, Quadro, A100, H100 等。

```bash
# CUDA 12.4 / 12.6
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (旧卡)
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118
```

**验证安装**:

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
# 应输出: CUDA: True  GPU: NVIDIA GeForce RTX 3090
```

---

### 1.3 AMD GPU 环境 (ROCm)

适用显卡: Radeon RX 7000 系列 (RX 7900 XTX 等), Radeon Pro W7900, Instinct MI 系列 (MI250X, MI300X 等)，以及部分 Radeon RX 6000 系列。

> MiniMind 已针对 AMD GPU 做了兼容性处理：`trainer_utils.py` 中的 cuDNN 设置会自动跳过、`auto_config.py` 的硬件探测会识别 AMD 显卡型号。

```bash
# ROCm 6.x 官方安装（推荐）
pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/rocm6.2

# ROCm 5.x 旧版
pip install torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/rocm5.7
```

**验证安装**:

```bash
python -c "
import torch
print(f'ROCm available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'HIP version: {torch.version.hip}')
# 应输出: ROCm available: True  GPU: AMD Radeon RX 7900 XTX  HIP version: 6.x
"
```

**已知注意事项**:

| 事项 | 说明 |
|------|------|
| API 兼容 | ROCm 提供 `torch.cuda.*` 全套 API，代码无需修改 |
| 混合精度 | bfloat16 和 float16 均支持 |
| 显存 | `torch.cuda.get_device_properties(0).total_memory` 正常获取 |
| 多卡 (DDP) | `torchrun --nproc_per_node N` 正常使用 |
| cuDNN 调用 | `trainer_utils.py` 自动检测跳过 |
| 性能 | 与同等 NVIDIA 卡持平（7900 XTX ≈ RTX 4080） |

**AMD GPU 推荐配置**:

| GPU | 显存 | 推荐用途 |
|-----|------|---------|
| RX 7900 XTX | 24GB | 全流程训练 |
| RX 7900 XT | 20GB | 全流程训练 |
| RX 7800 XT | 16GB | Pretrain + SFT |
| Instinct MI250X | 128GB | 大 batch / 长序列 |
| Instinct MI300X | 192GB | 全部极致 |

---

### 1.4 CPU 环境

无 GPU 时，PyTorch 自动使用 CPU 推理和训练（训练速度约为 GPU 的 10-50 倍）。

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

所有训练脚本的 `--device` 参数默认为 `cuda:0` 或自动降级到 `cpu`：

```bash
# 自动检测
python trainer/train_full_sft.py   # GPU 不存在时自动用 CPU

# 强制指定
python trainer/train_full_sft.py --device cpu
```

> CPU 训练仅适用于体验流程。完整训练 MiniMind 仍建议使用 GPU（NVIDIA 或 AMD 均可）。

---

### 1.5 数据下载

从 [ModelScope 数据集](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) 下载所需数据，放入 `./dataset` 目录。

**最小复现所需** (2 个文件):

| 文件 | 大小 | 用途 |
|------|------|------|
| `pretrain_t2t_mini.jsonl` | ~200MB | 预训练 |
| `sft_t2t_mini.jsonl` | ~50MB | 指令微调 |

**完整数据集** (可选):

| 文件 | 用途 |
|------|------|
| `pretrain_t2t.jsonl` | 完整预训练数据 |
| `sft_t2t.jsonl` | 完整 SFT 数据 |
| `dpo.jsonl` | 偏好对齐 (DPO) |
| `rlaif.jsonl` | 强化学习 (GRPO/PPO) |
| `agent_rl.jsonl` | Agent 工具调用 RL |
| `agent_rl_math.jsonl` | Agent 数学推理 RL |
| `lora_medical.jsonl` | LoRA 医疗微调 |
| `lora_identity.jsonl` | LoRA 自我认知微调 |

**下载方式**:

```bash
# 方式1: modelscope CLI
pip install modelscope
modelscope download --model gongjy/minimind_dataset \
    pretrain_t2t_mini.jsonl sft_t2t_mini.jsonl --local_dir ./dataset

# 方式2: 网页下载
# 浏览器打开 https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files
# 手动下载需要的文件放到 ./dataset/
```

现在你还支持从**任意格式**的数据集中加载数据（见[多数据集训练](#22-多数据集训练)），不再局限于 JSONL 格式。

---

## 二、使用方法

### 2.1 标准训练 Pipeline

以下命令全部在 `./trainer` 目录下执行：

```bash
cd trainer
```

#### Step 1: 预训练 (必须)

```bash
# 最小数据集快速复现 (~2h on RTX 3090 / RX 7900 XTX)
python train_pretrain.py \
    --data_path ../dataset/pretrain_t2t_mini.jsonl \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --epochs 2

# 完整数据集
python train_pretrain.py \
    --data_path ../dataset/pretrain_t2t.jsonl \
    --batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 3
```

> 输出权重: `../out/pretrain_768.pth`

#### Step 2: SFT 指令微调 (必须)

```bash
# 最小数据集快速复现 (~30min on RTX 3090)
python train_full_sft.py \
    --data_path ../dataset/sft_t2t_mini.jsonl \
    --batch_size 32 \
    --learning_rate 1e-5 \
    --epochs 3

# 完整数据集
python train_full_sft.py \
    --data_path ../dataset/sft_t2t.jsonl \
    --batch_size 32 \
    --learning_rate 1e-5 \
    --epochs 5
```

> 输出权重: `../out/full_sft_768.pth`

#### Step 3: 测试模型

```bash
# 回到项目根目录
cd ..

# CLI 交互推理
python eval_llm.py --weight full_sft

# WebUI (需 pip install streamlit)
cp -r ../out ./scripts/  # 或下载 pretrained 权重
cd scripts && streamlit run web_demo.py
```

#### Step 4-6: 进阶训练 (可选)

```bash
# DPO 偏好对齐
python train_dpo.py --data_path ../dataset/dpo.jsonl --batch_size 4 --epochs 1

# GRPO 强化学习
python train_grpo.py --data_path ../dataset/rlaif.jsonl --batch_size 2 --num_generations 6

# LoRA 微调 (医疗)
python train_lora.py --data_path ../dataset/lora_medical.jsonl --batch_size 32 --epochs 10
```

---

### 2.2 多数据集训练

这是本次增强的核心功能：所有训练脚本的 `--data_path` 现在支持多文件、多格式。

#### 基础用法

```bash
# 列出多个文件
python train_full_sft.py \
    --data_path ../dataset/sft_a.jsonl \
                ../dataset/sft_b.jsonl \
                ../dataset/sft_c.csv

# 相同效果（每个文件独立传参）
python train_full_sft.py \
    --data_path ../dataset/sft_a.jsonl ../dataset/sft_b.jsonl ../dataset/sft_c.csv
```

#### 混合格式

```bash
# JSONL + CSV + JSON 同一个训练中
python train_pretrain.py \
    --data_path ../dataset/pretrain_t2t_mini.jsonl \
                ../dataset/wiki_zh.json \
                ../dataset/code_corpus.csv \
    --batch_size 32
```

#### 支持的格式详情

| 格式 | 自动检测方式 | 列别名映射 |
|------|-------------|-----------|
| `.jsonl` | 扩展名 | `content`→`text`, `messages`→`conversations` 等 |
| `.json` | 扩展名 | 同上 |
| `.csv` | 扩展名 | 同上 |
| `.parquet` | 扩展名 | 同上 |
| `.txt` | 扩展名 (降级兜底) | 每行作为 `{'text': line}` |

> 详细列别名映射见 [第 3.1 节](#31-dataloaderhub--多格式数据加载)。

#### 数据处理 Pipeline (Python API)

```python
from dataset.data_loader import DataLoaderHub
from dataset.data_processor import process_dataset, describe_dataset

hub = DataLoaderHub()

# 加载 + 合并
raw = hub.load_multi(["../dataset/sft_*.jsonl", "../dataset/extra.csv"],
                     dataset_type="sft")

# 查看统计
print(describe_dataset(raw))
# 输出: total=100000, avg_len=385.2chars, p50=280...

# 清洗 + 去重
clean = process_dataset(
    raw, min_length=10, max_length=2048,
    dedup_threshold=3, clean=True, validate=True
)
print(f"清洗前: {len(raw)} → 清洗后: {len(clean)}")
```

---

### 2.3 RL 训练

GRPO、PPO 和 Agent RL 使用 Rollout Engine 在训练过程中生成数据，因此有更多参数：

```bash
# GRPO 强化学习
python train_grpo.py \
    --data_path ../dataset/rlaif.jsonl \
    --from_weight full_sft \
    --batch_size 2 \
    --num_generations 6 \
    --max_gen_len 1024 \
    --beta 0.1 \
    --learning_rate 3e-7

# PPO 强化学习
python train_ppo.py \
    --data_path ../dataset/rlaif.jsonl \
    --from_weight full_sft \
    --batch_size 2 \
    --num_generations 6 \
    --learning_rate 3e-7

# Agent RL (工具调用场景)
python train_agent.py \
    --data_path ../dataset/agent_rl.jsonl \
    --from_weight full_sft \
    --batch_size 2 \
    --num_generations 4 \
    --thinking_ratio 0.1 \
    --max_total_len 2500
```

**RL 训练参数说明**:

| 参数 | 默认 | 说明 |
|------|------|------|
| `--num_generations` | 6 (GRPO) / 4 (Agent) | 每个 prompt 生成多少个 response |
| `--beta` | 0.1 | KL 散度惩罚系数 (越大越不离谱) |
| `--max_gen_len` | 1024 | 单个 response 的最大 token 数 |
| `--loss_type` | `cispo` | `cispo` (推荐) 或 `grpo` |
| `--epsilon` | 0.2 | GRPO clip epsilon |
| `--epsilon_high` | 5.0 | CISPO epsilon 上界 |
| `--thinking_ratio` | 0.9 (GRPO) / 0.1 (Agent) | 随机启用 `<think>` 的概率 |

---

### 2.4 模型推理

#### CLI 推理

```bash
# 方式1: 使用本地 PyTorch 权重
python eval_llm.py --weight full_sft

# 方式2: 使用 HuggingFace / ModelScope 格式模型
python eval_llm.py --load_from ./minimind-3
```

#### Web UI

```bash
# 安装 streamlit
pip install streamlit

# 启动 (模型文件夹需放在 ./scripts/ 下)
cd scripts && streamlit run web_demo.py
```

Web UI 支持：
- 思考过程展示 (`<think>` 标签解析)
- 工具调用 (`<tool_call>` 标签解析)
- 工具选择 (下拉菜单)

#### OpenAI 兼容 API

```bash
python scripts/serve_openai_api.py
```

启动后可用于：
- `curl http://localhost:8000/v1/chat/completions ...`
- 接入 FastGPT、Open-WebUI 等 Chat UI
- 支持 `reasoning_content`、`tool_calls`、`open_thinking`

#### 第三方推理框架

```bash
# ollama
ollama run jingyaogong/minimind-3

# vllm
vllm serve /path/to/model --served-model-name "minimind"
```

---

### 2.5 多卡训练 (DDP)

如果你的设备有 `N` 张 GPU（NVIDIA 和 AMD 均支持），使用 `torchrun` 启动：

```bash
# N 卡并行
torchrun --nproc_per_node N train_full_sft.py \
    --data_path ../dataset/sft_t2t_mini.jsonl \
    --batch_size 32 \
    --learning_rate 1e-5

# 多机多卡 (示例: 2 机器各 4 卡)
# 在主节点:
torchrun --nproc_per_node 4 --nnodes 2 \
    --node_rank 0 --master_addr 192.168.1.100 --master_port 29500 \
    train_pretrain.py --data_path ../dataset/pretrain_t2t_mini.jsonl

# 在从节点:
torchrun --nproc_per_node 4 --nnodes 2 \
    --node_rank 1 --master_addr 192.168.1.100 --master_port 29500 \
    train_pretrain.py --data_path ../dataset/pretrain_t2t_mini.jsonl
```

**DDP 与 compile 的兼容性**: 已通过 `wrap_model()` 自动处理正确顺序（compile → DDP → rollout_engine.update）。

---

### 2.5 断点续训

所有训练脚本均支持检查点：

```bash
# 启动续训
python train_pretrain.py --from_resume 1

# 自动检测 ../checkpoints/ 下的 checkpoint 并恢复
```

- 支持跨 GPU 数量恢复
- 支持 wandb/swanlab 训练记录连续性
- 检查点保存: `../checkpoints/<weight>_<hidden_size>_resume.pth`

---

## 三、新功能详解

### 3.1 DataLoaderHub — 多格式数据加载

**文件**: `dataset/data_loader.py`

```python
from dataset.data_loader import DataLoaderHub

hub = DataLoaderHub()

# 单文件自动检测
ds = hub.load("../dataset/sft_data.csv")     # CSV → Dataset
ds = hub.load("../dataset/extra.parquet")    # Parquet → Dataset

# 多文件合并
ds = hub.load_multi(
    ["../dataset/sft_a.jsonl", "../dataset/sft_b.jsonl", "../dataset/sft_c.csv"],
    dataset_type="sft"
)

# 便利函数
from dataset.data_loader import load_sft_data, load_pretrain_data
ds = load_sft_data(["../dataset/sft_*.jsonl"])
```

#### 列别名映射表

| 规范列名 | 自动识别的列名 |
|----------|--------------|
| `text` | `content`, `document`, `passage`, `sentence`, `body`, `article` |
| `conversations` | `messages`, `dialog`, `dialogue`, `chat`, `conversation`, `history` |
| `chosen` | `preferred`, `positive` |
| `rejected` | `dispreferred`, `negative` |
| `gt` | `ground_truth`, `answer` |

---

### 3.2 DataProcessor — 数据处理工具集

**文件**: `dataset/data_processor.py`

```python
from dataset.data_processor import (
    auto_detect_schema,      # 自动识别数据结构类型
    validate_conversations,  # 校验对话格式
    clean_text,              # 文本清洗
    filter_by_length,        # 按长度过滤
    deduplicate,             # SimHash 近似去重
    describe_dataset,        # 生成统计报告
    process_dataset,         # 一键全流程
)

# 一键全流程清洗
ds = process_dataset(
    raw_dataset,
    min_length=10,           # 最短字符数
    max_length=2048,         # 最长字符数
    dedup_threshold=3,       # SimHash 汉明距离阈值
    clean=True,              # Unicode 规范化 + 空白清理
    validate=True,           # 校验对话结构
)
# 内部顺序: 清洗 → 验证 → 长度过滤 → 去重
```

---

### 3.3 Accumulator — 统一梯度累积

**文件**: `trainer/training_loop.py`

替代了原项目中分布在 8 个训练脚本中的 3 种不同的梯度累积逻辑。

```python
from trainer.training_loop import Accumulator

acc = Accumulator(model, optimizer, scaler,
                  accumulation_steps=8, grad_clip=1.0, scheduler=scheduler)

for step, batch in enumerate(loader):
    loss = compute_loss(batch)
    acc.backward(loss)                  # 一行替代:
                                        #   loss = loss / accumulation_steps
                                        #   loss.backward()
                                        #   if step % accumulation_steps == 0:
                                        #       clip_grad_norm_(...)
                                        #       optimizer.step()
                                        #       scheduler.step()
                                        #       optimizer.zero_grad(set_to_none=True)

    print(f"loss: {acc.loss_value}")    # 自动修复累积偏移

acc.finalize()                          # epoch 结束排空
```

**相关 API**:

| 函数/类 | 用途 |
|---------|------|
| `Accumulator` | 单优化器梯度累积 |
| `MultiOptimizerAccumulator` | 双优化器梯度累积 (PPO actor+critic) |
| `create_scheduler(opt, iters, epochs, accum, lr)` | 自动计算 T_max, 创建 CosineAnnealingLR |
| `wrap_model(model, compile, rank, engine)` | compile → DDP → rollout_engine 正确顺序 |
| `save_model_weights(model, dir, name, dim, moe)` | 统一权重保存 |
| `get_raw_model(model)` | 解开 DDP 和 compile wrapper |

---

### 3.4 智能训练配置工具 auto_config.py

**文件**: `scripts/auto_config.py`

自动检测硬件 + 分析数据集 + 调用大模型 API = 输出推荐训练参数。

```bash
# 基础用法
python scripts/auto_config.py \
    --data_path ../dataset/sft_t2t_mini.jsonl \
    --task full_sft \
    --api_key sk-xxx \
    --output config.json

# 使用自定义 API
python scripts/auto_config.py \
    --task grpo --use_moe 1 \
    --api_base https://api.deepseek.com/v1 \
    --api_key sk-xxx \
    --model_name deepseek-chat

# Dry run: 仅查看 Prompt，不调用 API
python scripts/auto_config.py --data_path ../dataset/sft_t2t_mini.jsonl --dry_run
```

**功能模块**:

| 模块 | 函数 | 说明 |
|------|------|------|
| 硬件探测 | `hardware_probe()` | GPU (CUDA/ROCm)、CPU、内存、磁盘 |
| 数据集分析 | `analyze_dataset()` | 采样数、token 分布、长度分布 |
| 显存估算 | `estimate_memory()` | MiniMind 特定参数公式 |
| LLM 调用 | `get_config_from_llm()` | OpenAI 兼容 API |
| 输出 | `apply_config()` | JSON 文件 + 训练命令 |

**输出示例**:

```json
{
  "batch_size": 16,
  "learning_rate": 1e-5,
  "max_seq_len": 768,
  "epochs": 3,
  "accumulation_steps": 2,
  "dtype": "bfloat16",
  "estimated_gpu_memory_gb": 5.2,
  "reasoning": "RX 7900 XTX 24GB 显存充足，10万条数据建议3轮，batch 16 + 累积2步 = 等效batch 32"
}
```

同时输出可直接运行的训练命令：

```bash
python trainer/train_full_sft.py \
    --data_path ../dataset/sft_t2t_mini.jsonl \
    --hidden_size 768 --num_hidden_layers 8 --use_moe 0 \
    --batch_size 16 --learning_rate 1e-05 --max_seq_len 768 \
    --epochs 3 --accumulation_steps 2 --dtype bfloat16
```

---

## 四、文件结构

```
minimind-master/
│
├── dataset/
│   ├── data_loader.py        [NEW]  多格式数据加载适配层 (DataLoaderHub)
│   ├── data_processor.py     [NEW]  数据处理工具集 (process_dataset 等)
│   ├── lm_dataset.py         [MOD]  所有 Dataset 类支持多文件路径
│   └── dataset.md
│
├── trainer/
│   ├── training_loop.py      [NEW]  统一训练基础设施
│   │   ├── Accumulator              梯度累积器
│   │   ├── MultiOptimizerAccumulator 双优化器累积器 (PPO)
│   │   ├── create_scheduler()       统一 CosineAnnealingLR
│   │   ├── wrap_model()             compile → DDP → rollout
│   │   ├── save_model_weights()     统一权重保存
│   │   └── get_raw_model()          解包 DDP/compile
│   ├── trainer_utils.py      [MOD]  cuDNN 调用加 ROCm 兼容守卫
│   ├── train_pretrain.py     [MOD]  + Accumulator + Scheduler + wrap_model
│   ├── train_full_sft.py     [MOD]  + Accumulator + Scheduler + wrap_model
│   ├── train_lora.py         [MOD]  + Accumulator + Scheduler + wrap_model
│   ├── train_distillation.py [MOD]  + Accumulator + Scheduler + wrap_model
│   ├── train_dpo.py          [MOD]  + Accumulator + Scheduler + wrap_model
│   ├── train_grpo.py         [MOD]  + Accumulator + Scheduler + wrap_model
│   ├── train_ppo.py          [MOD]  + MultiOptimizerAccumulator + wrap_model
│   ├── train_agent.py        [MOD]  + Accumulator + Scheduler + wrap_model
│   ├── train_tokenizer.py
│   └── rollout_engine.py
│
├── scripts/
│   ├── auto_config.py        [NEW]  智能训练配置工具
│   ├── chat_api.py
│   ├── convert_model.py
│   ├── eval_toolcall.py
│   ├── serve_openai_api.py
│   └── web_demo.py
│
├── model/
│   ├── model_minimind.py
│   └── model_lora.py
│
├── README.md                 [KEPT] 原始 README (未修改)
└── README_NEW.md             [NEW]  本文件
```

---

## 五、API 参考

### DataLoaderHub

```python
class DataLoaderHub:
    def load(self, path: str) -> Dataset:
        """加载单个文件，自动检测格式 (jsonl/json/csv/parquet/txt)。"""

    def load_multi(self, paths: Union[str, List[str]],
                   dataset_type: str = "pretrain",
                   shuffle: bool = True, seed: int = 42,
                   process_fn: Optional[Callable] = None) -> Dataset:
        """加载多个文件，归一化 Schema，合并，可选清洗。"""

# 便利函数
load_pretrain_data(paths) -> Dataset  # 'text' 列
load_sft_data(paths) -> Dataset       # 'conversations' 列
load_dpo_data(paths) -> Dataset       # 'chosen'/'rejected' 列
load_rlaif_data(paths) -> Dataset     # 'conversations' 列
load_agent_data(paths) -> Dataset     # 'conversations'/'gt' 列
```

### DataProcessor

```python
# 文本处理
clean_text(text: str) -> str
clean_conversations(convs: List[Dict]) -> List[Dict]

# 验证
validate_conversations(convs: List[Dict]) -> bool
auto_detect_schema(sample: Dict) -> str  # 'pretrain'|'sft'|'dpo'|'rlaif'|'agent'|'unknown'

# 过滤
filter_by_length(ds: Dataset, min_len: int, max_len: int) -> Dataset
filter_by_token_length(ds: Dataset, tokenizer, min_tokens: int, max_tokens: int) -> Dataset

# 去重
deduplicate(ds: Dataset, threshold: int = 3) -> Dataset  # SimHash 近似去重

# 统计
describe_dataset(ds: Dataset, tokenizer=None) -> Dict[str, Any]

# 一键处理
process_dataset(ds: Dataset, min_length=10, max_length=2048,
                dedup_threshold=3, clean=True, validate=True) -> Dataset
```

### Training Loop

```python
# 梯度累积器
acc = Accumulator(model, optimizer, scaler, accumulation_steps, grad_clip, scheduler)
acc.backward(loss)       # scaled backward + 条件 step
acc.loss_value           # -> float (真实 loss)
acc.finalize()           # 排空剩余梯度

# 双优化器版本 (PPO)
ppo_acc = MultiOptimizerAccumulator(model, actor_opt, critic_opt, scaler, accum_steps, clip)
ppo_acc.backward(loss)   # 同步执行两个优化器的 step
ppo_acc.finalize()

# 调度器工厂
scheduler = create_scheduler(optimizer, iters_per_epoch=100, epochs=3,
                             accumulation_steps=2, learning_rate=1e-5)

# 模型包装 (compile → DDP → rollout)
model = wrap_model(model, use_compile=True, local_rank=0, rollout_engine=engine)

# 权重保存与解包
save_model_weights(model, save_dir="../out", name="full_sft", hidden_size=768, use_moe=False)
raw = get_raw_model(model)
```

### 智能配置工具

```python
from scripts.auto_config import (
    hardware_probe,         # -> Dict[gpu, cpu, disk, torch_version]
    analyze_dataset,        # -> Dict[total_samples, columns, ...]
    estimate_memory,        # -> Dict[total_params_m, peak_estimate_gb, ...]
    build_prompt,           # -> str (LLM prompt)
    get_config_from_llm,    # -> Dict (推荐的训练配置)
    apply_config,           # 写入 JSON + 打印训练命令
)
```

---

## 设计原则

1. **无侵入性**: 所有新功能为增量添加，原有 API 完全兼容
2. **GPU 后端无关**: 代码在 CUDA 和 ROCm 上同等运行，cuDNN 调用有守卫
3. **渐进降级**: `auto_config.py` 在缺少 torch/psutil/datasets 时仍可部分运行
4. **统一而非分裂**: Accumulator / create_scheduler / wrap_model 消除了 8 个脚本的实现分歧
5. **教学友好**: 注释精简，函数职责单一
