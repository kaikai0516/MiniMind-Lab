# MiniMind-Lab 完整操作指南

> 从零开始训练微型大语言模型 (LLM) — 从环境搭建到提交流程的全流程指南

---

## 目录

- [一、环境检测与部署](#一环境检测与部署)
  - [1.1 硬件需求](#11-硬件需求)
  - [1.2 自动硬件探测](#12-自动硬件探测)
  - [1.3 环境安装 (Windows / Linux / macOS)](#13-环境安装-windows--linux--macos)
  - [1.4 Docker 部署](#14-docker-部署)
  - [1.5 安装验证](#15-安装验证)
- [二、数据准备](#二数据准备)
  - [2.1 数据格式说明](#21-数据格式说明)
  - [2.2 下载官方数据集](#22-下载官方数据集)
  - [2.3 自定义数据集](#23-自定义数据集)
  - [2.4 DataLoaderHub 多格式加载](#24-dataloaderhub-多格式加载)
  - [2.5 DataProcessor 清洗管线](#25-dataprocessor-清洗管线)
- [三、AI 分析 + 配置设定](#三ai-分析--配置设定)
  - [3.1 配置系统架构](#31-配置系统架构)
  - [3.2 auto_config 智能配置工具](#32-auto_config-智能配置工具)
  - [3.3 硬件探测 (--probe)](#33-硬件探测---probe)
  - [3.4 数据集自动分析](#34-数据集自动分析)
  - [3.5 本地规则引擎 (--local)](#35-本地规则引擎---local)
  - [3.6 AI API 模式](#36-ai-api-模式)
  - [3.7 手动修改 config.yaml](#37-手动修改-configyaml)
  - [3.8 完整配置参数参考](#38-完整配置参数参考)
- [四、各类训练步骤 + Jupyter 分析](#四各类训练步骤--jupyter-分析)
  - [4.1 训练管线全景](#41-训练管线全景)
  - [4.2 Step 1: 预训练 (Pretrain) [必须]](#42-step-1-预训练-pretrain-必须)
  - [4.3 Step 2: 全量 SFT [必须]](#43-step-2-全量-sft-必须)
  - [4.4 Step 3: 知识蒸馏 (Distillation) [可选]](#44-step-3-知识蒸馏-distillation-可选)
  - [4.5 Step 4: LoRA 微调 [可选]](#45-step-4-lora-微调-可选)
  - [4.6 Step 5: DPO 偏好对齐 [可选]](#46-step-5-dpo-偏好对齐-可选)
  - [4.7 Step 6: GRPO 组相对策略优化 [可选]](#47-step-6-grpo-组相对策略优化-可选)
  - [4.8 Step 7: PPO 近端策略优化 [可选]](#48-step-7-ppo-近端策略优化-可选)
  - [4.9 Step 8: Agent RL 工具调用 [可选]](#49-step-8-agent-rl-工具调用-可选)
  - [4.10 Jupyter 训练监控面板](#410-jupyter-训练监控面板)
- [五、推理与服务部署](#五推理与服务部署)
  - [5.1 CLI 对话测试](#51-cli-对话测试)
  - [5.2 OpenAI 兼容 API](#52-openai-兼容-api)
  - [5.3 Streamlit WebUI](#53-streamlit-webui)
  - [5.4 LLM 评估](#54-llm-评估)
- [六、常见问题与故障排查](#六常见问题与故障排查)

---

## 一、环境检测与部署

### 1.1 硬件需求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **GPU** | 无 (CPU可运行) | NVIDIA RTX 3090 / RX 7900 XTX (24GB+) |
| **显存** | 0 GB (CPU模式) | 8GB+ (bfloat16 全量训练) |
| **内存** | 8 GB | 32 GB+ |
| **磁盘** | 5 GB | 20 GB+ (含数据集与checkpoint) |
| **CUDA** | 无 | CUDA 12.4+ / ROCm 6.2+ |

**GPU 后端支持矩阵：**

| 后端 | 状态 | 说明 |
|------|------|------|
| NVIDIA CUDA 12.x | ✅ 完整支持 | 混合精度 bfloat16 + torch.compile |
| AMD ROCm 6.x | ✅ 完整支持 | 自动检测 `torch.version.hip` |
| Apple MPS | ✅ 部分支持 | 仅推理，训练建议用 CPU |
| CPU | ✅ 可用 | 自动降级，batch_size 自动缩小 |

### 1.2 自动硬件探测

项目内置硬件探测工具，可随时查看当前环境：

```bash
# 方式1: auto_config CLI
python scripts/auto_config.py --probe

# 方式2: Python 代码
python -c "
from scripts.auto_config import hardware_probe
import json
hw = hardware_probe()
print(json.dumps(hw, indent=2, ensure_ascii=False))
"
```

**输出示例 (Windows, 无GPU):**

```
  GPU: N/A (0GB) [不可用]
  Backend: N/A
  CPU: 16 核, RAM: 31.3GB
  磁盘: 53.4GB 可用
  PyTorch: 2.12.0+cpu
```

**输出示例 (Linux, RTX 3090):**

```
  GPU: NVIDIA GeForce RTX 3090 (24.0GB) [可用]
  Backend: CUDA
  CPU: 32 核, RAM: 64.0GB
  磁盘: 200.0GB 可用
  PyTorch: 2.6.0+cu124
```

### 1.3 环境安装 (Windows / Linux / macOS)

#### 1.3.1 创建虚拟环境

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

#### 1.3.2 安装 PyTorch

根据你的硬件选择安装命令：

```bash
# === NVIDIA CUDA 12.x ===
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# === AMD ROCm 6.x ===
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# === CPU only (无GPU) ===
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 1.3.3 安装 MiniMind-Lab

```bash
# 方式1: pip 可编辑安装 (推荐开发)
pip install -e ".[train]"      # 仅训练依赖
pip install -e ".[all]"        # 全部依赖 (含推理/评估/API)

# 方式2: 使用清华镜像加速
pip install -e ".[all]" -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方式3: 直接安装 requirements.txt
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方式4: Makefile 安装
make install-all
```

**依赖组说明：**

| 组名 | 内容 | 适用场景 |
|------|------|---------|
| `[train]` | PyYAML, wandb, swanlab, peft, trl | 训练 |
| `[serve]` | Flask, streamlit, openai | 推理服务 |
| `[eval]` | jieba, nltk, scikit-learn, sentence-transformers | 模型评估 |
| `[data]` | datasketch, simhash, marshmallow, ujson | 数据处理 |
| `[all]` | 以上全部 | 完整安装 |

### 1.4 Docker 部署

```bash
# 构建开发镜像
make docker-build
# 或:
docker build -t minimind:dev --target dev .

# 运行容器 (GPU)
make docker-run
# 或:
docker run --gpus all -p 8000:8000 -v $(pwd)/out:/app/out minimind:dev

# 运行容器 (CPU)
docker run -p 8000:8000 -v $(pwd)/out:/app/out minimind:dev
```

Docker 多阶段构建说明：
- **dev 阶段**: 完整环境，包含所有依赖，适合开发和训练
- **prod 阶段**: 仅推理依赖，镜像更小，适合部署 API 服务

### 1.5 安装验证

完成安装后，运行测试确认一切就绪：

```bash
# 完整测试套件 (17 个测试)
make test
# 或:
pytest tests/ -v

# 快速验证核心模块
python -c "
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.config import load_config
from dataset.data_loader import DataLoaderHub
from transformers import PreTrainedTokenizerFast

# 模型
config = MiniMindConfig()
model = MiniMindForCausalLM(config)
print(f'[OK] Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params')

# 配置
cfg = load_config('pretrain', [])
print(f'[OK] Config: task={cfg.task}, batch={cfg.batch_size}')

# Tokenizer
tok = PreTrainedTokenizerFast.from_pretrained('./model')
print(f'[OK] Tokenizer: vocab_size={tok.vocab_size}')

print('All checks passed!')
"
```

**预期输出：**

```
[OK] Model: 63.9M params
[OK] Config: task=pretrain, batch=32
[OK] Tokenizer: vocab_size=6400
All checks passed!
```

---

## 二、数据准备

### 2.1 数据格式说明

MiniMind-Lab 支持 5 种文件格式，**自动识别，无需手动指定**：

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| **JSONL** | `.jsonl` | 每行一个 JSON 对象，推荐格式 |
| **JSON** | `.json` | JSON 对象数组或单对象 |
| **CSV** | `.csv`, `.tsv` | 逗号/制表符分隔 |
| **Parquet** | `.parquet`, `.pq` | 列式存储，加载最快 |
| **TXT** | `.txt`, `.text`, `.md` | 每行作为一条 text |

不同任务类型需要不同的数据字段：

#### Pretrain 数据格式

```jsonl
{"text": "人工智能是计算机科学的一个重要分支..."}
{"text": "Transformer架构基于自注意力机制..."}
```

必填字段: `text` (或 `content`, `document`, `passage`, `body`, `article`)

#### SFT 数据格式

```jsonl
{"conversations": [{"role": "user", "content": "什么是机器学习？"}, {"role": "assistant", "content": "机器学习是人工智能的一个子集..."}]}
{"conversations": [{"role": "user", "content": "解释神经网络"}, {"role": "assistant", "content": "神经网络是..."}]}
```

必填字段: `conversations` (或 `messages`, `dialog`, `chat`, `history`)
每条消息必须包含 `role` 和 `content` 字段。

#### DPO 数据格式

```jsonl
{"chosen": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "好的回答"}], "rejected": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "差的回答"}]}
```

必填字段: `chosen`, `rejected` (或 `preferred`/`dispreferred`)

#### RLAIF 数据格式 (GRPO/PPO)

```jsonl
{"conversations": [{"role": "user", "content": "请解释量子计算"}]}
```

#### Agent RL 数据格式

```jsonl
{"conversations": [{"role": "user", "content": "查询今天天气"}], "gt": "天气查询结果: 晴, 25°C"}
```

必填字段: `conversations`, `gt` (或 `ground_truth`, `answers`)

### 2.2 下载官方数据集

使用 ModelScope 下载 MiniMind 官方数据集：

```bash
# 安装 modelscope (已在 requirements.txt 中包含)
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

# 下载预训练数据
modelscope download --model gongjy/minimind_dataset \
    pretrain_t2t_mini.jsonl --local_dir ./dataset

# 下载 SFT 数据
modelscope download --model gongjy/minimind_dataset \
    sft_t2t_mini.jsonl --local_dir ./dataset

# 下载全部数据集
modelscope download --model gongjy/minimind_dataset \
    pretrain_t2t_mini.jsonl \
    sft_t2t_mini.jsonl \
    dpo.jsonl \
    rlaif.jsonl \
    lora_medical.jsonl \
    --local_dir ./dataset
```

### 2.3 自定义数据集

你可以将自己的数据放在 `./dataset/` 目录下，支持任意格式 (JSONL/JSON/CSV/Parquet/TXT)：

```bash
# 示例: 将自己整理的 JSONL 数据放入 dataset 目录
cp /path/to/your/my_pretrain_data.jsonl ./dataset/
cp /path/to/your/my_sft_data.json ./dataset/
```

然后在 `config.yaml` 中更新路径：

```yaml
pretrain:
  data_path:
    - dataset/my_pretrain_data.jsonl
    - dataset/extra_data.csv
```

### 2.4 DataLoaderHub 多格式加载

`DataLoaderHub` 是统一的数据加载入口，支持自动格式检测、多文件合并、Schema 标准化：

```python
from dataset.data_loader import DataLoaderHub

hub = DataLoaderHub()

# 单文件加载
ds = hub.load("./dataset/pretrain_t2t_mini.jsonl")

# 多文件加载 + 多格式混合
ds = hub.load_multi(
    [
        "./dataset/pretrain_t2t_mini.jsonl",
        "./dataset/extra_data.csv",
        "./dataset/more_data.parquet",
    ],
    dataset_type="pretrain",   # 自动标准化列名
    shuffle=True,               # 随机打乱
    seed=42,
)

print(f"总样本数: {len(ds)}")
print(f"列名: {ds.column_names}")
```

**Schema 别名映射表：**

| 任务类型 | 标准列名 | 支持的别名 |
|---------|---------|-----------|
| pretrain | `text` | `content`, `document`, `passage`, `body`, `article` |
| sft | `conversations` | `messages`, `dialog`, `chat`, `conversation`, `history` |
| dpo | `chosen`, `rejected` | `preferred`/`dispreferred`, `chosen_response`/`rejected_response` |
| rlaif | `conversations` | `messages`, `dialog`, `chat` |
| agent | `conversations`, `gt` | `messages`/`dialog`, `ground_truth`/`answers`/`answer` |

### 2.5 DataProcessor 清洗管线

`DataProcessor` 提供数据清洗、验证和去重功能：

```python
from dataset.data_processor import (
    auto_detect_schema,       # 自动检测数据类型
    validate_conversations,   # 验证对话格式完整性
    # SimHash 去重将在后续版本开放
)

# 自动检测样本的数据类型
sample = {"text": "这是一段文本"}
task_type = auto_detect_schema(sample)
# -> "pretrain"

sample = {"conversations": [{"role": "user", "content": "你好"}]}
task_type = auto_detect_schema(sample)
# -> "sft"

# 验证对话格式
is_valid = validate_conversations([
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的？"}
])
# -> True
```

---

## 三、AI 分析 + 配置设定

### 3.1 配置系统架构

MiniMind-Lab 使用 **三层配置合并机制**：

```
config_defaults.yaml (内置默认值)
       ↓ 覆盖
config.yaml (用户配置文件)
       ↓ 覆盖
CLI 参数 (命令行)
       ↓
TrainingConfig 数据类 (最终配置)
```

优先级: **CLI 参数 > config.yaml > config_defaults.yaml**

配置文件位置：

| 文件 | 路径 | 用途 |
|------|------|------|
| 内置默认 | `trainer/config_defaults.yaml` | 8 种任务的预设值 |
| 用户配置 | `./config.yaml` | 用户修改的主入口 |
| 全局配置 | `config.yaml → global` 段 | 模型尺寸、dtype、num_workers |
| 任务配置 | `config.yaml → [task_name]` 段 | 各任务的 batch/lr/epochs |

### 3.2 auto_config 智能配置工具

`auto_config` 是 MiniMind-Lab 的核心特色工具，自动分析你的硬件和数据，推荐最优训练参数。

**完整命令格式：**

```bash
python scripts/auto_config.py \
    --data_path <数据路径> \
    --task <任务类型> \
    [--hidden_size 768] \
    [--num_layers 8] \
    [--use_moe 0|1] \
    [--local | --api_key sk-xxx] \
    [--output config.yaml]
```

### 3.3 硬件探测 (`--probe`)

只探测硬件信息，不生成配置：

```bash
python scripts/auto_config.py --probe

# 输出:
# =================================================================
#   MiniMind 智能训练配置工具
# =================================================================
#   硬件信息:
#     GPU: NVIDIA GeForce RTX 3090 (24.0GB) [可用]
#     Backend: CUDA
#     CPU: 32 核, RAM: 64.0GB
#     磁盘: 200.0GB 可用
#     PyTorch: 2.6.0+cu124
```

### 3.4 数据集自动分析

`auto_config` 会自动分析数据集并提供统计信息：

```
[2/4] 分析数据集...
  样本数: 10,000
  列: ['text']
  平均长度: 512 字符 (P50=480, P90=890)
  [自动检测] 任务类型: pretrain
```

分析内容包括：
- **样本总数** — 影响 epoch 和 batch_size 推荐
- **列名** — 用于 Schema 自动检测
- **长度分布** — P50 / P90 / 最大值 / 平均值
- **Token 估算** — 基于中英文混合约 2 字符/token
- **任务自动检测** — 根据文件名推断任务类型

### 3.5 本地规则引擎 (`--local`)

**不需要 API key**，基于规则的智能推荐：

```bash
python scripts/auto_config.py \
    --data_path ./dataset/pretrain_t2t_mini.jsonl \
    --task pretrain \
    --hidden_size 768 \
    --num_layers 8 \
    --local
```

规则引擎的调整逻辑：

| 条件 | 调整 |
|------|------|
| 样本数 < 1000 | epochs × 2, batch_size ÷ 2 |
| 样本数 > 50000 | epochs ÷ 2, batch_size × 2 |
| 平均长度 > 600 tokens | max_seq_len 提升到 1024 |
| 平均长度 < 150 tokens | max_seq_len 降低到 512 |
| GPU 显存不足 | 自动减少 batch_size 或增加 accumulation_steps |
| CPU only | batch_size ≤ 2, num_workers = 0 |

**8 种任务的规则推荐默认值：**

| 任务 | batch_size | learning_rate | epochs | max_seq_len | 特殊参数 |
|------|-----------|---------------|--------|-------------|---------|
| pretrain | 32 | 5e-4 | 2 | 340 | from_weight=none |
| full_sft | 16 | 1e-5 | 2 | 768 | from_weight=pretrain |
| lora | 32 | 1e-4 | 10 | 340 | lora_name=lora_weights |
| distillation | 32 | 5e-6 | 6 | 340 | α=0.5, T=1.5 |
| dpo | 4 | 4e-8 | 1 | 1024 | β=0.15 |
| grpo | 2 | 3e-7 | 1 | 768 | 生成6次, β=0.1 |
| ppo | 2 | 3e-7 | 1 | 768 | 生成6次, β=0.1 |
| agent | 2 | 3e-7 | 1 | 1024 | 生成4次, 工具调用 |

### 3.6 AI API 模式

使用 OpenAI 兼容 API 让 LLM 推荐训练参数：

```bash
# 使用 OpenAI
python scripts/auto_config.py \
    --data_path ./dataset/pretrain_t2t_mini.jsonl \
    --task pretrain \
    --api_key sk-your-openai-key \
    --model_name gpt-4o

# 使用本地 LLM (如 vLLM / Ollama)
python scripts/auto_config.py \
    --data_path ./dataset/pretrain_t2t_mini.jsonl \
    --task pretrain \
    --api_base http://localhost:8000/v1 \
    --api_key not-needed \
    --model_name qwen2.5-7b

# 也可通过环境变量设置
export OPENAI_API_KEY="sk-xxx"
python scripts/auto_config.py --data_path ./dataset/pretrain_t2t_mini.jsonl --task pretrain
```

**AI 分析流程：**

1. 收集硬件信息 (GPU 型号/显存/CPU/RAM)
2. 分析数据集 (样本数/长度分布/估算 token 数)
3. 估算显存需求 (模型+梯度+优化器+激活)
4. 构建 Prompt 发送给 LLM
5. 解析 LLM 返回的 JSON 配置
6. 写入 `config.yaml` 文件

如果 AI API 调用失败，**自动回退到本地规则引擎**。

### 3.7 手动修改 config.yaml

`config.yaml` 是用户配置的唯一入口，位于项目根目录。

**文件结构：**

```yaml
# -- 全局默认 (所有任务共用) --
global:
  hidden_size: 768          # 隐藏层维度 (512/768)
  num_hidden_layers: 8      # Transformer 层数 (4/8)
  use_moe: false            # 是否启用 MoE
  max_seq_len: 768          # 最大序列长度
  dtype: bfloat16           # 训练精度 (bfloat16/float16/float32)
  num_workers: 8            # 数据加载进程数
  device: ""                # 留空=自动检测 (cuda:0/cpu/mps)

# -- 路径 --
paths:
  data_dir: ./dataset
  save_dir: ./out           # 模型权重输出目录
  checkpoint_dir: ./checkpoints
  tokenizer_path: ./model

# -- Checkpoint --
checkpoint:
  max_keep: 3               # 最多保留 checkpoint 数
  format: torch             # 保存格式 (torch/safetensors)
  save_interval: 100        # 每 N 步保存一次

# -- 日志与监控 --
logging:
  log_interval: 100         # 每 N 步打印一次日志
  use_wandb: false          # 是否启用 Wandb 云端追踪
  wandb_project: MiniMind-Lab
  metrics_csv: ./metrics.csv

# -- 预训练 --
pretrain:
  data_path:
    - dataset/pretrain_t2t_mini.jsonl
  from_weight: "none"       # 从哪个权重开始 (none=随机初始化)
  save_weight: pretrain     # 保存权重的名称
  batch_size: 32
  learning_rate: 0.0005
  epochs: 2
  accumulation_steps: 1     # 梯度累积步数
  grad_clip: 1.0            # 梯度裁剪阈值

# -- 全量 SFT --
full_sft:
  data_path:
    - dataset/sft_t2t_mini.jsonl
  from_weight: pretrain     # 基于预训练权重
  save_weight: full_sft
  batch_size: 16
  learning_rate: 0.00001
  epochs: 2
  # ... 其他任务类似 ...
```

### 3.8 完整配置参数参考

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hidden_size` | int | 768 | 隐藏层维度 (512=小模型, 768=标准模型) |
| `num_hidden_layers` | int | 8 | Transformer 层数 (4/8/16) |
| `use_moe` | bool | false | 启用混合专家 (参数量 1.8x) |
| `max_seq_len` | int | 768 | 最大序列长度 |
| `batch_size` | int | 32 | 每批样本数 |
| `learning_rate` | float | 5e-4 | 学习率 |
| `epochs` | int | 2 | 训练轮数 |
| `accumulation_steps` | int | 1 | 梯度累积步数 (等效增大 batch) |
| `grad_clip` | float | 1.0 | 梯度裁剪阈值 |
| `dtype` | str | bfloat16 | 训练精度 (bfloat16/float16/float32) |
| `num_workers` | int | 8 | DataLoader 进程数 |
| `device` | str | "" | 设备 (空=自动检测 cuda:0/cpu/mps) |
| `use_compile` | bool | false | torch.compile 加速 |
| `from_resume` | bool | false | 断点续训 |
| `log_interval` | int | 100 | 日志输出间隔 (步) |
| `save_interval` | int | 100 | 模型保存间隔 (步) |
| `use_wandb` | bool | false | Wandb 云端日志 |

---

## 四、各类训练步骤 + Jupyter 分析

### 4.1 训练管线全景

MiniMind-Lab 提供 **8 种训练模式**，按建议顺序执行：

```
┌──────────────────────────────────────────────────────────────┐
│  训练路径:                                                    │
│                                                              │
│  [必须] Pretrain → [必须] Full SFT → [可选] 蒸馏 / LoRA       │
│                                          ↓                   │
│                         [可选] DPO → GRPO → PPO → Agent RL   │
│                                                              │
│  启动命令:                                                    │
│  make <task> ARGS="--epochs 5 --batch_size 64"               │
│  或:                                                         │
│  python trainer/train_<task>.py [args...]                    │
└──────────────────────────────────────────────────────────────┘
```

**所有命令的执行方式：**

```bash
# 方式1: Makefile (简洁)
make pretrain ARGS="--epochs 5 --batch_size 64"

# 方式2: Python 直接运行
python trainer/train_pretrain.py --epochs 5 --batch_size 64

# 方式3: pip 安装后的 CLI 命令
minimind-pretrain --epochs 5 --batch_size 64
```

### 4.2 Step 1: 预训练 (Pretrain) [必须]

**目标**: 在海量文本上训练模型的通用语言能力。

**建议数据量**: 1万条以上

**默认参数:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 32 | 标准预训练 |
| `learning_rate` | 5e-4 | 较高的初始学习率 |
| `epochs` | 2 | |
| `max_seq_len` | 340 | 预训练阶段使用较短序列 |
| `from_weight` | "none" | 从头训练 |

**完整命令示例：**

```bash
# 基础用法 — 读取 config.yaml 中的 pretrain 段配置
python trainer/train_pretrain.py

# 自定义参数
python trainer/train_pretrain.py \
    --data_path dataset/pretrain_t2t_mini.jsonl \
    --epochs 2 \
    --batch_size 32 \
    --learning_rate 0.0005 \
    --max_seq_len 340 \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --dtype bfloat16 \
    --save_dir ./out \
    --log_interval 100 \
    --save_interval 500

# CPU 训练 (自动降级)
python trainer/train_pretrain.py --device cpu --batch_size 2 --dtype float32

# 小模型快速实验 (3M 参数)
python trainer/train_pretrain.py --hidden_size 256 --num_hidden_layers 2 --batch_size 8

# GPU 训练
python trainer/train_pretrain.py --device cuda:0 --batch_size 64 --dtype bfloat16

# 多卡训练 (DDP)
torchrun --nproc_per_node=2 trainer/train_pretrain.py --batch_size 64
```

**训练流程内部实现：**

1. 读取 `config.yaml` → 创建 `TrainingConfig`
2. 调用 `DataLoaderHub.load_multi()` 加载数据
3. 使用 `PretrainedTokenizer` 进行 tokenization
4. 创建 `MiniMindForCausalLM` 模型
5. 创建 AdamW 优化器 + CosineAnnealingLR 调度器
6. 创建 `Accumulator` 管理梯度累积和混合精度
7. 循环训练: forward → loss → backward → clip → step → log
8. 每 `save_interval` 步保存 checkpoint
9. 训练完成保存最终权重到 `./out/pretrain_768.pth`

**预期输出示例 (GPU, 768 dims, 8 layers):**

```
[INFO] Model Params: 63.9M
[INFO] Trainable Params: 63.9M
[INFO] Starting training for 2 epochs on pretrain task
Epoch 1, Step 100/500, Loss=5.234, LR=4.8e-4
Epoch 1, Step 200/500, Loss=4.891, LR=4.2e-4
...
```

**输出权重文件：**

```
./out/pretrain_768.pth          # 标准模型
./out/pretrain_768_moe.pth      # MoE 模型 (如果 use_moe=true)
```

### 4.3 Step 2: 全量 SFT [必须]

**目标**: 在对话数据上微调预训练模型，使其学会对话格式。

**前置条件**: 已完成 Pretrain，`./out/pretrain_768.pth` 存在。

**默认参数:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 16 | SFT 使用更小的 batch |
| `learning_rate` | 1e-5 | 比 pretrain 低 50 倍 |
| `epochs` | 2 | |
| `max_seq_len` | 768 | SFT 使用完整序列长度 |
| `from_weight` | "pretrain" | 加载预训练权重 |

**完整命令示例：**

```bash
# 基础用法
python trainer/train_full_sft.py

# 自定义参数
python trainer/train_full_sft.py \
    --data_path dataset/sft_t2t_mini.jsonl \
    --from_weight pretrain \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --epochs 3 \
    --max_seq_len 768

# 基于已微调的模型继续训练
python trainer/train_full_sft.py \
    --from_weight full_sft \
    --save_weight full_sft_v2

# 从头开始 SFT (不使用预训练权重)
python trainer/train_full_sft.py --from_weight none
```

**输出：** `./out/full_sft_768.pth`

### 4.4 Step 3: 知识蒸馏 (Distillation) [可选]

**目标**: 用更大的教师模型指导小模型学习，提升小模型性能。

**前置条件**: 已完成 SFT，需要教师模型权重。

**默认参数:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 32 | |
| `learning_rate` | 5e-6 | 极低学习率 |
| `epochs` | 6 | 蒸馏需要更多轮次 |
| `alpha` | 0.5 | 硬标签 vs 软标签权重 |
| `temperature` | 1.5 | 蒸馏温度 (越高越软) |

**完整命令：**

```bash
python trainer/train_distillation.py \
    --data_path dataset/sft_t2t_mini.jsonl \
    --from_student_weight full_sft \
    --from_teacher_weight full_sft \
    --alpha 0.3 \
    --temperature 2.0 \
    --epochs 6 \
    --student_hidden_size 512 \
    --teacher_hidden_size 768
```

**蒸馏原理：**

```
Loss = α × Hard_Loss(学生输出, 真实标签) + (1-α) × Soft_Loss(学生输出, 教师输出)
```

- `alpha=1.0` → 完全使用硬标签 (退化为普通 SFT)
- `alpha=0.0` → 完全模仿教师
- `temperature` 越高 → 教师输出分布越平滑

**输出：** `./out/full_dist_768.pth`

### 4.5 Step 4: LoRA 微调 [可选]

**目标**: 使用低秩适配器进行参数高效的领域微调。

**前置条件**: 已完成 SFT。

**默认参数:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 32 | |
| `learning_rate` | 1e-4 | LoRA 可以使用更高的学习率 |
| `epochs` | 10 | |
| `lora_name` | lora_medical | LoRA 适配器名称 |

**完整命令：**

```bash
python trainer/train_lora.py \
    --data_path dataset/lora_medical.jsonl \
    --from_weight full_sft \
    --lora_name lora_medical \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --epochs 10

# 加载已有 LoRA 权重继续训练
python trainer/train_lora.py --from_weight lora_medical --lora_name lora_medical_v2
```

**LoRA 特点：**
- 仅训练 ~1% 参数，显存需求大幅降低
- 适配器独立保存，可叠加多个领域
- 推理时可与基础模型合并

**输出：** `./out/lora_medical_768.pth`

### 4.6 Step 5: DPO 偏好对齐 [可选]

**目标**: 直接在人类偏好数据上优化模型，使回答更符合人类期望。

**前置条件**: 已完成 SFT。

**数据格式**: 需要 `chosen` (好回答) 和 `rejected` (差回答) 配对。

**默认参数:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 4 | DPO 显存消耗大 |
| `learning_rate` | 4e-8 | 极低学习率防止 catastrophic forgetting |
| `epochs` | 1 | |
| `beta` | 0.15 | KL 散度惩罚系数 |

**完整命令：**

```bash
python trainer/train_dpo.py \
    --data_path dataset/dpo.jsonl \
    --from_weight full_sft \
    --batch_size 4 \
    --learning_rate 4e-8 \
    --beta 0.15 \
    --epochs 1

# 自定义 beta
python trainer/train_dpo.py --beta 0.3
```

**DPO 算法简介：**

DPO 直接优化策略模型，使其给 chosen 回答的概率高于 rejected 回答，同时通过 β 控制与原始模型的 KL 散度防止偏离过远。

**输出：** `./out/dpo_768.pth`

### 4.7 Step 6: GRPO 组相对策略优化 [可选]

**目标**: 使用强化学习方法进一步对齐模型，无需训练奖励模型。

**前置条件**: 已完成 SFT (或 DPO)。

**默认参数:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 2 | RL 训练显存消耗极大 |
| `learning_rate` | 3e-7 | 极低学习率 |
| `epochs` | 1 | |
| `num_generations` | 6 | 每次生成 N 个回答比较 |
| `beta` | 0.1 | KL 惩罚系数 |
| `loss_type` | cispo | 损失函数类型 |
| `epsilon` | 0.2 | 策略裁剪范围 |
| `epsilon_high` | 5.0 | 高奖励裁剪范围 |
| `max_gen_len` | 1024 | 最大生成长度 |
| `thinking_ratio` | 0.9 | 思考预算比例 |

**完整命令：**

```bash
# 使用 Torch 引擎 (默认)
python trainer/train_grpo.py \
    --data_path dataset/rlaif.jsonl \
    --from_weight full_sft \
    --batch_size 2 \
    --num_generations 6 \
    --max_gen_len 1024 \
    --loss_type cispo \
    --beta 0.1 \
    --epsilon 0.2

# 使用 SGLang 加速推理
python trainer/train_grpo.py \
    --rollout_engine sglang \
    --sglang_base_url http://localhost:8998 \
    --sglang_model_path ./model
```

**GRPO 算法简介：**

对于每个 prompt，模型生成 N 个回答，使用组内相对排名计算优势值，然后用 CISPO 损失优化策略模型。

**Rollout Engine 配置：**

```yaml
rollout:
  engine: torch                          # 推理引擎 (torch/sglang)
  sglang_base_url: "http://localhost:8998"
  sglang_model_path: "../model"
  sglang_shared_path: "./sglang_ckpt"
  reward_model_path: "../../internlm2-1_8b-reward"
```

**输出：** `./out/grpo_768.pth`

### 4.8 Step 7: PPO 近端策略优化 [可选]

**目标**: 使用经典的 PPO 算法进行强化学习对齐。

**前置条件**: 已完成 SFT，需要奖励模型 (Reward Model)。

**默认参数** (与 GRPO 基本一致):

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 2 | |
| `learning_rate` | 3e-7 | |
| `epochs` | 1 | |
| `num_generations` | 6 | |
| `beta` | 0.1 | KL 惩罚系数 |

**完整命令：**

```bash
python trainer/train_ppo.py \
    --data_path dataset/rlaif.jsonl \
    --from_weight full_sft \
    --batch_size 2 \
    --num_generations 6 \
    --reward_model_path ../../internlm2-1_8b-reward
```

**PPO 与 GRPO 的区别：**

| 维度 | PPO | GRPO |
|------|-----|------|
| 奖励信号 | 需要显式奖励模型 | 组内相对比较 |
| 优势估计 | GAE (广义优势估计) | 组内标准化 |
| 稳定性 | 较高 (clip objective) | 较高 (CISPO loss) |
| 训练速度 | 较慢 (需奖励模型推理) | 较快 (无需奖励模型) |

**输出：** `./out/ppo_768.pth`

### 4.9 Step 8: Agent RL 工具调用 [可选]

**目标**: 训练模型使用外部工具 (API 调用、搜索、计算器等)。

**前置条件**: 已完成 SFT。

**默认参数:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 2 | |
| `learning_rate` | 3e-7 | |
| `epochs` | 1 | |
| `num_generations` | 4 | Agent 生成较少样本 |
| `max_gen_len` | 768 | |
| `max_total_len` | 2500 | 工具结果+对话总长度 |
| `thinking_ratio` | 0.1 | Agent 任务占用较少思考预算 |

**完整命令：**

```bash
python trainer/train_agent.py \
    --data_path dataset/agent_rl.jsonl \
    --from_weight full_sft \
    --batch_size 2 \
    --num_generations 4 \
    --max_total_len 2500 \
    --thinking_ratio 0.1
```

**Agent 数据格式特殊要求：**

每条数据需要 `gt` 字段 (ground truth)，用于验证工具调用结果是否正确。

**输出：** `./out/agent_768.pth`

### 4.10 Jupyter 训练监控面板

**启动 Notebook：**

```bash
cd notebooks
jupyter notebook training_monitor.ipynb
```

或在 VS Code 中直接打开 `notebooks/training_monitor.ipynb`。

**Notebook 包含 3 个 Cell：**

**Cell 1: 环境导入**

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from IPython.display import clear_output, display
import time

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

CSV_PATH = "../metrics.csv"
REFRESH_SEC = 5

print('✅ 已加载，运行下一个 Cell 开始监控')
```

**Cell 2: 绘图函数定义**

定义 `plot_metrics(csv_path)` 函数，功能包括：
- 自动识别所有数值列 (loss, lr, perplexity 等)
- 自动布局多子图
- 绘制原始数据线 + 红色滚动均值平滑线
- 打印最新指标

**Cell 3: 实时监控循环**

```python
print('🟢 开始实时监控...')
while True:
    plot_metrics(CSV_PATH)
    time.sleep(REFRESH_SEC)  # 每 5 秒刷新
```

**监控面板实际效果：**

生成包含以下子图的图表（以 `metrics.csv` 中的列动态生成）：

```
┌─────────────────────┐ ┌─────────────────────┐
│   loss over Steps   │ │    lr over Steps    │
│  ╭╮ 原始数据 (蓝色)  │ │  ╭╮ 原始数据 (蓝色)  │
│  ── 平滑均值 (红色)  │ │  ── 平滑均值 (红色)  │
└─────────────────────┘ └─────────────────────┘
┌─────────────────────┐ ┌─────────────────────┐
│ perplexity over Steps│ │    (其他指标...)     │
│  ╭╮ 原始数据 (蓝色)  │ │                     │
│  ── 平滑均值 (红色)  │ │                     │
└─────────────────────┘ └─────────────────────┘
📊 Step 200 | loss: 2.1906 | lr: 0.0000 | perplexity: 8.94 | (200 条记录)
⏱️  下次刷新: 5s 后
```

**配置说明：**

- `CSV_PATH`: metrics CSV 文件路径 (默认 `../metrics.csv`)
- `REFRESH_SEC`: 刷新间隔秒数 (默认 5 秒)
- 按 Jupyter 的 ■ 停止按钮可退出监控循环

**训练过程中 metrics.csv 的生成：**

每 `log_interval` 步，训练脚本会自动追加一行到 `metrics.csv`：

```csv
timestamp,step,epoch,loss,lr,perplexity
2026-05-31T15:00:01,1,0,8.5000,1.000e-03,4914.77
2026-05-31T15:00:02,2,0,8.3200,9.950e-04,4099.58
...
```

---

## 五、推理与服务部署

### 5.1 CLI 对话测试

训练完成后，测试模型效果：

```bash
# 交互式对话
python eval_llm.py --weight full_sft

# 指定模型参数
python eval_llm.py --weight full_sft --hidden_size 768 --num_layers 8

# 测试特定 prompt
python eval_llm.py --weight full_sft --prompt "请介绍一下深度学习"
```

### 5.2 OpenAI 兼容 API

启动兼容 OpenAI API 格式的 HTTP 服务：

```bash
# 启动 API 服务
python scripts/serve_openai_api.py --weight full_sft --port 8000

# 或
minimind-serve --weight full_sft --port 8000
```

使用任何 OpenAI SDK 调用：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="minimind",
    messages=[{"role": "user", "content": "你好！"}],
    temperature=0.7,
    max_tokens=256,
)
print(response.choices[0].message.content)
```

```bash
# curl 直接调用
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"minimind","messages":[{"role":"user","content":"你好"}]}'
```

### 5.3 Streamlit WebUI

启动 Web 聊天界面：

```bash
streamlit run scripts/web_demo.py -- --weight full_sft
```

浏览器访问 `http://localhost:8501` 即可使用可视化聊天界面。

### 5.4 LLM 评估

对模型进行全面评估：

```bash
python eval_llm.py --weight full_sft --eval_mode all
```

---

## 六、常见问题与故障排查

### 环境问题

**Q: `pip install -e ".[all]"` 提示构建失败？**
- 使用 `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
- 部分包 (如 scikit-learn) 需要 MSVC 编译器，使用清华镜像可获取预编译 wheel

**Q: CUDA out of memory？**
- 减小 `batch_size` (如 32 → 8)
- 增加 `accumulation_steps` (如 1 → 4)
- 减小 `max_seq_len` (如 768 → 340)
- 使用 `--dtype float16` 代替 `bfloat16`
- 使用 `--use_moe 0` 关闭 MoE

**Q: ROCm 环境下报错？**
- 确认 `torch.version.hip` 存在
- 使用 `--device cpu` 降级到 CPU 模式

### 训练问题

**Q: loss 不下降？**
- 检查 `learning_rate` 是否过大或过小
- 检查数据是否正确加载 (`log_interval` 调小查看数据)
- pretrain 阶段 loss 从 ~8.0 开始是正常的

**Q: 模型生成乱码？**
- 确认 tokenizer 路径正确 (`./model/tokenizer.json` 存在)
- 检查 `from_weight` 指向的权重文件存在

**Q: 断点续训怎么恢复？**
```bash
python trainer/train_pretrain.py --from_resume 1
```

### 权重文件路径映射

| from_weight | 期望文件 | 说明 |
|-------------|---------|------|
| `none` | (无) | 随机初始化 |
| `pretrain` | `./out/pretrain_768.pth` | 预训练权重 |
| `full_sft` | `./out/full_sft_768.pth` | SFT 权重 |
| `dpo` | `./out/dpo_768.pth` | DPO 权重 |
| `grpo` | `./out/grpo_768.pth` | GRPO 权重 |

### 常用命令速查表

```bash
# ▸ 环境检测
python scripts/auto_config.py --probe

# ▸ 智能配置生成
python scripts/auto_config.py --data_path dataset/pretrain_t2t_mini.jsonl --task pretrain --local

# ▸ 预训练
python trainer/train_pretrain.py --epochs 2 --batch_size 32

# ▸ SFT 微调
python trainer/train_full_sft.py --epochs 2 --batch_size 16

# ▸ LoRA 微调
python trainer/train_lora.py --epochs 10 --batch_size 32

# ▸ 蒸馏
python trainer/train_distillation.py --epochs 6

# ▸ DPO
python trainer/train_dpo.py --epochs 1 --batch_size 4

# ▸ GRPO
python trainer/train_grpo.py --epochs 1 --batch_size 2

# ▸ PPO
python trainer/train_ppo.py --epochs 1 --batch_size 2

# ▸ Agent RL
python trainer/train_agent.py --epochs 1 --batch_size 2

# ▸ 对话测试
python eval_llm.py --weight full_sft

# ▸ 启动 API
python scripts/serve_openai_api.py --weight full_sft --port 8000

# ▸ 启动 WebUI
streamlit run scripts/web_demo.py -- --weight full_sft

# ▸ Jupyter 监控
cd notebooks && jupyter notebook training_monitor.ipynb

# ▸ 运行测试
pytest tests/ -v

# ▸ 代码检查
ruff check .
```

---

> **文档版本**: 1.0
> **最后更新**: 2026-05-31
> **适用项目**: MiniMind-Lab v1.0.0
