# MiniMind-Lab 完整项目白皮书

> 本文档面向将要重构此项目的开发者。覆盖项目每一个文件、每一个类、每一个函数、数据流、架构决策、已知问题。

**总代码量**: ~6500行 Python + ~1500行 YAML/TOML/Markdown 配置
**核心模型**: 64M参数 Decoder-Only Transformer (Qwen3 风格)
**训练模式**: 8种 (Pretrain / SFT / LoRA / Distillation / DPO / GRPO / PPO / Agent RL)

---

## 目录

1. [项目总览](#一项目总览)
2. [完整目录结构](#二完整目录结构)
3. [模型架构详解](#三模型架构详解)
4. [Tokenizer](#四tokenizer)
5. [数据管线](#五数据管线)
6. [训练管线](#六训练管线)
7. [RL 基础设施](#七rl-基础设施)
8. [推理与服务](#八推理与服务)
9. [企业级基础设施](#九企业级基础设施)
10. [配置系统](#十配置系统)
11. [打包与CLI](#十一打包与cli)
12. [设计模式与代码约定](#十二设计模式与代码约定)
13. [完整依赖清单](#十三完整依赖清单)
14. [已知问题与技术债](#十四已知问题与技术债)
15. [重构建议](#十五重构建议)

---

## 一、项目总览

### 1.1 项目定位

MiniMind-Lab 是一个从零训练微型大语言模型 (LLM) 的完整实验平台。核心模型仅 **64M 参数**，2小时内可在消费级 GPU (RTX 3090 / RX 7900 XTX) 上完成完整训练，成本约3元人民币。

项目基于 [MiniMind](https://github.com/jingyaogong/minimind) 进行了深度企业级工程化改造。原项目是一个教学导向的单一代码库，本项目将其重构为工程化平台。

### 1.2 核心能力矩阵

```
训练:  Pretrain → SFT → LoRA → Distillation → DPO → GRPO → PPO → Agent RL
数据:  JSONL / JSON / CSV / Parquet / TXT + 列别名映射 + SimHash去重
GPU:   NVIDIA CUDA 12.x / AMD ROCm 6.x / CPU 自动降级
推理:  CLI / Streamlit WebUI / OpenAI兼容API / Ollama / vLLM
工程:  YAML配置 / Rich日志 / safetensors / 原子写入 / pre-commit / pytest / Docker
```

### 1.3 技术栈

| 层 | 技术 |
|----|------|
| 深度学习框架 | PyTorch 2.x + torch.compile |
| 模型后端 | HuggingFace Transformers (PreTrainedModel, GenerationMixin) |
| 数据处理 | HuggingFace Datasets + SimHash |
| RL算法 | DPO / GRPO / CISPO / PPO (GAE) |
| 推理加速 | Flash Attention / SGLang / vLLM / Ollama |
| 日志 | Python logging + RichHandler |
| 配置 | YAML + argparse (三层合并) |
| 测试 | pytest |
| 代码质量 | ruff + pre-commit |
| 容器化 | Docker 多阶段构建 |
| 可视化 | Jupyter Notebook / SwanLab(Wandb兼容) |

---

## 二、完整目录结构

```
MiniMind-Lab/                          # 项目根目录
│
├── config.yaml                        # 【核心】用户唯一配置入口。所有8种任务的全部参数带中文注释。
│                                        # 优先级: 此文件默认 → 用户YAML → CLI参数
│
├── pyproject.toml                     # Python项目打包配置
│                                        # [project]: name=minimind-lab, version=1.0.0
│                                        # [project.scripts]: 12个CLI入口点
│                                        # [project.optional-dependencies]: train/serve/eval/data/all
│                                        # [tool.ruff]: ruff配置
│
├── Makefile                           # 20+快捷命令 (训练/推理/质量/Docker)
├── Dockerfile                         # 多阶段: pytorch:2.6.0-cuda12.4 → dev/prod
├── .dockerignore                      # 排除 out/ checkpoints/ __pycache__/
├── .env.example                       # 环境变量模板 (MINIMIND_LOG_FILE/LEVEL等)
├── .pre-commit-config.yaml            # ruff + check-yaml/toml/ast + detect-private-key
├── requirements.txt                   # 完整依赖清单 (33行, 含版本号)
│
├── README.md                          # 项目README (240行, 快速开始+文档导航)
├── PROJECT.md                         # 【本文件】完整项目白皮书
│
├── docs/                              # 详细文档 (5个文件, 总计2850行)
│   ├── 1_数据准备与智能配置.md          # 数据下载/格式说明/DataLoaderHub/DataProcessor/auto_config
│   ├── 2_训练流程.md                   # 8种训练模式完整参数+原理+RL算法详解
│   ├── 3_企业级基础设施.md              # 日志/配置/Checkpoint/异常/CLI/测试/代码质量
│   ├── 4_部署指南.md                   # NVIDIA/AMD/CPU/Docker/Makefile/故障排查
│   └── 5_API参考.md                   # 所有模块完整API文档
│
├── notebooks/                         # Jupyter可视化
│   └── training_monitor.ipynb         # 训练监控面板: 读取metrics.csv → 实时绘制loss/lr曲线
│
├── model/                             # 模型定义
│   ├── __init__.py                    # 空文件
│   ├── model_minimind.py              # 【核心】完整模型定义 (~420行)
│   │   ├── MiniMindConfig             # 模型配置 (PretrainedConfig子类, 27个参数)
│   │   ├── RMSNorm                    # RMS归一化
│   │   ├── precompute_freqs_cis()     # RoPE频率预计算 (含YaRN缩放)
│   │   ├── apply_rotary_pos_emb()     # 应用旋转位置编码
│   │   ├── repeat_kv()                # GQA键值头重复
│   │   ├── Attention                  # 注意力层 (GQA, QK-Norm, FlashAttn)
│   │   ├── FeedForward                # SwiGLU前馈网络
│   │   ├── MOEFeedForward             # MoE前馈网络 (4专家, Top-1路由)
│   │   ├── MiniMindBlock              # Transformer块
│   │   ├── MiniMindModel              # 纯Transformer (不含LM head)
│   │   └── MiniMindForCausalLM        # 完整因果语言模型 (含generate方法)
│   ├── model_lora.py                  # LoRA实现 (~120行)
│   │   ├── LoRA                       # 低秩适配模块
│   │   ├── apply_lora()               # 注入LoRA (monkey-patch方式)
│   │   ├── load_lora() / save_lora()  # 加载/保存LoRA权重
│   │   └── merge_lora()               # 合并LoRA到基础权重
│   ├── tokenizer.json                 # BPE tokenizer (SentencePiece, vocab=6400)
│   └── tokenizer_config.json          # Tokenizer配置 (chat_template等)
│
├── dataset/                           # 数据管线
│   ├── __init__.py                    # 空文件
│   ├── data_loader.py                 # 【核心】多格式数据加载器 (~260行)
│   │   ├── detect_format()            # 扩展名+文件头检测格式
│   │   ├── expand_paths()             # glob展开+目录遍历
│   │   ├── normalize_schema()         # 列别名映射
│   │   ├── validate_schema()          # 必需字段校验
│   │   ├── DataLoaderHub              # 数据加载Hub (load/load_multi)
│   │   └── 5个便利函数                 # load_pretrain/sft/dpo/rlaif/agent_data()
│   ├── data_processor.py              # 【核心】数据处理管线 (~220行)
│   │   ├── auto_detect_schema()       # 自动识别数据类型
│   │   ├── validate_conversations()   # 对话格式校验
│   │   ├── clean_text()               # Unicode规范化+空白清理
│   │   ├── filter_by_length()         # 字符数过滤
│   │   ├── filter_by_token_length()   # Token数过滤
│   │   ├── deduplicate()              # SimHash近似去重
│   │   ├── describe_dataset()         # 统计报告生成
│   │   └── process_dataset()          # 一键全流程
│   ├── lm_dataset.py                  # 【核心】PyTorch Dataset类 (~360行)
│   │   ├── pre_processing_chat()      # 随机System Prompt注入
│   │   ├── post_processing_chat()     # 空think标签清理
│   │   ├── PretrainDataset            # 预训练: text→tokenize→bos/eos
│   │   ├── SFTDataset                 # SFT: conversations→chat_template→assistant-only labels
│   │   ├── DPODataset                 # DPO: chosen/rejected pairs
│   │   ├── RLAIFDataset               # RL: prompt+空answer, thinking toggle
│   │   └── AgentRLDataset             # Agent: messages+tools+gt
│   └── dataset.md                     # 数据集说明 (占位文件)
│
├── trainer/                           # 训练核心
│   ├── __init__.py                    # 空文件
│   │
│   ├── 【训练脚本 - 8个】
│   ├── train_pretrain.py              # 预训练 (~150行)
│   ├── train_full_sft.py              # 全量SFT (~150行, 含MetricsWriter)
│   ├── train_lora.py                  # LoRA微调 (~170行)
│   ├── train_distillation.py          # 知识蒸馏 (~190行, 双模型)
│   ├── train_dpo.py                   # DPO偏好对齐 (~180行, 双模型)
│   ├── train_grpo.py                  # GRPO强化学习 (~230行, Rollout+Reward)
│   ├── train_ppo.py                   # PPO强化学习 (~280行, Actor-Critic+GAE)
│   └── train_agent.py                 # Agent RL (~350行, 多轮工具调用)
│   │
│   ├── 【训练基础设施】
│   ├── training_loop.py               # 统一训练循环 (~240行)
│   │   ├── Accumulator                # 单优化器梯度累积
│   │   ├── MultiOptimizerAccumulator  # 双优化器梯度累积 (PPO)
│   │   ├── create_scheduler()         # CosineAnnealingLR工厂
│   │   ├── wrap_model()               # compile→DDP→rollout顺序包装
│   │   ├── save_model_weights()       # 统一权重保存
│   │   └── get_raw_model()            # 解包DDP/compile wrapper
│   │
│   ├── trainer_utils.py               # 训练工具函数 (~180行)
│   │   ├── get_model_params()         # 参数量统计 (含MoE分解)
│   │   ├── is_main_process()          # 分布式主进程判断
│   │   ├── Logger()                   # 向后兼容日志shim
│   │   ├── get_lr()                   # 预热+余弦衰减公式
│   │   ├── init_distributed_mode()    # DDP初始化
│   │   ├── setup_seed()               # 随机种子 (含ROCm守卫)
│   │   ├── lm_checkpoint()            # 旧版checkpoint (向后兼容)
│   │   ├── init_model()               # 模型初始化+权重加载
│   │   ├── SkipBatchSampler           # 续训批次跳过
│   │   └── LMForRewardModel           # 奖励模型封装
│   │
│   ├── rollout_engine.py              # RL生成引擎 (~200行)
│   │   ├── RolloutResult              # 生成结果dataclass
│   │   ├── compute_per_token_logps()  # Per-token log概率计算
│   │   ├── RolloutEngine (ABC)        # 抽象基类
│   │   ├── TorchRolloutEngine         # PyTorch原生生成
│   │   ├── SGLangRolloutEngine        # SGLang高性能生成 (HTTP)
│   │   └── create_rollout_engine()    # 工厂函数
│   │
│   ├── train_tokenizer.py             # Tokenizer训练脚本 (参考用, ~200行)
│   │
│   ├── 【企业级基础设施 - 6个模块】
│   ├── logger.py                      # 结构化日志 (~110行)
│   │   ├── get_logger()               # 获取minimind层级logger
│   │   ├── set_level()                # 运行时级别切换
│   │   └── Logger()                   # 向后兼容shim
│   │
│   ├── config.py                      # 配置系统 (~320行)
│   │   ├── TrainingConfig             # 60+字段dataclass
│   │   ├── load_config()              # YAML→用户YAML→CLI三层合并
│   │   ├── create_arg_parser()        # 向后兼容argparse
│   │   └── 内部函数: _load_yaml, _dict_to_args, _merge_namespace, _namespace_to_dataclass
│   │
│   ├── config_defaults.yaml           # 内置默认配置 (~280行)
│   │   └── 8个section: pretrain/full_sft/lora/distillation/dpo/grpo/ppo/agent
│   │
│   ├── checkpoint.py                  # Checkpoint管理器 (~220行)
│   │   ├── LMCheckpointer             # safetensors优先/原子写入/轮转
│   │   └── 内部: _get_raw_model, _config_hash
│   │
│   ├── exceptions.py                  # 异常体系 (~25行)
│   │   └── MiniMindError→ConfigError/CheckpointError/DataLoadError/ModelBuildError/TrainingError
│   │
│   ├── metrics.py                     # 训练指标CSV导出 (~70行)
│   │   └── MetricsWriter              # 线程安全CSV写入, 自动header
│   │
│   └── cli.py                         # CLI入口点分发器 (~80行)
│       └── 12个入口函数 (pretrain/full_sft/lora/distillation/dpo/grpo/ppo/agent/eval/serve/chat/auto_config)
│
├── scripts/                           # 推理/服务/工具脚本
│   ├── auto_config.py                 # 智能配置工具 (~300行)
│   │   ├── hardware_probe()           # GPU/CPU/RAM/磁盘探测
│   │   ├── analyze_dataset()          # 数据集统计分析
│   │   ├── estimate_memory()          # 显存需求估算
│   │   ├── build_prompt()             # LLM prompt构建
│   │   ├── get_config_from_llm()      # OpenAI API调用
│   │   ├── format_training_command()  # 训练命令生成
│   │   └── apply_config()             # JSON输出+命令打印
│   │
│   ├── serve_openai_api.py            # OpenAI兼容API服务 (~300行)
│   │   ├── ChatRequest                # Pydantic请求模型
│   │   ├── CustomStreamer             # 流式输出队列
│   │   ├── parse_response()           # think/tool_call解析
│   │   ├── generate_stream_response() # SSE流式生成
│   │   └── /v1/chat/completions       # FastAPI路由
│   │
│   ├── web_demo.py                    # Streamlit WebUI (~400行)
│   │   ├── 8个Mock工具定义             # calculate_math/get_weather等
│   │   ├── execute_tool()             # 工具模拟执行
│   │   ├── process_assistant_content()# HTML渲染(think折叠/tool_call样式)
│   │   └── 多轮工具调用循环
│   │
│   ├── chat_api.py                    # 交互式API对话客户端 (~60行)
│   ├── convert_model.py               # 模型格式转换 (~200行)
│   │   ├── torch↔transformers互转
│   │   ├── LoRA合并
│   │   └── Jinja/JSON chat_template转换
│   └── eval_toolcall.py               # 工具调用评测 (~300行)
│
├── tests/                             # 自动化测试
│   ├── __init__.py                    # 空文件
│   ├── test_config.py                 # 配置系统测试 (7个测试用例)
│   ├── test_logger.py                 # 日志系统测试 (4个测试用例)
│   └── test_model.py                  # 模型测试 (5个测试用例, 含forward+MoE)
│
├── eval_llm.py                        # 【根级】模型推理入口 (~140行)
│                                      # 支持自动测试(8个prompt)/手动对话/历史对话/LoRA
│
├── CODE_OF_CONDUCT.md                 # 贡献者行为准则
└── dataset/                           # (运行时) 数据集存放目录
```

---

## 三、模型架构详解

### 3.1 总体架构

MiniMind 是一个 **Decoder-Only Transformer**，遵循 Qwen3/LLaMA 风格架构：

```
输入: token IDs (batch, seq_len)
  ↓
Embedding: nn.Embedding(6400, 768) → (batch, seq_len, 768)
  ↓
Dropout (rate=0.0, 实际不生效)
  ↓
×8 MiniMindBlock:
  ├── RMSNorm → Attention (GQA, QK-Norm, RoPE, FlashAttn) → Residual(+)
  └── RMSNorm → FeedForward(SwiGLU) or MOEFeedForward → Residual(+)
  ↓
RMSNorm (final)
  ↓
LM Head: nn.Linear(768, 6400, bias=False)  [权重与Embedding共享: tie_word_embeddings=True]
  ↓
输出: logits (batch, seq_len, 6400)
```

### 3.2 关键架构细节

#### 3.2.1 Grouped Query Attention (GQA)

- **8个Query头** + **4个KV头** = 2:1 重复比例
- 每头维度 = 768/8 = **96**
- KV头通过 `repeat_kv()` 复制以匹配Q头数量

#### 3.2.2 QK Normalization (QK-Norm)

在应用RoPE之前，对Q和K分别做RMSNorm归一化。这是Qwen3的关键创新之一，提升训练稳定性。

#### 3.2.3 Rotary Position Embedding (RoPE)

- Base frequency: **1,000,000** (1e6)
- 预计算到 **32,768** 位置
- 支持 **YaRN** 外推 (当 `inference_rope_scaling=True`):
  - beta_fast=32, beta_slow=1
  - factor=16
  - original_max_position_embeddings=2048

#### 3.2.4 SwiGLU FeedForward

```
gate_proj: 768 → intermediate_size (≈2304, 由 ceil(768*π/64)*64 计算)
up_proj:   768 → intermediate_size
down_proj: intermediate_size → 768
act_fn: SiLU

output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
```

`intermediate_size` 的计算公式 `math.ceil(hidden_size * math.pi / 64) * 64` 产生约3倍的扩展比 (768×π≈2413, align to 64 = 2304)。

#### 3.2.5 MoE 混合专家

当 `use_moe=True` 时，每个Transformer块的FFN替换为 `MOEFeedForward`:

```
Router Gate: 768 → 4 (softmax over experts)
4个独立Expert: 每个是完整的SwiGLU FeedForward
Top-1路由: 每个token只经过1个expert

辅助Loss: aux_loss = load_balance_loss × 5e-4
  load = one_hot(topk_idx).float().mean(0)   # 每个expert的token比例
  aux_loss = (load * router_scores.mean(0)).sum() * num_experts * router_aux_loss_coef
```

MoE参数量约为基础模型的1.8倍 (4个expert但每个token只用1个)。

#### 3.2.6 模型规格

| 配置 | hidden_size=768, num_layers=8 | hidden_size=512, num_layers=4 |
|------|------|------|
| 参数量 (非MoE) | ~64M | ~25M |
| 参数量 (MoE) | ~115M | ~45M |
| 词表大小 | 6400 | 6400 |
| 最大位置 | 32768 | 32768 |
| 训练序列 (Pretrain) | 340-768 | 340-512 |
| 训练序列 (SFT) | 768 | 512 |

### 3.3 MiniMindConfig 完整参数

```python
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    # 基础架构
    hidden_size: int = 768             # 隐藏层维度
    num_hidden_layers: int = 8         # Transformer层数
    intermediate_size: int             # FFN中间维度 (自动计算: ceil(hidden*π/64)*64)
    num_attention_heads: int = 8       # Q头数
    num_key_value_heads: int = 4       # KV头数 (GQA)
    head_dim: int                      # 每头维度 (自动: hidden_size // num_attention_heads)

    # 词表
    vocab_size: int = 6400
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = True   # 共享embedding和lm_head权重

    # 位置编码
    rope_theta: float = 1_000_000.0    # RoPE base frequency
    max_position_embeddings: int = 32768
    inference_rope_scaling: bool = False  # 启用YaRN外推
    rope_scaling: Optional[dict] = None   # YaRN参数 (自动配置)

    # 注意力
    flash_attn: bool = True            # 优先使用Flash Attention
    dropout: float = 0.0               # Dropout率 (训练时0.0)

    # 激活与归一化
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6

    # MoE (仅 use_moe=True 时生效)
    use_moe: bool = False
    num_experts: int = 4
    num_experts_per_tok: int = 1       # Top-1路由
    moe_intermediate_size: int         # Expert内部维度 (默认同intermediate_size)
    norm_topk_prob: bool = True        # 归一化top-k概率
    router_aux_loss_coef: float = 5e-4 # 负载均衡loss系数
```

### 3.4 LoRA 实现

LoRA通过**monkey-patching**方式注入，而非使用HuggingFace PEFT库:

```python
class LoRA(nn.Module):
    """低秩适配: output = B @ A @ x, 其中 A∈R^(in×rank), B∈R^(rank×out)"""
    def __init__(self, in_features, out_features, rank):
        self.A = nn.Linear(in_features, rank, bias=False)    # Kaiming初始化
        self.B = nn.Linear(rank, out_features, bias=False)   # 零初始化

def apply_lora(model, rank=16):
    """对所有 in_features == out_features 的 nn.Linear 层注入LoRA
       即: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head, gate(router)
       注意: k_proj和v_proj不是方阵 (768→384), 所以不会被注入!"""
```

**已知限制**: Monkey-patch方式与 `torch.compile` 不兼容，因此LoRA训练脚本自动禁用compile。

---

## 四、Tokenizer

### 4.1 基本信息

- **类型**: BPE (Byte-Pair Encoding), SentencePiece后端
- **词表大小**: 6400
- **文件**: `model/tokenizer.json` (tokenizer) + `model/tokenizer_config.json` (配置)

### 4.2 特殊Token

| Token ID | Token String | 用途 |
|----------|-------------|------|
| 0 | `<\|endoftext\|>` | pad_token, unk_token |
| 1 | `<\|im_start\|>` | bos_token, 对话开始标记 |
| 2 | `<\|im_end\|>` | eos_token, 对话结束标记 |
| 3-16 | `<\|object_ref_start\|>`, `<\|box_start\|>`, `<\|vision_start\|>`, `<\|audio_start\|>` 等 | 多模态标记 (视觉/音频) |
| 17-20 | `<tts_pad>`, `<tts_text_bos>` 等 | TTS标记 |
| 21-26 | `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`, `<think>`, `</think>` | 工具调用和思考标记 |
| 27-35 | `<\|buffer1\|>` ~ `<\|buffer9\|>` | 缓冲标记 |

### 4.3 Chat Template

Tokenizer包含完整的Jinja2 chat template，支持:
- **System Prompt**: `role: "system"` → `<|im_start|>system\n...<|im_end|>`
- **对话轮次**: `role: "user"/"assistant"` → `<|im_start|>user\n...<|im_end|>`
- **工具调用**: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
- **工具返回**: `<tool_response>...</tool_response>`
- **思考模式**: `<think>...</think>` (通过 `open_thinking` 参数控制)
- **多工具支持**: 支持并行工具调用

---

## 五、数据管线

### 5.1 数据加载流程

```
原始文件 (JSONL/JSON/CSV/Parquet/TXT)
  ↓ expand_paths()         ← glob展开, 目录遍历, 扩展名过滤
  ↓ detect_format()        ← 扩展名检测 + 文件头魔数检测
  ↓ _load_single()         ← datasets.load_dataset()
  ↓ normalize_schema()     ← 列别名映射 (content→text, messages→conversations等)
  ↓ validate_schema()      ← 必需字段检查, 缺失行过滤
  ↓ concatenate_datasets() ← 多文件合并
  ↓ (可选) process_fn       ← 自定义处理
  ↓ shuffle()              ← 随机打乱
  → HuggingFace Dataset
```

### 5.2 列别名映射表

DataLoaderHub内置的完整列名映射 (在 `SCHEMA_ALIASES` 中定义):

| 规范列 | 自动识别的源列名 |
|--------|----------------|
| `text` | `content`, `document`, `passage`, `sentence`, `body`, `article`, `paragraph` |
| `conversations` | `messages`, `dialog`, `dialogue`, `chat`, `conversation`, `history`, `turns` |
| `chosen` | `preferred`, `positive`, `good`, `best` |
| `rejected` | `dispreferred`, `negative`, `bad`, `worst` |
| `gt` | `ground_truth`, `answer`, `label`, `target` |

### 5.3 数据处理管线

```
process_dataset() 全流程 (按顺序):
  1. clean_text()           ← Unicode NFKC, 控制字符移除, 空白合并
  2. validate_conversations() ← role/content检查
  3. filter_by_length()     ← 字符数范围过滤
  4. filter_by_token_length() ← Token数范围过滤 (需tokenizer)
  5. deduplicate()          ← SimHash近似去重 (汉明距离阈值)
```

### 5.4 五种Dataset类

#### PretrainDataset
- **输入**: `{'text': '...'}`
- **处理**: `bos + text + eos` → tokenize → 右padding
- **输出**: `(input_ids, labels)` — labels中pad位置为-100

#### SFTDataset
- **输入**: `{'conversations': [{role, content}, ...]}`
- **处理**: 随机注入system prompt → apply_chat_template → tokenize → generate_labels
- **Labels**: 只对assistant回复计算loss (user/system部分mask为-100)
- **关键**: 通过查找 `<|im_start|>assistant\n` 和 `<|im_end|>` 的token序列来定位assistant位置

#### DPODataset
- **输入**: `{'conversations': [...], 'chosen': '...', 'rejected': '...'}`
- **处理**: 分别构建chosen和rejected的完整对话 → tokenize → 生成mask
- **输出**: `(x_chosen, y_chosen, mask_chosen, x_rejected, y_rejected, mask_rejected)`

#### RLAIFDataset
- **输入**: `{'conversations': [{role, content}, ...]}`
- **处理**: 去掉最后一条消息 → 添加generation_prompt → 按thinking_ratio随机启用thinking
- **输出**: `{'prompt': '...', 'answer': ''}` (answer为空, 由RL rollout生成)

#### AgentRLDataset
- **输入**: `{'conversations': [...], 'gt': '...'}`
- **处理**: 去掉最后一条消息 → 提取tools定义 → 保留gt
- **输出**: `{'messages': [...], 'tools': [...], 'gt': '...'}`

---

## 六、训练管线

### 6.1 所有训练脚本的共同结构

8个训练脚本遵循完全相同的10步结构:

```
1. init_distributed_mode() + setup_seed(42)
2. 解析参数 → 创建 MiniMindConfig → 建目录
3. 设置混合精度 autocast (bfloat16/float16)
4. 初始化 Wandb/SwanLab + (仅full_sft) MetricsWriter
5. init_model() + Dataset + DistributedSampler + AdamW
6. create_scheduler() + Accumulator/MultiOptimizerAccumulator
7. 可选的断点续训恢复 (model/optimizer/scaler/scheduler/epoch/step)
8. wrap_model() (compile → DDP → rollout_engine.update)
9. Epoch循环: train_epoch() → 每log_interval打印指标 → 每save_interval保存
10. dist.destroy_process_group()
```

### 6.2 训练模式参数速查

| 模式 | 文件 | Batch | LR | Epochs | 特殊参数 |
|------|------|-------|-----|--------|---------|
| Pretrain | train_pretrain.py | 32-64 | 5e-4 | 2-3 | from_weight="none" |
| Full SFT | train_full_sft.py | 16-32 | 1e-5 | 2-3 | from_weight="pretrain" |
| LoRA | train_lora.py | 32 | 1e-4 | 10 | lora_name, 仅训练LoRA参数 |
| Distillation | train_distillation.py | 32 | 5e-6 | 6 | alpha=0.5, temperature=1.5, 双模型 |
| DPO | train_dpo.py | 4 | 4e-8 | 1 | beta=0.15, 参考模型 |
| GRPO | train_grpo.py | 2 | 3e-7 | 1 | num_generations=6, beta=0.1, rollout+reward |
| PPO | train_ppo.py | 2 | 3e-7 | 1 | Actor+Critic双优化器, GAE |
| Agent RL | train_agent.py | 2 | 3e-7 | 1 | 多轮工具调用, max_total_len=2500 |

### 6.3 Accumulator 梯度累积机制

```python
class Accumulator:
    def backward(self, loss):
        scaled_loss = loss / accumulation_steps
        scaler.scale(scaled_loss).backward()  # 或 loss.backward()
        counter += 1
        if counter % accumulation_steps == 0:
            _step()  # 梯度裁剪 + optimizer.step() + scheduler.step() + zero_grad

    @property
    def loss_value(self):
        return last_loss * accumulation_steps  # 补偿累积缩放
```

关键细节:
- `loss_value` 属性返回 `last_loss * accumulation_steps` — 确保打印的loss值不受累积步数影响
- `finalize()` 在epoch结束时排空剩余梯度 (处理不整除情况)
- 支持混合精度: float16用GradScaler, bfloat16不用

### 6.4 分布式训练 (DDP)

```python
def wrap_model(model, use_compile, local_rank, rollout_engine):
    if use_compile:
        model = torch.compile(model)                    # 1. 先compile
    if dist.is_initialized():
        model = DistributedDataParallel(model, ...)     # 2. 再DDP
    if rollout_engine:
        rollout_engine.update_policy(model)             # 3. 最后更新rollout
    return model
```

编译→DDP的顺序至关重要 (compile必须先于DDP)。

### 6.5 断点续训

`lm_checkpoint()` (旧版, trainer_utils.py中):
- 保存: `{checkpoint_dir}/{weight}_{hidden_size}[_moe]_resume.pth`
- 内容: model state_dict, optimizer, scaler, scheduler, epoch, step, world_size, wandb_id
- 跨GPU恢复: 检测world_size变化, 自动调整step计数

`LMCheckpointer` (新版, checkpoint.py中):
- 保存: safetensors格式 + 元数据JSON + 优化器状态
- 自动轮转: max_keep=3, 超量时删除最旧的
- 原子写入: .tmp → os.replace

### 6.6 CSV指标输出 (仅train_full_sft.py)

```python
from trainer.metrics import MetricsWriter
writer = MetricsWriter("./metrics.csv")
# 每次log_interval:
writer.write(step=step, loss=loss, logits_loss=logits_loss, aux_loss=aux_loss, lr=lr)
```

Jupyter notebook `notebooks/training_monitor.ipynb` 读取此CSV并实时绘图。

---

## 七、RL 基础设施

### 7.1 Rollout Engine

两种引擎可选:

**TorchRolloutEngine** (默认):
- 直接调用 `model.generate()` 生成response
- 计算per-token log probabilities
- 慢但无需额外服务

**SGLangRolloutEngine** (高性能):
- 通过HTTP调用外部SGLang服务器
- `/generate` 生成response (含logprobs)
- `/update_weights_from_disk` 同步模型权重
- `/flush_cache` 清除KV缓存
- 需要独立部署SGLang服务

### 7.2 奖励模型

```python
class LMForRewardModel:
    """封装internlm2-1_8b-reward作为奖励模型"""
    def get_score(self, messages, response):
        # 拼接历史+最后一轮query+response → 模型打分 → clip到[-3, 3]
```

### 7.3 RL算法对比

| | DPO | GRPO | CISPO | PPO | Agent RL |
|------|-----|------|-------|-----|---------|
| 需要奖励模型 | ❌ | ✅ | ✅ | ✅ | ✅ |
| 需要Critic | ❌ | ❌ | ❌ | ✅ | ❌ |
| 每prompt生成数 | 0 (离线) | 6 | 6 | 1 | 4 |
| 优势估计 | — | 组内归一化 | 组内归一化 | GAE | 组内归一化 |
| KL惩罚 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Clip方式 | — | 对称ε | 不对称(ε_low, ε_high) | 对称ε | 对称ε |

### 7.4 DPO损失

```python
def dpo_loss(ref_logps, policy_logps, mask, beta):
    # policy: log-softmax of training model
    # ref: log-softmax of frozen reference model
    pi_logratios = chosen_policy - reject_policy
    ref_logratios = chosen_ref - reject_ref
    loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
    return loss.mean()
```

### 7.5 GRPO/CISPO 损失

```python
# 组内相对优势
advantages = (rewards - rewards.group_mean()) / (rewards.group_std() + 1e-4)

# GRPO: 对称clip
ratio = exp(policy_logps - old_logps)
clipped = clamp(ratio, 1-epsilon, 1+epsilon)
loss = -min(ratio * advantages, clipped * advantages) + beta * KL

# CISPO: 不对称clamp + detach
clamped_ratio = clamp(ratio, 1-epsilon_low, 1+epsilon_high)
loss = -(clamped_ratio.detach() * advantages * policy_logps) + beta * KL
```

### 7.6 PPO with GAE

```python
# GAE: 逆向累积
for t in reversed(range(T)):
    delta = rewards[t] + gamma * next_values[t] - values[t]
    gae = delta + gamma * lam * next_gae
    advantages[t] = gae
    returns[t] = advantages[t] + values[t]

# PPO update (inner loop):
for _ in range(ppo_update_iters):
    for mini_batch in batches:
        # Policy loss (clipped)
        ratio = exp(new_logps - old_logps)
        policy_loss = -min(ratio*adv, clamp(ratio, 1-ε, 1+ε)*adv)

        # Value loss (clipped)
        value_loss = 0.5 * mean(max((v-return)^2, (v_clipped-return)^2))

        # Total
        total = policy_loss + vf_coef * value_loss + kl_coef * KL
```

PPO是唯一需要Critic (Value)模型的算法, 使用 `MultiOptimizerAccumulator` 管理actor和critic双优化器。

### 7.7 Agent RL 工具调用流程

```
1. 解析prompt中的tools定义
2. 第1轮: 模型生成 → 解析 <tool_call>...</tool_call>
3. 如果有tool_call:
   a. 执行mock工具 (MOCK_RESULTS)
   b. 拼装 <tool_response>...</tool_response>
   c. 追加到messages (工具返回不参与loss计算, mask=0)
   d. 返回步骤2继续生成 (最多3轮)
4. 计算奖励:
   - 无工具调用: 格式分 + 奖励模型分
   - 有工具调用: 工具对齐分 + GT验证分
   - 所有分clip到[-3, 3]
```

6个Mock工具: calculate_math, unit_converter, get_current_weather, get_current_time, get_exchange_rate, translate_text

---

## 八、推理与服务

### 8.1 eval_llm.py (根级CLI推理)

```
参数:
  --load_from (默认'model')   'model'=加载本地权重, 否则=HF模型路径
  --weight (默认'full_sft')   本地权重名
  --lora_weight               可选的LoRA权重
  --hidden_size (默认768)
  --num_hidden_layers (默认8)
  --use_moe (默认0)
  --inference_rope_scaling    启用YaRN外推
  --max_new_tokens (默认8192)
  --temperature (默认0.85)
  --top_p (默认0.95)
  --open_thinking (默认0)     启用思考模式
  --historys (默认0)          历史对话轮数 (必须偶数)
  --show_speed (默认1)        显示生成速度
  --device

两种模式:
  模式0: 自动测试8个预设中文prompt
  模式1: 手动输入交互对话
```

### 8.2 serve_openai_api.py (OpenAI兼容API)

FastAPI服务, 端口8998:

- **POST /v1/chat/completions**: 标准OpenAI chat completions接口
  - 支持流式 (SSE) 和非流式
  - 支持 `reasoning_content` (从 `<think>` 标签解析)
  - 支持 `tool_calls` (从 `<tool_call>` 标签解析)
  - 通过 `chat_template_kwargs.open_thinking` 控制思考模式

### 8.3 web_demo.py (Streamlit WebUI)

- 自动扫描 `scripts/` 目录下的模型文件夹
- 支持8种工具的开关选择
- 多轮工具调用 (最多16次迭代)
- 思考过程折叠显示 (HTML `<details>`)
- 工具调用样式化显示 (彩色HTML div)
- 中英文双语界面

### 8.4 eval_toolcall.py (工具调用评测)

- 8个预定义测试用例
- 支持local模型和API两种后端
- 多轮工具调用循环
- 自动校验工具调用格式
- Mock工具执行

### 8.5 convert_model.py (模型格式转换)

支持的转换:
- `torch → transformers (MiniMind原生)`: 保存为MiniMindConfig格式
- `torch → transformers (Qwen3兼容)`: 保存为Qwen3Config/Qwen3MoeConfig格式
- `transformers → torch`: 提取state_dict
- `base + lora → merged`: 合并LoRA到基础权重
- `jinja → json / json → jinja`: Chat template格式互转

---

## 九、企业级基础设施

### 9.1 日志系统 (`trainer/logger.py`)

```
minimind (root logger)
  ├── StreamHandler → RichHandler (彩色终端, traceback美化)
  ├── FileHandler → (可选, MINIMIND_LOG_FILE环境变量)
  └── Logger() shim → 委托到 log.info()
```

**环境变量控制**:
- `MINIMIND_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR` (默认INFO)
- `MINIMIND_LOG_FILE=path/to/log` (不设则不写文件)

**向后兼容**: `trainer_utils.py` 中的 `Logger()` 函数已改为调用 `trainer.logger.Logger()`, 所有旧代码无需修改。

### 9.2 配置系统 (`trainer/config.py` + `config_defaults.yaml` + `config.yaml`)

三层合并策略:

```
Layer 1: config_defaults.yaml (内置, 8种任务的默认参数)
    ↓ 覆盖
Layer 2: 用户自定义YAML (--config my.yaml)
    ↓ 覆盖
Layer 3: CLI参数 (--epochs 5 --batch_size 64)
    ↓
最终: TrainingConfig dataclass (60+字段)
```

**关键API**:
```python
load_config("full_sft", sys.argv[1:], user_yaml="my.yaml")  # 三层合并
create_arg_parser("full_sft")  # 向后兼容的argparse
config.auto_device()           # 惰性设备检测
```

### 9.3 Checkpoint管理器 (`trainer/checkpoint.py`)

```
LMCheckpointer:
  save() → 原子写入:
    {weight_name}.safetensors   (或 .pth)
    {weight_name}_meta.json     (epoch/step/config_hash/timestamp)
    {weight_name}_optim.pth     (优化器状态)
    (可选) {weight_name}_lora.pth

  load() → 自动尝试 .safetensors → .pth
  load_metadata() → 只读元数据, 不加载权重
  list_checkpoints() → 按时间倒序列出所有checkpoint
  _rotate() → 自动清理超出max_keep的旧checkpoint
```

**对比旧版**:

| 特性 | 旧 `lm_checkpoint()` | 新 `LMCheckpointer` |
|------|---------------------|---------------------|
| 格式 | .pth (pickle) | .safetensors (安全) / .pth |
| 元数据 | 无 | JSON (epoch/step/hash/time) |
| 轮转 | 手动 | 自动 (max_keep) |
| 跨格式 | 否 | 自动尝试 |

### 9.4 异常体系 (`trainer/exceptions.py`)

```
MiniMindError (基类)
  ├── ConfigError         — 配置错误
  ├── CheckpointError     — 检查点错误
  ├── DataLoadError       — 数据加载错误
  ├── ModelBuildError     — 模型构建错误
  └── TrainingError       — 运行时训练错误
```

目前仅定义了异常类层次结构。训练脚本中大多数仍使用裸 `except:` 或 `except Exception:`, 尚未大范围迁移到自定义异常。

### 9.5 测试 (`tests/`)

3个测试文件, 16个测试用例, 全部CPU运行:

- `test_config.py` (7 tests): load_config各预设, CLI覆盖, 未知任务回退
- `test_logger.py` (4 tests): logger创建, 文件输出, 级别切换, Logger shim
- `test_model.py` (5 tests): Config创建, 模型构建, forward pass, MoE aux_loss

### 9.6 代码质量

- **ruff**: 配置在pyproject.toml中 (`E,F,W,I,N,UP,B,C4`)
- **pre-commit**: ruff + check-yaml/toml/ast + detect-private-key + end-of-file-fixer

---

## 十、配置系统

### 10.1 三层合并逻辑

```python
def load_config(defaults="full_sft", cli_args=None, user_yaml=None) -> TrainingConfig:
    # 1. 加载内置config_defaults.yaml, 取defaults对应的section
    builtin = _load_yaml("trainer/config_defaults.yaml")
    ns = _dict_to_args(builtin[defaults])

    # 2. 叠加用户YAML
    if user_yaml:
        user_ns = _dict_to_args(_load_yaml(user_yaml))
        ns = _merge_namespace(ns, user_ns)

    # 3. 叠加CLI参数 (--config参数在这里被消费)
    if cli_args:
        parser = _build_arg_parser(defaults)
        cli_ns, _ = parser.parse_known_args(cli_args)
        ns = _merge_namespace(ns, cli_ns)

    return _namespace_to_dataclass(ns, TrainingConfig)
```

### 10.2 根config.yaml结构

```yaml
global:             # 全局默认 (hidden_size, num_layers, dtype, device...)
paths:              # 路径配置 (data_dir, save_dir, checkpoint_dir...)
checkpoint:         # Checkpoint配置 (max_keep, format, save_interval)
logging:            # 日志配置 (log_interval, use_wandb, metrics_csv)

# 8个任务专属section:
pretrain:           # 预训练参数
full_sft:           # SFT参数
lora:               # LoRA参数
distillation:       # 蒸馏参数 (含alpha, temperature)
dpo:                # DPO参数 (含beta)
grpo:               # GRPO参数 (含num_generations, beta, epsilon等)
ppo:                # PPO参数
agent:              # Agent RL参数

rollout:            # Rollout Engine配置 (sglang地址等)
advanced:           # 高级配置 (compile, debug, resume)
```

---

## 十一、打包与CLI

### 11.1 pyproject.toml

```toml
[project]
name = "minimind-lab"
version = "1.0.0"
requires-python = ">=3.10"

[project.optional-dependencies]
train = ["peft", "trl", "wandb", "swanlab", "pyyaml", "python-dotenv"]
serve = ["flask", "flask-cors", "openai", "streamlit"]
eval = ["jieba", "jsonlines", "nltk", "scikit-learn", "sentence-transformers"]
data = ["datasketch", "marshmallow", "simhash", "ujson", "modelscope"]

[project.scripts]
minimind-pretrain = "trainer.cli:pretrain"
minimind-full-sft = "trainer.cli:full_sft"
... (共12个入口点)
```

### 11.2 CLI入口点实现

`trainer/cli.py` 使用 `runpy.run_module()` 运行原有训练脚本, 不修改脚本本身:

```python
def _run(module, args=None):
    if args is not None:
        saved = sys.argv[:]
        sys.argv[1:] = args
    try:
        runpy.run_module(module, run_name="__main__")
    finally:
        if saved is not None:
            sys.argv[:] = saved
```

### 11.3 Makefile

20+目标, 通过 `ARGS` 变量透传参数:

```makefile
pretrain:
	python trainer/train_pretrain.py $(ARGS)

full-sft:
	python trainer/train_full_sft.py $(ARGS)
# ... 其他训练目标类似

lint:  ruff check .
test:  pytest tests/ -v
clean: 清理缓存
docker-build / docker-run: Docker操作
```

---

## 十二、设计模式与代码约定

### 12.1 现有模式

**Monkey-Patching (LoRA)**: LoRA不是通过标准PEFT接口注入, 而是直接替换 `nn.Linear.forward` 方法。简单但导致与 `torch.compile` 不兼容。

**sys.path 操作**: 所有训练脚本在开头进行 `sys.path.append(os.path.join(...))` 来保证导入路径。这是 `pip install -e .` 之前的遗留模式。现在有pyproject.toml后已不需要, 但保留以兼容直接 `python trainer/train_xxx.py` 的运行方式。

**全局 args 变量**: 8个训练脚本中, `args` 是全局变量, `train_epoch()` 等函数通过闭包访问。这是从原始MiniMind继承的模式。

**Dual Logger系统**: 两个Logger并存 — `trainer.logger` (新) 和 `trainer_utils.Logger` (旧, 向后兼容shim)。

**Dual Checkpoint系统**: 两个checkpoint系统并存 — `lm_checkpoint()` (旧, trainer_utils.py) 和 `LMCheckpointer` (新, checkpoint.py)。

### 12.2 命名约定

- 训练脚本: `train_<mode>.py` (小写+下划线)
- 模型文件: `model_<variant>.py`
- 数据集类: `<Mode>Dataset` (PascalCase)
- 工具函数: `snake_case()`
- 私有函数: `_leading_underscore()`

### 12.3 HuggingFace集成

- `MiniMindConfig` 继承 `PretrainedConfig`
- `MiniMindForCausalLM` 继承 `PreTrainedModel, GenerationMixin`
- 使用 `transformers` 的 `AutoTokenizer`, `AutoModel`
- 使用 `datasets` 库的 `Dataset`, `load_dataset`
- `generate()` 方法自定义实现 (不用transformers的generate, 而是手写采样循环)

---

## 十三、完整依赖清单

### 13.1 核心依赖 (pyproject.toml)

```
torch>=2.1             # 深度学习框架
transformers>=4.40     # HuggingFace模型库
datasets>=3.0          # HuggingFace数据集
einops>=0.8            # 张量操作
numpy>=1.26            # 数值计算
rich>=13.7             # 终端美化
tiktoken>=0.10         # Token计数
jinja2>=3.1            # Chat template
```

### 13.2 可选依赖

```
训练: peft, trl, wandb, swanlab, pyyaml, python-dotenv
推理: flask, flask-cors, openai, streamlit
评估: jieba, jsonlines, nltk, scikit-learn, sentence-transformers
数据: datasketch, marshmallow, simhash, ujson, modelscope
```

### 13.3 requirements.txt 完整清单

```
datasets==3.6.0, datasketch==1.6.4, Flask==3.0.3, Flask_Cors==4.0.0,
jieba==0.42.1, jsonlines==4.0.0, marshmallow==3.22.0, ngrok==1.4.0,
nltk==3.8, numpy==1.26.4, openai==1.59.6, psutil==5.9.8,
pydantic==2.11.5, python-dotenv==1.0.1, pyyaml==6.0.2, rich==13.7.1,
scikit_learn==1.5.1, sentence_transformers==2.3.1, simhash==2.1.2,
tiktoken==0.10.0, transformers==4.57.6, jinja2==3.1.2, trl==0.13.0,
ujson==5.1.0, wandb==0.18.3, streamlit==1.50.0, einops==0.8.1,
swanlab==0.7.11, modelscope==1.37.0
```

---

## 十四、已知问题与技术债

### 14.1 架构层面

1. **全局args变量**: 8个训练脚本使用全局 `args` 变量, 函数通过闭包访问。这使得代码难以测试和复用。

2. **sys.path hack**: 所有脚本开头的 `sys.path.append(...)` 在pip安装后已不需要, 但未清理。

3. **训练脚本高度重复**: 8个训练脚本有约60%的重复代码 (argparse定义、DDP初始化、模型加载、训练循环框架)。差异主要在 train_epoch 内部逻辑。

4. **双系统并存**: Logger (新旧两套)、Checkpoint (新旧两套) 同时存在, 增加维护负担。

### 14.2 LoRA实现

5. **Monkey-patching而非PEFT**: LoRA通过直接替换 `nn.Linear.forward` 实现, 不与 `torch.compile` 兼容, 且与HuggingFace PEFT生态隔离。

6. **LoRA注入范围**: 只注入 `in_features == out_features` 的Linear层, 遗漏了 `k_proj` 和 `v_proj` (768→384, 非方阵)。

### 14.3 训练脚本

7. **硬编码参数**: 部分脚本存在硬编码值 (如 `max_seq_len=340` 在train_lora.py中)。

8. **Dataloader不一致**: 训练脚本自己构建DataLoader, 而非通过统一工厂。

9. **checkpoint保存逻辑**: 训练脚本中同时调用 `save_model_weights()` (来自training_loop.py) 和 `lm_checkpoint()` (来自trainer_utils.py), 职责重叠。

### 14.4 推理

10. **自定义generate()**: `MiniMindForCausalLM.generate()` 是手写实现, 不经过transformers的 `GenerationMixin` 管线。这意味着不支持 `StoppingCriteria`, `LogitsProcessor` 等标准接口。

11. **eval_llm.py与serve_openai_api.py逻辑重复**: 两个文件有几乎相同的 `init_model()` 函数。

### 14.5 配置

12. **config.yaml与config_defaults.yaml重复**: 两个文件都定义了8种任务的参数。根 `config.yaml` 是用户界面, `config_defaults.yaml` 是代码加载的默认值, 容易不同步。

13. **TrainingConfig未完全集成**: 训练脚本仍使用自己的argparse定义, 未迁移到 `load_config()`。

### 14.6 测试

14. **测试覆盖率低**: 仅3个测试文件, 训练管线、RL算法、数据加载器等核心模块无测试。

15. **无CI配置**: 无GitHub Actions或其他CI流水线。

### 14.7 其他

16. **Mock工具是假的**: Agent RL中的6个工具全部是mock实现, 返回固定/随机结果。

17. **SwanLab/Wandb混合**: 代码中使用 `import swanlab as wandb` 来兼容国内用户, 但命名容易混淆。

18. **Windows兼容hack**: `import datasets` 在 `import torch` 之前 (train_full_sft.py line 7注释: "Windows pyarrow/torch DLL conflict workaround")。

19. **未使用的依赖**: requirements.txt中 `ngrok`, `marshmallow` 等可能未被实际使用。

---

## 十五、重构建议

### 15.1 优先级排序

**P0 (阻断性)**:
- 消除训练脚本重复 → 统一 `TrainingPipeline` 类
- 将args迁移到 `load_config()` → 消除全局变量

**P1 (重要)**:
- LoRA迁移到PEFT标准接口
- generate()使用transformers标准管线
- 统一Logger (移除旧Logger shim)
- 统一Checkpoint (移除旧lm_checkpoint)

**P2 (改善)**:
- 添加CI流水线
- 扩展测试覆盖率
- 清理sys.path hack
- 合并config.yaml和config_defaults.yaml

### 15.2 推荐的目录重构

```
minimind_lab/                    # 可安装包
├── __init__.py
├── config/
│   ├── defaults.yaml            # 唯一默认配置
│   └── config.py                # TrainingConfig + load_config
├── model/
│   ├── config.py                # MiniMindConfig
│   ├── model.py                 # MiniMindForCausalLM (精简)
│   ├── attention.py             # Attention, GQA, RoPE
│   ├── feedforward.py           # FeedForward, MOEFeedForward
│   └── lora.py                  # LoRA (PEFT兼容)
├── data/
│   ├── loader.py                # DataLoaderHub
│   ├── processor.py             # DataProcessor
│   └── datasets.py              # 5种Dataset类
├── training/
│   ├── pipeline.py              # 统一TrainingPipeline类
│   ├── accumulator.py           # Accumulator
│   ├── scheduler.py             # create_scheduler
│   ├── checkpoint.py            # LMCheckpointer
│   └── metrics.py               # MetricsWriter
├── rl/
│   ├── rollout.py               # RolloutEngine
│   ├── rewards.py               # calculate_rewards
│   ├── grpo.py                  # GRPO/CISPO loss
│   ├── ppo.py                   # PPO loss + GAE
│   └── agent.py                 # Agent RL逻辑
├── inference/
│   ├── cli.py                   # eval_llm逻辑
│   ├── server.py                # serve_openai_api
│   └── web.py                   # web_demo
├── utils/
│   ├── logging.py               # 统一日志
│   ├── exceptions.py            # 异常体系
│   └── helpers.py               # setup_seed, init_distributed等
└── cli/
    └── main.py                  # CLI入口点
```

### 15.3 TrainingPipeline 类设计建议

```python
class TrainingPipeline:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self._init_distributed()
        self._init_model()
        self._init_data()
        self._init_optimizer()
        self._init_scheduler()
        self._init_accumulator()

    def train(self):
        for epoch in range(self.start_epoch, self.config.epochs):
            self.train_epoch(epoch)
            self.save_checkpoint(epoch)

    def train_epoch(self, epoch):
        raise NotImplementedError  # 子类实现

    def resume(self, checkpoint_path):
        ...

    def save_checkpoint(self, epoch):
        ...
```

每种训练模式只需继承并实现 `train_epoch()` 方法。

---

## 附录A: 文件大小统计

| 目录 | 文件数 | 总行数 |
|------|--------|--------|
| model/ | 4 (.py) + 2 (.json) | ~540 |
| dataset/ | 4 (.py) + 1 (.md) | ~860 |
| trainer/ (训练脚本) | 8 (.py) | ~1860 |
| trainer/ (基础设施) | 8 (.py) + 1 (.yaml) | ~1720 |
| scripts/ | 6 (.py) | ~1560 |
| tests/ | 3 (.py) | ~130 |
| 根目录 | 9 (.py/.md/.toml/.yaml) | ~1050 |
| docs/ | 5 (.md) | ~2850 |
| **总计** | **~50文件** | **~10,570行** |

## 附录B: 代码调用关系图

```
eval_llm.py ─────────────────────────────────────────────┐
scripts/serve_openai_api.py ─────────────────────────────┤
scripts/web_demo.py ─────────────────────────────────────┤
scripts/eval_toolcall.py ────────────────────────────────┤
    │                                                     │
    ├── model/model_minimind.py ← MiniMindForCausalLM     │
    ├── model/model_lora.py ← LoRA                       │
    └── trainer/trainer_utils.py ← init_model, Logger    │
                                                          │
trainer/train_pretrain.py ────────────────────────────────┤
trainer/train_full_sft.py ────────────────────────────────┤
trainer/train_lora.py ────────────────────────────────────┤
trainer/train_distillation.py ────────────────────────────┤
trainer/train_dpo.py ─────────────────────────────────────┤
trainer/train_grpo.py ────────────────────────────────────┤
trainer/train_ppo.py ─────────────────────────────────────┤
trainer/train_agent.py ───────────────────────────────────┤
    │                                                     │
    ├── model/ ← MiniMindConfig, MiniMindForCausalLM      │
    ├── dataset/lm_dataset.py ← 5种Dataset                │
    ├── trainer/training_loop.py ← Accumulator等          │
    ├── trainer/trainer_utils.py ← 工具函数               │
    ├── trainer/rollout_engine.py ← RL生成引擎            │
    └── trainer/logger.py ← 结构化日志                    │
                                                          │
scripts/auto_config.py ───────────────────────────────────┤
    │                                                     │
    ├── dataset/data_loader.py ← DataLoaderHub            │
    └── dataset/data_processor.py ← 数据分析              │
```

---

**本文档版本**: v1.0
**最后更新**: 2026-05-31
**维护者**: MiniMind-Lab Contributors
