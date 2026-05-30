# 📖 API 参考

> 完整 API 文档 — 所有模块、类、函数、参数。适合二次开发和脚本编写。

---

## 目录

1. [DataLoaderHub — 数据加载](#一dataloaderhub--数据加载)
2. [DataProcessor — 数据处理](#二dataprocessor--数据处理)
3. [训练基础设施](#三训练基础设施)
4. [配置系统](#四配置系统)
5. [日志系统](#五日志系统)
6. [Checkpoint 管理器](#六checkpoint-管理器)
7. [异常体系](#七异常体系)
8. [MetricsWriter — 指标导出](#八metricswriter--指标导出)
9. [智能配置工具](#九智能配置工具)
10. [CLI 入口点](#十cli-入口点)
11. [训练脚本入口](#十一训练脚本入口)

---

## 一、DataLoaderHub — 数据加载

**文件**: `dataset/data_loader.py`

### DataLoaderHub 类

```python
class DataLoaderHub:
    def load(self, path: str) -> Dataset:
        """
        加载单个文件。
        - 自动检测格式: .jsonl / .json / .csv / .parquet / .txt
        - 自动应用列别名映射 (content→text, messages→conversations 等)
        - 返回 HuggingFace Dataset 对象
        """

    def load_multi(
        self,
        paths: Union[str, List[str]],
        dataset_type: str = "pretrain",  # pretrain | sft | dpo | rlaif | agent
        shuffle: bool = True,
        seed: int = 42,
        process_fn: Optional[Callable] = None,
    ) -> Dataset:
        """
        加载多个文件并合并。
        1. 每个文件独立加载 + 归一化 Schema
        2. concatenate_datasets 合并
        3. 可选 shuffle
        4. 可选 process_fn 自定义处理
        """
```

### 便利函数

```python
load_pretrain_data(paths)  -> Dataset  # 目标列: 'text'
load_sft_data(paths)       -> Dataset  # 目标列: 'conversations'
load_dpo_data(paths)       -> Dataset  # 目标列: 'chosen', 'rejected'
load_rlaif_data(paths)     -> Dataset  # 目标列: 'conversations'
load_agent_data(paths)     -> Dataset  # 目标列: 'conversations', 'gt'
```

### 使用示例

```python
from dataset.data_loader import DataLoaderHub, load_sft_data

hub = DataLoaderHub()

# 单文件
ds = hub.load("dataset/sft.jsonl")

# 多文件混合格式
ds = hub.load_multi(
    ["dataset/sft_*.jsonl", "dataset/extra.csv"],
    dataset_type="sft"
)

# 便利函数
ds = load_sft_data(["dataset/sft_*.jsonl"])
```

### 列别名映射

当源数据的列名与预期不同时，自动映射：

```
源列名                           → 规范列名
content/document/passage/body    → text
messages/dialog/chat/history     → conversations
preferred/positive               → chosen
dispreferred/negative            → rejected
answer/label/target              → gt
```

---

## 二、DataProcessor — 数据处理

**文件**: `dataset/data_processor.py`

### 文本处理

```python
clean_text(text: str) -> str
"""
Unicode NFKC 规范化 → 全角空格转半角 → 多余空白合并 → 首尾 trim
"""

clean_conversations(convs: List[Dict]) -> List[Dict]
"""
对每条消息的 'content' 字段执行 clean_text()
"""
```

### 验证

```python
validate_conversations(convs: List[Dict]) -> bool
"""
检查:
- 每条消息有 'role' 和 'content'
- role 是 user/assistant/system/tool 之一
- content 非空
"""

auto_detect_schema(sample: Dict) -> str
"""
返回: 'pretrain' | 'sft' | 'dpo' | 'rlaif' | 'agent' | 'unknown'
根据字段自动判断数据类型
"""
```

### 过滤

```python
filter_by_length(ds: Dataset, min_length: int, max_length: int) -> Dataset
"""
按字符数过滤 (统计所有文本内容的总字符数)
"""

filter_by_token_length(
    ds: Dataset, tokenizer, min_tokens: int, max_tokens: int
) -> Dataset
"""
按 token 数过滤 (需要 tokenizer)
"""
```

### 去重

```python
deduplicate(ds: Dataset, threshold: int = 3) -> Dataset
"""
SimHash 近似去重。
threshold = 汉明距离阈值 (0=精确去重, 3=默认, 6+=宽松)
"""
```

### 统计

```python
describe_dataset(ds: Dataset, tokenizer=None) -> Dict[str, Any]
"""
返回: {
    'total': int,          # 总样本数
    'avg_len_chars': float, # 平均字符数
    'p50_len_chars': float, # 中位数
    'p95_len_chars': float, # 95分位
    'min_len': int,
    'max_len': int,
    'columns': list[str],
}
"""
```

### 一键处理

```python
process_dataset(
    ds: Dataset,
    min_length: int = 10,
    max_length: int = 2048,
    dedup_threshold: int = 3,
    clean: bool = True,
    validate: bool = True,
) -> Dataset
"""
处理顺序: 清洗 → 验证 → 长度过滤 → 去重
失败数据自动丢弃并统计
"""
```

---

## 三、训练基础设施

**文件**: `trainer/training_loop.py`

### Accumulator — 单优化器梯度累积

```python
class Accumulator:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scaler: GradScaler,
        accumulation_steps: int = 1,
        grad_clip: float = 1.0,
        scheduler: Optional[LRScheduler] = None,
    ):
        """
        统一梯度累积器。
        每 accumulation_steps 次 backward 后自动:
          1. 梯度裁剪 (grad_clip)
          2. scaler.step(optimizer)
          3. scaler.update()
          4. scheduler.step()
          5. optimizer.zero_grad(set_to_none=True)
        """

    def backward(self, loss: Tensor) -> None:
        """自动 scale + backward + 条件 step。"""

    def finalize(self) -> None:
        """Epoch 结束排空剩余梯度。"""

    @property
    def loss_value(self) -> float:
        """真实 loss (已补偿 accumulation 偏移)。"""
```

### MultiOptimizerAccumulator — 双优化器梯度累积

```python
class MultiOptimizerAccumulator:
    """
    PPO 专用 — actor 和 critic 两个独立优化器。
    接口与 Accumulator 相同: .backward(loss), .finalize(), .loss_value
    """
    def __init__(
        self, model, actor_opt, critic_opt, scaler,
        accumulation_steps=1, grad_clip=1.0
    ):
        ...
```

### create_scheduler

```python
def create_scheduler(
    optimizer: optim.Optimizer,
    iters_per_epoch: int,
    epochs: int,
    accumulation_steps: int = 1,
    learning_rate: float = 1e-5,
) -> CosineAnnealingLR:
    """
    创建 CosineAnnealingLR。
    T_max = (iters_per_epoch * epochs) // accumulation_steps
    eta_min = learning_rate * 0.1  (默认)
    """
```

### wrap_model

```python
def wrap_model(
    model: nn.Module,
    use_compile: bool = False,
    local_rank: int = 0,
    rollout_engine=None,
) -> nn.Module:
    """
    按正确顺序包装模型:
    1. torch.compile (可选)
    2. DistributedDataParallel (可选)
    3. rollout_engine.update (可选)
    """
```

### save_model_weights

```python
def save_model_weights(
    model: nn.Module,
    save_dir: str,
    name: str,
    hidden_size: int,
    use_moe: bool = False,
) -> str:
    """
    保存半精度权重到 {save_dir}/{name}_{hidden_size}{_moe}.pth
    自动解包 DDP 和 torch.compile wrapper
    """
```

### get_raw_model

```python
def get_raw_model(model: nn.Module) -> nn.Module:
    """解包 DDP (.module) 和 compile (_orig_mod)，返回裸模型。"""
```

---

## 四、配置系统

**文件**: `trainer/config.py`

### load_config

```python
def load_config(
    defaults: str = "full_sft",
    cli_args: Optional[List[str]] = None,
    user_yaml: Optional[str] = None,
) -> TrainingConfig:
    """
    三层合并:
    1. 加载 config_defaults.yaml → 取 defaults 对应 section
    2. 叠加 user_yaml (可选)
    3. 叠加 cli_args (可选)
    返回 TrainingConfig 实例
    """
```

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    # 任务
    task: str = "full_sft"

    # 模型
    hidden_size: int = 768
    num_hidden_layers: int = 8
    use_moe: bool = False
    max_seq_len: int = 768
    max_gen_len: int = 1024
    max_total_len: int = 2500

    # 数据
    data_path: List[str] = field(default_factory=lambda: ["../dataset/sft_t2t_mini.jsonl"])
    from_weight: str = "full_sft"
    save_weight: str = "full_sft"

    # 训练
    batch_size: int = 32
    learning_rate: float = 1e-5
    epochs: int = 3
    accumulation_steps: int = 1
    grad_clip: float = 1.0
    dtype: str = "bfloat16"
    num_workers: int = 8

    # 日志
    log_interval: int = 100
    save_interval: int = 100
    save_dir: str = "../out"
    checkpoint_dir: str = "../checkpoints"

    # 设备
    device: str = ""  # 空=自动检测
    from_resume: bool = False
    use_compile: bool = False
    debug_mode: bool = False
    debug_interval: int = 20

    # 监控
    use_wandb: bool = False
    wandb_project: str = "MiniMind"

    # RL
    num_generations: int = 6
    beta: float = 0.1
    loss_type: str = "cispo"
    epsilon: float = 0.2
    epsilon_high: float = 5.0
    thinking_ratio: float = 0.9

    # Rollout
    rollout_engine: str = "torch"
    sglang_base_url: str = "http://localhost:8998"
    sglang_model_path: str = "../model"
    sglang_shared_path: str = "./sglang_ckpt"
    reward_model_path: str = "../../internlm2-1_8b-reward"

    # 蒸馏
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

    # LoRA
    lora_name: str = "lora_medical"

    # Checkpoint
    checkpoint_max_keep: int = 3
    checkpoint_format: str = "torch"

    def auto_device(self) -> str:
        """惰性设备检测。不用 torch 时返回 'cpu'。"""
```

### create_arg_parser

```python
def create_arg_parser(task: str) -> ArgumentParser:
    """
    返回标准的 argparse ArgumentParser。
    向后兼容 — 训练脚本可以继续使用 parser.parse_args()。
    同时支持 --config my.yaml 参数。
    """
```

---

## 五、日志系统

**文件**: `trainer/logger.py`

```python
def get_logger(name: str = "minimind") -> logging.Logger:
    """
    返回 minimind 层级下的子 logger。
    name 通常传 __name__。
    """

def set_level(level: str) -> None:
    """
    运行时切换日志级别。
    level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR'
    """

def Logger(content: str) -> None:
    """
    向后兼容 shim — 委托到 log.info(content)。
    旧代码无需修改。
    """
```

### 环境变量

```
MINIMIND_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR  → 默认 INFO
MINIMIND_LOG_FILE=/path/to/log               → 不启用文件写入
```

---

## 六、Checkpoint 管理器

**文件**: `trainer/checkpoint.py`

```python
class LMCheckpointer:
    def __init__(
        self,
        save_dir: str = "../checkpoints",
        max_keep: int = 3,
        fmt: str = "safetensors",
    ):
        """
        save_dir: checkpoint 目录
        max_keep: 最多保留数量 (0=不限制)
        fmt: "safetensors" (默认) | "torch"
        """

    def save(
        self,
        model,
        optimizer=None,
        epoch: int = 0,
        step: int = 0,
        weight_name: str = "checkpoint",
        lora_state: Optional[Dict] = None,
        extra: Optional[Dict] = None,
    ) -> str:
        """
        保存 checkpoint。
        返回权重文件路径。
        原子写入: .tmp → os.replace
        写入: .safetensors/.pth + _meta.json + _optim.pth + (可选)_lora.pth
        """

    def load(self, weight_name: str, map_location="cpu") -> Dict:
        """
        加载模型权重。
        自动尝试 .safetensors → .pth
        """

    def load_optimizer(self, weight_name: str, map_location="cpu") -> Optional[Dict]:
        """加载优化器状态"""

    def load_metadata(self, weight_name: str) -> Optional[Dict]:
        """加载元数据 {'epoch', 'step', 'config_hash', 'timestamp'}"""

    def list_checkpoints(self) -> list:
        """返回 [(name, step, timestamp), ...] 按时间倒序"""
```

---

## 七、异常体系

**文件**: `trainer/exceptions.py`

```python
class MiniMindError(Exception):
    """所有 MiniMind 异常的基类"""

class ConfigError(MiniMindError):
    """配置错误: 未知任务类型、无效参数值、YAML 格式问题"""

class CheckpointError(MiniMindError):
    """Checkpoint 错误: 文件损坏、路径不存在、格式不匹配"""

class DataLoadError(MiniMindError):
    """数据加载错误: 文件缺失、格式无效、Schema 不匹配"""

class ModelBuildError(MiniMindError):
    """模型构建错误: 架构参数无效、尺寸不合法"""

class TrainingError(MiniMindError):
    """运行时训练错误: NaN loss、OOM、梯度爆炸"""
```

---

## 八、MetricsWriter — 指标导出

**文件**: `trainer/metrics.py`

```python
class MetricsWriter:
    def __init__(self, path: str = "./metrics.csv", flush_every: int = 1):
        """
        path: CSV 输出路径
        flush_every: 每 N 次写入 flush 磁盘
        """

    def write(self, **metrics: Any) -> None:
        """
        写入一行指标。
        必须包含: step=<int>
        可选: loss, lr, aux_loss, logits_loss, ...
        首次调用自动写入 CSV header。
        """

    @property
    def csv_path(self) -> str:
        """返回 CSV 文件的绝对路径"""
```

### CSV 格式

```csv
timestamp,step,loss,logits_loss,aux_loss,lr
1780152418.099,100,2.3400,2.3000,0.0400,0.00001000
1780152420.512,200,2.1500,2.1200,0.0300,0.00000999
```

### Jupyter 集成

`notebooks/training_monitor.ipynb` 实时读取 CSV 并绘制所有数值列。

---

## 九、智能配置工具

**文件**: `scripts/auto_config.py`

```python
hardware_probe() -> Dict
"""
探测硬件信息:
{gpu_name, gpu_memory_gb, gpu_count, backend(cuda/rocm/cpu),
 cpu_cores, ram_gb, disk_free_gb, torch_version}
"""

analyze_dataset(data_path: str, task: str, max_seq_len: int = 768) -> Dict
"""
分析数据集:
{total_samples, columns, avg_chars_per_sample, p50_chars, p95_chars, estimated_tokens}
"""

estimate_memory(hidden_size, num_layers, use_moe, batch_size, max_seq_len, dtype) -> Dict
"""
估算显存:
{total_params_m, model_memory_gb, activation_memory_gb,
 optimizer_memory_gb, peak_estimate_gb}
"""

build_prompt(hw: Dict, data: Dict, mem: Dict, task: str) -> str
"""
组装 LLM prompt
"""

get_config_from_llm(prompt: str, api_key: str, api_base: str, model: str) -> Dict
"""
调用 OpenAI 兼容 API 获取推荐配置
"""

apply_config(config: Dict, output_path: str) -> None
"""
写入 JSON + 打印可直接运行的训练命令
"""
```

---

## 十、CLI 入口点

`pip install -e .` 后可用：

```bash
minimind-pretrain         # 预训练
minimind-full-sft         # 全量 SFT
minimind-lora             # LoRA 微调
minimind-distillation     # 知识蒸馏
minimind-dpo              # DPO 偏好对齐
minimind-grpo             # GRPO 强化学习
minimind-ppo              # PPO 强化学习
minimind-agent            # Agent RL
minimind-eval             # 模型推理
minimind-serve            # OpenAI API 服务
minimind-chat             # 交互对话
minimind-auto-config      # 智能配置
```

**文件**: `trainer/cli.py` — 基于 `runpy.run_module` 实现，不修改原有训练脚本。

---

## 十一、训练脚本入口

所有训练脚本在 `trainer/` 目录下，接受相同的通用参数 (加上各任务的专属参数)：

| 脚本 | 任务 | 专属参数 |
|------|------|---------|
| `train_pretrain.py` | 预训练 | — |
| `train_full_sft.py` | 全量 SFT | — |
| `train_lora.py` | LoRA 微调 | `--lora_name` |
| `train_distillation.py` | 知识蒸馏 | `--student_*`, `--teacher_*`, `--alpha`, `--temperature` |
| `train_dpo.py` | DPO | `--beta` |
| `train_grpo.py` | GRPO | `--num_generations`, `--beta`, `--loss_type`, `--epsilon*` |
| `train_ppo.py` | PPO | `--num_generations`, `--beta` |
| `train_agent.py` | Agent RL | `--num_generations`, `--thinking_ratio`, `--max_total_len` |
