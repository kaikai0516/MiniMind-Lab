# 📖 API 参考

---

## 数据加载

### DataLoaderHub

```python
from dataset.data_loader import DataLoaderHub

hub = DataLoaderHub()

# 单文件
ds = hub.load("dataset/data.csv")

# 多文件合并
ds = hub.load_multi(["dataset/a.jsonl", "dataset/b.csv"], dataset_type="sft")

# 便利函数
from dataset.data_loader import (
    load_pretrain_data,  # → 'text' 列
    load_sft_data,       # → 'conversations' 列
    load_dpo_data,       # → 'chosen'/'rejected' 列
    load_rlaif_data,     # → 'conversations' 列
    load_agent_data,     # → 'conversations'/'gt' 列
)
```

### DataProcessor

```python
from dataset.data_processor import (
    clean_text, clean_conversations,
    validate_conversations, auto_detect_schema,
    filter_by_length, filter_by_token_length,
    deduplicate, describe_dataset,
    process_dataset,  # 一键全流程
)

clean = process_dataset(raw, min_length=10, max_length=2048,
                        dedup_threshold=3, clean=True, validate=True)
stats = describe_dataset(clean)
```

---

## 训练基础设施

### Accumulator

```python
from trainer.training_loop import Accumulator

acc = Accumulator(model, optimizer, scaler,
                  accumulation_steps=8, grad_clip=1.0, scheduler=scheduler)

for batch in loader:
    loss = model(batch).loss
    acc.backward(loss)
    print(acc.loss_value)

acc.finalize()  # epoch 结束排空
```

### MultiOptimizerAccumulator (PPO)

```python
from trainer.training_loop import MultiOptimizerAccumulator

ppo_acc = MultiOptimizerAccumulator(
    model, actor_opt, critic_opt, scaler, accumulation_steps, grad_clip
)
```

### create_scheduler

```python
from trainer.training_loop import create_scheduler

scheduler = create_scheduler(optimizer, iters_per_epoch=100, epochs=3,
                             accumulation_steps=2, learning_rate=1e-5)
```

### wrap_model

```python
from trainer.training_loop import wrap_model, save_model_weights, get_raw_model

model = wrap_model(model, use_compile=True, local_rank=0, rollout_engine=None)
save_model_weights(model, save_dir="../out", name="full_sft", hidden_size=768, use_moe=False)
raw = get_raw_model(model)  # 解包 DDP/compile
```

---

## 配置系统

```python
from trainer.config import load_config, TrainingConfig, create_arg_parser

# 三层合并加载
config = load_config("full_sft", sys.argv[1:], user_yaml="my.yaml")

# 访问所有字段
print(config.learning_rate, config.batch_size, config.epochs)
print(config.hidden_size, config.max_seq_len, config.device)

# 向后兼容的 argparse
parser = create_arg_parser("full_sft")
args = parser.parse_args()
```

### TrainingConfig 字段一览

| 分类 | 字段 |
|------|------|
| 任务 | `task` |
| 模型 | `hidden_size`, `num_hidden_layers`, `use_moe`, `max_seq_len`, `max_gen_len`, `max_total_len` |
| 数据 | `data_path`, `from_weight`, `save_weight` |
| 训练 | `batch_size`, `learning_rate`, `epochs`, `accumulation_steps`, `grad_clip`, `dtype`, `num_workers` |
| 日志 | `log_interval`, `save_interval`, `save_dir`, `checkpoint_dir` |
| 设备 | `device` (auto), `use_compile`, `from_resume`, `debug_mode` |
| 监控 | `use_wandb`, `wandb_project` |
| RL | `num_generations`, `beta`, `loss_type`, `epsilon`, `epsilon_high`, `thinking_ratio` |
| Rollout | `rollout_engine`, `sglang_base_url`, `sglang_model_path`, `reward_model_path` |
| 蒸馏 | `student_hidden_size`, `teacher_hidden_size`, `alpha`, `temperature` 等 |
| LoRA | `lora_name` |
| Checkpoint | `checkpoint_max_keep`, `checkpoint_format` |

---

## 日志系统

```python
from trainer.logger import get_logger, set_level, Logger

log = get_logger(__name__)
log.debug / log.info / log.warning / log.error(...)

set_level("DEBUG")
Logger("legacy compat")  # 向后兼容
```

---

## Checkpoint 管理器

```python
from trainer.checkpoint import LMCheckpointer

ckp = LMCheckpointer("../checkpoints", max_keep=3, fmt="safetensors")

ckp.save(model, optimizer, epoch, step, weight_name="full_sft",
         lora_state=None, extra={"notes": "exp1"})

state = ckp.load("full_sft")
meta = ckp.load_metadata("full_sft")
opt = ckp.load_optimizer("full_sft")

for name, step, ts in ckp.list_checkpoints():
    print(name, step, ts)
```

---

## 异常

```python
from trainer.exceptions import (
    MiniMindError, ConfigError, CheckpointError,
    DataLoadError, ModelBuildError, TrainingError,
)

raise ConfigError(f"Unknown task: {task}")
raise CheckpointError(f"Corrupt file: {path}")
```

---

## 智能配置工具

```python
from scripts.auto_config import (
    hardware_probe,       # → GPU/CPU/RAM/磁盘信息
    analyze_dataset,      # → 样本数/列/token分布
    estimate_memory,      # → 参数量/峰值显存
    build_prompt,         # → LLM prompt
    get_config_from_llm,  # → 推荐配置 dict
    apply_config,         # → 写 JSON + 打印命令
)
```

---

## CLI 入口点

```bash
minimind-pretrain     minimind-full-sft    minimind-lora
minimind-distillation minimind-dpo         minimind-grpo
minimind-ppo          minimind-agent       minimind-eval
minimind-serve        minimind-chat        minimind-auto-config
```
