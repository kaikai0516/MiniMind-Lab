#!/usr/bin/env python3
"""
MiniMind 智能配置工具 — 分析硬件 + 数据集 + AI推理 → 输出统一 config.yaml

用法:
    # 一键：分析数据 + AI推荐 → 写入 config.yaml
    python scripts/auto_config.py --data_path ./dataset/pretrain_t2t_mini.jsonl --task pretrain

    # 使用本地规则（无需 API key）
    python scripts/auto_config.py --data_path ./dataset/sft_t2t_mini.jsonl --task full_sft --local

    # 自定义模型尺寸
    python scripts/auto_config.py --data_path ./dataset/pretrain_t2t_mini.jsonl --task pretrain --hidden_size 512 --num_layers 4

    # 仅探测硬件
    python scripts/auto_config.py --probe

    # 使用 API Key
    python scripts/auto_config.py --data_path ./dataset/sft_t2t_mini.jsonl --task full_sft --api_key sk-xxx
"""
import os, sys, json, argparse, math, warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

__package__ = "scripts"
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))

warnings.filterwarnings('ignore')

# ==============================================================================
# Constants — 各任务类型的推荐参数范围
# ==============================================================================
TASK_DEFAULTS = {
    "pretrain": {
        "batch_size": 32, "learning_rate": 5e-4, "epochs": 2,
        "accumulation_steps": 1, "max_seq_len": 340, "from_weight": "none",
    },
    "full_sft": {
        "batch_size": 16, "learning_rate": 1e-5, "epochs": 2,
        "accumulation_steps": 1, "max_seq_len": 768, "from_weight": "pretrain",
    },
    "lora": {
        "batch_size": 32, "learning_rate": 1e-4, "epochs": 10,
        "accumulation_steps": 1, "max_seq_len": 340, "from_weight": "full_sft",
        "lora_name": "lora_weights",
    },
    "distillation": {
        "batch_size": 32, "learning_rate": 5e-6, "epochs": 6,
        "accumulation_steps": 1, "max_seq_len": 340, "alpha": 0.5, "temperature": 1.5,
    },
    "dpo": {
        "batch_size": 4, "learning_rate": 4e-8, "epochs": 1,
        "accumulation_steps": 1, "max_seq_len": 1024, "beta": 0.15,
    },
    "grpo": {
        "batch_size": 2, "learning_rate": 3e-7, "epochs": 1,
        "accumulation_steps": 1, "max_seq_len": 768, "max_gen_len": 1024,
        "num_generations": 6, "beta": 0.1, "loss_type": "cispo",
        "epsilon": 0.2, "epsilon_high": 5.0, "thinking_ratio": 0.9,
    },
    "ppo": {
        "batch_size": 2, "learning_rate": 3e-7, "epochs": 1,
        "accumulation_steps": 1, "max_seq_len": 768, "max_gen_len": 1024,
        "num_generations": 6, "beta": 0.1, "loss_type": "cispo",
        "epsilon": 0.2, "epsilon_high": 5.0, "thinking_ratio": 0.9,
    },
    "agent": {
        "batch_size": 2, "learning_rate": 3e-7, "epochs": 1,
        "accumulation_steps": 1, "max_seq_len": 1024, "max_gen_len": 768,
        "max_total_len": 2500, "num_generations": 4, "beta": 0.1,
        "loss_type": "cispo", "epsilon": 0.2, "epsilon_high": 5.0, "thinking_ratio": 0.1,
    },
}


# ==============================================================================
# 1. 硬件探测
# ==============================================================================
def hardware_probe() -> Dict[str, Any]:
    info = {
        "gpu_available": False, "gpu_count": 0, "gpu_name": "N/A",
        "gpu_memory_gb": 0, "gpu_backend": "N/A",
        "cpu_cores": 0, "ram_gb": 0, "disk_free_gb": 0,
        "torch_version": "N/A",
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_memory_gb"] = round(props.total_memory / (1024**3), 1)
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                info["gpu_backend"] = "ROCm"
            else:
                info["gpu_backend"] = "CUDA"
    except ImportError:
        pass
    try:
        import psutil
        info["cpu_cores"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
        info["disk_free_gb"] = round(psutil.disk_usage(str(_project_root)).free / (1024**3), 1)
    except ImportError:
        pass
    return info


# ==============================================================================
# 2. 数据集分析与类型自动检测
# ==============================================================================
def _auto_detect_task(data_path: List[str]) -> str:
    """根据文件名字模式猜测任务类型"""
    paths_str = " ".join(data_path).lower()
    if any(k in paths_str for k in ("pretrain", "pretrain")):
        return "pretrain"
    if any(k in paths_str for k in ("sft", "chat", "instruct", "conversation")):
        return "full_sft"
    if any(k in paths_str for k in ("dpo", "preference", "chosen", "reject")):
        return "dpo"
    if any(k in paths_str for k in ("rl", "rlaif", "reward")):
        return "grpo"
    if any(k in paths_str for k in ("agent", "tool", "function_call")):
        return "agent"
    if any(k in paths_str for k in ("lora", "medical", "domain")):
        return "lora"
    return "full_sft"  # default fallback


def analyze_dataset(data_path: List[str]) -> Dict[str, Any]:
    """加载数据集并返回统计信息"""
    from dataset.data_loader import DataLoaderHub
    hub = DataLoaderHub()
    ds = hub.load_multi(data_path, dataset_type="pretrain", shuffle=False)

    stats = {
        "total_samples": len(ds),
        "columns": list(ds.column_names),
    }

    # 采样计算文本长度
    text_col = _find_text_column(ds)
    sample_size = min(300, len(ds))
    lengths = []
    for i in range(sample_size):
        val = ds[i].get(text_col, "")
        if isinstance(val, (list, dict)):
            val = json.dumps(val, ensure_ascii=False)
        else:
            val = str(val)
        lengths.append(len(val))

    if lengths:
        lengths.sort()
        stats["avg_chars"] = round(sum(lengths) / len(lengths))
        stats["char_p50"] = lengths[len(lengths)//2]
        stats["char_p90"] = lengths[9*len(lengths)//10] if len(lengths) >= 10 else lengths[-1]
        stats["char_max"] = lengths[-1]

    # 估算 token 数 (中英文混合约 2 字符/token)
    stats["est_tokens_per_sample"] = round(stats.get("avg_chars", 0) / 2)
    stats["est_total_tokens"] = stats["total_samples"] * stats["est_tokens_per_sample"]

    # 自动检测任务类型
    stats["detected_task"] = _auto_detect_task(data_path)

    return stats


def _find_text_column(ds) -> str:
    for col in ds.column_names:
        if col.lower() in ("text", "content", "prompt", "conversations", "messages", "input"):
            return col
    return ds.column_names[0] if ds.column_names else "text"


# ==============================================================================
# 3. 显存估算
# ==============================================================================
def estimate_memory(hidden_size: int, num_layers: int, use_moe: bool,
                    batch_size: int, max_seq_len: int, dtype: str = "bfloat16") -> Dict[str, Any]:
    bytes_per_param = 2 if "16" in dtype else 4
    vocab = 6400
    per_layer = 12 * hidden_size * hidden_size
    if use_moe:
        per_layer = int(per_layer * 1.8)
    total_params = vocab * hidden_size + num_layers * per_layer
    model_gb = total_params * bytes_per_param / (1024**3)
    grad_gb = model_gb
    optimizer_gb = total_params * 8 / (1024**3)  # AdamW fp32 states
    activation_gb = batch_size * max_seq_len * hidden_size * num_layers * 2 / (1024**3)
    peak_gb = model_gb + grad_gb + optimizer_gb + activation_gb

    return {
        "total_params_m": round(total_params / 1e6, 1),
        "peak_gb": round(peak_gb, 2),
        "model_gb": round(model_gb, 2),
        "optimizer_gb": round(optimizer_gb, 2),
        "activation_gb": round(activation_gb, 2),
    }


# ==============================================================================
# 4. 规则引擎 — 无需 API 的智能推荐
# ==============================================================================
def rule_based_config(task: str, hardware: Dict, dataset: Dict,
                      memory: Dict, hidden_size: int, num_layers: int,
                      use_moe: bool, data_path: List[str]) -> Dict[str, Any]:
    """基于规则的本地配置推荐"""
    defaults = TASK_DEFAULTS.get(task, TASK_DEFAULTS["full_sft"]).copy()
    n_samples = dataset.get("total_samples", 1000)
    avg_chars = dataset.get("avg_chars", 500)

    config = {
        "task": task,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "use_moe": use_moe,
        "data_path": data_path,
        "save_weight": defaults.get("save_weight", task),
        "batch_size": defaults["batch_size"],
        "learning_rate": defaults["learning_rate"],
        "epochs": defaults["epochs"],
        "accumulation_steps": defaults["accumulation_steps"],
        "max_seq_len": defaults["max_seq_len"],
        "grad_clip": 1.0,
        "dtype": "bfloat16" if hardware["gpu_available"] else "float32",
        "num_workers": min(8, hardware.get("cpu_cores", 4)),
        "log_interval": 100,
        "save_interval": 100,
        "from_weight": defaults.get("from_weight", "none"),
        "use_compile": False,
        "from_resume": False,
    }

    # -- 根据数据集规模和硬件做调整 --
    gpu_mem = hardware.get("gpu_memory_gb", 0)

    # 小数据集 → 更多 epochs, 小 batch
    if n_samples < 1000:
        config["epochs"] = max(config["epochs"] * 2, 3)
        config["batch_size"] = max(config["batch_size"] // 2, 1)
    elif n_samples > 50000:
        config["epochs"] = max(config["epochs"] // 2, 1)
        config["batch_size"] = min(config["batch_size"] * 2, 128)

    # 根据平均长度调整 max_seq_len
    if avg_chars > 0:
        est_tokens = avg_chars / 2  # rough estimate
        if est_tokens > 600:
            config["max_seq_len"] = min(defaults["max_seq_len"], 1024)
        elif est_tokens < 150:
            config["max_seq_len"] = min(defaults["max_seq_len"], 512)

    # GPU 显存约束
    if gpu_mem > 0 and memory["peak_gb"] > gpu_mem * 0.85:
        # 减少 batch_size 或增加 accumulation_steps
        safety_factor = (gpu_mem * 0.85) / memory["peak_gb"]
        if config["batch_size"] > 4:
            config["batch_size"] = max(int(config["batch_size"] * safety_factor), 1)
        else:
            config["accumulation_steps"] = max(int(2 / safety_factor), 1)

    # CPU only → 极小 batch
    if not hardware["gpu_available"]:
        config["batch_size"] = min(config["batch_size"], 2)
        config["num_workers"] = 0

    # -- 任务特定参数 --
    if task == "lora":
        config["lora_name"] = defaults.get("lora_name", "lora_weights")
    if task == "distillation":
        config["alpha"] = defaults.get("alpha", 0.5)
        config["temperature"] = defaults.get("temperature", 1.5)
    if task == "dpo":
        config["beta"] = defaults.get("beta", 0.15)
    if task in ("grpo", "ppo"):
        config["max_gen_len"] = defaults.get("max_gen_len", 1024)
        config["num_generations"] = defaults.get("num_generations", 6)
        config["beta"] = defaults.get("beta", 0.1)
        config["loss_type"] = defaults.get("loss_type", "cispo")
        config["epsilon"] = defaults.get("epsilon", 0.2)
        config["epsilon_high"] = defaults.get("epsilon_high", 5.0)
        config["thinking_ratio"] = defaults.get("thinking_ratio", 0.9)
    if task == "agent":
        config["max_gen_len"] = defaults.get("max_gen_len", 768)
        config["max_total_len"] = defaults.get("max_total_len", 2500)
        config["num_generations"] = defaults.get("num_generations", 4)
        config["beta"] = defaults.get("beta", 0.1)
        config["loss_type"] = defaults.get("loss_type", "cispo")
        config["epsilon"] = defaults.get("epsilon", 0.2)
        config["epsilon_high"] = defaults.get("epsilon_high", 5.0)
        config["thinking_ratio"] = defaults.get("thinking_ratio", 0.1)
        config["from_weight"] = "full_sft"

    config["_source"] = "rule_engine"
    config["_reasoning"] = f"本地规则: 数据{n_samples}条/均长{avg_chars}字符, "
    if gpu_mem > 0:
        config["_reasoning"] += f"GPU {hardware['gpu_name']} ({gpu_mem}GB), "
    config["_reasoning"] += f"峰值显存估算 {memory['peak_gb']}GB"

    return config


# ==============================================================================
# 5. AI API 配置推荐
# ==============================================================================
def ai_based_config(task: str, hardware: Dict, dataset: Dict,
                    memory: Dict, hidden_size: int, num_layers: int,
                    use_moe: bool, api_base: str, api_key: str,
                    model_name: str, data_path: List[str]) -> Dict[str, Any]:
    """通过 LLM API 获取推荐配置"""
    try:
        from openai import OpenAI
    except ImportError:
        print("[WARNING] openai 未安装，回退到本地规则模式")
        return rule_based_config(task, hardware, dataset, memory, hidden_size, num_layers, use_moe, data_path)

    defaults = TASK_DEFAULTS.get(task, TASK_DEFAULTS["full_sft"])

    prompt = f"""你是一个深度学习训练配置专家。请为 MiniMind 微型语言模型推荐最优训练超参数。

## 任务
{task}

## 硬件
- GPU: {hardware['gpu_name']} ({hardware['gpu_memory_gb']}GB, {hardware['gpu_backend']})
  {'无可用GPU!' if not hardware['gpu_available'] else ''}
- CPU: {hardware['cpu_cores']}核, RAM: {hardware['ram_gb']}GB

## 模型
- hidden_size={hidden_size}, num_layers={num_layers}, MoE={use_moe}
- 参数量: {memory['total_params_m']}M, 峰值显存估算: {memory['peak_gb']}GB

## 数据集
- 样本数: {dataset['total_samples']}
- 平均长度: {dataset.get('avg_chars', '?')}字符, P50: {dataset.get('char_p50', '?')}字符
- 估算token数/样本: {dataset.get('est_tokens_per_sample', '?')}

## 默认参数
{json.dumps(defaults, indent=2)}

请以 JSON 格式返回推荐配置，包含: batch_size(int), learning_rate(float), max_seq_len(int), epochs(int), accumulation_steps(int), dtype(str), reasoning(str, <100字)。
仅输出JSON，不要其他内容。"""

    client = OpenAI(base_url=api_base, api_key=api_key)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=1024,
    )
    content = resp.choices[0].message.content.strip()
    # Extract JSON
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    ai_config = json.loads(content)

    # Merge with defaults
    config = defaults.copy()
    config.update({k: v for k, v in ai_config.items() if k != "reasoning"})
    config["task"] = task
    config["hidden_size"] = hidden_size
    config["num_hidden_layers"] = num_layers
    config["use_moe"] = use_moe
    config["data_path"] = data_path
    config["save_weight"] = defaults.get("save_weight", task)
    config["from_weight"] = defaults.get("from_weight", "none")
    config["grad_clip"] = 1.0
    config["num_workers"] = min(8, hardware.get("cpu_cores", 4))
    config["log_interval"] = 100
    config["save_interval"] = 100
    config["use_compile"] = False
    config["from_resume"] = False
    config["_source"] = f"ai:{model_name}"
    config["_reasoning"] = ai_config.get("reasoning", "AI推荐")
    return config


# ==============================================================================
# 6. 写入 config.yaml
# ==============================================================================
def write_config_yaml(config: Dict[str, Any], output_path: str):
    """将配置写入统一的 config.yaml 文件"""
    task = config.get("task", "full_sft")
    hidden_size = config.get("hidden_size", 768)
    num_layers = config.get("num_hidden_layers", 8)
    use_moe = config.get("use_moe", False)

    # 提取任务专属参数
    task_keys = {
        "data_path", "from_weight", "save_weight", "batch_size", "learning_rate",
        "epochs", "accumulation_steps", "grad_clip", "max_seq_len", "dtype",
    }
    rl_keys = {"max_gen_len", "max_total_len", "num_generations", "beta",
               "loss_type", "epsilon", "epsilon_high", "thinking_ratio"}
    extra_keys = {"lora_name", "alpha", "temperature"}

    task_params = {}
    for k in task_keys | rl_keys | extra_keys:
        if k in config:
            task_params[k] = config[k]

    source = config.get("_source", "rule_engine")
    reasoning = config.get("_reasoning", "")

    lines = [
        "# =============================================================================",
        "#  MiniMind 统一训练配置",
        f"#  生成方式: {source}",
        f"#  理由: {reasoning}",
        f"#  生成时间: {__import__('datetime').datetime.now().isoformat()}",
        "# =============================================================================",
        "",
        "# ── 全局默认 ────────────────────────────────────────────────────────────",
        "global:",
        f"  hidden_size: {hidden_size}",
        f"  num_hidden_layers: {num_layers}",
        f"  use_moe: {str(use_moe).lower()}",
        f"  max_seq_len: {task_params.get('max_seq_len', 768)}",
        f"  dtype: {task_params.get('dtype', 'bfloat16')}",
        f"  num_workers: {config.get('num_workers', 8)}",
        '  device: ""',
        "",
        "# ── 路径 ────────────────────────────────────────────────────────────────",
        "paths:",
        "  data_dir: ./dataset",
        "  save_dir: ./out",
        "  checkpoint_dir: ./checkpoints",
        "  tokenizer_path: ./model",
        "",
        "# ── Checkpoint ───────────────────────────────────────────────────────────",
        "checkpoint:",
        "  max_keep: 3",
        "  format: torch",
        f"  save_interval: {config.get('save_interval', 100)}",
        "",
        "# ── 日志与监控 ───────────────────────────────────────────────────────────",
        "logging:",
        f"  log_interval: {config.get('log_interval', 100)}",
        "  use_wandb: false",
        "  wandb_project: MiniMind-Lab",
        "  metrics_csv: ./metrics.csv",
        "",
        "# ╔════════════════════════════════════════════════════════════════════════╗",
        "# ║                     当前任务: {:<43s} ║".format(task),
        "# ╚════════════════════════════════════════════════════════════════════════╝",
        "",
        f"# ── 当前任务配置 ────────────────────────────────────────────────────────",
        f"{task}:",
    ]

    # Write task-specific params
    for k, v in task_params.items():
        if isinstance(v, list):
            lines.append(f"  {k}:")
            for item in v:
                lines.append(f"    - {item}")
        elif isinstance(v, str):
            lines.append(f"  {k}: \"{v}\"")
        elif isinstance(v, bool):
            lines.append(f"  {k}: {str(v).lower()}")
        elif isinstance(v, float):
            lines.append(f"  {k}: {v}")
        else:
            lines.append(f"  {k}: {v}")

    lines += [
        "",
        "# ── Rollout Engine ──────────────────────────────────────────────────────",
        "rollout:",
        '  engine: torch',
        '  sglang_base_url: "http://localhost:8998"',
        '  sglang_model_path: "../model"',
        '  sglang_shared_path: "./sglang_ckpt"',
        '  reward_model_path: "../../internlm2-1_8b-reward"',
        "",
        "# ── 高级选项 ─────────────────────────────────────────────────────────────",
        "advanced:",
        f"  use_compile: false",
        f"  debug_mode: false",
        "  debug_interval: 20",
        f"  from_resume: false",
        "",
    ]

    content = "\n".join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\n[OK] 配置已写入: {output_path}")


# ==============================================================================
# 7. CLI 入口
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="MiniMind 智能配置工具 — 分析硬件+数据 → 生成最优 config.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/auto_config.py --data_path ./dataset/pretrain_t2t_mini.jsonl --task pretrain
  python scripts/auto_config.py --data_path ./dataset/sft_data --task full_sft --local
  python scripts/auto_config.py --data_path ./dataset/rlaif.jsonl --task grpo --api_key sk-xxx
  python scripts/auto_config.py --probe
        """
    )
    # Data & Task
    parser.add_argument("--data_path", type=str, nargs='+', default=[],
                        help="训练数据路径（支持多个文件/目录）")
    parser.add_argument("--task", type=str, default="auto",
                        choices=["auto", "pretrain", "full_sft", "lora", "distillation",
                                 "dpo", "grpo", "ppo", "agent"],
                        help="训练任务类型 (auto=自动检测)")
    # Model
    parser.add_argument("--hidden_size", type=int, default=768,
                        help="隐藏层维度 (512/768)")
    parser.add_argument("--num_layers", type=int, default=8,
                        help="Transformer 层数 (4/8)")
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1],
                        help="是否使用 MoE 架构")
    # AI
    parser.add_argument("--api_base", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--local", action="store_true",
                        help="使用本地规则引擎（无需API key）")
    # Output
    parser.add_argument("--output", type=str, default="config.yaml",
                        help="输出配置文件路径 (默认: ./config.yaml)")
    parser.add_argument("--probe", action="store_true",
                        help="仅探测硬件信息，不生成配置")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")

    print("=" * 65)
    print("  MiniMind 智能训练配置工具")
    print("=" * 65)

    # --probe mode
    if args.probe:
        hw = hardware_probe()
        print("\n  硬件信息:")
        print(f"    GPU: {hw['gpu_name']} ({hw['gpu_memory_gb']}GB) {'[可用]' if hw['gpu_available'] else '[不可用]'}")
        print(f"    Backend: {hw['gpu_backend']}")
        print(f"    CPU: {hw['cpu_cores']} 核, RAM: {hw['ram_gb']}GB")
        print(f"    磁盘: {hw['disk_free_gb']}GB 可用")
        print(f"    PyTorch: {hw['torch_version']}")
        return

    if not args.data_path:
        print("\n[ERROR] 请指定 --data_path 参数。")
        print("示例: python scripts/auto_config.py --data_path ./dataset/my_data.jsonl --task pretrain")
        sys.exit(1)

    # ── Step 1: 硬件探测 ──
    print("\n[1/4] 探测硬件...")
    hw = hardware_probe()
    if hw["gpu_available"]:
        print(f"  GPU: {hw['gpu_name']} ({hw['gpu_memory_gb']}GB) [{hw['gpu_backend']}]")
    else:
        print("  GPU: 无可用 GPU (CPU模式)")
    print(f"  CPU: {hw['cpu_cores']} 核 / {hw['ram_gb']}GB RAM")

    # ── Step 2: 数据分析 ──
    print("\n[2/4] 分析数据集...")
    try:
        ds_stats = analyze_dataset(args.data_path)
        print(f"  样本数: {ds_stats['total_samples']:,}")
        print(f"  列: {ds_stats['columns']}")
        if ds_stats.get('avg_chars'):
            print(f"  平均长度: {ds_stats['avg_chars']} 字符 (P50={ds_stats['char_p50']}, P90={ds_stats['char_p90']})")
    except Exception as e:
        print(f"  [ERROR] {e}")
        sys.exit(1)

    # Auto-detect task
    task = args.task
    if task == "auto":
        task = ds_stats.get("detected_task", "full_sft")
        print(f"  [自动检测] 任务类型: {task}")
    else:
        print(f"  任务类型: {task}")

    # ── Step 3: 显存估算 ──
    print("\n[3/4] 估算显存需求...")
    default_bs = TASK_DEFAULTS.get(task, {}).get("batch_size", 16)
    default_seq = TASK_DEFAULTS.get(task, {}).get("max_seq_len", 768)
    mem = estimate_memory(args.hidden_size, args.num_layers, bool(args.use_moe),
                          default_bs, default_seq)
    print(f"  参数量: {mem['total_params_m']}M")
    print(f"  训练峰值显存估算: {mem['peak_gb']}GB (batch={default_bs}, seq={default_seq})")
    if hw["gpu_available"] and mem["peak_gb"] > hw["gpu_memory_gb"]:
        print(f"  [WARNING] 峰值超过GPU显存 ({hw['gpu_memory_gb']}GB)，将自动调整 batch_size")

    # ── Step 4: 生成配置 ──
    print("\n[4/4] 生成训练配置...")
    use_ai = bool(api_key) and not args.local

    if use_ai:
        print("  调用 AI API...")
        try:
            config = ai_based_config(task, hw, ds_stats, mem,
                                     args.hidden_size, args.num_layers,
                                     bool(args.use_moe), args.api_base,
                                     api_key, args.model_name, args.data_path)
        except Exception as e:
            print(f"  [WARNING] AI 调用失败 ({e})，回退到本地规则")
            config = rule_based_config(task, hw, ds_stats, mem,
                                       args.hidden_size, args.num_layers,
                                       bool(args.use_moe), args.data_path)
    else:
        print("  使用本地规则引擎...")
        config = rule_based_config(task, hw, ds_stats, mem,
                                   args.hidden_size, args.num_layers,
                                   bool(args.use_moe), args.data_path)

    # ── 写入 config.yaml ──
    write_config_yaml(config, args.output)

    # ── 打印摘要 ──
    print("\n" + "=" * 65)
    print("  推荐配置")
    print("=" * 65)
    for k, v in config.items():
        if k.startswith("_"):
            continue
        print(f"  {k:24s}: {v}")

    print(f"\n  [推理] {config.get('_reasoning', '')}")
    print(f"\n  配置已写入: {args.output}")
    print(f"  启动训练: python trainer/train_{task if task != 'full_sft' else 'full_sft'}.py")


if __name__ == "__main__":
    main()
