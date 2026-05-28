"""
MiniMind 自动训练配置工具

通过检测硬件性能 + 分析数据集特征 + 调用大模型 API → 生成最优训练超参数配置。

Usage:
    python scripts/auto_config.py \
        --data_path ../dataset/sft_t2t_mini.jsonl \
        --model_size 768 --num_layers 8 \
        --task full_sft \
        --api_key sk-xxx \
        --output config.json

    # 也可以 dry-run 模式（只输出prompt不调用API）
    python scripts/auto_config.py --data_path ../dataset/sft_t2t_mini.jsonl --dry_run
"""

import os
import sys
import json
import argparse
import warnings

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

warnings.filterwarnings('ignore')


# ============================================================================
# 1. 硬件探测
# ============================================================================

def _guess_gpu_vendor(name: str) -> str:
    """从 GPU 名称推断厂商。"""
    name_lower = name.lower()
    if any(k in name_lower for k in ("amd", "radeon", "instinct", "mi25", "mi50", "mi100", "mi200", "mi300")):
        return "AMD"
    if any(k in name_lower for k in ("nvidia", "geforce", "rtx", "gtx", "tesla", "quadro", "a100", "a10", "a40", "h100", "h200", "b100", "b200", "l40", "t4", "v100", "p100")):
        return "NVIDIA"
    if any(k in name_lower for k in ("intel", "arc", "gaudi", "max")):
        return "Intel"
    return "Unknown"


def hardware_probe():
    """探测 GPU、CPU、内存、磁盘信息，返回结构化 dict。"""
    info = {
        "gpu": {"available": False, "count": 0, "devices": []},
        "cpu": {"cores_physical": 0, "cores_logical": 0, "ram_total_gb": 0},
        "disk": {"free_gb": 0},
        "torch_version": "N/A",
    }

    # GPU — 需要 torch
    try:
        import torch
        info["torch_version"] = torch.__version__

        # 检测 GPU 后端类型
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            info["gpu"]["backend"] = "ROCm (AMD)"
        elif hasattr(torch.backends, 'cudnn'):
            info["gpu"]["backend"] = "CUDA (NVIDIA)"
        elif torch.cuda.is_available():
            info["gpu"]["backend"] = "CUDA-compatible"

        if torch.cuda.is_available():
            info["gpu"]["available"] = True
            info["gpu"]["count"] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vendor = _guess_gpu_vendor(props.name)
                info["gpu"]["devices"].append({
                    "name": props.name,
                    "vendor": vendor,
                    "memory_gb": round(props.total_memory / (1024 ** 3), 1),
                    "sm_count": props.multi_processor_count,
                    "compute_capability": f"{props.major}.{props.minor}",
                })
    except ImportError:
        info["gpu"]["note"] = "torch 未安装，无法探测 GPU 信息"

    # CPU
    try:
        import psutil
        info["cpu"]["cores_physical"] = psutil.cpu_count(logical=False)
        info["cpu"]["cores_logical"] = psutil.cpu_count(logical=True)
        info["cpu"]["ram_total_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        info["disk"]["free_gb"] = round(psutil.disk_usage(sys.path[0]).free / (1024 ** 3), 1)
    except ImportError:
        info["cpu"]["note"] = "psutil 未安装，无法获取 CPU/内存详情"
        info["disk"]["free_gb"] = "unknown"

    return info


# ============================================================================
# 2. 数据集分析
# ============================================================================

def analyze_dataset(data_path, max_seq_len=768):
    """加载数据集，返回统计信息 dict。"""
    try:
        from dataset.data_loader import DataLoaderHub
        hub = DataLoaderHub()
        ds = hub.load_multi(data_path, dataset_type="pretrain", shuffle=False)
    except ImportError as e:
        return {"total_samples": 0, "columns": [], "estimated_total_tokens": 0,
                "avg_sample_chars": 0, "length_distribution": {}, "warnings": [f"依赖缺失: {e}"]}
    except Exception as e:
        return {"total_samples": 0, "columns": [], "estimated_total_tokens": 0,
                "avg_sample_chars": 0, "length_distribution": {}, "warnings": [f"数据加载失败: {e}"]}

    stats = {
        "total_samples": len(ds),
        "format": "unknown",
        "estimated_total_tokens": 0,
        "avg_sample_chars": 0,
        "length_distribution": {},
        "warnings": [],
    }

    # 列信息
    stats["columns"] = ds.column_names

    # 采样计算文本长度分布
    sample_size = min(300, len(ds))
    lengths = []
    text_col = _find_text_column(ds)

    for i in range(sample_size):
        text = str(ds[i].get(text_col, ""))
        lengths.append(len(text))

    if lengths:
        lengths.sort()
        stats["avg_sample_chars"] = round(sum(lengths) / len(lengths), 1)
        stats["length_distribution"] = {
            "min": lengths[0],
            "p25": lengths[len(lengths) // 4],
            "p50": lengths[len(lengths) // 2],
            "p75": lengths[3 * len(lengths) // 4],
            "max": lengths[-1],
        }
        # 粗略估算 token 数（中文约 1.5 字符/token，英文约 4 字符/token）
        avg_tokens_per_char = 0.5  # 混合中英文的粗略估计
        stats["estimated_avg_tokens_per_sample"] = round(stats["avg_sample_chars"] * avg_tokens_per_char)
        stats["estimated_total_tokens"] = stats["total_samples"] * stats["estimated_avg_tokens_per_sample"]

    # 检查潜在问题
    if stats["total_samples"] < 1000:
        stats["warnings"].append("数据量较少 (<1000)，建议增加数据或减少 epoch 数")
    if stats.get("estimated_avg_tokens_per_sample", 0) > max_seq_len:
        stats["warnings"].append(f"平均 token 数超过 max_seq_len({max_seq_len})，部分数据会被截断")

    return stats


def _find_text_column(ds):
    """找到数据集中的文本列。"""
    for col in ds.column_names:
        col_lower = col.lower()
        if col_lower in ("text", "content", "prompt", "conversations", "messages"):
            return col
    return ds.column_names[0] if ds.column_names else "text"


# ============================================================================
# 3. 硬件参数估计
# ============================================================================

def estimate_memory(hidden_size, num_layers, use_moe, batch_size, max_seq_len, dtype="bfloat16"):
    """估算训练峰值显存需求 (GB)。"""
    bytes_per_param = 2 if dtype in ("bfloat16", "float16") else 4

    # MiniMind 参数量近似公式
    # Embedding: vocab(6400) × hidden
    # Per layer: 4 × hidden² (QKV+out proj) + 8 × hidden² (FFN gate+up+down, intermediate≈4×hidden)
    # Final norm + lm_head
    vocab = 6400
    embed = vocab * hidden_size
    per_layer = 12 * hidden_size * hidden_size
    if use_moe:
        per_layer *= 1.8  # MoE 约增加 80% 参数量

    total_params = embed + num_layers * per_layer
    model_gb = total_params * bytes_per_param / (1024 ** 3)

    # 训练时额外需求
    grad_gb = model_gb  # 梯度
    optimizer_gb = total_params * 8 / (1024 ** 3)  # AdamW (momentum + variance, both fp32)
    # 激活内存粗略估算
    activation_gb = batch_size * max_seq_len * hidden_size * num_layers * 2 / (1024 ** 3)

    peak_gb = model_gb + grad_gb + optimizer_gb + activation_gb

    return {
        "total_params_m": round(total_params / 1e6, 1),
        "model_gb": round(model_gb, 2),
        "grad_gb": round(grad_gb, 2),
        "optimizer_gb": round(optimizer_gb, 2),
        "activation_gb": round(activation_gb, 2),
        "peak_estimate_gb": round(peak_gb, 2),
    }


# ============================================================================
# 4. 构建 LLM Prompt
# ============================================================================

def build_prompt(hardware, dataset_stats, memory, task, model_size, num_layers, use_moe):
    """构建发送给大模型的配置推荐 prompt。"""

    backend = hardware["gpu"].get("backend", "")
    backend_note = f" (后端: {backend})" if backend else ""
    gpu_info = "无可用 GPU"
    if hardware["gpu"]["available"]:
        gpu_info = "\n".join(
            f"  - [{d.get('vendor', '?')}] {d['name']}: {d['memory_gb']}GB 显存, "
            f"{d['sm_count']} SM, 计算能力 {d['compute_capability']}"
            for d in hardware["gpu"]["devices"]
        ) + backend_note

    cpu_info = (
        f"{hardware['cpu']['cores_physical']} 物理核 / {hardware['cpu']['cores_logical']} 逻辑核, "
        f"{hardware['cpu']['ram_total_gb']}GB RAM"
    )

    dataset_info = json.dumps(dataset_stats, ensure_ascii=False, indent=2)
    memory_info = json.dumps(memory, ensure_ascii=False, indent=2)

    prompt = f"""你是一个深度学习训练配置专家。请根据以下信息，为 MiniMind 语言模型推荐最优训练超参数。

## 任务类型
{task}

## 硬件环境
- PyTorch 版本: {hardware['torch_version']}
- GPU:
{gpu_info}
- CPU: {cpu_info}
- 可用磁盘: {hardware['disk']['free_gb']}GB

## 模型架构
- hidden_size: {model_size}
- num_hidden_layers: {num_layers}
- use_moe: {use_moe}

## 显存估算
{memory_info}

## 数据集统计
{dataset_info}

## MiniMind 训练参数参考范围

### 预训练 (pretrain)
- batch_size: 16-128 (小模型可更大)
- learning_rate: 1e-4 ~ 5e-4 (CosineAnnealingLR, min=lr/10)
- max_seq_len: 256-1024
- epochs: 1-5
- dtype: bfloat16 (推荐) / float16
- accumulation_steps: 1-4

### SFT 全量微调 (full_sft)
- batch_size: 16-64
- learning_rate: 5e-6 ~ 5e-5
- max_seq_len: 256-768
- epochs: 1-6
- dtype: bfloat16

### LoRA 微调 (lora)
- batch_size: 16-64
- learning_rate: 1e-5 ~ 1e-4
- lora_rank: 8 (默认)
- epochs: 6-10

### 蒸馏 (distillation)
- batch_size: 16-32
- learning_rate: 5e-6 ~ 1e-5
- temperature: 1.0-2.0
- alpha: 0.3-0.7

### DPO
- batch_size: 2-8
- learning_rate: 1e-8 ~ 1e-7
- beta: 0.1-0.5
- epochs: 1-2

### GRPO/PPO/Agent RL
- batch_size: 1-4
- learning_rate: 1e-7 ~ 5e-7
- num_generations: 4-8
- beta: 0.05-0.2
- epochs: 1-2
- max_gen_len: 512-1024

## 输出要求
请以 JSON 格式输出推荐配置，包含以下字段：
- batch_size: int
- learning_rate: float
- max_seq_len: int
- epochs: int
- accumulation_steps: int
- dtype: str
- estimated_gpu_memory_gb: float (使用上面估算的峰值)
- reasoning: str (推荐理由，100字以内)

只输出 JSON，不要包含其他文字。"""

    return prompt


# ============================================================================
# 5. 调用大模型 API
# ============================================================================

def get_config_from_llm(prompt, api_base, api_key, model_name):
    """通过 OpenAI 兼容 API 调用大模型获取推荐配置。"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请安装 openai: pip install openai")

    client = OpenAI(base_url=api_base, api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你是一个深度学习训练配置专家。请严格按照 JSON 格式输出。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    content = response.choices[0].message.content.strip()
    # 尝试提取 JSON（处理可能的 markdown 代码块）
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        config = json.loads(content)
        return config
    except json.JSONDecodeError:
        print(f"[ERROR] LLM 返回内容无法解析为 JSON，原始内容:\n{content}")
        return None


# ============================================================================
# 6. 输出配置
# ============================================================================

def format_training_command(config, task, data_path, model_size, num_layers, use_moe, save_weight):
    """根据任务类型生成对应的训练命令。"""
    task_map = {
        "pretrain": "train_pretrain.py",
        "full_sft": "train_full_sft.py",
        "lora": "train_lora.py",
        "distillation": "train_distillation.py",
        "dpo": "train_dpo.py",
        "grpo": "train_grpo.py",
        "ppo": "train_ppo.py",
        "agent": "train_agent.py",
    }

    script = task_map.get(task, f"train_{task}.py")
    data_str = " ".join(data_path) if isinstance(data_path, list) else data_path

    cmd_parts = [
        f"python trainer/{script}",
        f"--data_path {data_str}",
        f"--hidden_size {model_size}",
        f"--num_hidden_layers {num_layers}",
        f"--use_moe {use_moe}",
        f"--batch_size {config.get('batch_size', 16)}",
        f"--learning_rate {config.get('learning_rate', 1e-5)}",
        f"--max_seq_len {config.get('max_seq_len', 768)}",
        f"--epochs {config.get('epochs', 1)}",
        f"--accumulation_steps {config.get('accumulation_steps', 1)}",
        f"--dtype {config.get('dtype', 'bfloat16')}",
    ]

    if save_weight:
        cmd_parts.append(f"--save_weight {save_weight}")

    return " \\\n    ".join(cmd_parts)


def apply_config(config, output_path, task, data_path, model_size, num_layers, use_moe, save_weight):
    """将配置写入 JSON 文件并打印训练命令。"""
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"[OK] 配置已保存到 {output_path}")

    print("\n" + "=" * 70)
    print("推荐训练配置")
    print("=" * 70)
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print("=" * 70)

    if config.get("reasoning"):
        print(f"\n推荐理由: {config['reasoning']}")

    cmd = format_training_command(config, task, data_path, model_size, num_layers, use_moe, save_weight)
    print(f"\n推荐训练命令:\n{cmd}")
    print()


# ============================================================================
# CLI 入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind 自动训练配置工具")
    parser.add_argument("--data_path", type=str, nargs='+', default=["../dataset/sft_t2t_mini.jsonl"],
                        help="训练数据路径（支持多个文件/目录）")
    parser.add_argument("--model_size", type=int, default=768, help="隐藏层维度 (hidden_size)")
    parser.add_argument("--num_layers", type=int, default=8, help="Transformer 层数")
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1], help="是否使用 MoE 架构")
    parser.add_argument("--task", type=str, default="full_sft",
                        choices=["pretrain", "full_sft", "lora", "distillation", "dpo", "grpo", "ppo", "agent"],
                        help="训练任务类型")
    parser.add_argument("--save_weight", type=str, default="", help="保存权重的前缀名")
    parser.add_argument("--api_base", type=str, default="https://api.openai.com/v1",
                        help="大模型 API 地址（兼容 OpenAI 接口）")
    parser.add_argument("--api_key", type=str, default="", help="API Key（也可通过环境变量 OPENAI_API_KEY 设置）")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="大模型名称")
    parser.add_argument("--output", type=str, default="", help="输出 JSON 配置文件路径")
    parser.add_argument("--dry_run", action="store_true", help="仅打印发送给 LLM 的 prompt，不实际调用 API")
    args = parser.parse_args()

    # API key 优先级：命令行 > 环境变量
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")

    print("=" * 70)
    print("MiniMind 自动训练配置工具")
    print("=" * 70)

    # Step 1: 硬件探测
    print("\n[1/4] 探测硬件...")
    hardware = hardware_probe()
    if hardware["gpu"]["available"]:
        for d in hardware["gpu"]["devices"]:
            print(f"  GPU: {d['name']} ({d['memory_gb']}GB)")
    else:
        print("  GPU: 无可用 CUDA 设备")
    print(f"  CPU: {hardware['cpu'].get('cores_logical', 'N/A')} 逻辑核 / {hardware['cpu'].get('ram_total_gb', 'N/A')}GB RAM")

    # Step 2: 数据集分析
    print("\n[2/4] 分析数据集...")
    max_seq_len = 768  # 默认值，用于初步分析
    try:
        dataset_stats = analyze_dataset(args.data_path, max_seq_len)
        print(f"  样本数: {dataset_stats['total_samples']}")
        print(f"  列: {dataset_stats['columns']}")
        if dataset_stats.get("estimated_total_tokens"):
            print(f"  估算总 token 数: {dataset_stats['estimated_total_tokens']:,}")
        if dataset_stats["warnings"]:
            for w in dataset_stats["warnings"]:
                print(f"  [WARNING] {w}")
    except Exception as e:
        print(f"  [ERROR] 数据集分析失败: {e}")
        dataset_stats = {"total_samples": 0, "columns": [], "estimated_total_tokens": 0,
                         "avg_sample_chars": 0, "length_distribution": {}, "warnings": [str(e)]}

    # Step 3: 显存估算
    print("\n[3/4] 估算显存需求...")
    memory = estimate_memory(
        args.model_size, args.num_layers, bool(args.use_moe),
        batch_size=16,  # 用默认 batch_size 做初步估算
        max_seq_len=max_seq_len,
        dtype="bfloat16",
    )
    print(f"  模型参数量: {memory['total_params_m']}M")
    print(f"  训练峰值显存估算: {memory['peak_estimate_gb']}GB")
    gpu_ok = True
    if hardware["gpu"]["available"]:
        max_gpu_mem = max(d["memory_gb"] for d in hardware["gpu"]["devices"])
        if memory['peak_estimate_gb'] > max_gpu_mem:
            print(f"  [WARNING] 估算峰值 ({memory['peak_estimate_gb']}GB) 超过 GPU 显存 ({max_gpu_mem}GB)，"
                  f"建议减小 batch_size 或使用梯度累积")
            gpu_ok = False

    # Step 4: 构建 Prompt 并调用 LLM
    prompt = build_prompt(
        hardware, dataset_stats, memory, args.task,
        args.model_size, args.num_layers, bool(args.use_moe),
    )

    if args.dry_run:
        print("\n[4/4] Dry Run — 仅打印 Prompt\n")
        print("=" * 70)
        print(prompt)
        print("=" * 70)
    else:
        print("\n[4/4] 调用大模型 API...")
        if not api_key:
            print("  [ERROR] 未设置 API Key。请通过 --api_key 或环境变量 OPENAI_API_KEY 提供。")
            print("  提示：可以使用 --dry_run 查看发送给 LLM 的 prompt。")
            sys.exit(1)

        try:
            config = get_config_from_llm(prompt, args.api_base, api_key, args.model_name)
            if config:
                # 补充显存估算到配置中
                config["estimated_gpu_memory_gb"] = memory["peak_estimate_gb"]
                apply_config(config, args.output, args.task, args.data_path,
                           args.model_size, args.num_layers, bool(args.use_moe), args.save_weight)
            else:
                print("[ERROR] 未能获取有效配置，请重试或使用 --dry_run 检查 prompt。")
                sys.exit(1)
        except Exception as e:
            print(f"[ERROR] API 调用失败: {e}")
            print("提示：可以使用 --dry_run 查看发送给 LLM 的 prompt。")
            sys.exit(1)
