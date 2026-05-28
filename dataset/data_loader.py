"""
Multi-format data loading hub. Supports JSONL, JSON, CSV, Parquet, TXT with
auto-detection, multi-file merging, and schema normalization.
"""
import os
import glob
import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Callable
from datasets import load_dataset, Dataset, concatenate_datasets


# Canonical field names per dataset type
PRETRAIN_FIELDS = {"text"}
SFT_FIELDS = {"conversations"}
DPO_FIELDS = {"chosen", "rejected"}
RLAIF_FIELDS = {"conversations"}
AGENT_FIELDS = {"conversations", "gt"}

# Maps common alternative column names to canonical ones
SCHEMA_ALIASES: Dict[str, Dict[str, str]] = {
    "pretrain": {
        "content": "text", "document": "text", "passage": "text",
        "sentence": "text", "body": "text", "article": "text",
    },
    "sft": {
        "messages": "conversations", "dialog": "conversations",
        "dialogue": "conversations", "chat": "conversations",
        "conversation": "conversations", "history": "conversations",
    },
    "dpo": {
        "preferred": "chosen", "chosen_response": "chosen",
        "dispreferred": "rejected", "rejected_response": "rejected",
        "positive": "chosen", "negative": "rejected",
    },
    "rlaif": {
        "messages": "conversations", "dialog": "conversations",
        "chat": "conversations",
    },
    "agent": {
        "messages": "conversations", "dialog": "conversations",
        "ground_truth": "gt", "answers": "gt", "answer": "gt",
    },
}


def detect_format(path: str) -> str:
    """Detect file format by extension, with content sniffing fallback."""
    ext = Path(path).suffix.lower()
    mapping = {
        ".jsonl": "json",
        ".json": "json",
        ".csv": "csv",
        ".tsv": "csv",
        ".parquet": "parquet",
        ".pq": "parquet",
        ".txt": "text",
        ".text": "text",
        ".md": "text",
    }
    if ext in mapping:
        return mapping[ext]

    # Fallback: sniff first bytes
    try:
        with open(path, "rb") as f:
            head = f.read(4)
        if head == b"PAR1":
            return "parquet"
        if head.startswith(b"{"):
            return "json"
        if head.startswith(b"["):
            return "json"
    except Exception:
        pass
    return "json"  # default fallback


def _load_single(path: str, fmt: str) -> Dataset:
    """Load a single file into a HuggingFace Dataset."""
    if fmt == "json":
        try:
            return load_dataset("json", data_files=path, split="train")
        except Exception:
            # Maybe it's a JSON array; try line-by-line fallback
            samples = []
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content.startswith("["):
                samples = json.loads(content)
            else:
                for line in content.splitlines():
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
            return Dataset.from_list(samples)

    if fmt == "csv":
        return load_dataset("csv", data_files=path, split="train")

    if fmt == "parquet":
        return load_dataset("parquet", data_files=path, split="train")

    if fmt == "text":
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append({"text": line})
        return Dataset.from_list(samples)

    raise ValueError(f"Unsupported format: {fmt}")


def expand_paths(paths: Union[str, List[str]]) -> List[str]:
    """Expand a path spec (single file, list, directory, glob) into concrete file paths."""
    if isinstance(paths, str):
        paths = [paths]

    expanded = []
    for p in paths:
        if os.path.isdir(p):
            expanded.extend(sorted(
                f for f in glob.glob(os.path.join(p, "*"))
                if os.path.isfile(f)
            ))
        elif "*" in p or "?" in p:
            expanded.extend(sorted(glob.glob(p)))
        else:
            expanded.append(p)

    # Filter: only keep supported formats
    supported = {".jsonl", ".json", ".csv", ".tsv", ".parquet", ".pq", ".txt", ".text", ".md"}
    return [f for f in expanded if Path(f).suffix.lower() in supported or os.path.isfile(f)]


def normalize_schema(ds: Dataset, dataset_type: str) -> Dataset:
    """Rename columns to canonical names based on dataset type."""
    aliases = SCHEMA_ALIASES.get(dataset_type, {})
    rename = {}
    for col in ds.column_names:
        if col in aliases:
            rename[col] = aliases[col]
    if rename:
        ds = ds.rename_columns(rename)
    return ds


def validate_schema(ds: Dataset, dataset_type: str) -> Dataset:
    """Remove samples missing required fields for the dataset type."""
    required_map = {
        "pretrain": PRETRAIN_FIELDS,
        "sft": SFT_FIELDS,
        "dpo": DPO_FIELDS,
        "rlaif": RLAIF_FIELDS,
        "agent": AGENT_FIELDS,
    }
    required = required_map.get(dataset_type, set())
    if not required:
        return ds

    available = set(ds.column_names)

    def has_required(sample):
        return all(f in sample and sample[f] is not None for f in required & available)

    return ds.filter(has_required)


class DataLoaderHub:
    """Unified multi-format, multi-file data loader.

    Usage:
        hub = DataLoaderHub()
        ds = hub.load_multi(["data/train.jsonl", "data/extra.csv"], dataset_type="pretrain")
    """

    def __init__(self):
        self._cache: Dict[str, Dataset] = {}

    def load(self, path: str) -> Dataset:
        """Load a single file with format auto-detection."""
        if path in self._cache:
            return self._cache[path]
        fmt = detect_format(path)
        ds = _load_single(path, fmt)
        self._cache[path] = ds
        return ds

    def load_multi(
        self,
        paths: Union[str, List[str]],
        dataset_type: str = "pretrain",
        shuffle: bool = True,
        seed: int = 42,
        process_fn: Optional[Callable] = None,
    ) -> Dataset:
        """Load and merge multiple datasets with schema normalization.

        Args:
            paths: File path(s), directory path(s), or glob pattern(s).
            dataset_type: One of 'pretrain', 'sft', 'dpo', 'rlaif', 'agent'.
            shuffle: Shuffle the merged dataset.
            seed: Random seed for shuffling.
            process_fn: Optional per-sample preprocessing function.
        """
        files = expand_paths(paths)
        if not files:
            raise FileNotFoundError(f"No supported files found in: {paths}")

        datasets = []
        for f in files:
            ds = self.load(f)
            ds = normalize_schema(ds, dataset_type)
            ds = validate_schema(ds, dataset_type)
            if process_fn is not None:
                ds = ds.map(process_fn, desc=f"Processing {os.path.basename(f)}")
            datasets.append(ds)

        if len(datasets) == 1:
            merged = datasets[0]
        else:
            merged = concatenate_datasets(datasets)

        if shuffle:
            merged = merged.shuffle(seed=seed)

        return merged


def load_pretrain_data(paths: Union[str, List[str]], **kwargs) -> Dataset:
    """Convenience: load pretrain dataset(s)."""
    return DataLoaderHub().load_multi(paths, dataset_type="pretrain", **kwargs)


def load_sft_data(paths: Union[str, List[str]], **kwargs) -> Dataset:
    """Convenience: load SFT dataset(s)."""
    return DataLoaderHub().load_multi(paths, dataset_type="sft", **kwargs)


def load_dpo_data(paths: Union[str, List[str]], **kwargs) -> Dataset:
    """Convenience: load DPO dataset(s)."""
    return DataLoaderHub().load_multi(paths, dataset_type="dpo", **kwargs)


def load_rlaif_data(paths: Union[str, List[str]], **kwargs) -> Dataset:
    """Convenience: load RLAIF dataset(s)."""
    return DataLoaderHub().load_multi(paths, dataset_type="rlaif", **kwargs)


def load_agent_data(paths: Union[str, List[str]], **kwargs) -> Dataset:
    """Convenience: load Agent RL dataset(s)."""
    return DataLoaderHub().load_multi(paths, dataset_type="agent", **kwargs)
