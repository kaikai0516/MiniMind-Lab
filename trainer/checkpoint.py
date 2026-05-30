"""
Checkpoint manager for MiniMind.

Provides ``LMCheckpointer`` — a drop-in replacement for the legacy
``lm_checkpoint()`` function with safetensors support, metadata tracking,
and automatic rotation of old checkpoints.

Usage::

    from trainer.checkpoint import LMCheckpointer
    ckp = LMCheckpointer(config=training_config)
    ckp.save(model, optimizer, epoch=0, step=100)
    state = ckp.load("pretrain")
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from trainer.exceptions import CheckpointError

# torch is imported lazily — this module is importable without torch installed
if TYPE_CHECKING:
    import torch
    from torch.nn.parallel import DistributedDataParallel

# safetensors is optional — fall back to torch.save
try:
    from safetensors.torch import save_file as _safe_save
    from safetensors.torch import load_file as _safe_load

    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False


def _import_torch():
    import torch
    from torch.nn.parallel import DistributedDataParallel  # noqa: F811
    return torch, DistributedDataParallel


def _get_raw_model(model):
    """Unwrap DDP / torch.compile to get the bare model."""
    _, DistributedDataParallel = _import_torch()
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(raw, "_orig_mod", raw)


def _config_hash(config: Any) -> str:
    """Short hash of config for metadata tracking."""
    data = str(vars(config) if hasattr(config, "__dict__") else config).encode()
    return hashlib.sha256(data).hexdigest()[:8]


class LMCheckpointer:
    """Checkpoint manager with safetensors support and rotation.

    Parameters
    ----------
    save_dir:
        Directory for checkpoint files.
    max_keep:
        Maximum number of checkpoints to retain (oldest removed first).
        Set to 0 for unlimited.
    fmt:
        ``"safetensors"`` (default) or ``"torch"``.
    """

    def __init__(
        self,
        save_dir: str = "../checkpoints",
        max_keep: int = 3,
        fmt: str = "safetensors",
    ):
        self.save_dir = Path(save_dir)
        self.max_keep = max_keep
        self.fmt = fmt if fmt == "torch" or _HAS_SAFETENSORS else "torch"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ── save ──────────────────────────────────────────────────────

    def save(
        self,
        model,
        optimizer=None,
        epoch: int = 0,
        step: int = 0,
        weight_name: str = "checkpoint",
        lora_state: Optional[Dict] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save model weights + optimizer state + metadata.

        Returns the path to the saved weight file.
        """
        torch, _ = _import_torch()
        raw = _get_raw_model(model)
        state_dict = {k: v.cpu() for k, v in raw.state_dict().items()}

        suffix = ".safetensors" if self.fmt == "safetensors" else ".pth"
        weight_path = self.save_dir / f"{weight_name}{suffix}"

        # atomic write
        tmp = str(weight_path) + ".tmp"
        if self.fmt == "safetensors":
            _safe_save(state_dict, tmp)
        else:
            torch.save(state_dict, tmp)
        os.replace(tmp, str(weight_path))

        # metadata
        meta: Dict[str, Any] = {
            "weight_name": weight_name,
            "epoch": epoch,
            "step": step,
            "config_hash": _config_hash(raw.config if hasattr(raw, "config") else "unknown"),
            "timestamp": time.time(),
            "format": self.fmt,
        }
        if extra:
            meta["extra"] = extra

        meta_path = self.save_dir / f"{weight_name}_meta.json"
        with open(str(meta_path) + ".tmp", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        os.replace(str(meta_path) + ".tmp", str(meta_path))

        # optimizer (always torch format due to complex state)
        if optimizer is not None:
            opt_path = self.save_dir / f"{weight_name}_optim.pth"
            torch.save(optimizer.state_dict(), str(opt_path) + ".tmp")
            os.replace(str(opt_path) + ".tmp", str(opt_path))

        # LoRA weights
        if lora_state is not None:
            lora_path = self.save_dir / f"{weight_name}_lora.pth"
            torch.save(lora_state, str(lora_path) + ".tmp")
            os.replace(str(lora_path) + ".tmp", str(lora_path))

        # rotation
        self._rotate(weight_name)

        return str(weight_path)

    # ── load ──────────────────────────────────────────────────────

    def load(
        self,
        weight_name: str = "checkpoint",
        map_location: str = "cpu",
    ) -> Dict:
        """Load model weights. Tries safetensors first, then torch."""
        torch, _ = _import_torch()
        for ext in (".safetensors", ".pth"):
            path = self.save_dir / f"{weight_name}{ext}"
            if path.exists():
                if ext == ".safetensors":
                    return _safe_load(str(path))
                return torch.load(str(path), map_location=map_location, weights_only=False)

        raise CheckpointError(f"No checkpoint found for '{weight_name}' in {self.save_dir}")

    def load_optimizer(
        self, weight_name: str = "checkpoint", map_location: str = "cpu"
    ) -> Optional[Dict]:
        torch, _ = _import_torch()
        path = self.save_dir / f"{weight_name}_optim.pth"
        if path.exists():
            return torch.load(str(path), map_location=map_location, weights_only=False)
        return None

    def load_metadata(self, weight_name: str = "checkpoint") -> Optional[Dict]:
        path = self.save_dir / f"{weight_name}_meta.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    # ── internal ──────────────────────────────────────────────────

    def _rotate(self, current_name: str) -> None:
        if self.max_keep <= 0:
            return

        # collect all checkpoint base names sorted by timestamp
        metas = sorted(
            self.save_dir.glob("*_meta.json"),
            key=lambda p: p.stat().st_mtime,
        )
        # don't count the current checkpoint
        keep_basenames = {current_name}
        for m in metas:
            bn = m.stem.replace("_meta", "")
            keep_basenames.add(bn)

        # remove oldest first
        removed = 0
        target = max(0, len(keep_basenames) - self.max_keep)
        for m in metas:
            bn = m.stem.replace("_meta", "")
            if bn == current_name:
                continue
            if removed >= target:
                break
            self._remove_one(bn)
            keep_basenames.discard(bn)
            removed += 1

    def _remove_one(self, base_name: str) -> None:
        for pattern in (f"{base_name}.safetensors", f"{base_name}.pth",
                        f"{base_name}_meta.json", f"{base_name}_optim.pth",
                        f"{base_name}_lora.pth"):
            p = self.save_dir / pattern
            if p.exists():
                p.unlink()

    # ── list ──────────────────────────────────────────────────────

    def list_checkpoints(self) -> list:
        """Return sorted list of (name, step, timestamp) for all checkpoints."""
        result = []
        for m in sorted(self.save_dir.glob("*_meta.json")):
            name = m.stem.replace("_meta", "")
            meta = self.load_metadata(name) or {}
            result.append((name, meta.get("step", 0), meta.get("timestamp", 0)))
        return sorted(result, key=lambda x: x[2], reverse=True)
