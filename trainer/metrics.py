"""
Training metrics CSV writer for Jupyter visualization.

Writes one row per ``log_interval`` step so notebooks can live-plot
training progress without needing wandb/swanlab.

Usage (in training scripts)::

    from trainer.metrics import MetricsWriter
    writer = MetricsWriter("./metrics.csv")
    writer.write(step=100, loss=2.34, lr=5e-5, ...)
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


class MetricsWriter:
    """Thread-safe CSV writer for training metrics.

    Parameters
    ----------
    path:
        CSV file path (overwritten on first write).
    flush_every:
        Flush to disk every N writes (default 1 = every write).
    """

    def __init__(self, path: str = "./metrics.csv", flush_every: int = 1):
        self.path = Path(path)
        self.flush_every = flush_every
        self._write_count = 0
        self._header_written = False
        self._fieldnames: Optional[list] = None

    def write(self, **metrics: Any) -> None:
        """Write one row of metrics.  Extra keys are added as new columns."""
        row = {"timestamp": time.time(), "step": metrics.pop("step", 0)}
        # determine field order: timestamp, step, then remaining keys
        fields = ["timestamp", "step"] + [k for k in metrics if k not in ("timestamp", "step")]
        row.update(metrics)

        if not self._header_written or fields != self._fieldnames:
            # write header on first call or when fields change
            self._fieldnames = fields
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
            self._header_written = True

        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._fieldnames)
            w.writerow(row)

            self._write_count += 1
            if self._write_count % self.flush_every == 0:
                f.flush()
                os.fsync(f.fileno())

    @property
    def csv_path(self) -> str:
        return str(self.path.resolve())
