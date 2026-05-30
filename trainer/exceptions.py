"""
Custom exception hierarchy for MiniMind.

Usage::

    from trainer.exceptions import ConfigError, CheckpointError
    raise ConfigError(f"Unknown task type: {task}")
"""


class MiniMindError(Exception):
    """Base exception for all MiniMind errors."""


class ConfigError(MiniMindError):
    """Configuration-related errors (missing keys, invalid values)."""


class CheckpointError(MiniMindError):
    """Checkpoint save/load errors (corrupt file, missing data)."""


class DataLoadError(MiniMindError):
    """Dataset loading errors (missing files, invalid format)."""


class ModelBuildError(MiniMindError):
    """Model construction errors (invalid architecture params)."""


class TrainingError(MiniMindError):
    """Runtime training errors (NaN loss, OOM specifics)."""
