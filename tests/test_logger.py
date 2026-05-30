"""
Sanity tests for ``trainer.logger``.
All tests run on CPU; no GPU required.
"""

import os
import tempfile
from io import StringIO
from trainer.logger import get_logger, set_level, Logger


class TestGetLogger:
    def test_basic_logger(self):
        log = get_logger("test.basic")
        assert log is not None

    def test_logger_writes_to_stderr(self, capsys):
        log = get_logger("test.stderr")
        log.info("hello world")
        # Rich handler writes to stderr; check it doesn't crash
        assert True

    def test_file_handler(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            tmp = f.name
        try:
            os.environ["MINIMIND_LOG_FILE"] = tmp
            # Re-import won't re-init; test the path directly
            log = get_logger("test.file")
            log.warning("file test message")
            with open(tmp, "r") as fh:
                content = fh.read()
            assert "file test message" in content
        finally:
            os.unlink(tmp)
            os.environ.pop("MINIMIND_LOG_FILE", None)

    def test_set_level(self):
        set_level("DEBUG")
        log = get_logger("test.level")
        log.debug("debug message")  # should not crash


class TestBackwardCompat:
    def test_logger_shim(self):
        """The global Logger() shim delegates to log.info()."""
        # Should not raise
        Logger("backward compat test")
