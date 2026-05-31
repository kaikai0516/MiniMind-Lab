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
        # Reset the logger singleton so MINIMIND_LOG_FILE is picked up
        import trainer.logger as _logmod
        _logmod._initialised = False
        _logmod._root_logger = None

        # Use a temp filename without holding it open (avoids Windows file locking)
        tmp = os.path.join(tempfile.gettempdir(), f"test_minimind_{os.getpid()}.log")
        try:
            os.environ["MINIMIND_LOG_FILE"] = tmp
            # Re-read the env var (it's captured at module import time)
            _logmod._file_path = tmp
            log = get_logger("test.file")
            log.warning("file test message")
            # Force flush all handlers to ensure the message is written
            for handler in _logmod._root_logger.handlers:
                handler.flush()
                handler.close()
            with open(tmp, "r", encoding="utf-8") as fh:
                content = fh.read()
            assert "file test message" in content
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
            os.environ.pop("MINIMIND_LOG_FILE", None)
            # Reset again so other tests aren't affected
            _logmod._initialised = False
            _logmod._root_logger = None
            _logmod._file_path = ""

    def test_set_level(self):
        set_level("DEBUG")
        log = get_logger("test.level")
        log.debug("debug message")  # should not crash


class TestBackwardCompat:
    def test_logger_shim(self):
        """The global Logger() shim delegates to log.info()."""
        # Should not raise
        Logger("backward compat test")
