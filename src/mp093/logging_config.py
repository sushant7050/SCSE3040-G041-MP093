"""JSON logging (lab P10): one JSON object per line, written to stdout only.

Usage:
    from mp093.logging_config import get_logger
    log = get_logger("mp093.api")
    log.info("prediction", extra={"event": "prediction", "latency_ms": 1.2})
"""

import json
import logging
import sys
from datetime import datetime, timezone

# Attributes every LogRecord has. Anything else on a record came from extra={...}.
_STANDARD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {
    "message", "asctime", "taskName",
}


class JsonFormatter(logging.Formatter):
    """Format a record as one JSON line: timestamp, level, logger, message, extras."""

    def format(self, record):
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                entry[key] = value
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


class StdoutHandler(logging.StreamHandler):
    """StreamHandler on whatever sys.stdout is right now (pytest's capsys swaps it)."""

    def emit(self, record):
        self.stream = sys.stdout
        super().emit(record)


def get_logger(name="mp093", level=logging.INFO):
    """A logger that writes JSON lines to sys.stdout. Never to a file."""
    logger = logging.getLogger(name)
    if not any(isinstance(h, StdoutHandler) for h in logger.handlers):
        handler = StdoutHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False   # no duplicate lines through the root logger
    return logger
