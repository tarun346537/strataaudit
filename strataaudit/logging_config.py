"""JSON logging configuration for audit and training jobs."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from uuid import uuid4

from .config import Settings
from .types import JSONRecord

_trace_id: ContextVar[str] = ContextVar("audit_trace_id", default="")


class JSONFormatter(logging.Formatter):
    """Serialize log events with a consistent run/trace context."""

    def format(self, record: logging.LogRecord) -> str:
        """Write a JSON record without interpolating credentials or input prompts."""
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": _trace_id.get(),
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, allow_nan=False)


def setup_logging(level: str | None = None) -> logging.Logger:
    """Configure the package logger once per CLI job."""
    logger = logging.getLogger("strataaudit")
    logger.setLevel((level or Settings.load().log_level).upper())
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JSONFormatter())
    logger.handlers[:] = [handler]
    logger.propagate = False
    _trace_id.set(uuid4().hex)
    return logger


def record_trace(payload: JSONRecord, trace_id: str) -> JSONRecord:
    """Attach trace metadata and emit one structured audit record."""
    record = {
        **payload,
        "timestamp": datetime.now(UTC).isoformat(),
        "level": "INFO",
        "trace_id": trace_id,
    }
    logging.getLogger("strataaudit.trace").info("audit_trace", extra={"fields": record})
    return record


@contextmanager
def trace_context(trace_id: str) -> Iterator[str]:
    """Group all log events for one audit case or experiment run."""
    token = _trace_id.set(trace_id)
    try:
        yield trace_id
    finally:
        _trace_id.reset(token)
