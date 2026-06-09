"""Bounded retries and a circuit breaker for read-only model requests."""

from __future__ import annotations

import json
import logging
import math
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from uuid import uuid4

from .constants import MAX_RESPONSE_BYTES
from .types import JSONRecord

logger = logging.getLogger(__name__)
_attempt_limit: ContextVar[int | None] = ContextVar("model_attempt_limit", default=None)


class InferenceError(RuntimeError):
    """A model service is unavailable or returned an invalid response."""


@contextmanager
def model_attempt_limit(limit: int) -> Iterator[None]:
    """Bound physical HTTP attempts when a caller accounts per-model-call cost."""
    if limit < 1:
        raise ValueError("model attempt limit must be positive")
    token = _attempt_limit.set(limit)
    try:
        yield
    finally:
        _attempt_limit.reset(token)


@dataclass
class HTTPTransport:
    """Retry transient inference failures while refusing repeated outages."""

    timeout: float
    retries: int
    backoff: float
    breaker_failures: int
    breaker_reset: float
    attempts: int = field(default=0, init=False)
    consecutive_failures: int = field(default=0, init=False)
    blocked_until: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) for value in [self.timeout, self.backoff, self.breaker_reset]
        ):
            raise ValueError(
                "http timing settings must be finite. check model client configuration"
            )
        if self.timeout <= 0 or self.backoff < 0 or self.breaker_reset <= 0:
            raise ValueError("use positive HTTP timeout/reset and nonnegative retry backoff")
        if self.retries < 0 or self.breaker_failures < 1:
            raise ValueError(
                "use nonnegative HTTP retries and a positive circuit failure threshold"
            )

    def request(self, url: str, body: JSONRecord, headers: dict[str, str]) -> JSONRecord:
        """Send an inference request with a stable correlation ID across retries."""
        if time.monotonic() < self.blocked_until:
            raise InferenceError(
                "model circuit is open. wait for its reset interval or check the server"
            )
        maximum = self.retries + 1
        limit = _attempt_limit.get()
        if limit is not None:
            maximum = min(maximum, limit)
        request = urllib.request.Request(
            url,
            data=json.dumps(body, allow_nan=False).encode(),
            headers={**headers, "X-Request-ID": uuid4().hex},
        )
        for attempt in range(maximum):
            self.attempts += 1
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = response.read(MAX_RESPONSE_BYTES + 1)
                if len(payload) > MAX_RESPONSE_BYTES:
                    raise InferenceError("model response exceeds its size limit. reduce max_tokens")
                value = json.loads(payload)
                if not isinstance(value, dict):
                    raise InferenceError(
                        "model response must be a JSON object. check the serving endpoint"
                    )
                self.consecutive_failures = 0
                self.blocked_until = 0.0
                return value
            except urllib.error.HTTPError as error:
                if error.code not in {408, 429, 500, 502, 503, 504}:
                    raise InferenceError(
                        f"model HTTP {error.code}. check endpoint, model ID and credentials"
                    ) from error
                cause: BaseException = error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                cause = error
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise InferenceError(
                    "model server returned invalid JSON. check the compatible API endpoint"
                ) from error
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.breaker_failures:
                self.blocked_until = time.monotonic() + self.breaker_reset
                break
            if attempt + 1 < maximum:
                delay = self.backoff * 2**attempt
                logger.warning(
                    "model_transport_retry",
                    extra={
                        "fields": {
                            "attempt": attempt + 1,
                            "retry_delay_seconds": delay,
                            "error_type": type(cause).__name__,
                        }
                    },
                )
                time.sleep(delay)
        raise InferenceError(
            "model service unavailable. inspect server health and retry after the breaker reset"
        ) from cause
