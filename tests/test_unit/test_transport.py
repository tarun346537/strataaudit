"""Inference failures must neither duplicate billed calls nor stall all audits."""

import io
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from strataaudit import transport
from strataaudit.transport import HTTPTransport, InferenceError, model_attempt_limit


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.delays: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.delays.append(seconds)
        self.now += seconds


def test_transient_outage_keeps_request_identity_and_uses_exponential_delays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr(transport, "time", clock)
    sent: list[Request] = []

    def send(request: Request, timeout: float) -> io.BytesIO:
        assert timeout == 1.0
        sent.append(request)
        if len(sent) < 3:
            raise URLError("temporary model outage")
        return io.BytesIO(json.dumps({"ok": True}).encode())

    monkeypatch.setattr(transport.urllib.request, "urlopen", send)
    client = HTTPTransport(1.0, retries=3, backoff=0.25, breaker_failures=4, breaker_reset=10.0)
    assert client.request("http://local-model/v1/chat/completions", {"seed": 7}, {}) == {"ok": True}
    assert clock.delays == [0.25, 0.5]
    assert client.attempts == 3
    assert client.consecutive_failures == 0
    assert len({request.get_header("X-request-id") for request in sent}) == 1
    assert all(json.loads(request.data or b"{}")["seed"] == 7 for request in sent)


def test_authentication_failure_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def deny(request: Request, timeout: float) -> io.BytesIO:
        nonlocal calls
        calls += 1
        raise HTTPError(request.full_url, 401, "denied", {}, None)

    monkeypatch.setattr(transport.urllib.request, "urlopen", deny)
    client = HTTPTransport(1.0, retries=5, backoff=0.0, breaker_failures=5, breaker_reset=10.0)
    with pytest.raises(InferenceError, match="HTTP 401"):
        client.request(
            "http://local-model/v1/chat/completions", {}, {"Authorization": "fixture-key"}
        )
    assert calls == client.attempts == 1


def test_broken_server_opens_circuit_then_recovers_after_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr(transport, "time", clock)
    calls = 0
    unavailable = True

    def send(request: Request, timeout: float) -> io.BytesIO:
        nonlocal calls
        calls += 1
        if unavailable:
            raise URLError("offline")
        return io.BytesIO(b'{"ok":true}')

    monkeypatch.setattr(transport.urllib.request, "urlopen", send)
    client = HTTPTransport(1.0, retries=5, backoff=0.0, breaker_failures=2, breaker_reset=10.0)
    with pytest.raises(InferenceError, match="unavailable"):
        client.request("http://local-model", {}, {})
    with pytest.raises(InferenceError, match="circuit is open"):
        client.request("http://local-model", {}, {})
    assert calls == 2
    clock.now = 10.1
    unavailable = False
    assert client.request("http://local-model", {}, {}) == {"ok": True}
    assert calls == 3
    assert client.consecutive_failures == 0


def test_audit_attempt_limit_restores_retry_policy_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def send(request: Request, timeout: float) -> io.BytesIO:
        nonlocal calls
        calls += 1
        if calls in {1, 2}:
            raise URLError("temporary failure")
        return io.BytesIO(b'{"ok":true}')

    monkeypatch.setattr(transport.urllib.request, "urlopen", send)
    client = HTTPTransport(1.0, retries=3, backoff=0.0, breaker_failures=5, breaker_reset=10.0)
    with pytest.raises(InferenceError), model_attempt_limit(1):
        client.request("http://local-model", {}, {})
    assert calls == 1
    assert client.request("http://local-model", {}, {}) == {"ok": True}
    assert calls == client.attempts == 3


@pytest.mark.parametrize("raw", [b"not-json", b"[]"])
def test_successful_http_status_still_requires_a_json_object(
    monkeypatch: pytest.MonkeyPatch,
    raw: bytes,
) -> None:
    monkeypatch.setattr(
        transport.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(raw)
    )
    client = HTTPTransport(1.0, retries=1, backoff=0.0, breaker_failures=2, breaker_reset=10.0)
    with pytest.raises(InferenceError, match="JSON"):
        client.request("http://local-model", {}, {})
    assert client.attempts == 1
