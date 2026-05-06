"""Runtime settings for model training and geological audits."""

from __future__ import annotations

import math
import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from .constants import (
    DEFAULT_BOUNDARY_TOLERANCE,
    DEFAULT_BREAKER_FAILURES,
    DEFAULT_BREAKER_RESET,
    DEFAULT_HTTP_RETRIES,
    DEFAULT_HTTP_TIMEOUT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RETRY_BACKOFF,
    DEFAULT_SEED,
    DEFAULT_VERIFICATION_FRACTION,
    DEFAULT_WORLDS,
)
from .types import JSONRecord


def load_env_file(path: Path) -> None:
    """Load task-specific variables without replacing an existing shell setting."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        key = key.strip()
        if not separator or not (key.startswith("STRATA_") or key == "VLLM_API_KEY"):
            raise ValueError(f"invalid setting in {path}: use STRATA_* or VLLM_API_KEY assignments")
        os.environ.setdefault(key, value.strip().strip("\"'"))


@dataclass(frozen=True)
class Settings:
    """Settings shared by CLI, training and served-model jobs."""

    seed: int = DEFAULT_SEED
    worlds: int = DEFAULT_WORLDS
    verification_fraction: float = DEFAULT_VERIFICATION_FRACTION
    tolerance: float = DEFAULT_BOUNDARY_TOLERANCE
    log_level: str = "INFO"
    device: str = "cuda"
    output_dir: str = DEFAULT_OUTPUT_DIR
    judge_url: str = "http://127.0.0.1:8001/v1"
    judge_model: str = ""
    challenger_url: str = "http://127.0.0.1:8000/v1"
    challenger_model: str = ""
    http_timeout: float = DEFAULT_HTTP_TIMEOUT
    http_retries: int = DEFAULT_HTTP_RETRIES
    retry_backoff: float = DEFAULT_RETRY_BACKOFF
    breaker_failures: int = DEFAULT_BREAKER_FAILURES
    breaker_reset: float = DEFAULT_BREAKER_RESET
    tracking: str = "json"
    early_stopping_patience: int = 3
    train_steps: int = 500
    lora_rank: int = 16
    max_length: int = 2048
    learning_rate: float = 5e-5
    gradient_accumulation_steps: int = 4
    dpo_beta: float = 0.1
    lora_dropout: float = 0.05
    audit_budget_units: int = 100
    judge_cost: int = 1
    challenger_cost: int = 1
    verification_cost: int = 4
    escalation_confidence: float = 0.9
    verification_reserve: float = 0.4

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> Settings:
        """Read defaults, an optional TOML file, then environment overrides."""
        load_env_file(Path(".env"))
        path = config_path or os.environ.get("STRATA_CONFIG")
        values: JSONRecord = {}
        if path:
            payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
            values = payload.get("strataaudit", payload)
            if not isinstance(values, dict):
                raise ValueError("audit config must contain a TOML table")
        known = {item.name: item for item in fields(cls)}
        unknown = set(values) - set(known)
        if unknown:
            raise ValueError(f"unknown audit config keys: {sorted(unknown)}")
        defaults = cls()
        for name, value in values.items():
            expected = type(getattr(defaults, name))
            valid = isinstance(value, expected) or (expected is float and isinstance(value, int))
            if isinstance(value, bool) or not valid:
                raise ValueError(f"invalid audit setting {name}: expected {expected.__name__}")
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"invalid audit setting {name}: expected a finite number")
        for name in known:
            raw = os.environ.get("STRATA_" + name.upper())
            if raw is None or raw == "":
                continue
            current = getattr(defaults, name)
            try:
                values[name] = (
                    int(raw)
                    if isinstance(current, int)
                    else float(raw)
                    if isinstance(current, float)
                    else raw
                )
            except ValueError as error:
                raise ValueError(
                    f"invalid STRATA_{name.upper()}: expected {type(current).__name__}"
                ) from error
            if isinstance(values[name], (int, float)) and not math.isfinite(values[name]):
                raise ValueError(f"invalid STRATA_{name.upper()}: expected a finite number")
        result = cls(**values)
        if result.worlds < 3 or not 0 <= result.verification_fraction <= 1 or result.tolerance <= 0:
            raise ValueError(
                "invalid world count, verification fraction or boundary tolerance. check audit config"
            )
        if (
            min(result.http_timeout, result.breaker_reset) <= 0
            or result.http_retries < 0
            or result.retry_backoff < 0
        ):
            raise ValueError(
                "invalid HTTP timeout/retry settings. use positive timeouts and nonnegative retries"
            )
        if result.breaker_failures < 1 or result.early_stopping_patience < 1:
            raise ValueError("breaker failures and early-stopping patience must be positive")
        if (
            min(result.train_steps, result.lora_rank, result.gradient_accumulation_steps) < 1
            or result.max_length < 64
        ):
            raise ValueError(
                "training steps, LoRA rank and accumulation must be positive. max_length must be >=64"
            )
        if result.learning_rate <= 0 or result.dpo_beta <= 0 or not 0 <= result.lora_dropout < 1:
            raise ValueError("invalid learning rate, DPO beta or LoRA dropout in training settings")
        if (
            result.audit_budget_units < 0
            or min(result.judge_cost, result.challenger_cost, result.verification_cost) < 1
        ):
            raise ValueError(
                "audit budget must be nonnegative and each external call cost must be positive"
            )
        if (
            not 0.5 <= result.escalation_confidence <= 1
            or not 0 <= result.verification_reserve <= 1
        ):
            raise ValueError(
                "audit confidence must lie in [.5,1] and verification reserve in [0,1]"
            )
        return result
