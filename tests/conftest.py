"""Shared geological cases for unit and integration checks."""

import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from strataaudit.data import Case, synthetic_cases


@pytest.fixture(scope="session")
def geological_cases() -> list[Case]:
    """Return independent worlds with visible and withheld boreholes."""
    return synthetic_cases(seed=7, worlds=30)


@pytest.fixture
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Keep shell credentials and a developer's .env out of local model checks."""
    previous = {
        name: value
        for name, value in os.environ.items()
        if name.startswith("STRATA_") or name == "VLLM_API_KEY"
    }
    for name in list(previous):
        if name.startswith("STRATA_") or name == "VLLM_API_KEY":
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    try:
        yield
    finally:
        for name in list(os.environ):
            if name.startswith("STRATA_") or name == "VLLM_API_KEY":
                os.environ.pop(name)
        os.environ.update(previous)


@pytest.fixture
def preserved_package_logging() -> Iterator[None]:
    """Return CLI logging to its previous state before pytest closes capture."""
    logger = logging.getLogger("strataaudit")
    handlers, level, propagation = logger.handlers[:], logger.level, logger.propagate
    try:
        yield
    finally:
        logger.handlers[:] = handlers
        logger.setLevel(level)
        logger.propagate = propagation
