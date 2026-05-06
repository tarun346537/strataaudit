"""Numerical aliases and contracts at model-service boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NotRequired, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .data import Evidence, Hypothesis

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]
type JSONRecord = dict[str, Any]
type Messages = list[dict[str, str]]


class JudgeResult(TypedDict):
    """A checked judge decision grounded in cited visible evidence."""

    supported: bool
    confidence: float
    rationale: str
    evidence_ids: list[str]
    raw_probability_supported: NotRequired[float]
    temperature: NotRequired[float]


class ChallengeResult(TypedDict):
    """A critique and two-boundary model proposed from visible observations."""

    critique: str
    alternative_parameters: list[float]
    evidence_ids: list[str]


class LogitJudge(Protocol):
    """Evidence-only scoring contract for inexpensive numerical judges."""

    def logit(self, e: Evidence, h: Hypothesis, /) -> float:
        """Score whether a hypothesis is supported by the visible evidence."""
        ...


class Judge(Protocol):
    """Structured assessment contract shared by served and calibrated judges."""

    def assess(self, e: Evidence, h: Hypothesis, /) -> JudgeResult:
        """Return a checked assessment using visible boreholes only."""
        ...


class Challenger(Protocol):
    """Proposal contract shared by a local adapter and remote model."""

    def propose(self, evidence: Evidence, hypothesis: Hypothesis, /) -> ChallengeResult:
        """Propose an alternative model grounded in the visible observations."""
        ...
