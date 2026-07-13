"""A corrupt baseline must never disable a drift check."""

import pytest

from strataaudit.regression_gate import check_thresholds
from strataaudit.types import JSONRecord


@pytest.mark.parametrize("baseline", [{}, {"ece": float("nan")}, {"ece": True}, None])
def test_missing_or_nonfinite_baseline_fails_instead_of_skipping_drift(
    baseline: JSONRecord | None,
) -> None:
    failures = check_thresholds(
        {"ece": 0.5}, {"thresholds": {"ece": {"maximum_increase": 0.01}}}, baseline
    )
    assert failures
    assert "baseline" in failures[0]


@pytest.mark.parametrize("thresholds", [{}, {"ece": {}}, {"ece": {"max": float("nan")}}])
def test_empty_or_nonfinite_gate_rules_cannot_pass(thresholds: JSONRecord) -> None:
    assert check_thresholds({"ece": 0.2}, {"thresholds": thresholds})
