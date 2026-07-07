"""Check metric denominators with hand-defined probabilities and decisions."""

import numpy as np
import pytest

from strataaudit.audit import ece, evaluate
from strataaudit.data import Case, Evidence, Hypothesis, Verification


def test_ece_distinguishes_calibration_from_perfect_classification_accuracy() -> None:
    labels = np.array([0, 0, 1, 1])
    # fixed probability inputs isolate metric arithmetic
    before = np.array([0.19, 0.19, 0.81, 0.81])
    after = np.array([0.06, 0.06, 0.94, 0.94])
    assert np.array_equal(before >= 0.5, labels)
    assert np.array_equal(after >= 0.5, labels)
    assert ece(before, labels) == pytest.approx(0.19)
    assert ece(after, labels) == pytest.approx(0.06)
    assert ece([0, 0.25, 0.75, 1], labels, bins=4) == pytest.approx(0.125)


def test_false_acceptance_uses_all_false_hypotheses_not_all_accepted_claims() -> None:
    truths = [True, True, False, False, False, False]
    decisions = [True, False, True, True, False, False]
    cases = []
    for i, correct in enumerate(truths):
        world = f"known-{i}"
        visible = Evidence(
            world,
            (f"v{i}-a", f"v{i}-b"),
            np.array([-1.0, 1.0]),
            np.array([[5.0, 15.0], [5.0, 15.0]]),
        )
        withheld = Verification(
            world,
            (f"h{i}-a", f"h{i}-b"),
            np.array([-0.5, 0.5]),
            np.array([[5.0, 15.0], [5.0, 15.0]]),
        )
        parameters = np.array([5.0, 15.0, 0.0, 0.0]) if correct else np.array([7.0, 17.0, 0.0, 0.0])
        cases.append(Case("test", visible, Hypothesis(str(i), world, parameters), withheld))

    class KnownJudge:
        def logit(self, e: Evidence, h: Hypothesis) -> float:
            return 2.0 if decisions[int(h.case_id)] else -2.0

    metrics, traces = evaluate(cases, KnownJudge(), 1.0, budget_fraction=0.0)
    direct = metrics["direct_judge"]
    assert direct["false_acceptances"] == 2
    assert direct["false_hypotheses"] == 4
    assert direct["false_acceptance_rate"] == pytest.approx(0.5)
    assert direct["false_discovery_rate"] == pytest.approx(2 / 3)
    assert metrics["online_verifier_calls"] == 0
    assert metrics["evaluation_label_oracle_calls"] == 6
    assert all(row["trace_id"] == row["case_id"] for row in traces)
