import unittest

import numpy as np

from strataaudit.audit import (
    EvidenceJudge,
    challenger,
    ece,
    evaluate,
    fit_temperature,
    hidden_error,
    oracle,
    sigmoid,
)
from strataaudit.data import (
    Case,
    Evidence,
    Hypothesis,
    Verification,
    synthetic_cases,
    validate_splits,
)
from strataaudit.types import FloatArray


class AuditTests(unittest.TestCase):
    def test_verification_is_independent_of_visible_fit(self) -> None:
        e = Evidence("w", ("v1", "v2"), [-0.01, 0.01], [[5, 15], [5, 15]])
        h = Hypothesis("h", "w", [5, 15, 2, 2])
        v = Verification("w", ("hidden1", "hidden2"), [-1, 1], [[5, 15], [5, 15]])
        self.assertLess(np.abs(h.predict(e.x) - e.boundaries).max(), 0.03)
        self.assertFalse(oracle(h, v))
        self.assertGreater(hidden_error(h, v), 1)

    def test_challenger_recovers_visible_planes(self) -> None:
        e = Evidence("w", ("a", "b", "c"), [-1, 0, 1], [[4, 15.5], [5, 15], [6, 14.5]])
        alternative = challenger(e, "challenge")
        np.testing.assert_allclose(alternative.parameters, [5, 15, 1, -0.5], atol=1e-12)

    def test_no_borehole_or_world_overlap(self) -> None:
        cases = synthetic_cases(worlds=30)
        validate_splits(cases)
        case = cases[0]
        changed = Case(
            "test",
            case.evidence,
            Hypothesis("distinct-case", case.evidence.world_id, case.hypothesis.parameters),
            case.verification,
        )
        with self.assertRaisesRegex(ValueError, "leaks"):
            validate_splits(cases + [changed])
        with self.assertRaises(ValueError):
            Case(
                "train",
                case.evidence,
                case.hypothesis,
                Verification(
                    case.evidence.world_id,
                    case.evidence.borehole_ids,
                    case.evidence.x,
                    case.evidence.boundaries,
                ),
            )

    def test_temperature_improves_calibration_loss_without_changing_decisions(self) -> None:
        logits = np.array([-8.0, -4.0, 4.0, 8.0])
        labels = np.array([0.0, 1.0, 1.0, 1.0])
        t = fit_temperature(logits, labels)

        def loss(values: FloatArray) -> float:
            return float(np.mean(np.logaddexp(0, values) - labels * values))

        self.assertLessEqual(loss(logits / t), loss(logits))
        np.testing.assert_array_equal(sigmoid(logits) >= 0.5, sigmoid(logits / t) >= 0.5)
        self.assertAlmostEqual(ece([0, 1], [0, 1]), 0)

    def test_full_budget_verification_is_exact_and_counted(self) -> None:
        cases = synthetic_cases(worlds=30)
        judge = EvidenceJudge.fit([c for c in cases if c.split == "train"])
        test = [c for c in cases if c.split == "test"]
        metrics, traces = evaluate(test, judge, 1.0, budget_fraction=1.0)
        self.assertEqual(metrics["online_verifier_calls"], len(test))
        self.assertEqual(metrics["budgeted_audit"]["false_acceptances"], 0)
        self.assertEqual(metrics["budgeted_audit"]["accuracy"], 1)
        self.assertTrue(all(t["accepted"] == t["evaluation_only_truth"] for t in traces))
        low, _ = evaluate(test, judge, 1.0, budget_fraction=0.23)
        self.assertEqual(low["online_verifier_calls"], int(0.23 * len(test)))

    def test_training_cannot_take_test_cases(self) -> None:
        with self.assertRaisesRegex(ValueError, "training"):
            EvidenceJudge.fit([c for c in synthetic_cases(worlds=15) if c.split == "test"])

    def test_invalid_probabilities_and_parameters(self) -> None:
        with self.assertRaises(ValueError):
            ece([float("nan")], [1])
        with self.assertRaises(ValueError):
            Hypothesis("h", "w", [5, 15, float("nan"), 0])
        with self.assertRaises(ValueError):
            fit_temperature([1], [2])


if __name__ == "__main__":
    unittest.main()
