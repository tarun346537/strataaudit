import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from strataaudit.artifacts import adapter_artifact_hashes
from strataaudit.audit import challenger
from strataaudit.data import Case, Evidence, Hypothesis, Verification
from strataaudit.orchestrate import (
    AuditCosts,
    CalibratedServedJudge,
    compare_at_equal_budget,
    validate_adapter_holdouts,
)
from strataaudit.regression_gate import check_thresholds
from strataaudit.torch_calibration import fit_temperature_torch, group_holdout_split
from strataaudit.types import ChallengeResult, JudgeResult
from strataaudit.usgs_import import import_intervals


class ResearchPipelineTests(unittest.TestCase):
    def test_group_split_never_separates_rows_from_same_group(self) -> None:
        groups = [f"region-{i // 5}" for i in range(30)]
        a = group_holdout_split(groups)
        self.assertEqual(a, group_holdout_split(list(reversed(groups))))
        self.assertEqual(set(a.values()), {"train", "calibration", "test"})
        self.assertEqual(len(a), 6)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional")
    def test_torch_temperature_is_fitted_without_test_leakage(self) -> None:
        fitted = fit_temperature_torch(
            [-8, -4, 4, 8],
            [0, 1, 1, 1],
            calibration_groups=["cal"] * 4,
            training_groups=["train"],
            test_groups=["test"],
        )
        self.assertGreater(fitted.temperature, 0)
        self.assertLessEqual(fitted.calibration_nll_after, fitted.calibration_nll_before)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            fit_temperature_torch([1], [1], calibration_groups=["leak"], test_groups=["leak"])

    def test_orchestration_invokes_judge_and_challenger_under_same_budget(self) -> None:
        cases = []
        for i in range(6):
            world = f"w{i}"
            x = np.array([-0.5, 0, 0.5])
            y = np.column_stack([5 + 0.5 * x, 15 - 0.5 * x])
            e = Evidence(world, tuple(f"{world}-v{j}" for j in range(3)), x, y)
            v = Verification(
                world,
                (f"{world}-hidden-a", f"{world}-hidden-b"),
                [-1, 1],
                [[4.5, 15.5], [5.5, 14.5]],
            )
            cases.append(Case("test", e, Hypothesis(f"h{i}", world, [7, 17, 0, 0]), v))

        class Cheap:
            def logit(self, *_: object) -> float:
                return 0.0

        class Judge:
            calls = 0

            def assess(self, evidence: Evidence, hypothesis: Hypothesis) -> JudgeResult:
                self.calls += 1
                self.assert_visible(evidence)
                correct = np.max(np.abs(hypothesis.predict(evidence.x) - evidence.boundaries)) < 0.1
                return {
                    "supported": bool(correct),
                    "confidence": 0.85,
                    "rationale": "residual",
                    "evidence_ids": [evidence.borehole_ids[0]],
                }

            def assert_visible(self, evidence: Evidence) -> None:
                assert all("hidden" not in key for key in evidence.borehole_ids)

        class Challenger:
            calls = 0

            def propose(self, evidence: Evidence, hypothesis: Hypothesis) -> ChallengeResult:
                self.calls += 1
                h = challenger(evidence, hypothesis.case_id + "-alternative")
                return {
                    "alternative_parameters": h.parameters.tolist(),
                    "critique": "visible misfit",
                    "evidence_ids": [evidence.borehole_ids[0]],
                }

        judge, challenge = Judge(), Challenger()
        metrics, traces = compare_at_equal_budget(
            cases, Cheap(), 1.0, lambda: judge, challenge, budget_units=24, costs=AuditCosts()
        )
        self.assertGreater(judge.calls, 0)
        self.assertGreater(challenge.calls, 0)
        for strategy in ["direct_judging", "challenger_assisted"]:
            report = metrics[strategy]
            self.assertEqual(report["budget_limit_units"], 24)
            self.assertLessEqual(report["budget_used_units"], 24)
            counts = report["external_calls"]
            self.assertEqual(
                report["budget_used_units"],
                counts["judge"] + counts["challenger"] + 4 * counts["verification"],
            )
        self.assertGreater(
            metrics["challenger_assisted"]["hidden_borehole_predictive_accuracy"],
            metrics["direct_judging"]["hidden_borehole_predictive_accuracy"],
        )
        self.assertTrue(
            any(
                event["type"] == "challenger"
                for trace in traces["challenger_assisted"]
                for event in trace["events"]
            )
        )

    def test_regression_gate_catches_silent_quality_and_budget_regressions(self) -> None:
        specification = {
            "thresholds": {
                "metrics.ece": {"max": 0.1, "maximum_increase": 0.02},
                "metrics.verification": {"max": 0.25},
            }
        }
        baseline = {"metrics": {"ece": 0.05}}
        self.assertEqual(
            check_thresholds(
                {"metrics": {"ece": 0.06, "verification": 0.2}}, specification, baseline
            ),
            [],
        )
        failures = check_thresholds(
            {"metrics": {"ece": 0.09, "verification": 0.5}}, specification, baseline
        )
        self.assertEqual(len(failures), 2)
        self.assertTrue(check_thresholds({"metrics": {"verification": 0.2}}, specification))

    def test_three_independent_fixed_benchmark_regressions_are_rejected(self) -> None:
        specification = {
            "thresholds": {
                "ece": {"max": 0.08},
                "false_acceptance": {"max": 0.03},
                "verification_fraction": {"max": 0.25},
            }
        }
        good = {"ece": 0.05, "false_acceptance": 0.0, "verification_fraction": 0.25}
        for key in good:
            mutated = dict(good, **{key: 0.9})
            self.assertTrue(check_thresholds(mutated, specification), key)

    def test_served_judge_temperature_changes_confidence_not_class(self) -> None:
        class Judge:
            def assess(self, *_: object) -> JudgeResult:
                return {
                    "supported": True,
                    "confidence": 0.9,
                    "rationale": "evidence",
                    "evidence_ids": ["v"],
                }

        result = CalibratedServedJudge(Judge(), 2.0).assess(None, None)
        self.assertTrue(result["supported"])
        self.assertLess(result["confidence"], 0.9)
        self.assertEqual(result["raw_probability_supported"], 0.9)

    def test_adapter_manifest_enforces_holdouts_across_artifacts(self) -> None:
        from strataaudit.data import synthetic_cases

        cases = synthetic_cases(worlds=15)
        train_world = next(c.evidence.world_id for c in cases if c.split == "train")
        cal_world = next(c.evidence.world_id for c in cases if c.split == "calibration")
        test_world = next(c.evidence.world_id for c in cases if c.split == "test")
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory, "training_manifest.json")
            with self.assertRaisesRegex(ValueError, "actual training_manifest"):
                validate_adapter_holdouts(directory, cases)
            Path(directory, "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "unit-test-base", "r": 2}), encoding="utf-8"
            )
            Path(directory, "adapter_model.safetensors").write_bytes(b"unit-test-export")
            manifest.write_text(
                json.dumps(
                    {
                        "worlds": {train_world: "train", cal_world: "calibration"},
                        "status": "complete",
                        "completed_optimizer_steps": 1,
                        "base_model": "unit-test-base",
                        "rank": 2,
                        "adapter_artifacts_sha256": adapter_artifact_hashes(directory),
                    }
                ),
                encoding="utf-8",
            )
            validation = validate_adapter_holdouts(directory, cases)
            self.assertEqual(validation["test_world_overlap"], 0)
            self.assertEqual(len(validation["training_manifest_sha256"]), 64)
            for fitted_split in ["train", "calibration"]:
                manifest.write_text(
                    json.dumps({"worlds": {test_world: fitted_split}}), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "test world"):
                    validate_adapter_holdouts(directory, cases)
            manifest.write_text(
                json.dumps({"worlds": {train_world: "calibration"}}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "split mismatch"):
                validate_adapter_holdouts(directory, cases)
            manifest.write_text(json.dumps({"worlds": {train_world: "test"}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "train/calibration"):
                validate_adapter_holdouts(directory, cases)

    def test_interval_import_records_source_hash_and_withholds_boreholes(self) -> None:
        mapping = {
            key: key
            for key in ["world_id", "borehole_id", "x", "top_depth", "bottom_depth", "lithology"]
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "fixture-intervals.csv")
            with source.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(mapping))
                writer.writeheader()
                for i in range(3):
                    for j in range(4):
                        for top, bottom, unit in [
                            (0, 5, "sand"),
                            (5, 15, "shale"),
                            (15, 25, "basement"),
                        ]:
                            writer.writerow(
                                {
                                    "world_id": f"region-{i}",
                                    "borehole_id": f"r{i}-h{j}",
                                    "x": j,
                                    "top_depth": top,
                                    "bottom_depth": bottom,
                                    "lithology": unit,
                                }
                            )
            # the interval fixture uses synthetic rows
            provenance = import_intervals(
                source,
                Path(directory, "normalized"),
                source_url="https://example.org/test-fixture",
                dataset_title="Controlled interval fixture",
                columns=mapping,
                middle_unit="shale",
                lower_unit="basement",
                provider="test_fixture",
            )
            self.assertEqual(provenance["boreholes"], 12)
            self.assertEqual(provenance["visible_boreholes"], 6)
            self.assertEqual(provenance["verification_boreholes"], 6)
            self.assertEqual(len(provenance["source_sha256"]), 64)
            with self.assertRaisesRegex(ValueError, "USGS provenance"):
                import_intervals(
                    source,
                    Path(directory, "invalid"),
                    source_url="https://example.org/unknown",
                    dataset_title="Unknown",
                    columns=mapping,
                    middle_unit="shale",
                    lower_unit="basement",
                )


if __name__ == "__main__":
    unittest.main()
