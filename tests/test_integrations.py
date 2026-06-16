import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Self
from unittest.mock import patch
from urllib.request import Request

from strataaudit.data import load_csv_cases, synthetic_cases
from strataaudit.fixtures import export
from strataaudit.llm import (
    ChatClient,
    InferenceError,
    ServedChallenger,
    ServedJudge,
    parse_challenge,
)
from strataaudit.preferences import generate
from strataaudit.train_challenger import load_preferences
from strataaudit.types import JSONRecord, Messages


class Response(io.BytesIO):
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cases = synthetic_cases(worlds=15)
        self.evidence = self.cases[0].evidence
        self.hypothesis = self.cases[0].hypothesis

    def test_csv_roundtrip_keeps_oracle_data_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            export(self.cases, directory)
            restored = load_csv_cases(
                Path(directory, "boreholes.csv"), Path(directory, "hypotheses.jsonl")
            )
        self.assertEqual(len(restored), len(self.cases))
        payload = json.dumps(restored[0].evidence.prompt_payload())
        self.assertNotIn("hidden", payload)
        self.assertEqual(
            restored[0].verification.borehole_ids, self.cases[0].verification.borehole_ids
        )

    def test_http_judge_schema_and_no_hidden_evidence(self) -> None:
        calls = []
        answer = {
            "supported": True,
            "confidence": 0.8,
            "rationale": "fits observations",
            "evidence_ids": [self.evidence.borehole_ids[0]],
        }

        def send(request: Request, timeout: float) -> Response:
            calls.append(json.loads(request.data))
            self.assertEqual(timeout, 2)
            return Response(
                json.dumps(
                    {
                        "choices": [
                            {"finish_reason": "stop", "message": {"content": json.dumps(answer)}}
                        ]
                    }
                ).encode()
            )

        with patch("urllib.request.urlopen", side_effect=send):
            judge = ServedJudge(ChatClient("http://localhost:8000/v1", "judge", timeout=2))
            self.assertAlmostEqual(judge.probability(self.evidence, self.hypothesis), 0.8)
        self.assertEqual(calls[0]["response_format"]["type"], "json_schema")
        self.assertNotIn("hidden", json.dumps(calls[0]))

    def test_invalid_citations_and_truncated_outputs_fail(self) -> None:
        with self.assertRaises(InferenceError):
            parse_challenge(
                {
                    "critique": "claim",
                    "alternative_parameters": [5, 15, 0, 0],
                    "evidence_ids": ["hidden-99"],
                },
                self.evidence,
            )
        truncated = {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}
        with patch("urllib.request.urlopen", return_value=Response(json.dumps(truncated).encode())):
            with self.assertRaises(InferenceError):
                ChatClient("http://localhost/v1", "m").complete([], {}, "x")

    def test_judge_cannot_reverse_its_decision_with_confidence_below_half(self) -> None:
        answer = {
            "supported": False,
            "confidence": 0.2,
            "rationale": "uncertain",
            "evidence_ids": [self.evidence.borehole_ids[0]],
        }
        envelope = {
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer)}}]
        }
        with patch("urllib.request.urlopen", return_value=Response(json.dumps(envelope).encode())):
            with self.assertRaisesRegex(InferenceError, "confidence must be"):
                ServedJudge(ChatClient("http://localhost/v1", "judge")).assess(
                    self.evidence, self.hypothesis
                )

    def test_rlaif_generation_excludes_test_worlds_and_loads_for_dpo(self) -> None:
        class FakeClient:
            def __init__(self, model: str) -> None:
                self.model = model
                self.seen = []

            def complete(
                self,
                messages: Messages,
                schema: JSONRecord,
                name: str,
                temperature: float = 0,
                seed: int = 7,
            ) -> JSONRecord:
                content = json.loads(messages[-1]["content"])
                self.seen.append(content)
                evidence_id = content["evidence"]["observations"][0]["id"]
                if name == "challenge_preference":
                    return {
                        "winner": "A",
                        "rationale": "better evidence fit",
                        "evidence_ids": [evidence_id],
                    }
                return {
                    "critique": f"candidate {seed}",
                    "alternative_parameters": [5, 15, 0, 0],
                    "evidence_ids": [evidence_id],
                }

        candidate, judge = FakeClient("challenger"), FakeClient("independent-judge")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "pairs.jsonl")
            report = generate(self.cases, ServedChallenger(candidate), judge, output)
            splits, worlds = load_preferences(output)
        self.assertGreater(report["pairs"], 0)
        self.assertTrue(splits["train"] and splits["calibration"])
        self.assertNotIn("test", worlds.values())
        self.assertNotIn("hidden", json.dumps(candidate.seen))

    def test_dpo_loader_rejects_world_leakage(self) -> None:
        record = {
            "world_id": "same",
            "split": "train",
            "prompt": [{"role": "user", "content": "p"}],
            "chosen": [{"role": "assistant", "content": "a"}],
            "rejected": [{"role": "assistant", "content": "b"}],
        }
        other = dict(record, split="calibration")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "pairs.jsonl")
            output.write_text(json.dumps(record) + "\n" + json.dumps(other), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "leakage"):
                load_preferences(output)


if __name__ == "__main__":
    unittest.main()
