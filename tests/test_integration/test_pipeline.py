"""Audit the real HTTP-client path with controlled geological observations."""

import io
import json
from urllib.error import URLError
from urllib.request import Request

import numpy as np
import pytest

from strataaudit.artifacts import calibrate_served_judge
from strataaudit.audit import challenger as fit_alternative
from strataaudit.data import Case, Evidence, Hypothesis, Verification
from strataaudit.llm import ChatClient, ServedChallenger, ServedJudge
from strataaudit.pipeline import AuditCosts, compare_at_equal_budget
from strataaudit.types import ChallengeResult, JSONRecord, JudgeResult


def known_case(world: str, split: str = "test") -> Case:
    x = np.array([-0.5, 0.0, 0.5])
    visible = Evidence(
        world,
        tuple(f"{world}-visible-{i}" for i in range(3)),
        x,
        np.column_stack([5 + 0.5 * x, 15 - 0.5 * x]),
    )
    withheld = Verification(
        world,
        (f"{world}-hidden-a", f"{world}-hidden-b"),
        np.array([-1.0, 1.0]),
        np.array([[4.5, 15.5], [5.5, 14.5]]),
    )
    return Case(
        split, visible, Hypothesis(f"{world}-h", world, np.array([7.0, 17.0, 0.0, 0.0])), withheld
    )


@pytest.mark.integration
@pytest.mark.usefixtures("isolated_settings")
def test_actual_judge_and_challenger_clients_share_budget_without_hidden_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[JSONRecord] = []

    def model_service(request: Request, timeout: float) -> io.BytesIO:
        assert request.data is not None
        payload = json.loads(request.data)
        sent.append(payload)
        data = json.loads(payload["messages"][-1]["content"])
        observations = data["evidence"]["observations"]
        x = np.array([row["x"] for row in observations])
        boundaries = np.array([row["boundary_depths"] for row in observations])
        if payload["model"] == "challenger":
            fit = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), boundaries, rcond=None)[0]
            result = {
                "critique": "candidate does not fit the visible boundaries",
                "alternative_parameters": np.r_[fit[0], fit[1]].tolist(),
                "evidence_ids": [observations[0]["id"]],
            }
        else:
            parameters = np.array(data["hypothesis"]["parameters"])
            predicted = parameters[:2] + x[:, None] * parameters[2:]
            result = {
                "supported": bool(np.max(np.abs(predicted - boundaries)) < 0.1),
                "confidence": 0.85,
                "rationale": "checked visible boundary residuals",
                "evidence_ids": [observations[0]["id"]],
            }
        return io.BytesIO(
            json.dumps(
                {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}]}
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", model_service)
    judges: list[ServedJudge] = []

    def make_judge() -> ServedJudge:
        judge = ServedJudge(ChatClient("http://local-judge/v1", "judge"))
        judges.append(judge)
        return judge

    challenger = ServedChallenger(ChatClient("http://local-challenger/v1", "challenger"))
    cases = [known_case(f"world-{i}") for i in range(6)]

    class UncertainJudge:
        def logit(self, e: Evidence, h: Hypothesis) -> float:
            return 0.0

    report, traces = compare_at_equal_budget(
        cases, UncertainJudge(), 1.0, make_judge, challenger, budget_units=24, costs=AuditCosts()
    )
    external_attempts = 0
    for strategy in ["direct_judging", "challenger_assisted"]:
        result = report[strategy]
        counts = result["external_calls"]
        external_attempts += counts["judge"] + counts["challenger"]
        assert result["budget_limit_units"] == 24
        assert result["budget_used_units"] <= 24
        assert (
            result["budget_used_units"]
            == counts["judge"] + counts["challenger"] + 4 * counts["verification"]
        )
        assert all(row["trace_id"] == row["case_id"] for row in traces[strategy])
    assert len(sent) == external_attempts
    assert sum(
        judge.client.transport.attempts for judge in judges
    ) + challenger.client.transport.attempts == len(sent)
    assert report["direct_judging"]["hidden_borehole_predictive_accuracy"] == 0.0
    assert report["challenger_assisted"]["hidden_borehole_predictive_accuracy"] == 1.0
    serialized = json.dumps(sent)
    assert all(
        identity not in serialized for case in cases for identity in case.verification.borehole_ids
    )


def test_overlapping_calibration_world_is_rejected_before_a_remote_call() -> None:
    class NeverCallJudge:
        def assess(self, e: Evidence, h: Hypothesis) -> JudgeResult:
            raise AssertionError("test-world evidence must not reach calibration scoring")

    calibration = [known_case("shared-world", "calibration")]
    with pytest.raises(ValueError, match="disjoint"):
        calibrate_served_judge(calibration, NeverCallJudge(), ["training-world"], ["shared-world"])


@pytest.mark.integration
@pytest.mark.usefixtures("isolated_settings")
def test_direct_arm_circuit_failure_does_not_block_the_assisted_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STRATA_BREAKER_FAILURES", "1")
    attempts = 0

    def model_service(request: Request, timeout: float) -> io.BytesIO:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise URLError("transient outage during direct judging")
        assert request.data is not None
        data = json.loads(json.loads(request.data)["messages"][-1]["content"])
        answer = {
            "supported": True,
            "confidence": 0.85,
            "rationale": "consistent alternative",
            "evidence_ids": [data["evidence"]["observations"][0]["id"]],
        }
        return io.BytesIO(
            json.dumps({"choices": [{"message": {"content": json.dumps(answer)}}]}).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", model_service)
    clients: list[ServedJudge] = []

    def make_judge() -> ServedJudge:
        judge = ServedJudge(ChatClient("http://local-judge/v1", "judge"))
        clients.append(judge)
        return judge

    class Cheap:
        def logit(self, e: Evidence, h: Hypothesis) -> float:
            return 0.0

    class Alternative:
        def propose(self, evidence: Evidence, hypothesis: Hypothesis) -> ChallengeResult:
            fitted = fit_alternative(evidence, hypothesis.case_id)
            return {
                "critique": "visible residuals disagree",
                "alternative_parameters": fitted.parameters.tolist(),
                "evidence_ids": [evidence.borehole_ids[0]],
            }

    compare_at_equal_budget(
        [known_case(f"w-{i}") for i in range(6)],
        Cheap(),
        1.0,
        make_judge,
        Alternative(),
        budget_units=24,
    )
    assert len(clients) == 2
    assert clients[0].client.transport.attempts == 1
    assert clients[0].client.transport.blocked_until > 0
    assert clients[1].client.transport.attempts > 0
    assert clients[1].client.transport.consecutive_failures == 0
