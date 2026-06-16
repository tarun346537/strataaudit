"""Run the public audit command through calibration and both inference arms."""

import importlib.util
import io
import json
import sys
from pathlib import Path
from urllib.request import Request

import numpy as np
import pytest

from strataaudit import orchestrate
from strataaudit.data import Case
from strataaudit.fixtures import export
from strataaudit.types import JSONRecord


@pytest.mark.integration
@pytest.mark.usefixtures("isolated_settings", "preserved_package_logging")
@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="PyTorch calibration optional"
)
def test_audit_cli_retains_data_identity_calibration_cost_and_frozen_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, geological_cases: list[Case]
) -> None:
    """Use controlled HTTP outputs to verify the served-model audit contract."""
    inputs = tmp_path / "inputs"
    export(geological_cases, inputs)
    sent: list[JSONRecord] = []

    def model_service(request: Request, timeout: float) -> io.BytesIO:
        assert request.data is not None
        payload = json.loads(request.data)
        data = json.loads(payload["messages"][-1]["content"])
        sent.append(payload)
        rows = data["evidence"]["observations"]
        x = np.array([row["x"] for row in rows])
        boundaries = np.array([row["boundary_depths"] for row in rows])
        result: JSONRecord
        if payload["model"] == "controlled-challenger":
            fit = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), boundaries, rcond=None)[0]
            result = {
                "critique": "propose the visible least-squares fit",
                "alternative_parameters": np.r_[fit[0], fit[1]].tolist(),
                "evidence_ids": [rows[0]["id"]],
            }
        else:
            p = np.array(data["hypothesis"]["parameters"])
            rmse = np.sqrt(np.mean((p[:2] + x[:, None] * p[2:] - boundaries) ** 2))
            result = {
                "supported": bool(rmse <= 0.45),
                "confidence": 0.85,
                "rationale": "checked visible residuals",
                "evidence_ids": [rows[0]["id"]],
            }
        return io.BytesIO(
            json.dumps({"choices": [{"message": {"content": json.dumps(result)}}]}).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", model_service)
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "strataaudit.orchestrate",
            "--boreholes",
            str(inputs / "boreholes.csv"),
            "--hypotheses",
            str(inputs / "hypotheses.jsonl"),
            "--judge-model",
            "controlled-judge",
            "--challenger-model",
            "controlled-challenger",
            "--escalation-confidence",
            "1.0",
            "--budget-units",
            "40",
            "--output",
            str(output),
        ],
    )
    orchestrate.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    calibration_count = sum(c.split == "calibration" for c in geological_cases)
    assert report["served_calibration_http_attempts"] == calibration_count
    counted_model_calls = calibration_count
    for strategy in ["direct_judging", "challenger_assisted"]:
        result = report[strategy]
        assert result["budget_limit_units"] == 40
        assert result["budget_used_units"] <= 40
        counts = result["external_calls"]
        counted_model_calls += counts["judge"] + counts["challenger"]
    assert len(sent) == counted_model_calls
    assert report["provenance"]["cases_by_split"]["calibration"] == calibration_count
    assert len(report["provenance"]["inputs"]["hypotheses"]["sha256"]) == 64
    assert report["audit_settings"]["correctness_tolerance_metres"] == 0.45
    assert report["served_judge_temperature_fit"]["observations"] == calibration_count
    assert output.with_suffix(".traces.jsonl").is_file()
    transmitted = json.dumps(sent)
    assert all(
        identity not in transmitted
        for case in geological_cases
        for identity in case.verification.borehole_ids
    )
