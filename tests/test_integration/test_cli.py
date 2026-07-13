"""Exercise documented file-based evaluation and CI exit behavior."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

import strataaudit.evaluate as evaluation_cli
import strataaudit.regression_gate as gate_cli
from strataaudit.data import Case
from strataaudit.fixtures import export


@pytest.mark.integration
@pytest.mark.usefixtures("isolated_settings", "preserved_package_logging")
@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="PyTorch calibration optional"
)
def test_normalized_evaluation_cli_uses_torch_and_writes_case_traces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    geological_cases: list[Case],
) -> None:
    inputs = tmp_path / "normalized"
    export(geological_cases, inputs)
    output = tmp_path / "evaluation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "strataaudit.evaluate",
            "--boreholes",
            str(inputs / "boreholes.csv"),
            "--hypotheses",
            str(inputs / "hypotheses.jsonl"),
            "--calibration-backend",
            "torch",
            "--budget",
            "0.23",
            "--output",
            str(output),
        ],
    )
    evaluation_cli.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    traces = [
        json.loads(line) for line in output.with_suffix(".traces.jsonl").read_text().splitlines()
    ]
    count = sum(case.split == "test" for case in geological_cases)
    assert report["calibration_backend"] == "torch"
    assert report["metrics"]["cases"] == len(traces) == count
    assert report["metrics"]["online_verifier_calls"] == int(0.23 * count)
    assert report["metrics"]["online_verification_fraction"] <= 0.23
    assert report["metrics"]["temperature"] > 0
    assert (
        report["provenance"]["inputs"]["boreholes"]["sha256"]
        == hashlib.sha256((inputs / "boreholes.csv").read_bytes()).hexdigest()
    )
    assert all(row["split"] == "test" and row["level"] == "INFO" for row in traces)


@pytest.mark.integration
@pytest.mark.usefixtures("isolated_settings", "preserved_package_logging")
@pytest.mark.parametrize("limit,expected_pass", [(1.0, True), (-1.0, False)])
def test_fixed_gate_writes_traces_and_failure_report_before_exiting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    limit: float,
    expected_pass: bool,
) -> None:
    spec = tmp_path / "gate.json"
    spec.write_text(
        json.dumps(
            {
                "benchmark": {"seed": 7, "worlds": 30, "budget": 0.25},
                "thresholds": {"metrics.ece_calibrated": {"max": limit}},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "gate-result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "strataaudit.regression_gate",
            "--spec",
            str(spec),
            "--baseline",
            "",
            "--output",
            str(output),
        ],
    )
    if expected_pass:
        gate_cli.main()
    else:
        with pytest.raises(SystemExit) as error:
            gate_cli.main()
        assert error.value.code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["regression_gate"]["passed"] == expected_pass
    assert bool(report["regression_gate"]["failures"]) != expected_pass
    assert output.with_suffix(".traces.jsonl").is_file()
