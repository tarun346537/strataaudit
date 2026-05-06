"""Runtime settings and audit logs must preserve run identity."""

import json
import logging
from datetime import datetime
from pathlib import Path

import pytest

from strataaudit.config import Settings
from strataaudit.logging_config import record_trace, setup_logging, trace_context


@pytest.mark.usefixtures("isolated_settings")
def test_shell_settings_win_over_dotenv_and_toml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "audit.toml").write_text(
        "[strataaudit]\nseed=12\nverification_fraction=0.23\n", encoding="utf-8"
    )
    (tmp_path / ".env").write_text(
        "STRATA_CONFIG=audit.toml\nSTRATA_SEED=13\nVLLM_API_KEY=fixture-key\n", encoding="utf-8"
    )
    monkeypatch.setenv("STRATA_SEED", "14")
    settings = Settings.load()
    assert settings.seed == 14
    assert settings.verification_fraction == 0.23


@pytest.mark.usefixtures("isolated_settings")
@pytest.mark.parametrize(
    "name,value", [("STRATA_HTTP_TIMEOUT", "inf"), ("STRATA_TRAIN_STEPS", "many")]
)
def test_bad_numeric_environment_settings_fail_before_model_calls(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        Settings.load()


@pytest.mark.usefixtures("isolated_settings")
def test_boolean_seed_in_a_toml_file_is_not_silently_treated_as_one(tmp_path: Path) -> None:
    path = tmp_path / "audit.toml"
    path.write_text("[strataaudit]\nseed=true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="seed"):
        Settings.load(path)


@pytest.mark.usefixtures("isolated_settings")
def test_json_logs_keep_trace_identity_and_do_not_expose_environment_credentials(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("VLLM_API_KEY", "fixture-do-not-log")
    package_logger = logging.getLogger("strataaudit")
    old_handlers, old_level, old_propagation = (
        package_logger.handlers[:],
        package_logger.level,
        package_logger.propagate,
    )
    try:
        logger = setup_logging()
        with trace_context("run-4"):
            logger.info("audit_started")
            record = record_trace({"case_id": "case-1", "accepted": True}, "case-1")
        output = capsys.readouterr().err
        rows = [json.loads(line) for line in output.splitlines()]
        assert [row["trace_id"] for row in rows] == ["run-4", "case-1"]
        assert all(row["level"] == "INFO" for row in rows)
        assert all(datetime.fromisoformat(row["timestamp"]).tzinfo is not None for row in rows)
        assert record["trace_id"] == "case-1"
        assert "fixture-do-not-log" not in output
    finally:
        package_logger.handlers[:] = old_handlers
        package_logger.setLevel(old_level)
        package_logger.propagate = old_propagation


def test_caller_payload_cannot_replace_the_trace_metadata() -> None:
    record = record_trace({"timestamp": "stale", "level": "DEBUG", "trace_id": "wrong"}, "case-2")
    assert record["trace_id"] == "case-2"
    assert record["level"] == "INFO"
    assert record["timestamp"] != "stale"
