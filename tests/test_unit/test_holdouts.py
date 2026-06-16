"""Reject renamed copies of observed boreholes and mismatched exports."""

import json
from pathlib import Path

import numpy as np
import pytest

from strataaudit.artifacts import adapter_artifact_hashes, validate_adapter_holdouts
from strataaudit.data import Case, Evidence, Hypothesis, Verification, validate_splits
from strataaudit.fixtures import export
from strataaudit.gempy_worlds import borehole_coordinates


def test_hidden_site_cannot_reappear_with_a_different_id() -> None:
    evidence = Evidence("w", ("observed",), [0.2], [[5, 15]])
    hidden = Verification("w", ("renamed",), [0.2 + 1e-10], [[5, 15]])
    with pytest.raises(ValueError, match="coordinates overlap"):
        Case("test", evidence, Hypothesis("h", "w", [5, 15, 0, 0]), hidden)


def test_gempy_grids_do_not_reuse_any_visible_site() -> None:
    visible, hidden = borehole_coordinates()
    assert len(visible) == 5
    assert len(hidden) == 16
    assert np.ptp(hidden) == pytest.approx(2.0)
    assert not np.isclose(visible[:, None], hidden[None, :], atol=1e-8, rtol=0).any()


@pytest.mark.parametrize("record_type", [Evidence, Verification])
def test_relabelled_repeats_cannot_overweight_a_borehole(
    record_type: type[Evidence] | type[Verification],
) -> None:
    with pytest.raises(ValueError, match="duplicate borehole coordinates"):
        record_type("world", ("first", "relabelled"), [0.0, 1e-10], [[5, 15], [5, 15]])


def test_renaming_worlds_and_reordering_boreholes_cannot_cross_a_split(
    geological_cases: list[Case],
) -> None:
    c = geological_cases[0]
    copy = Case(
        "test",
        Evidence(
            "renamed-world",
            tuple(f"v{i}" for i in range(len(c.evidence.x))),
            c.evidence.x[::-1],
            c.evidence.boundaries[::-1],
        ),
        Hypothesis("renamed-case", "renamed-world", c.hypothesis.parameters),
        Verification(
            "renamed-world",
            tuple(f"h{i}" for i in range(len(c.verification.x))),
            c.verification.x[::-1],
            c.verification.boundaries[::-1],
        ),
    )
    with pytest.raises(ValueError, match="duplicates observations"):
        validate_splits([*geological_cases, copy])


def test_export_refuses_to_silently_keep_only_the_first_world_observations(
    geological_cases: list[Case], tmp_path: Path
) -> None:
    first = geological_cases[0]
    changed = Case(
        first.split,
        Evidence(
            first.evidence.world_id,
            first.evidence.borehole_ids,
            first.evidence.x,
            first.evidence.boundaries + 0.5,
        ),
        Hypothesis("new-case", first.evidence.world_id, first.hypothesis.parameters),
        first.verification,
    )
    cases = [*geological_cases, changed]
    with pytest.raises(ValueError, match="inconsistent observations"):
        validate_splits(cases)
    with pytest.raises(ValueError, match="inconsistent observations"):
        export(cases, tmp_path)
    assert not (tmp_path / "boreholes.csv").exists()


@pytest.fixture
def adapter_export(tmp_path: Path, geological_cases: list[Case]) -> Path:
    """Check adapter export integrity with controlled metadata files."""
    config = {"base_model_name_or_path": "unit-test-base", "r": 2}
    (tmp_path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"controlled-test-export")
    train_world = next(c.evidence.world_id for c in geological_cases if c.split == "train")
    manifest = {
        "worlds": {train_world: "train"},
        "status": "complete",
        "completed_optimizer_steps": 2,
        "base_model": "unit-test-base",
        "rank": 2,
        "adapter_artifacts_sha256": adapter_artifact_hashes(tmp_path),
    }
    (tmp_path / "training_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_incomplete_adapter_cannot_be_used_for_audit(
    adapter_export: Path, geological_cases: list[Case]
) -> None:
    path = adapter_export / "training_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["status"] = "running"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        validate_adapter_holdouts(adapter_export, geological_cases)


def test_copied_weights_cannot_keep_an_unrelated_training_manifest(
    adapter_export: Path, geological_cases: list[Case]
) -> None:
    validate_adapter_holdouts(adapter_export, geological_cases)
    (adapter_export / "adapter_model.safetensors").write_bytes(b"different-export")
    with pytest.raises(ValueError, match="artifacts differ"):
        validate_adapter_holdouts(adapter_export, geological_cases)


def test_changing_the_tokenizer_also_invalidates_the_saved_export(
    adapter_export: Path, geological_cases: list[Case]
) -> None:
    validate_adapter_holdouts(adapter_export, geological_cases)
    (adapter_export / "chat_template.jinja").write_text(
        "different message format", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="artifacts differ"):
        validate_adapter_holdouts(adapter_export, geological_cases)


@pytest.mark.parametrize(
    "field,value",
    [("base_model", "different-base"), ("rank", 4), ("base_model_revision", "different-revision")],
)
def test_adapter_base_and_rank_must_match_the_training_record(
    adapter_export: Path, geological_cases: list[Case], field: str, value: str | int
) -> None:
    path = adapter_export / "training_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="base model/rank differs"):
        validate_adapter_holdouts(adapter_export, geological_cases)
