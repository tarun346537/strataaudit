"""Calibration and holdout checks shared across audit entry points."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .audit import oracle, sigmoid
from .constants import DEFAULT_BOUNDARY_TOLERANCE, PROBABILITY_EPSILON
from .data import Case, Evidence, Hypothesis
from .torch_calibration import TemperatureFit, fit_temperature_torch
from .types import JSONRecord, Judge, JudgeResult


def dataset_provenance(
    boreholes: str | Path, hypotheses: str | Path, cases: list[Case]
) -> JSONRecord:
    """Record input content and split identity alongside every evaluation."""
    inputs = {}
    for name, path in {"boreholes": Path(boreholes), "hypotheses": Path(hypotheses)}.items():
        inputs[name] = {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {
        "inputs": inputs,
        "world_splits": {case.evidence.world_id: case.split for case in cases},
        "cases_by_split": {
            split: sum(case.split == split for case in cases)
            for split in ["train", "calibration", "test"]
        },
    }


def adapter_artifact_hashes(adapter_path: str | Path) -> dict[str, str]:
    """Bind PEFT weights, configuration and tokenizer files to their training run."""
    output = Path(adapter_path)
    paths = [output / "adapter_config.json", *sorted(output.glob("adapter_model*.safetensors"))]
    if len(paths) < 2 or any(not path.is_file() for path in paths):
        raise ValueError("adapter config/weights missing. finish training and export the adapter")
    token_files = {
        path
        for pattern in (
            "tokenizer*",
            "vocab.*",
            "merges.txt",
            "special_tokens_map.json",
            "chat_template*",
            "adapter_model*.index.json",
        )
        for path in output.glob(pattern)
        if path.is_file()
    }
    paths.extend(sorted(token_files))
    hashes = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[path.name] = digest.hexdigest()
    return hashes


class CalibratedServedJudge:
    """Apply a held-out temperature to actual served judge probabilities."""

    def __init__(self, delegate: Judge, temperature: float) -> None:
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("positive finite judge temperature required")
        self.delegate, self.temperature = delegate, temperature

    def assess(self, e: Evidence, h: Hypothesis) -> JudgeResult:
        """Rescale confidence while preserving the judge's decision boundary."""
        result = self.delegate.assess(e, h).copy()
        raw = result["confidence"] if result["supported"] else 1 - result["confidence"]
        raw = float(np.clip(raw, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON))
        probability = float(sigmoid(np.log(raw / (1 - raw)) / self.temperature))
        result.update(
            {
                "supported": probability >= 0.5,
                "confidence": max(probability, 1 - probability),
                "raw_probability_supported": raw,
                "temperature": self.temperature,
            }
        )
        return result


def calibrate_served_judge(
    cases: list[Case],
    served_judge: Judge,
    training_groups: Sequence[str],
    test_groups: Sequence[str],
    tolerance: float = DEFAULT_BOUNDARY_TOLERANCE,
) -> tuple[CalibratedServedJudge, TemperatureFit]:
    """Fit served-model confidence using independently held-out calibration worlds."""
    if not cases or any(c.split != "calibration" for c in cases):
        raise ValueError("served judge calibration accepts calibration cases only")
    calibration_groups = {case.evidence.world_id for case in cases}
    if (
        calibration_groups & set(training_groups)
        or calibration_groups & set(test_groups)
        or set(training_groups) & set(test_groups)
    ):
        raise ValueError(
            "training, calibration, and test groups must be disjoint before model scoring"
        )
    logits: list[float] = []
    for c in cases:
        response = served_judge.assess(c.evidence, c.hypothesis)
        probability = (
            response["confidence"] if response["supported"] else 1 - response["confidence"]
        )
        probability = float(np.clip(probability, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON))
        logits.append(float(np.log(probability / (1 - probability))))
    fit = fit_temperature_torch(
        logits,
        [oracle(c.hypothesis, c.verification, tolerance) for c in cases],
        calibration_groups=[c.evidence.world_id for c in cases],
        training_groups=training_groups,
        test_groups=test_groups,
    )
    return CalibratedServedJudge(served_judge, fit.temperature), fit


def validate_adapter_holdouts(adapter_path: str | Path, cases: list[Case]) -> JSONRecord:
    """Check saved training metadata before loading an adapter for evaluation."""
    path = Path(adapter_path) / "training_manifest.json"
    if not path.is_file():
        raise ValueError(
            "challenger evaluation requires its actual training_manifest.json with world splits"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("adapter training manifest must contain a JSON object")
    fitted_worlds = manifest.get("worlds")
    if (
        not isinstance(fitted_worlds, dict)
        or not fitted_worlds
        or not all(
            isinstance(world, str)
            and world
            and isinstance(split, str)
            and split in {"train", "calibration"}
            for world, split in fitted_worlds.items()
        )
    ):
        raise ValueError(
            "adapter training manifest requires nonempty world IDs and train/calibration assignments"
        )
    current_splits: dict[str, str] = {}
    for case in cases:
        world = case.evidence.world_id
        if world in current_splits and current_splits[world] != case.split:
            raise ValueError("evaluation dataset has inconsistent world splits")
        current_splits[world] = case.split
    for world, fitted_split in fitted_worlds.items():
        current = current_splits.get(world)
        if current == "test":
            raise ValueError(f"test world {world!r} was used in adapter training/calibration")
        if current is not None and current != fitted_split:
            raise ValueError(f"adapter/dataset split mismatch for world {world!r}")
    if (
        manifest.get("status") != "complete"
        or type(manifest.get("completed_optimizer_steps")) is not int
        or manifest["completed_optimizer_steps"] < 1
    ):
        raise ValueError("adapter training is incomplete. use a completed training export")
    actual_hashes = adapter_artifact_hashes(adapter_path)
    if manifest.get("adapter_artifacts_sha256") != actual_hashes:
        raise ValueError("adapter artifacts differ from the training manifest. restore the export")
    config = json.loads((Path(adapter_path) / "adapter_config.json").read_text(encoding="utf-8"))
    base_model = manifest.get("base_model")
    if (
        not isinstance(base_model, str)
        or not base_model
        or config.get("base_model_name_or_path") != base_model
        or config.get("r") != manifest.get("rank")
        or config.get("revision") != manifest.get("base_model_revision")
    ):
        raise ValueError("adapter base model/rank differs from the training manifest or revision")
    return {
        "training_manifest": str(path),
        "training_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fitted_worlds": len(fitted_worlds),
        "test_world_overlap": 0,
        "adapter_artifacts_sha256": actual_hashes,
        "completed_optimizer_steps": manifest["completed_optimizer_steps"],
    }
