"""Validated schemas and grouped synthetic/CSV datasets.

Visible boreholes and verification boreholes have distinct types. Judges receive
Evidence. Evaluation and oracle code receive Verification.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .constants import (
    COORDINATE_ATOL_METRES,
    DEFAULT_HYPOTHESES_PER_WORLD,
    DEFAULT_SEED,
    DEFAULT_WORLDS,
)
from .types import FloatArray, JSONRecord


def finite_array(value: ArrayLike, shape_tail: tuple[int, ...], name: str) -> FloatArray:
    """Check observation arrays before they cross an audit boundary."""
    a = np.asarray(value, dtype=np.float64)
    if (
        a.ndim != len(shape_tail) + 1
        or a.shape[1:] != shape_tail
        or not len(a)
        or not np.isfinite(a).all()
    ):
        raise ValueError(f"{name} must be nonempty, finite, with trailing shape {shape_tail}")
    return a


@dataclass(frozen=True)
class Evidence:
    """Borehole boundaries available to the judge and challenger."""

    world_id: str
    borehole_ids: tuple[str, ...]
    x: FloatArray
    boundaries: FloatArray

    def __post_init__(self) -> None:
        x = finite_array(self.x, (), "x")
        b = finite_array(self.boundaries, (2,), "boundaries")
        if not self.world_id or len(x) != len(b) or len(set(self.borehole_ids)) != len(x):
            raise ValueError("unique borehole IDs must match the observations")
        if len(self.borehole_ids) != len(x) or np.any(b[:, 0] >= b[:, 1]):
            raise ValueError("ordered boundary depths are required at every borehole")
        if np.any(np.diff(np.sort(x)) <= COORDINATE_ATOL_METRES):
            raise ValueError("duplicate borehole coordinates. use distinct transect sites")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "boundaries", b)

    def prompt_payload(self) -> JSONRecord:
        """Serialize visible observations without verification data."""
        return {
            "world_id": self.world_id,
            "observations": [
                {"id": h, "x": float(x), "boundary_depths": b.tolist()}
                for h, x, b in zip(self.borehole_ids, self.x, self.boundaries, strict=True)
            ],
        }


@dataclass(frozen=True)
class Verification:
    """Withheld boreholes owned by the independent verifier/evaluator."""

    world_id: str
    borehole_ids: tuple[str, ...]
    x: FloatArray
    boundaries: FloatArray

    def __post_init__(self) -> None:
        checked = Evidence(self.world_id, self.borehole_ids, self.x, self.boundaries)
        object.__setattr__(self, "x", checked.x)
        object.__setattr__(self, "boundaries", checked.boundaries)


@dataclass(frozen=True)
class Hypothesis:
    """Two planar geological boundaries proposed for a specific world."""

    case_id: str
    world_id: str
    parameters: FloatArray  # [depth1, depth2, slope1, slope2], where depth(x)=depth+slope*x

    def __post_init__(self) -> None:
        p = np.asarray(self.parameters, dtype=float)
        if not self.case_id or not self.world_id or p.shape != (4,) or not np.isfinite(p).all():
            raise ValueError("hypothesis requires IDs and four finite parameters")
        object.__setattr__(self, "parameters", p)

    def predict(self, x: ArrayLike) -> FloatArray:
        """Evaluate both boundary depths at the supplied coordinates."""
        xx = np.asarray(x, dtype=np.float64)
        return np.asarray(
            self.parameters[:2] + xx[..., None] * self.parameters[2:], dtype=np.float64
        )

    def payload(self) -> JSONRecord:
        """Serialize the proposed model independently of evidence or truth."""
        return {
            "case_id": self.case_id,
            "world_id": self.world_id,
            "parameters": self.parameters.tolist(),
            "parameterization": "two boundary depths = intercept + slope*x",
        }


@dataclass(frozen=True)
class Case:
    """Join one hypothesis to separate visible and withheld observations."""

    split: str
    evidence: Evidence
    hypothesis: Hypothesis
    verification: Verification

    def __post_init__(self) -> None:
        ids = {self.evidence.world_id, self.hypothesis.world_id, self.verification.world_id}
        if len(ids) != 1 or self.split not in {"train", "calibration", "test"}:
            raise ValueError("case IDs/split are inconsistent")
        if set(self.evidence.borehole_ids) & set(self.verification.borehole_ids):
            raise ValueError("visible and verification boreholes overlap")
        if np.any(
            np.isclose(
                self.evidence.x[:, None],
                self.verification.x[None, :],
                atol=COORDINATE_ATOL_METRES,
                rtol=0,
            )
        ):
            raise ValueError(
                "visible and verification coordinates overlap. withhold distinct sites"
            )


def validate_splits(cases: list[Case]) -> None:
    """Reject duplicate hypotheses and geological worlds shared across splits."""
    seen: dict[str, str] = {}
    observations: dict[str, tuple[Evidence | Verification, ...]] = {}
    contents: dict[tuple[tuple[float, ...], ...], tuple[str, str]] = {}
    case_ids: set[str] = set()
    for case in cases:
        if case.hypothesis.case_id in case_ids:
            raise ValueError("duplicate case_id")
        case_ids.add(case.hypothesis.case_id)
        w = case.evidence.world_id
        if w in seen and seen[w] != case.split:
            raise ValueError(f"world {w} leaks across splits")
        seen[w] = case.split
        current: tuple[Evidence | Verification, ...] = (case.evidence, case.verification)
        if w in observations:
            for previous, supplied in zip(observations[w], current, strict=True):
                if (
                    previous.borehole_ids != supplied.borehole_ids
                    or not np.array_equal(previous.x, supplied.x)
                    or not np.array_equal(previous.boundaries, supplied.boundaries)
                ):
                    raise ValueError(f"world {w} has inconsistent observations or borehole roles")
        observations[w] = current
        rows = np.vstack([np.column_stack([record.x, record.boundaries]) for record in current])
        signature = tuple(sorted(tuple(float(v) for v in row) for row in rows))
        previous_split, previous_world = contents.get(signature, (case.split, w))
        if previous_split != case.split:
            raise ValueError(
                f"world {w} duplicates observations from {previous_world} across splits"
            )
        contents[signature] = (case.split, w)
    if set(seen.values()) != {"train", "calibration", "test"}:
        raise ValueError("all three nonempty world-disjoint splits are required")


def synthetic_cases(
    seed: int = DEFAULT_SEED,
    worlds: int = DEFAULT_WORLDS,
    per_world: int = DEFAULT_HYPOTHESES_PER_WORLD,
) -> list[Case]:
    """Create small independent worlds for numerical checks and local examples."""
    if worlds < 15 or per_world < 1:
        raise ValueError("at least 15 worlds and one hypothesis per world are required")
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(worlds):
        w = f"synthetic-{i:04d}"
        split = (
            "train" if i < int(0.6 * worlds) else "calibration" if i < int(0.8 * worlds) else "test"
        )
        truth = np.array([6 + rng.normal(), 16 + rng.normal(), rng.normal(), rng.normal()])
        visible_x = np.sort(rng.uniform(-0.45, 0.45, size=5))
        hidden_x = np.linspace(-1, 1, 11)
        evidence = Evidence(
            w,
            tuple(f"{w}-visible-{j}" for j in range(5)),
            visible_x,
            truth[:2] + visible_x[:, None] * truth[2:] + rng.normal(0, 0.08, size=(5, 2)),
        )
        verification = Verification(
            w,
            tuple(f"{w}-hidden-{j}" for j in range(11)),
            hidden_x,
            truth[:2] + hidden_x[:, None] * truth[2:],
        )
        for j in range(per_world):
            scale = rng.choice([0.05, 0.2, 0.5, 1.0, 2.0])
            p = truth + rng.normal(0, scale, size=4) * np.array([0.6, 0.6, 1.5, 1.5])
            cases.append(Case(split, evidence, Hypothesis(f"{w}-h{j}", w, p), verification))
    validate_splits(cases)
    return cases


def load_csv_cases(boreholes_path: str | Path, hypotheses_path: str | Path) -> list[Case]:
    """Read normalized boundary logs from the release-independent CSV schema.

    CSV: world_id,split,borehole_id,role,x,depth_1,depth_2.
    role is visible or verification. JSONL: case_id,world_id,parameters.
    """
    worlds: dict[str, JSONRecord] = {}
    with Path(boreholes_path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"world_id", "split", "borehole_id", "role", "x", "depth_1", "depth_2"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"csv needs {sorted(required)}")
        for line, row in enumerate(reader, 2):
            w = row["world_id"]
            entry = worlds.setdefault(w, {"split": row["split"], "visible": [], "verification": []})
            if row["split"] != entry["split"] or row["role"] not in {"visible", "verification"}:
                raise ValueError(f"bad split/role at CSV line {line}")
            entry[row["role"]].append(
                (
                    row["borehole_id"],
                    float(row["x"]),
                    [float(row["depth_1"]), float(row["depth_2"])],
                )
            )
    cases = []
    with Path(hypotheses_path).open(encoding="utf-8") as f:
        for line, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            h = json.loads(raw)
            hyp = Hypothesis(h["case_id"], h["world_id"], h["parameters"])
            if hyp.world_id not in worlds:
                raise ValueError(f"unknown world at JSONL line {line}")
            info = worlds[hyp.world_id]
            visible, withheld = info["visible"], info["verification"]
            evidence = Evidence(
                hyp.world_id,
                tuple(r[0] for r in visible),
                np.array([r[1] for r in visible]),
                np.array([r[2] for r in visible]),
            )
            verification = Verification(
                hyp.world_id,
                tuple(r[0] for r in withheld),
                np.array([r[1] for r in withheld]),
                np.array([r[2] for r in withheld]),
            )
            cases.append(Case(info["split"], evidence, hyp, verification))
    validate_splits(cases)
    return cases
