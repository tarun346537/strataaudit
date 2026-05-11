"""Optional GemPy 3 world generation and sampled borehole-boundary export.

Creates two planar stratigraphic surfaces from local CSVs, then samples the
GemPy lithology model along boreholes. Synthetic transition depths use the
configured sampling resolution.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .config import Settings
from .constants import COORDINATE_ATOL_METRES, DEFAULT_SEED
from .data import Case, Evidence, Hypothesis, Verification, validate_splits
from .fixtures import export
from .logging_config import setup_logging
from .types import FloatArray

if TYPE_CHECKING:
    from gempy.core.data import GeoModel


def borehole_coordinates() -> tuple[FloatArray, FloatArray]:
    """Keep withheld sites distinct even when regular sampling grids line up."""
    visible = np.linspace(-0.4, 0.4, 5)
    candidates = np.linspace(-1, 1, 21)
    repeated = np.isclose(
        candidates[:, None], visible[None, :], atol=COORDINATE_ATOL_METRES, rtol=0
    ).any(axis=1)
    return visible, candidates[~repeated]


def sample_boundaries(
    model: GeoModel, x: ArrayLike, zmax: float = 25.0, z_samples: int = 301
) -> FloatArray:
    """Extract two discretized boundary transitions from sampled GemPy units."""
    import gempy as gp

    x = np.asarray(x, float)
    z = np.linspace(0, zmax, z_samples)
    points = np.column_stack([np.repeat(x, len(z)), np.zeros(len(x) * len(z)), np.tile(z, len(x))])
    values = np.asarray(gp.compute_model_at(gempy_model=model, at=points), float).reshape(
        len(x), len(z)
    )
    if not np.isfinite(values).all():
        raise ValueError("GemPy returned nonfinite lithology")
    # GemPy custom-grid values may be continuous near interfaces. Nearest unit
    # boundary depths reflect borehole sampling resolution
    units = np.rint(values).astype(int)
    results = []
    for row in units:
        changes = np.flatnonzero(np.diff(row) != 0)
        if len(changes) != 2:
            raise ValueError(f"expected two boundary crossings, got {len(changes)}. inspect model")
        results.append((z[changes] + z[changes + 1]) / 2)
    return np.array(results)


def generate_worlds(output_dir: str | Path, seed: int = DEFAULT_SEED, worlds: int = 15) -> None:
    """Construct independent GemPy worlds and export their borehole boundaries."""
    import gempy as gp

    if worlds < 15:
        raise ValueError("at least 15 worlds required for grouped evaluation")
    output = Path(output_dir)
    source_dir = output / "gempy_inputs"
    source_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(worlds):
        world = f"gempy-synthetic-{i:04d}"
        truth = np.array([7.0, 17.0, rng.uniform(-1, 1), rng.uniform(-1, 1)])
        points = source_dir / f"{world}-points.csv"
        orientations = source_dir / f"{world}-orientations.csv"
        with points.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["X", "Y", "Z", "formation"])
            for j, formation in enumerate(["upper", "lower"]):
                for x, y in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    w.writerow([x, y, truth[j] + truth[j + 2] * x, formation])
        with orientations.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["X", "Y", "Z", "G_x", "G_y", "G_z", "formation"])
            for j, formation in enumerate(["upper", "lower"]):
                w.writerow([0, 0, truth[j], -truth[j + 2], 0, 1, formation])
        model = gp.create_geomodel(
            project_name=world,
            extent=[-1.1, 1.1, -1.1, 1.1, 0.0, 25.0],
            refinement=2,
            importer_helper=gp.data.ImporterHelper(
                path_to_surface_points=str(points),
                path_to_orientations=str(orientations),
                pandas_reader_kwargs={"dtype": {"formation": object}},
            ),
        )
        gp.map_stack_to_surfaces(model, mapping_object={"stratigraphy": ("upper", "lower")})
        visible_x, hidden_x = borehole_coordinates()
        boundaries = sample_boundaries(model, np.r_[visible_x, hidden_x])
        e = Evidence(
            world, tuple(f"{world}-visible-{j}" for j in range(5)), visible_x, boundaries[:5]
        )
        v = Verification(
            world,
            tuple(f"{world}-hidden-{j}" for j in range(len(hidden_x))),
            hidden_x,
            boundaries[len(visible_x) :],
        )
        split = (
            "train" if i < int(0.6 * worlds) else "calibration" if i < int(0.8 * worlds) else "test"
        )
        for j, scale in enumerate([0.05, 0.2, 0.6, 1.2]):
            h = Hypothesis(f"{world}-h{j}", world, truth + rng.normal(0, scale, size=4))
            cases.append(Case(split, e, h, v))
    validate_splits(cases)
    export(cases, output)
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "kind": "new_gempy_synthetic_worlds",
                "gempy_version": gp.__version__,
                "worlds": worlds,
                "seed": seed,
                "boundary_resolution": 25 / 300,
                "visible_boreholes_per_world": len(visible_x),
                "verification_boreholes_per_world": len(hidden_x),
                "coordinate_overlap": 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Write configured GemPy inputs and normalized synthetic audit data."""
    config = Settings.load()
    logger = setup_logging(config.log_level)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=str(Path(config.output_dir, "gempy")))
    p.add_argument("--worlds", type=int, default=15)
    p.add_argument("--seed", type=int, default=config.seed)
    a = p.parse_args()
    generate_worlds(a.output_dir, a.seed, a.worlds)
    logger.info(
        "gempy_worlds_written", extra={"fields": {"output_dir": a.output_dir, "worlds": a.worlds}}
    )


if __name__ == "__main__":
    main()
