"""Normalize sourced drill-core interval logs without guessing a release schema.

Users select the source CSV, column mapping, geological boundary units, and source
record. Distinct boreholes are withheld within each world. Worlds are split by
group. Coordinates and depths are converted to metres before export.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from .config import Settings
from .constants import DEFAULT_SEED
from .logging_config import setup_logging
from .torch_calibration import group_holdout_split
from .types import JSONRecord


def import_intervals(
    source: str | Path,
    output_dir: str | Path,
    *,
    source_url: str,
    dataset_title: str,
    columns: dict[str, str],
    middle_unit: str,
    lower_unit: str,
    depth_unit: str = "m",
    coordinate_unit: str = "m",
    seed: int = DEFAULT_SEED,
    provider: str = "USGS",
    visible_fraction: float = 0.6,
) -> JSONRecord:
    """Normalize interval boundaries while retaining source and split provenance."""
    host = (urlparse(source_url).hostname or "").lower()
    if not host or not source_url.startswith("https://") or not dataset_title.strip():
        raise ValueError("explicit HTTPS source URL and dataset title required")
    if provider == "USGS" and not (
        host.endswith(".usgs.gov")
        or host == "usgs.gov"
        or host.endswith(".sciencebase.gov")
        or host == "sciencebase.gov"
    ):
        raise ValueError("USGS provenance requires an actual USGS/ScienceBase source record URL")
    required = {"world_id", "borehole_id", "x", "top_depth", "bottom_depth", "lithology"}
    if set(columns) != required or middle_unit == lower_unit or not middle_unit or not lower_unit:
        raise ValueError(
            "complete explicit interval column mapping and distinct boundary units required"
        )
    factors = {"m": 1.0, "ft": 0.3048}
    if depth_unit not in factors or coordinate_unit not in factors or not 0 < visible_fraction < 1:
        raise ValueError(
            "supported units are m/ft. visible_fraction must lie strictly between 0 and 1"
        )
    source = Path(source)
    groups: dict[str, dict[str, list[tuple[float, float, float, str]]]] = {}
    with source.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not set(columns.values()).issubset(reader.fieldnames or []):
            raise ValueError("source CSV is missing mapped columns")
        for line, raw in enumerate(reader, 2):
            row = {key: raw[value].strip() for key, value in columns.items()}
            world, hole = row["world_id"], row["borehole_id"]
            x, top, bottom = (float(row[k]) for k in ["x", "top_depth", "bottom_depth"])
            if (
                not world
                or not hole
                or not np.isfinite([x, top, bottom]).all()
                or top < 0
                or bottom <= top
            ):
                raise ValueError(f"invalid interval at source row {line}")
            groups.setdefault(world, {}).setdefault(hole, []).append(
                (x, top, bottom, row["lithology"])
            )
    assignments = group_holdout_split(groups, seed)
    rng = np.random.default_rng(seed)
    rows = []
    for world in sorted(groups):
        holes = groups[world]
        if len(holes) < 4:
            raise ValueError(
                f"world {world} needs at least four boreholes (two visible, two verification)"
            )
        order = rng.permutation(sorted(holes)).tolist()
        n_visible = min(len(order) - 2, max(2, int(len(order) * visible_fraction)))
        for i, hole in enumerate(order):
            intervals = sorted(holes[hole], key=lambda item: item[1])
            if len({r[0] for r in intervals}) != 1:
                raise ValueError(f"inconsistent coordinate for borehole {hole}")
            if any(a[2] > b[1] + 1e-8 for a, b in zip(intervals, intervals[1:], strict=False)):
                raise ValueError(f"overlapping intervals in borehole {hole}")
            depths = []
            for unit in [middle_unit, lower_unit]:
                occurrences = [r[1] for r in intervals if r[3] == unit]
                if not occurrences:
                    raise ValueError(f"boundary unit {unit!r} absent in borehole {hole}")
                depths.append(min(occurrences) * factors[depth_unit])
            if depths[0] >= depths[1]:
                raise ValueError(f"boundary units have inconsistent order in borehole {hole}")
            rows.append(
                [
                    world,
                    assignments[world],
                    hole,
                    "visible" if i < n_visible else "verification",
                    intervals[0][0] * factors[coordinate_unit],
                    *depths,
                ]
            )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "boreholes.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["world_id", "split", "borehole_id", "role", "x", "depth_1", "depth_2"])
        writer.writerows(rows)
    provenance = {
        "kind": "normalized_drill_core_intervals",
        "provider": provider,
        "dataset_title": dataset_title,
        "source_url": source_url,
        "source_file": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "column_mapping": columns,
        "boundary_interpretation": {
            "depth_1": f"first top of {middle_unit}",
            "depth_2": f"first top of {lower_unit}",
        },
        "input_units": {"depth": depth_unit, "coordinate": coordinate_unit},
        "output_units": "metres",
        "seed": seed,
        "world_splits": assignments,
        "worlds": len(groups),
        "boreholes": len(rows),
        "visible_boreholes": sum(r[3] == "visible" for r in rows),
        "verification_boreholes": sum(r[3] == "verification" for r in rows),
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def main() -> None:
    """Import an explicitly selected geological release using its column mapping."""
    config = Settings.load()
    logger = setup_logging(config.log_level)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--intervals", required=True)
    p.add_argument(
        "--columns", required=True, help="JSON mapping canonical fields to source CSV column names"
    )
    p.add_argument("--source-url", required=True)
    p.add_argument("--dataset-title", required=True)
    p.add_argument("--provider", default="USGS")
    p.add_argument("--middle-unit", required=True)
    p.add_argument("--lower-unit", required=True)
    p.add_argument("--depth-unit", choices=["m", "ft"], default="m")
    p.add_argument("--coordinate-unit", choices=["m", "ft"], default="m")
    p.add_argument("--seed", type=int, default=config.seed)
    p.add_argument("--output-dir", default=str(Path(config.output_dir, "imported-core")))
    a = p.parse_args()
    provenance = import_intervals(
        a.intervals,
        a.output_dir,
        source_url=a.source_url,
        dataset_title=a.dataset_title,
        columns=json.loads(Path(a.columns).read_text(encoding="utf-8")),
        middle_unit=a.middle_unit,
        lower_unit=a.lower_unit,
        depth_unit=a.depth_unit,
        coordinate_unit=a.coordinate_unit,
        seed=a.seed,
        provider=a.provider,
    )
    logger.info("core_intervals_imported", extra={"fields": provenance})


if __name__ == "__main__":
    main()
