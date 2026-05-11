"""Export synthetic normalized CSV/JSONL for the research CLI."""

import argparse
import csv
import json
from pathlib import Path

from .config import Settings
from .data import Case, Evidence, Verification, synthetic_cases, validate_splits
from .logging_config import setup_logging


def export(cases: list[Case], output_dir: str | Path) -> None:
    """Write normalized inputs and an explicit synthetic provenance manifest."""
    validate_splits(cases)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "boreholes.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["world_id", "split", "borehole_id", "role", "x", "depth_1", "depth_2"])
        seen = set()
        for c in cases:
            if c.evidence.world_id in seen:
                continue
            seen.add(c.evidence.world_id)
            observations: list[tuple[str, Evidence | Verification]] = [
                ("visible", c.evidence),
                ("verification", c.verification),
            ]
            for role, data in observations:
                for identity, x, depths in zip(
                    data.borehole_ids, data.x, data.boundaries, strict=True
                ):
                    writer.writerow([data.world_id, c.split, identity, role, x, *depths])
    with (output / "hypotheses.jsonl").open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(
                json.dumps(
                    {
                        "case_id": c.hypothesis.case_id,
                        "world_id": c.hypothesis.world_id,
                        "parameters": c.hypothesis.parameters.tolist(),
                    }
                )
                + "\n"
            )
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "kind": "synthetic_fixture",
                "worlds": len(seen),
                "cases": len(cases),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Write synthetic planar-world inputs and their provenance manifest."""
    config = Settings.load()
    logger = setup_logging(config.log_level)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=str(Path(config.output_dir, "fixtures")))
    p.add_argument("--worlds", type=int, default=config.worlds)
    p.add_argument("--seed", type=int, default=config.seed)
    a = p.parse_args()
    export(synthetic_cases(a.seed, a.worlds), a.output_dir)
    logger.info("sample_inputs_written", extra={"fields": {"output_dir": a.output_dir}})


if __name__ == "__main__":
    main()
