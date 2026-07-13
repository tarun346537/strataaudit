"""Fail CI on fixed benchmark quality, budget or runtime regressions."""

import argparse
import json
import math
import time
from pathlib import Path

from .config import Settings
from .demo import run
from .logging_config import setup_logging
from .types import JSONRecord


def nested(document: JSONRecord, path: str) -> object:
    """Read a metric by dotted path without assuming a fixed report layout."""
    value: object = document
    for component in path.split("."):
        if not isinstance(value, dict):
            raise KeyError(path)
        value = value[component]
    return value


def check_thresholds(
    report: JSONRecord, specification: JSONRecord, baseline: JSONRecord | None = None
) -> list[str]:
    """Check quality, runtime and budget limits against fixed benchmark settings."""
    failures = []
    thresholds = specification.get("thresholds")
    if not isinstance(thresholds, dict) or not thresholds:
        return ["gate specification needs nonempty thresholds"]
    for path, rule in thresholds.items():
        if (
            not isinstance(rule, dict)
            or not rule
            or set(rule) - {"max", "min", "maximum_increase"}
            or any(
                isinstance(limit, bool)
                or not isinstance(limit, (int, float))
                or not math.isfinite(limit)
                for limit in rule.values()
            )
        ):
            failures.append(f"{path}: invalid threshold rule")
            continue
        try:
            value = nested(report, path)
        except (KeyError, TypeError):
            failures.append(f"{path}: measurement missing")
            continue
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            failures.append(f"{path}: finite numeric measurement required")
            continue
        if not math.isfinite(value):
            failures.append(f"{path}: nonfinite measurement")
            continue
        if "max" in rule and value > rule["max"]:
            failures.append(f"{path}: {value} exceeds maximum {rule['max']}")
        if "min" in rule and value < rule["min"]:
            failures.append(f"{path}: {value} below minimum {rule['min']}")
        if "maximum_increase" in rule:
            try:
                previous = nested(baseline or {}, path)
            except (KeyError, TypeError):
                failures.append(f"{path}: baseline measurement missing")
                continue
            if (
                isinstance(previous, bool)
                or not isinstance(previous, (int, float))
                or not math.isfinite(previous)
            ):
                failures.append(f"{path}: finite numeric baseline measurement required")
                continue
            if value > previous + rule["maximum_increase"]:
                failures.append(
                    f"{path}: increase exceeds baseline tolerance {rule['maximum_increase']}"
                )
    return failures


def main() -> None:
    """Run the fixed benchmark and fail the process when a gate is violated."""
    config = Settings.load()
    logger = setup_logging(config.log_level)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="benchmarks/gate.json")
    parser.add_argument("--baseline", default="benchmarks/baseline.json")
    parser.add_argument("--output", default=str(Path(config.output_dir, "regression-gate.json")))
    a = parser.parse_args()
    specification = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    baseline = json.loads(Path(a.baseline).read_text(encoding="utf-8")) if a.baseline else None
    started = time.perf_counter()
    result, traces = run(**specification["benchmark"])
    result["runtime_seconds"] = time.perf_counter() - started
    failures = check_thresholds(result, specification, baseline)
    result["regression_gate"] = {
        "passed": not failures,
        "failures": failures,
        "benchmark": specification["benchmark"],
        "specification": a.spec,
        "baseline": a.baseline,
    }
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output.with_suffix(".traces.jsonl").write_text(
        "".join(json.dumps(t, allow_nan=False) + "\n" for t in traces), encoding="utf-8"
    )
    logger.info("regression_gate_complete", extra={"fields": result["regression_gate"]})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
