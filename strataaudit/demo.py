"""Run a deterministic audit with synthetic planar worlds and a numerical judge."""

import argparse
import json
from pathlib import Path

from .audit import EvidenceJudge, evaluate, fit_temperature, oracle
from .config import Settings
from .constants import DEFAULT_SEED, DEFAULT_VERIFICATION_FRACTION, DEFAULT_WORLDS
from .data import synthetic_cases
from .logging_config import setup_logging
from .types import JSONRecord


def run(
    seed: int = DEFAULT_SEED,
    worlds: int = DEFAULT_WORLDS,
    budget: float = DEFAULT_VERIFICATION_FRACTION,
) -> tuple[JSONRecord, list[JSONRecord]]:
    """Fit and evaluate a small grouped numerical audit end to end."""
    cases = synthetic_cases(seed, worlds)
    train = [c for c in cases if c.split == "train"]
    calibration = [c for c in cases if c.split == "calibration"]
    test = [c for c in cases if c.split == "test"]
    judge = EvidenceJudge.fit(train)
    temperature = fit_temperature(
        [judge.logit(c.evidence, c.hypothesis) for c in calibration],
        [oracle(c.hypothesis, c.verification) for c in calibration],
    )
    metrics, traces = evaluate(test, judge, temperature, budget)
    result = {
        "provenance": {
            "kind": "synthetic_demo",
            "seed": seed,
            "simulator": "NumPy planar-boundary simulator",
            "judge": "logistic evidence baseline",
        },
        "splits": {
            s: {
                "cases": sum(c.split == s for c in cases),
                "worlds": len({c.evidence.world_id for c in cases if c.split == s}),
            }
            for s in ["train", "calibration", "test"]
        },
        "metrics": metrics,
        "judge": judge.to_dict(),
    }
    return result, traces


def main() -> None:
    """Write numerical metrics and traces using configured local settings."""
    config = Settings.load()
    logger = setup_logging(config.log_level)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(Path(config.output_dir, "demo.json")))
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--worlds", type=int, default=config.worlds)
    parser.add_argument("--budget", type=float, default=config.verification_fraction)
    args = parser.parse_args()
    result, traces = run(args.seed, args.worlds, args.budget)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    trace_path = output.with_suffix(".traces.jsonl")
    trace_path.write_text(
        "".join(json.dumps(t, allow_nan=False) + "\n" for t in traces), encoding="utf-8"
    )
    logger.info(
        "numerical_audit_complete",
        extra={
            "fields": {
                "output": str(output),
                "traces": str(trace_path),
                "metrics": result["metrics"],
            }
        },
    )


if __name__ == "__main__":
    main()
