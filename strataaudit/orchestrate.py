"""Run calibrated judge escalation and a trained challenger at equal audit budgets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .artifacts import CalibratedServedJudge as CalibratedServedJudge
from .artifacts import calibrate_served_judge, dataset_provenance, validate_adapter_holdouts
from .audit import EvidenceJudge, oracle
from .config import Settings
from .data import load_csv_cases
from .llm import ChatClient, ServedChallenger, ServedJudge
from .logging_config import setup_logging
from .pipeline import AuditCosts, compare_at_equal_budget
from .pipeline import audit_strategy as audit_strategy
from .torch_calibration import fit_temperature_torch
from .types import Challenger, Judge


def main() -> None:
    """Evaluate direct and challenger-assisted strategies on withheld worlds."""
    config = Settings.load()
    logger = setup_logging(config.log_level)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boreholes", required=True)
    parser.add_argument("--hypotheses", required=True)
    parser.add_argument("--judge-url", default=config.judge_url)
    parser.add_argument("--judge-model", default=config.judge_model)
    parser.add_argument("--challenger-url", default=config.challenger_url)
    parser.add_argument("--challenger-model", default=config.challenger_model)
    parser.add_argument("--challenger-adapter")
    parser.add_argument("--device", default=config.device)
    parser.add_argument("--budget-units", type=int, default=config.audit_budget_units)
    parser.add_argument("--judge-cost", type=int, default=config.judge_cost)
    parser.add_argument("--challenger-cost", type=int, default=config.challenger_cost)
    parser.add_argument("--verification-cost", type=int, default=config.verification_cost)
    parser.add_argument("--escalation-confidence", type=float, default=config.escalation_confidence)
    parser.add_argument("--skip-served-calibration", action="store_true")
    parser.add_argument("--output", default=str(Path(config.output_dir, "orchestrated-audit.json")))
    args = parser.parse_args()
    if not args.judge_model:
        parser.error("set --judge-model or STRATA_JUDGE_MODEL")
    cases = load_csv_cases(args.boreholes, args.hypotheses)
    provenance = dataset_provenance(args.boreholes, args.hypotheses, cases)
    adapter_validation = (
        validate_adapter_holdouts(args.challenger_adapter, cases)
        if args.challenger_adapter
        else None
    )
    train, calibration, test = (
        [c for c in cases if c.split == s] for s in ["train", "calibration", "test"]
    )
    cheap_judge = EvidenceJudge.fit(train, tolerance=config.tolerance)
    fit = fit_temperature_torch(
        [cheap_judge.logit(c.evidence, c.hypothesis) for c in calibration],
        [oracle(c.hypothesis, c.verification, config.tolerance) for c in calibration],
        calibration_groups=[c.evidence.world_id for c in calibration],
        training_groups=[c.evidence.world_id for c in train],
        test_groups=[c.evidence.world_id for c in test],
    )
    served = ServedJudge(ChatClient(args.judge_url, args.judge_model))
    served_fit = None
    if not args.skip_served_calibration:
        _, served_fit = calibrate_served_judge(
            calibration,
            served,
            [c.evidence.world_id for c in train],
            [c.evidence.world_id for c in test],
            tolerance=config.tolerance,
        )
    calibration_attempts = served.client.transport.attempts

    def make_judge() -> Judge:
        fresh = ServedJudge(ChatClient(args.judge_url, args.judge_model))
        return CalibratedServedJudge(fresh, served_fit.temperature) if served_fit else fresh

    challenger: Challenger
    if args.challenger_adapter:
        from .inference import LocalChallenger

        challenger = LocalChallenger(args.challenger_adapter, args.device)
    elif args.challenger_model:
        challenger = ServedChallenger(ChatClient(args.challenger_url, args.challenger_model))
    else:
        parser.error("provide --challenger-adapter or --challenger-model")
    report, traces = compare_at_equal_budget(
        test,
        cheap_judge,
        fit.temperature,
        make_judge,
        challenger,
        budget_units=args.budget_units,
        costs=AuditCosts(args.judge_cost, args.challenger_cost, args.verification_cost),
        escalation_confidence=args.escalation_confidence,
        verification_reserve=config.verification_reserve,
        tolerance=config.tolerance,
    )
    report["temperature_fit"] = asdict(fit)
    report["provenance"] = provenance
    report["audit_settings"] = {
        "correctness_tolerance_metres": config.tolerance,
        "escalation_confidence": args.escalation_confidence,
        "verification_reserve": config.verification_reserve,
        "cheap_judge": cheap_judge.to_dict(),
        "served_judge_model": args.judge_model,
        "challenger_model": args.challenger_model if not args.challenger_adapter else None,
    }
    report["served_judge_temperature_fit"] = asdict(served_fit) if served_fit else None
    report["served_calibration_cases_outside_test_budget"] = len(calibration) if served_fit else 0
    report["served_calibration_http_attempts"] = calibration_attempts
    report["challenger_adapter_holdout_validation"] = adapter_validation
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output.with_suffix(".traces.jsonl").write_text(
        "".join(
            json.dumps({"strategy": s, **t}, allow_nan=False) + "\n"
            for s, rows in traces.items()
            for t in rows
        ),
        encoding="utf-8",
    )
    logger.info("budgeted_audit_complete", extra={"fields": report})


if __name__ == "__main__":
    main()
