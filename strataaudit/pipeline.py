"""Cost-accounted audit strategies with independent final evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Literal, TypedDict, Unpack

import numpy as np

from .audit import hidden_error, oracle, sigmoid
from .constants import DEFAULT_BOUNDARY_TOLERANCE
from .data import Case, Hypothesis
from .logging_config import record_trace
from .transport import InferenceError, model_attempt_limit
from .types import Challenger, JSONRecord, Judge, LogitJudge


@dataclass(frozen=True)
class AuditCosts:
    """Relative accounting weights for expensive model and verification calls."""

    judge: int = 1
    challenger: int = 1
    verification: int = 4

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("external call costs must be positive integers")


DEFAULT_COSTS = AuditCosts()


class AuditOptions(TypedDict, total=False):
    """Explicit budget parameters shared by compared strategies."""

    budget_units: int
    costs: AuditCosts
    escalation_confidence: float
    verification_reserve: float
    tolerance: float


def audit_strategy(
    cases: list[Case],
    cheap_judge: LogitJudge,
    temperature: float,
    served_judge: Judge,
    *,
    challenger: Challenger | None = None,
    budget_units: int = 100,
    costs: AuditCosts = DEFAULT_COSTS,
    escalation_confidence: float = 0.9,
    verification_reserve: float = 0.4,
    tolerance: float = DEFAULT_BOUNDARY_TOLERANCE,
) -> tuple[JSONRecord, list[JSONRecord]]:
    """Audit without allowing hidden transport retries to overspend the budget.

    Each budgeted model call permits one physical HTTP attempt. Unbudgeted jobs
    retain bounded retries. Offline oracle labels are computed after decisions.
    """
    if not cases or any(c.split != "test" for c in cases):
        raise ValueError("nonempty held-out test cases required")
    if type(budget_units) is not int or budget_units < 0 or not 0.5 <= escalation_confidence <= 1:
        raise ValueError("nonnegative integer budget and confidence in [.5,1] required")
    if not 0 <= verification_reserve <= 1 or temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("invalid reserve/temperature")
    p = np.array(
        [float(sigmoid(cheap_judge.logit(c.evidence, c.hypothesis) / temperature)) for c in cases]
    )
    final_hypotheses = [c.hypothesis for c in cases]
    accepted = p >= 0.5
    traces: list[JSONRecord] = [
        {
            "case_id": c.hypothesis.case_id,
            "world_id": c.evidence.world_id,
            "cheap_probability_supported": float(q),
            "events": [],
        }
        for c, q in zip(cases, p, strict=True)
    ]
    used, counts = 0, {"judge": 0, "challenger": 0, "verification": 0}
    stage_limit = int(budget_units * (1 - verification_reserve))
    unit_costs = {
        "judge": costs.judge,
        "challenger": costs.challenger,
        "verification": costs.verification,
    }

    def consume(kind: Literal["judge", "challenger", "verification"], limit: int) -> bool:
        nonlocal used
        cost = unit_costs[kind]
        if used + cost > limit:
            return False
        used += cost
        counts[kind] += 1
        return True

    for index in np.argsort(np.abs(p - 0.5), kind="stable"):
        i = int(index)
        if max(p[i], 1 - p[i]) >= escalation_confidence:
            continue
        c = cases[i]
        proposal = None
        if challenger is not None:
            if used + costs.challenger + costs.judge > stage_limit:
                continue
            consume("challenger", stage_limit)
            try:
                with model_attempt_limit(1):
                    proposal = challenger.propose(c.evidence, c.hypothesis)
                candidate = Hypothesis(
                    c.hypothesis.case_id + "-challenge",
                    c.evidence.world_id,
                    np.asarray(proposal["alternative_parameters"], dtype=np.float64),
                )
                traces[i]["events"].append({"type": "challenger", "proposal": proposal})
            except (InferenceError, ValueError, KeyError) as error:
                traces[i]["events"].append(
                    {"type": "challenger_failure", "error": type(error).__name__}
                )
                continue
        else:
            candidate = c.hypothesis
        if not consume("judge", stage_limit):
            continue
        try:
            with model_attempt_limit(1):
                result = served_judge.assess(c.evidence, candidate)
            q = result["confidence"] if result["supported"] else 1 - result["confidence"]
            traces[i]["events"].append(
                {"type": "served_judge", "response": result, "candidate": candidate.payload()}
            )
            if proposal is None or q >= 0.5:
                final_hypotheses[i] = candidate
                p[i], accepted[i] = q, q >= 0.5
        except InferenceError as error:
            traces[i]["events"].append({"type": "judge_failure", "error": type(error).__name__})
    for index in np.argsort(np.abs(p - 0.5), kind="stable"):
        i = int(index)
        if max(p[i], 1 - p[i]) >= escalation_confidence:
            continue
        if not consume("verification", budget_units):
            break
        accepted[i] = oracle(final_hypotheses[i], cases[i].verification, tolerance)
        traces[i]["events"].append({"type": "verification", "accepted": bool(accepted[i])})
    truths, errors = [], []
    for c, h, decision, trace in zip(cases, final_hypotheses, accepted, traces, strict=True):
        correct = oracle(h, c.verification, tolerance)
        error_value = hidden_error(h, c.verification)
        truths.append(correct)
        errors.append(error_value)
        trace.update(
            {
                "accepted": bool(decision),
                "final_hypothesis": h.payload(),
                "evaluation_only_correct_hypothesis": correct,
                "evaluation_only_hidden_rmse": error_value,
            }
        )
    traces = [record_trace(trace, str(trace["case_id"])) for trace in traces]
    truth = np.array(truths)
    false = int(np.sum(accepted & ~truth))
    return {
        "strategy": "challenger_assisted" if challenger else "direct_judging",
        "budget_limit_units": budget_units,
        "budget_used_units": used,
        "cost_units": asdict(costs),
        "external_calls": counts,
        "model_http_attempts_per_budgeted_call": 1,
        "case_count": len(cases),
        "hidden_borehole_predictive_accuracy": float(np.mean(truth)),
        "mean_hidden_boundary_rmse": float(np.mean(errors)),
        "audit_decision_accuracy": float(np.mean(accepted == truth)),
        "false_acceptances": false,
        "false_hypotheses": int(np.sum(~truth)),
        "false_acceptance_rate": false / int(np.sum(~truth)) if np.any(~truth) else None,
        "verification_fraction": counts["verification"] / len(cases),
        "evaluation_label_oracle_calls": len(cases),
    }, traces


def compare_at_equal_budget(
    cases: list[Case],
    cheap_judge: LogitJudge,
    temperature: float,
    judge_factory: Callable[[], Judge],
    challenger: Challenger,
    **kwargs: Unpack[AuditOptions],
) -> tuple[JSONRecord, dict[str, list[JSONRecord]]]:
    """Compare equal caps with separate served-client state for each strategy."""
    direct, direct_traces = audit_strategy(
        cases, cheap_judge, temperature, judge_factory(), **kwargs
    )
    assisted, assisted_traces = audit_strategy(
        cases, cheap_judge, temperature, judge_factory(), challenger=challenger, **kwargs
    )
    return {
        "budget_policy": "shared cheap judge with equal maximum weighted external-call budgets",
        "actual_spending_matched": direct["budget_used_units"] == assisted["budget_used_units"],
        "direct_judging": direct,
        "challenger_assisted": assisted,
    }, {"direct_judging": direct_traces, "challenger_assisted": assisted_traces}
