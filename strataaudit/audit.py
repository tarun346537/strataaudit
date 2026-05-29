"""Numerical evidence judge, temperature calibration, and budgeted verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from .constants import DEFAULT_BOUNDARY_TOLERANCE, DEFAULT_ECE_BINS, DEFAULT_JUDGE_STEPS
from .data import Case, Evidence, Hypothesis, Verification
from .logging_config import record_trace
from .types import BoolArray, FloatArray, JSONRecord, LogitJudge


def sigmoid(x: ArrayLike) -> FloatArray:
    """Map finite logits to probabilities without exponential overflow."""
    return np.asarray(1 / (1 + np.exp(-np.clip(x, -40, 40))), dtype=np.float64)


def hidden_error(h: Hypothesis, v: Verification) -> float:
    """Measure independent borehole error for a proposed model."""
    if h.world_id != v.world_id:
        raise ValueError("verification world mismatch")
    return float(np.sqrt(np.mean((h.predict(v.x) - v.boundaries) ** 2)))


def oracle(h: Hypothesis, v: Verification, tolerance: float = DEFAULT_BOUNDARY_TOLERANCE) -> bool:
    """Verify boundary order and error against withheld observations."""
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive")
    predictions = h.predict(v.x)
    return bool(np.all(predictions[:, 0] < predictions[:, 1]) and hidden_error(h, v) <= tolerance)


def features(e: Evidence, h: Hypothesis) -> FloatArray:
    """Accept visible evidence features without Case or Verification objects."""
    if e.world_id != h.world_id:
        raise ValueError("hypothesis/evidence world mismatch")
    residual = h.predict(e.x) - e.boundaries
    rmse = float(np.sqrt(np.mean(residual**2)))
    return np.array(
        [
            1.0,
            rmse,
            rmse**2,
            np.max(np.abs(residual)),
            np.std(e.x),
            np.max(np.abs(h.parameters[2:])),
        ]
    )


def challenger(e: Evidence, case_id: str) -> Hypothesis:
    """Fit planar boundary parameters to visible boreholes by least squares."""
    design = np.column_stack([np.ones(len(e.x)), e.x])
    if np.linalg.matrix_rank(design) < 2:
        raise ValueError("challenger needs at least two distinct visible x coordinates")
    fit = np.linalg.lstsq(design, e.boundaries, rcond=None)[0]
    return Hypothesis(case_id, e.world_id, np.r_[fit[0], fit[1]])


@dataclass
class EvidenceJudge:
    """Regularized numerical support model using only visible residuals."""

    mean: FloatArray
    scale: FloatArray
    weights: FloatArray

    @classmethod
    def fit(
        cls,
        cases: list[Case],
        tolerance: float = DEFAULT_BOUNDARY_TOLERANCE,
        steps: int = DEFAULT_JUDGE_STEPS,
    ) -> Self:
        """Fit normalization and logistic weights on training worlds only."""
        if not cases or any(c.split != "train" for c in cases):
            raise ValueError("judge fitting accepts training cases only")
        xx = np.stack([features(c.evidence, c.hypothesis) for c in cases])
        yy = np.array([oracle(c.hypothesis, c.verification, tolerance) for c in cases], dtype=float)
        if len(np.unique(yy)) != 2:
            raise ValueError("training labels need both classes")
        mean, scale = xx.mean(0), xx.std(0)
        mean[0], scale[0] = 0, 1
        scale[scale < 1e-6] = 1
        x = (xx - mean) / scale
        weights = np.zeros(x.shape[1])
        for _ in range(steps):
            regularizer = 0.01 * weights
            regularizer[0] = 0
            weights -= 0.08 * (x.T @ (sigmoid(x @ weights) - yy) / len(yy) + regularizer)
        return cls(mean, scale, weights)

    def logit(self, e: Evidence, h: Hypothesis) -> float:
        """Score visible consistency without exposing withheld boreholes."""
        return float(((features(e, h) - self.mean) / self.scale) @ self.weights)

    def to_dict(self) -> JSONRecord:
        """Serialize fitted numerical parameters for evaluation reports."""
        return {
            "kind": "numpy_evidence_logistic_judge",
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
        }


def fit_temperature(logits: ArrayLike, labels: ArrayLike) -> float:
    """Fit a positive scalar on a separate calibration set by grid search."""
    ll, yy = np.asarray(logits, float), np.asarray(labels, float)
    if ll.ndim != 1 or ll.shape != yy.shape or not len(ll) or not np.isfinite(ll).all():
        raise ValueError("calibration requires equal nonempty finite arrays")
    if not np.isin(yy, [0.0, 1.0]).all():
        raise ValueError("binary labels required")
    candidates = np.exp(np.linspace(-3, 3, 301))
    losses = [np.mean(np.logaddexp(0, ll / t) - yy * ll / t) for t in candidates]
    return float(candidates[int(np.argmin(losses))])


def ece(probabilities: ArrayLike, labels: ArrayLike, bins: int = DEFAULT_ECE_BINS) -> float:
    """Compute class-probability calibration error with explicit bin weighting."""
    p, y = np.asarray(probabilities, float), np.asarray(labels, float)
    if bins < 1 or p.shape != y.shape or p.ndim != 1 or not len(p):
        raise ValueError("ece requires equal nonempty 1D arrays and positive bins")
    if not np.isfinite(p).all() or not np.isin(y, [0.0, 1.0]).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid probabilities/labels")
    indices = np.minimum((p * bins).astype(int), bins - 1)
    return float(
        sum(
            np.mean(indices == b) * abs(p[indices == b].mean() - y[indices == b].mean())
            for b in range(bins)
            if np.any(indices == b)
        )
    )


def evaluate(
    cases: list[Case],
    judge: LogitJudge,
    temperature: float,
    budget_fraction: float = 0.25,
    tolerance: float = DEFAULT_BOUNDARY_TOLERANCE,
) -> tuple[JSONRecord, list[JSONRecord]]:
    """Evaluate frozen decisions and count online verification separately."""
    if not cases or any(c.split != "test" for c in cases):
        raise ValueError("evaluation accepts nonempty held-out test cases only")
    if not np.isfinite(temperature) or temperature <= 0 or not 0 <= budget_fraction <= 1:
        raise ValueError("temperature/budget out of bounds")
    logits = np.array([judge.logit(c.evidence, c.hypothesis) for c in cases])
    probabilities = sigmoid(logits / temperature)
    # The batch budget is selected exclusively from observable probabilities.
    chosen = np.argsort(np.abs(probabilities - 0.5), kind="stable")[
        : int(len(cases) * budget_fraction)
    ]
    selected = set(chosen.tolist())
    baseline = probabilities >= 0.5
    accepted = baseline.copy()
    traces, labels, errors, proposed_errors = [], [], [], []
    for i, (c, p) in enumerate(zip(cases, probabilities, strict=True)):
        verifier_verdict = (
            oracle(c.hypothesis, c.verification, tolerance) if i in selected else None
        )
        if verifier_verdict is not None:
            accepted[i] = verifier_verdict
        alternative = (
            challenger(c.evidence, c.hypothesis.case_id + "-challenge")
            if i in selected and not accepted[i]
            else None
        )
        # Evaluator computes ground truth for metrics AFTER decisions. These calls
        # are separate from the algorithm's counted online verification budget.
        truth = oracle(c.hypothesis, c.verification, tolerance)
        err = hidden_error(c.hypothesis, c.verification)
        alt_err = hidden_error(alternative, c.verification) if alternative else None
        labels.append(truth)
        errors.append(err)
        if alt_err is not None:
            proposed_errors.append(alt_err)
        traces.append(
            {
                "case_id": c.hypothesis.case_id,
                "world_id": c.evidence.world_id,
                "split": "test",
                "probability_supported": float(p),
                "routed": i in selected,
                "accepted": bool(accepted[i]),
                "verifier_verdict": verifier_verdict,
                "online_verifier_calls": int(i in selected),
                "challenger_proposal": alternative.payload() if alternative else None,
                "challenger_automatically_accepted": False,
                "evaluation_only_truth": truth,
                "evaluation_only_hidden_rmse": err,
                "evaluation_only_challenger_rmse": alt_err,
            }
        )
    y = np.array(labels, dtype=bool)

    def rates(decisions: BoolArray) -> JSONRecord:
        false_count = int(np.sum(decisions & ~y))
        return {
            "accepted": int(np.sum(decisions)),
            "false_acceptances": false_count,
            "false_hypotheses": int(np.sum(~y)),
            "false_acceptance_rate": false_count / int(np.sum(~y)) if np.any(~y) else None,
            "false_discovery_rate": false_count / int(np.sum(decisions))
            if np.any(decisions)
            else None,
            "accuracy": float(np.mean(decisions == y)),
        }

    traces = [record_trace(trace, str(trace["case_id"])) for trace in traces]
    return {
        "cases": len(cases),
        "worlds": len({c.evidence.world_id for c in cases}),
        "temperature": temperature,
        "ece_uncalibrated": ece(sigmoid(logits), y),
        "ece_calibrated": ece(probabilities, y),
        "direct_judge": rates(baseline),
        "budgeted_audit": rates(accepted),
        "online_verifier_calls": len(selected),
        "online_verification_fraction": len(selected) / len(cases),
        "evaluation_label_oracle_calls": len(cases),
        "mean_original_hidden_rmse": float(np.mean(errors)),
        "challenger_proposals": len(proposed_errors),
        "mean_proposed_challenger_rmse": float(np.mean(proposed_errors))
        if proposed_errors
        else None,
    }, traces
