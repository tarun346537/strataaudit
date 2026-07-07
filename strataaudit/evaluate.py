"""Calibrate and evaluate normalized borehole hypotheses with a local/served judge."""

import argparse
import json
import math
from pathlib import Path

from .artifacts import dataset_provenance
from .audit import EvidenceJudge, evaluate, fit_temperature, oracle
from .config import Settings
from .constants import PROBABILITY_EPSILON
from .data import Evidence, Hypothesis, load_csv_cases
from .llm import ChatClient, ServedJudge
from .logging_config import setup_logging
from .types import JSONRecord, JudgeResult, LogitJudge


class CachedJudge:
    """Reuse served assessments during calibration and held-out evaluation."""

    def __init__(self, responses: dict[str, JudgeResult]) -> None:
        self.responses = responses

    def logit(self, e: Evidence, h: Hypothesis) -> float:
        """Convert an already collected model response into a finite logit."""
        response = self.responses[h.case_id]
        q = response["confidence"] if response["supported"] else 1 - response["confidence"]
        q = min(1 - PROBABILITY_EPSILON, max(PROBABILITY_EPSILON, q))
        return math.log(q / (1 - q))


def main() -> None:
    """Evaluate numerical or served judgments against independent boreholes."""
    config = Settings.load()
    logger = setup_logging(config.log_level)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--boreholes", required=True)
    p.add_argument("--hypotheses", required=True)
    p.add_argument("--judge-mode", choices=["numpy", "vllm"], default="numpy")
    p.add_argument("--judge-url", default=config.judge_url)
    p.add_argument("--judge-model", default=config.judge_model)
    p.add_argument("--tolerance", type=float, default=config.tolerance)
    p.add_argument("--calibration-backend", choices=["numpy", "torch"], default="numpy")
    p.add_argument("--budget", type=float, default=config.verification_fraction)
    p.add_argument("--output", default=str(Path(config.output_dir, "evaluation.json")))
    a = p.parse_args()
    cases = load_csv_cases(a.boreholes, a.hypotheses)
    provenance = dataset_provenance(a.boreholes, a.hypotheses, cases)
    train, calibration, test = (
        [c for c in cases if c.split == s] for s in ["train", "calibration", "test"]
    )
    model_info: JSONRecord = {}
    responses: dict[str, JudgeResult] = {}
    judge: LogitJudge
    if a.judge_mode == "numpy":
        judge = EvidenceJudge.fit(train, tolerance=a.tolerance)
        model_info = judge.to_dict()
    else:
        if not a.judge_model:
            p.error("--judge-model is required for vllm")
        served = ServedJudge(ChatClient(a.judge_url, a.judge_model))
        for c in calibration + test:
            responses[c.hypothesis.case_id] = served.assess(c.evidence, c.hypothesis)
        judge = CachedJudge(responses)
        model_info = {"kind": "served_llm_judge", "model": a.judge_model}
    calibration_logits = [judge.logit(c.evidence, c.hypothesis) for c in calibration]
    calibration_labels = [oracle(c.hypothesis, c.verification, a.tolerance) for c in calibration]
    if a.calibration_backend == "torch":
        from .torch_calibration import fit_temperature_torch

        temperature = fit_temperature_torch(
            calibration_logits,
            calibration_labels,
            calibration_groups=[c.evidence.world_id for c in calibration],
            training_groups=[c.evidence.world_id for c in train],
            test_groups=[c.evidence.world_id for c in test],
        ).temperature
    else:
        temperature = fit_temperature(calibration_logits, calibration_labels)
    metrics, traces = evaluate(test, judge, temperature, a.budget, a.tolerance)
    for c, trace in zip(test, traces, strict=True):
        trace["visible_evidence"] = c.evidence.prompt_payload()
        trace["hypothesis"] = c.hypothesis.payload()
        if responses:
            trace["judge_response"] = responses[c.hypothesis.case_id]
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "provenance": {
            "kind": "dataset_evaluation",
            **provenance,
        },
        "judge": model_info,
        "calibration_backend": a.calibration_backend,
        "tolerance": a.tolerance,
        "metrics": metrics,
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output.with_suffix(".traces.jsonl").write_text(
        "".join(json.dumps(t, allow_nan=False) + "\n" for t in traces), encoding="utf-8"
    )
    logger.info("heldout_evaluation_complete", extra={"fields": report})


if __name__ == "__main__":
    main()
