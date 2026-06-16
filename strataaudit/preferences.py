"""Generate evidence-constrained RLAIF pairs using a separate model judge."""

import argparse
import json
from pathlib import Path

from .config import Settings
from .constants import DEFAULT_SEED
from .data import Case, load_csv_cases
from .llm import (
    ChatClient,
    InferenceError,
    ServedChallenger,
    challenge_messages,
    compare_challenges,
)
from .logging_config import setup_logging
from .types import JSONRecord


def generate(
    cases: list[Case],
    challenger: ServedChallenger,
    judge: ChatClient,
    output: Path,
    seed: int = DEFAULT_SEED,
) -> JSONRecord:
    """Build complete preference files while withholding all test worlds."""
    if challenger.client.model == judge.model:
        raise ValueError("use a distinct judge model ID to avoid self-preference generation")
    output.parent.mkdir(parents=True, exist_ok=True)
    count, errors, ties = 0, 0, 0
    # Atomic completion prevents a partially written training set masquerading as complete.
    temporary = output.with_suffix(output.suffix + ".partial")
    failures = output.with_suffix(".errors.jsonl")
    with (
        temporary.open("w", encoding="utf-8") as f,
        failures.open("w", encoding="utf-8") as error_file,
    ):
        for i, c in enumerate(cases):
            if c.split == "test":
                continue  # Test worlds never generate training preferences.
            try:
                a = challenger.propose(c.evidence, c.hypothesis, seed + 2 * i)
                b = challenger.propose(c.evidence, c.hypothesis, seed + 2 * i + 1)
                preference = compare_challenges(judge, c.evidence, c.hypothesis, a, b)
                if preference["winner"] == "tie" or a == b:
                    ties += 1
                    continue
                chosen, rejected = (a, b) if preference["winner"] == "A" else (b, a)
                f.write(
                    json.dumps(
                        {
                            "prompt": challenge_messages(c.evidence, c.hypothesis),
                            "chosen": [{"role": "assistant", "content": json.dumps(chosen)}],
                            "rejected": [{"role": "assistant", "content": json.dumps(rejected)}],
                            "case_id": c.hypothesis.case_id,
                            "world_id": c.evidence.world_id,
                            "split": c.split,
                            "preference_rationale": preference["rationale"],
                            "provenance": {
                                "kind": "rlaif_generation",
                                "seed": seed,
                                "challenger_model": challenger.client.model,
                                "judge_model": judge.model,
                            },
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                count += 1
            except InferenceError as error:
                errors += 1
                error_file.write(
                    json.dumps({"case_id": c.hypothesis.case_id, "error": str(error)}) + "\n"
                )
    if not count:
        raise RuntimeError(
            f"no usable preferences generated ({errors} errors, {ties} ties). see {failures}"
        )
    temporary.replace(output)
    summary = {
        "pairs": count,
        "errors": errors,
        "ties": ties,
        "output": str(output),
        "test_worlds_excluded": True,
        "kind": "new_rlaif_generation",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    """Collect evidence-only challenger preferences from separate served models."""
    config = Settings.load()
    logger = setup_logging(config.log_level)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--boreholes", required=True)
    p.add_argument("--hypotheses", required=True)
    p.add_argument("--challenger-url", default=config.challenger_url)
    p.add_argument("--challenger-model", default=config.challenger_model)
    p.add_argument("--judge-url", default=config.judge_url)
    p.add_argument("--judge-model", default=config.judge_model)
    p.add_argument("--output", default=str(Path(config.output_dir, "preferences.jsonl")))
    p.add_argument("--seed", type=int, default=config.seed)
    a = p.parse_args()
    if not a.challenger_model or not a.judge_model:
        p.error("set challenger/judge model IDs with CLI flags or STRATA_*_MODEL")
    summary = generate(
        load_csv_cases(a.boreholes, a.hypotheses),
        ServedChallenger(ChatClient(a.challenger_url, a.challenger_model)),
        ChatClient(a.judge_url, a.judge_model),
        Path(a.output),
        a.seed,
    )
    logger.info("preference_collection_complete", extra={"fields": summary})


if __name__ == "__main__":
    main()
