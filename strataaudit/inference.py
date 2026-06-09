"""Run a locally trained PEFT challenger adapter and validate every structured output."""

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from .artifacts import validate_adapter_holdouts
from .config import Settings
from .data import Evidence, Hypothesis, load_csv_cases
from .llm import challenge_messages, parse_challenge
from .logging_config import setup_logging
from .types import ChallengeResult

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedTokenizerBase


class LocalChallenger:
    """Load a trained PEFT adapter for evidence-constrained proposals."""

    def __init__(self, adapter_path: str, device: str = "cpu") -> None:
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(adapter_path)
        self.model: PeftModel = AutoPeftModelForCausalLM.from_pretrained(adapter_path).to(device)
        self.model.eval()

    def propose(
        self, evidence: Evidence, hypothesis: Hypothesis, max_new_tokens: int = 512
    ) -> ChallengeResult:
        """Decode an alternative and reject output that violates the evidence schema."""
        import torch

        text = self.tokenizer.apply_chat_template(
            challenge_messages(evidence, hypothesis), tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            result = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        answer = self.tokenizer.decode(
            result[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        # keep schema validation at the model boundary
        if not isinstance(answer, str):
            raise ValueError(
                "challenger returned multiple decoded outputs. use one hypothesis per call"
            )
        return parse_challenge(json.loads(answer), evidence)


def main() -> None:
    """Write adapter proposals only after validating artifact/data holdouts."""
    config = Settings.load()
    logger = setup_logging(config.log_level)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", required=True)
    p.add_argument("--boreholes", required=True)
    p.add_argument("--hypotheses", required=True)
    p.add_argument("--device", default=config.device)
    p.add_argument("--output", default=str(Path(config.output_dir, "challenges.jsonl")))
    a = p.parse_args()
    cases = load_csv_cases(a.boreholes, a.hypotheses)
    validate_adapter_holdouts(a.adapter, cases)
    model = LocalChallenger(a.adapter, a.device)
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for c in cases:
            if c.split == "test":
                f.write(
                    json.dumps(
                        {
                            "kind": "new_adapter_inference",
                            "case_id": c.hypothesis.case_id,
                            "challenge": model.propose(c.evidence, c.hypothesis),
                        }
                    )
                    + "\n"
                )
    logger.info("challenger_inference_complete", extra={"fields": {"output": str(output)}})


if __name__ == "__main__":
    main()
