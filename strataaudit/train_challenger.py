"""TRL DPO with LoRA adapters for challenger preference training."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from .artifacts import adapter_artifact_hashes
from .config import Settings
from .logging_config import setup_logging
from .types import JSONRecord


def resolve_model_revision(model: str, revision: str | None) -> str | None:
    """Pin remote model loads to a commit and retain local revision settings."""
    if Path(model).is_dir():
        return revision
    from huggingface_hub import model_info

    commit = model_info(model, revision=revision).sha
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError(
            "can't pin the base model. pass --revision with a commit hash or check Hub access"
        )
    return commit


def load_preferences(path: str | Path) -> tuple[dict[str, list[JSONRecord]], dict[str, str]]:
    """Validate preference conversations and retain group assignments for audit checks."""
    records = [
        json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()
    ]
    splits: dict[str, list[JSONRecord]] = {"train": [], "calibration": []}
    worlds: dict[str, str] = {}
    for record in records:
        split = record["split"]
        if split not in splits:
            raise ValueError("preferences must exclude test worlds")
        world = record["world_id"]
        if world in worlds and worlds[world] != split:
            raise ValueError("world leakage across preference splits")
        worlds[world] = split
        for key in ("prompt", "chosen", "rejected"):
            messages = record[key]
            if (
                not isinstance(messages, list)
                or not messages
                or not all(
                    isinstance(m, dict)
                    and m.get("role") in {"system", "user", "assistant"}
                    and isinstance(m.get("content"), str)
                    for m in messages
                )
            ):
                raise ValueError(f"invalid conversational {key}")
        if record["chosen"] == record["rejected"]:
            raise ValueError("identical preference completions")
        splits[split].append({k: record[k] for k in ("prompt", "chosen", "rejected")})
    if not all(splits.values()):
        raise ValueError("nonempty training and calibration preference splits required")
    return splits, worlds


def main() -> None:
    """Train or resume a LoRA challenger with early stopping and JSON metrics."""
    settings = Settings.load()
    logger = setup_logging(settings.log_level)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", required=True)
    p.add_argument(
        "--model", required=True, help="Hugging Face model ID or local causal LM directory"
    )
    p.add_argument("--revision", help="base-model commit or revision resolved once for training")
    p.add_argument("--output", default=str(Path(settings.output_dir, "challenger-lora")))
    p.add_argument("--steps", type=int, default=settings.train_steps)
    p.add_argument("--rank", type=int, default=settings.lora_rank)
    p.add_argument("--max-length", type=int, default=settings.max_length)
    p.add_argument("--seed", type=int, default=settings.seed)
    p.add_argument("--resume-from-checkpoint")
    p.add_argument("--early-stopping-patience", type=int, default=settings.early_stopping_patience)
    p.add_argument("--tracking", choices=["json", "none"], default=settings.tracking)
    p.add_argument("--cpu", action="store_true", help="select CPU training")
    a = p.parse_args()
    if a.steps < 1 or a.rank < 1 or a.max_length < 64 or a.early_stopping_patience < 1:
        p.error("positive steps/rank and max-length >=64 required")
    splits, worlds = load_preferences(a.pairs)
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoTokenizer, EarlyStoppingCallback, TrainerCallback
        from trl.trainer.dpo_config import DPOConfig
        from trl.trainer.dpo_trainer import DPOTrainer
    except ImportError as error:
        raise SystemExit(
            "install training extras: python -m pip install -e '.[training]'"
        ) from error
    revision = resolve_model_revision(a.model, a.revision)
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=revision)
    if not tokenizer.chat_template:
        raise ValueError("choose a model/tokenizer with an explicit chat template")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(Path(a.pairs).read_bytes()).hexdigest()
    manifest_path = output / "training_manifest.json"
    parameters = {
        "rank": a.rank,
        "max_length": a.max_length,
        "seed": a.seed,
        "learning_rate": settings.learning_rate,
        "gradient_accumulation_steps": settings.gradient_accumulation_steps,
        "dpo_beta": settings.dpo_beta,
        "lora_dropout": settings.lora_dropout,
        "base_model_revision": revision,
    }
    if a.resume_from_checkpoint:
        if not manifest_path.is_file():
            p.error("resume requires the output directory's saved training_manifest.json")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("pairs_sha256") != digest or previous.get("base_model") != a.model:
            p.error("resume dataset/model differs from the saved training manifest")
        if previous.get("training_parameters") != parameters:
            p.error("resume training parameters differ from the saved training manifest")
        if not Path(a.resume_from_checkpoint, "trainer_state.json").is_file():
            p.error(
                "resume checkpoint needs trainer_state.json. select a saved checkpoint-* directory"
            )
    manifest: JSONRecord = {
        "kind": "local_dpo_run",
        "base_model": a.model,
        "base_model_revision": revision,
        "seed": a.seed,
        "steps": a.steps,
        "rank": a.rank,
        "training_parameters": parameters,
        "worlds": worlds,
        "pairs_sha256": digest,
        "pairs": {k: len(v) for k, v in splits.items()},
        "status": "running",
        "resume_from_checkpoint": a.resume_from_checkpoint,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    config = DPOConfig(
        output_dir=a.output,
        model_init_kwargs={"revision": revision},
        max_steps=a.steps,
        learning_rate=settings.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        per_device_eval_batch_size=1,
        eval_strategy="steps",
        eval_steps=max(1, a.steps // 5),
        save_steps=max(1, a.steps // 5),
        save_total_limit=2,
        logging_steps=1,
        max_length=a.max_length,
        beta=settings.dpo_beta,
        seed=a.seed,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        use_cpu=a.cpu,
        bf16=not a.cpu and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        optim="adamw_torch",
    )
    callbacks: list[TrainerCallback] = [
        EarlyStoppingCallback(early_stopping_patience=a.early_stopping_patience)
    ]
    if a.tracking == "json":
        from .tracking import JSONMetricsCallback

        callbacks.append(JSONMetricsCallback(output / "metrics.jsonl", digest[:16]))
    trainer = DPOTrainer(
        model=a.model,
        args=config,
        processing_class=tokenizer,
        callbacks=callbacks,
        train_dataset=Dataset.from_list(splits["train"]),
        eval_dataset=Dataset.from_list(splits["calibration"]),
        peft_config=LoraConfig(
            r=a.rank,
            lora_alpha=2 * a.rank,
            lora_dropout=settings.lora_dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            revision=revision,
        ),
    )
    trainer.train(resume_from_checkpoint=a.resume_from_checkpoint)
    trainer.save_model(a.output)
    tokenizer.save_pretrained(a.output)
    manifest["status"] = "complete"
    manifest["completed_optimizer_steps"] = trainer.state.global_step
    manifest["adapter_artifacts_sha256"] = adapter_artifact_hashes(output)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "challenger_training_complete",
        extra={
            "fields": {
                "output": str(output),
                "optimizer_steps": trainer.state.global_step,
                "tracking": a.tracking,
            }
        },
    )


if __name__ == "__main__":
    main()
