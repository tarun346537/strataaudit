"""CPU DPO checkpoint/resume check on a tiny initialized model.

Exercise CPU DPO training, checkpoint resume and PEFT adapter inference.
Run from the repository root with training extras installed.
"""

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Run the CPU DPO checkpoint/resume and adapter-loading workflow."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from strataaudit.config import Settings

    output = Path(Settings.load().output_dir) / "dpo-smoke"
    model_dir = output / "tiny-base"
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab = {
        word: i
        for i, word in enumerate(
            [
                "[PAD]",
                "[UNK]",
                "[BOS]",
                "[EOS]",
                "system",
                "user",
                "assistant",
                ":",
                "hypothesis",
                "consistent",
                "inconsistent",
                "with",
                "visible",
                "evidence",
                ".",
                "audit",
            ]
        )
    }
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}"
        "{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    )
    tokenizer.save_pretrained(model_dir)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocab),
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=256,
            bos_token_id=2,
            eos_token_id=3,
            pad_token_id=0,
        )
    )
    model.save_pretrained(model_dir)
    pairs = output / "pairs.jsonl"
    with pairs.open("w", encoding="utf-8") as f:
        for i in range(6):
            f.write(
                json.dumps(
                    {
                        "world_id": f"smoke-world-{i}",
                        "split": "train" if i < 4 else "calibration",
                        "prompt": [
                            {"role": "user", "content": "audit hypothesis with visible evidence"}
                        ],
                        "chosen": [
                            {"role": "assistant", "content": "consistent with visible evidence ."}
                        ],
                        "rejected": [{"role": "assistant", "content": "inconsistent ."}],
                    }
                )
                + "\n"
            )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "strataaudit.train_challenger",
            "--pairs",
            str(pairs),
            "--model",
            str(model_dir),
            "--output",
            str(output / "adapter"),
            "--steps",
            "1",
            "--rank",
            "2",
            "--max-length",
            "128",
            "--cpu",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "strataaudit.train_challenger",
            "--pairs",
            str(pairs),
            "--model",
            str(model_dir),
            "--output",
            str(output / "adapter"),
            "--steps",
            "2",
            "--rank",
            "2",
            "--max-length",
            "128",
            "--cpu",
            "--resume-from-checkpoint",
            str(output / "adapter" / "checkpoint-1"),
        ],
        check=True,
    )
    if not (output / "adapter" / "adapter_model.safetensors").is_file():
        raise RuntimeError("training did not export adapter weights")
    # Exercise the actual inference loader against the newly saved adapter.
    from strataaudit.artifacts import adapter_artifact_hashes
    from strataaudit.inference import LocalChallenger

    loaded = LocalChallenger(str(output / "adapter"))
    if not loaded.model.peft_config:
        raise RuntimeError("inference did not load the PEFT adapter")
    manifest = json.loads(
        (output / "adapter" / "training_manifest.json").read_text(encoding="utf-8")
    )
    if manifest["adapter_artifacts_sha256"] != adapter_artifact_hashes(output / "adapter"):
        raise RuntimeError("exported adapter/tokenizer files differ from the training record")
    (output / "validation.json").write_text(
        json.dumps(
            {
                "kind": "tiny_cpu_training_smoke",
                "optimizer_steps": 2,
                "checkpoint_resume_checked": True,
                "model": "randomly initialized local GPT2 (16 dimensions, 1 layer)",
                "adapter_exported": True,
                "adapter_reloaded_for_inference": True,
                "adapter_and_tokenizer_hashes_checked": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    from strataaudit.logging_config import setup_logging

    setup_logging().info(
        "tiny_dpo_smoke_passed",
        extra={
            "fields": {
                "optimizer_steps": 2,
                "adapter_exported": True,
                "adapter_reloaded": True,
                "checkpoint_resume_checked": True,
            }
        },
    )


if __name__ == "__main__":
    main()
