"""Persistent JSON metrics for TRL jobs."""

from __future__ import annotations

import json
from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from .logging_config import record_trace
from .types import JSONRecord


class JSONMetricsCallback(TrainerCallback):
    """Record optimizer and evaluation metrics as local JSONL."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path, self.run_id = path, run_id
        path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: JSONRecord | None = None,
        **kwargs: object,
    ) -> TrainerControl:
        """Append one durable metric record for the current trainer step."""
        record = record_trace(
            {"event": "training_metrics", "step": state.global_step, "metrics": logs or {}},
            self.run_id,
        )
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
        return control
