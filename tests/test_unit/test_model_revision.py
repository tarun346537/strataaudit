"""Require a consistent tokenizer and model repository revision for training."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from strataaudit.train_challenger import resolve_model_revision


def test_local_smoke_model_does_not_require_a_remote_revision(tmp_path: Path) -> None:
    assert resolve_model_revision(str(tmp_path), None) is None


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None, reason="training extra optional"
)
def test_remote_commit_is_frozen_from_a_resolved_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    calls: list[tuple[str, str | None]] = []

    def model_info(repo_id: str, *, revision: str | None) -> SimpleNamespace:
        calls.append((repo_id, revision))
        return SimpleNamespace(sha="a" * 40)

    monkeypatch.setattr(huggingface_hub, "model_info", model_info)
    assert resolve_model_revision("controlled/model", "main") == "a" * 40
    assert calls == [("controlled/model", "main")]


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None, reason="training extra optional"
)
@pytest.mark.parametrize("unresolved", [None, "main"])
def test_unresolved_remote_branch_cannot_be_saved_as_a_pinned_run(
    monkeypatch: pytest.MonkeyPatch, unresolved: str | None
) -> None:
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "model_info", lambda *args, **kwargs: SimpleNamespace(sha=unresolved)
    )
    with pytest.raises(ValueError, match="can't pin"):
        resolve_model_revision("controlled/model", unresolved)
