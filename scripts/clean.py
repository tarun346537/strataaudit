"""Remove local tool caches contained by the repository."""

import shutil
import sys
from pathlib import Path


def main() -> None:
    """Clean known disposable paths after checking their resolved locations."""
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from strataaudit.logging_config import setup_logging

    removed: list[str] = []
    for name in (".pytest_cache", ".ruff_cache", ".mypy_cache", ".coverage", "htmlcov"):
        path = root / name
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or resolved.parent != root:
            raise ValueError(f"cache path leaves the project directory: {name}")
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(name)
        elif path.exists():
            path.unlink()
            removed.append(name)
    setup_logging().info("tool_caches_removed", extra={"fields": {"paths": removed}})


if __name__ == "__main__":
    main()
