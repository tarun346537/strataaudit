"""Create small geological input files for local checks."""

import sys
from pathlib import Path


def main() -> None:
    """Run the same sample-input exporter used by the audit CLI."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from strataaudit.fixtures import main as export_main

    export_main()


if __name__ == "__main__":
    main()
