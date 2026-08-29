"""Top-level reproducibility launcher."""

from __future__ import annotations

import subprocess
import sys


def run(command):

    print()
    print("=" * 72)
    print(
        "RUNNING:",
        " ".join(command),
    )
    print("=" * 72)

    result = subprocess.run(
        command,
        check=False,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Command failed with "
            f"return code {result.returncode}"
        )


def main():

    python = sys.executable

    run(
        [
            python,
            "scripts/data/prepare_datasets.py",
        ]
    )

    run(
        [
            python,
            "scripts/experiments/run_ablations.py",
        ]
    )

    print()
    print(
        "Dataset preparation and experiment "
        "configuration completed."
    )

    print()
    print(
        "The full training matrix is intentionally "
        "launched separately because it may require "
        "substantial compute and dataset download time."
    )


if __name__ == "__main__":
    main()
