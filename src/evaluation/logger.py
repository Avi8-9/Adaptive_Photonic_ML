"""Structured experiment logging."""

from __future__ import annotations

import csv
import json
from pathlib import Path


class ExperimentLogger:
    """Persist epoch-level experiment records."""

    def __init__(
        self,
        directory: str | Path,
    ):
        self.directory = Path(
            directory
        )

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.records = []

    def log(
        self,
        **kwargs,
    ):
        self.records.append(
            dict(kwargs)
        )

    def save_json(
        self,
        filename="records.json",
    ):
        path = (
            self.directory
            / filename
        )

        path.write_text(
            json.dumps(
                self.records,
                indent=2,
            ),
            encoding="utf-8",
        )

    def save_csv(
        self,
        filename="records.csv",
    ):
        if not self.records:
            return

        path = (
            self.directory
            / filename
        )

        keys = sorted(
            {
                key
                for record in self.records
                for key in record
            }
        )

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=keys,
            )

            writer.writeheader()

            for record in self.records:
                writer.writerow(record)
