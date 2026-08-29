"""Utilities for converting experiment outputs into figure data."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def load_json(path):
    path = Path(path)

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_csv(path):

    path = Path(path)

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:

        return list(
            csv.DictReader(handle)
        )


def save_csv(
    rows,
    path,
    fields=None,
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    if fields is None:
        fields = list(
            rows[0].keys()
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)
