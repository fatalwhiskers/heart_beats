"""
dataset_loader.py
-----------------
Loads per-video crop settings from CSV manifests.

Two CSV formats are supported:
  - Dataset 1: filename, file_CSV, x1, y1, x2, y2
  - Dataset 2: subject, video_path, file_CSV, x1, y1, x2, y2

Typical usage
-------------
>>> loader = DatasetLoader()
>>> rows = loader.load_dataset1("data/dataset1.csv")
>>> rows = loader.load_dataset2("data/dataset2.csv")
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Row dataclasses — typed alternatives to bare tuples
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Dataset1Row:
    filename : str
    file_csv : str
    x1       : int
    y1       : int
    x2       : int
    y2       : int


@dataclass(frozen=True)
class Dataset2Row:
    subject    : str
    video_path : str
    file_csv   : str
    x1         : int
    y1         : int
    x2         : int
    y2         : int


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class DatasetLoader:
    """
    Parses CSV manifests that map video files to their manual crop boxes.

    Malformed rows are skipped with a warning rather than raising.
    """

    def load_dataset1(self, csv_path: str | Path) -> list[Dataset1Row]:
        """
        Load a Dataset-1 manifest.

        Expected columns: ``filename``, ``file_CSV``, ``x1``, ``y1``,
        ``x2``, ``y2``.

        Parameters
        ----------
        csv_path : str | Path

        Returns
        -------
        list[Dataset1Row]
        """
        rows: list[Dataset1Row] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=2):   # row 1 = header
                try:
                    rows.append(Dataset1Row(
                        filename = row["filename"].strip(),
                        file_csv = row["file_CSV"].strip(),
                        x1       = int(row["x1"]),
                        y1       = int(row["y1"]),
                        x2       = int(row["x2"]),
                        y2       = int(row["y2"]),
                    ))
                except (KeyError, ValueError) as exc:
                    print(f"[DatasetLoader] Skipping row {i} in {csv_path}: {exc}")
        return rows

    def load_dataset2(self, csv_path: str | Path) -> list[Dataset2Row]:
        """
        Load a Dataset-2 manifest.

        Expected columns: ``subject``, ``video_path``, ``file_CSV``,
        ``x1``, ``y1``, ``x2``, ``y2``.

        Parameters
        ----------
        csv_path : str | Path

        Returns
        -------
        list[Dataset2Row]
        """
        rows: list[Dataset2Row] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=2):
                try:
                    rows.append(Dataset2Row(
                        subject    = row["subject"].strip(),
                        video_path = row["video_path"].strip(),
                        file_csv   = row["file_CSV"].strip(),
                        x1         = int(row["x1"]),
                        y1         = int(row["y1"]),
                        x2         = int(row["x2"]),
                        y2         = int(row["y2"]),
                    ))
                except (KeyError, ValueError) as exc:
                    print(f"[DatasetLoader] Skipping row {i} in {csv_path}: {exc}")
        return rows
