"""One row per participant, recording, epoch, model and run.

The file this writes is the only thing downstream of a model. Metrics are
computed from it, tables are generated from it, and a claim that cannot be
recomputed from it is not made. It is CSV rather than a binary format because
the point is that someone can open it.

Every row carries all five probabilities, not just the predicted class. Without
them the confusion between N1 and REM cannot be examined afterwards, and no
recalibration or threshold study is possible without re-running the model.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sleepstatelab.labels import STAGES

PREDICTION_COLUMNS: tuple[str, ...] = (
    "run_id",
    "model",
    "split_id",
    "split_part",
    "seed",
    "participant_id",
    "recording_id",
    "epoch_index",
    "true_label",
    "pred_label",
    *(f"p_{name}" for name in STAGES),
    "qc_flags",
)


@dataclass(frozen=True, slots=True)
class PredictionRow:
    """One epoch's prediction, as it is written and read back."""

    run_id: str
    model: str
    split_id: str
    split_part: str
    seed: int
    participant_id: str
    recording_id: str
    epoch_index: int
    true_label: str
    pred_label: str
    probabilities: tuple[float, ...]
    qc_flags: int


class PredictionWriter:
    """Append rows for one model and run to a CSV, header first."""

    def __init__(self, path: Path | str, *, overwrite: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and not overwrite
        # The handle is deliberately held open across many write() calls and
        # closed by close() or the context manager; a `with` here would close
        # the file after the first block of predictions.
        self._handle = open(self.path, "a" if exists else "w", newline="")  # noqa: SIM115
        self._writer = csv.writer(self._handle)
        if not exists:
            self._writer.writerow(PREDICTION_COLUMNS)
        self.n_rows = 0

    def write(
        self,
        *,
        run_id: str,
        model: str,
        split_id: str,
        split_part: str,
        seed: int,
        participant_ids: Iterable[str],
        recording_ids: Iterable[str],
        epoch_indices: np.ndarray,
        true_labels: np.ndarray,
        probabilities: np.ndarray,
        qc_flags: np.ndarray | None = None,
    ) -> int:
        """Write one block of predictions.

        ``probabilities`` is ``[n, 5]`` in the order of ``labels.STAGES`` and is
        checked for it: a column order silently swapped between a model and its
        report is a mistake that produces plausible, wrong tables.
        """
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != len(STAGES):
            raise ValueError(
                f"probabilities must be [n, {len(STAGES)}] in the order {STAGES}, "
                f"got {probabilities.shape}"
            )
        epoch_indices = np.asarray(epoch_indices)
        true_labels = np.asarray(true_labels)
        participants = list(participant_ids)
        recordings = list(recording_ids)
        n = probabilities.shape[0]
        if not (
            len(participants) == len(recordings) == epoch_indices.size == true_labels.size == n
        ):
            raise ValueError("prediction block has mismatched lengths")
        flags = (
            np.zeros(n, dtype=int) if qc_flags is None else np.asarray(qc_flags, dtype=int)
        )
        predicted = probabilities.argmax(axis=1)

        for row in range(n):
            truth = true_labels[row]
            truth_name = STAGES[int(truth)] if not isinstance(truth, str) else truth
            self._writer.writerow(
                [
                    run_id,
                    model,
                    split_id,
                    split_part,
                    seed,
                    participants[row],
                    recordings[row],
                    int(epoch_indices[row]),
                    truth_name,
                    STAGES[int(predicted[row])],
                    *(f"{value:.6f}" for value in probabilities[row]),
                    int(flags[row]),
                ]
            )
        self.n_rows += n
        return n

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> PredictionWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def read_predictions(path: Path | str) -> list[dict[str, Any]]:
    """Read a prediction file back, with numbers as numbers."""
    rows: list[dict[str, Any]] = []
    with open(path, newline="") as handle:
        for raw in csv.DictReader(handle):
            missing = [c for c in PREDICTION_COLUMNS if c not in raw]
            if missing:
                raise ValueError(f"{path} is missing prediction column(s) {missing}")
            row = dict(raw)
            row["seed"] = int(raw["seed"])
            row["epoch_index"] = int(raw["epoch_index"])
            row["qc_flags"] = int(raw["qc_flags"])
            for name in STAGES:
                row[f"p_{name}"] = float(raw[f"p_{name}"])
            rows.append(row)
    return rows


def iter_prediction_files(directory: Path | str) -> Iterator[Path]:
    """Every prediction CSV under a directory, in a stable order."""
    return iter(sorted(Path(directory).rglob("predictions*.csv")))
