"""Scoring, with the two decisions that make sleep-staging numbers comparable.

**Participant-averaged, not pooled.** The primary metric is macro-F1 computed
*within* each participant and then averaged equally over people. A pooled
macro-F1 over all epochs is dominated by whoever contributed the most epochs,
and on Sleep Cassette that is whoever slept longest with the recorder running.
Both are reported, and they are labelled, because they are different quantities
and the literature is not consistent about which it means.

**Absent classes are handled explicitly, and disclosed.** Within one
participant, a stage can be absent from the truth (someone who reached no N3),
absent from the predictions, or both. The rule here:

* a class with no true epochs *and* no predicted epochs contributes nothing to
  that participant's macro average -- there was no question to get right;
* a class with no true epochs but some predicted epochs scores F1 = 0 and does
  count -- predicting a stage that never happened is an error;
* a class with true epochs and no predictions scores F1 = 0 and counts.

Every table records how many participants each stage was averaged over, so a
per-stage figure resting on three people cannot be read as though it rested on
twenty.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sleepstatelab.labels import STAGES

ABSENT_CLASS_RULE = (
    "Within a participant, a stage with no true and no predicted epochs is "
    "omitted from that participant's macro average; a stage present in either "
    "the truth or the predictions scores F1 = 0 when it is missed."
)


def _f1_per_class(truth: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-class precision, recall, F1 and support over the fixed class order."""
    n = len(STAGES)
    precision = np.zeros(n)
    recall = np.zeros(n)
    f1 = np.zeros(n)
    support = np.zeros(n, dtype=int)
    for index in range(n):
        true_positive = int(np.count_nonzero((truth == index) & (predicted == index)))
        predicted_positive = int(np.count_nonzero(predicted == index))
        actual_positive = int(np.count_nonzero(truth == index))
        support[index] = actual_positive
        precision[index] = true_positive / predicted_positive if predicted_positive else 0.0
        recall[index] = true_positive / actual_positive if actual_positive else 0.0
        denominator = precision[index] + recall[index]
        f1[index] = 2 * precision[index] * recall[index] / denominator if denominator else 0.0
    return precision, recall, f1, support


def participant_macro_f1(truth: np.ndarray, predicted: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Macro-F1 for one participant, plus its per-class F1 and which classes counted."""
    _, _, f1, support = _f1_per_class(truth, predicted)
    predicted_any = np.array(
        [np.count_nonzero(predicted == index) for index in range(len(STAGES))]
    )
    counted = (support > 0) | (predicted_any > 0)
    if not counted.any():
        return float("nan"), f1, counted
    return float(np.mean(f1[counted])), f1, counted


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Every number a report table is built from, for one model on one split part."""

    model: str
    run_id: str
    split_id: str
    split_part: str
    seed: int
    n_epochs: int
    n_participants: int
    labels: tuple[str, ...]

    participant_macro_f1_mean: float
    """The primary metric: macro-F1 per participant, averaged equally over people."""

    participant_macro_f1_sd: float
    participant_macro_f1: dict[str, float]
    pooled_macro_f1: float
    """Macro-F1 over all epochs pooled together. Reported beside the primary
    metric, never instead of it."""

    pooled_accuracy: float
    balanced_accuracy: float
    cohens_kappa: float
    per_stage_precision: dict[str, float]
    per_stage_recall: dict[str, float]
    per_stage_f1: dict[str, float]
    per_stage_support: dict[str, int]
    per_stage_participants: dict[str, int]
    """How many participants each per-stage figure was averaged over."""

    confusion: list[list[int]]
    qc_flagged_epochs: int
    qc_coverage: float
    absent_class_rule: str = ABSENT_CLASS_RULE
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"{self.model}: participant macro-F1 "
            f"{self.participant_macro_f1_mean:.3f} +/- {self.participant_macro_f1_sd:.3f} "
            f"over {self.n_participants} participant(s); pooled macro-F1 "
            f"{self.pooled_macro_f1:.3f}; kappa {self.cohens_kappa:.3f}; "
            f"n={self.n_epochs}"
        )


def evaluate_rows(rows: list[dict[str, Any]]) -> EvaluationResult:
    """Every metric, from prediction rows that all belong to one model and run."""
    if not rows:
        raise ValueError("no prediction rows to evaluate")
    models = {row["model"] for row in rows}
    if len(models) != 1:
        raise ValueError(f"rows mix models {sorted(models)}; evaluate one at a time")

    index_of = {name: i for i, name in enumerate(STAGES)}
    truth = np.array([index_of[row["true_label"]] for row in rows])
    predicted = np.array([index_of[row["pred_label"]] for row in rows])
    participants = np.array([row["participant_id"] for row in rows])
    qc = np.array([row["qc_flags"] for row in rows])

    per_participant: dict[str, float] = {}
    per_stage_f1_values: dict[str, list[float]] = defaultdict(list)
    for person in sorted(set(participants.tolist())):
        mask = participants == person
        score, f1, counted = participant_macro_f1(truth[mask], predicted[mask])
        if not np.isnan(score):
            per_participant[person] = score
        for index, name in enumerate(STAGES):
            if counted[index]:
                per_stage_f1_values[name].append(float(f1[index]))

    values = np.array(list(per_participant.values()), dtype=float)
    precision, recall, f1, support = _f1_per_class(truth, predicted)
    present = support > 0
    balanced = float(np.mean(recall[present])) if present.any() else 0.0

    confusion = np.zeros((len(STAGES), len(STAGES)), dtype=int)
    for actual, guess in zip(truth, predicted, strict=True):
        confusion[actual, guess] += 1

    first = rows[0]
    return EvaluationResult(
        model=first["model"],
        run_id=first["run_id"],
        split_id=first["split_id"],
        split_part=first["split_part"],
        seed=first["seed"],
        n_epochs=len(rows),
        n_participants=len(per_participant),
        labels=STAGES,
        participant_macro_f1_mean=float(values.mean()) if values.size else float("nan"),
        participant_macro_f1_sd=float(values.std(ddof=0)) if values.size else float("nan"),
        participant_macro_f1=per_participant,
        pooled_macro_f1=float(np.mean(f1[(support > 0) | (confusion.sum(axis=0) > 0)])),
        pooled_accuracy=float(np.mean(truth == predicted)),
        balanced_accuracy=balanced,
        cohens_kappa=cohens_kappa(confusion),
        per_stage_precision=dict(zip(STAGES, (float(x) for x in precision), strict=True)),
        per_stage_recall=dict(zip(STAGES, (float(x) for x in recall), strict=True)),
        per_stage_f1={
            name: float(np.mean(per_stage_f1_values[name])) if per_stage_f1_values[name] else float("nan")
            for name in STAGES
        },
        per_stage_support=dict(zip(STAGES, (int(x) for x in support), strict=True)),
        per_stage_participants={name: len(per_stage_f1_values[name]) for name in STAGES},
        confusion=confusion.tolist(),
        qc_flagged_epochs=int(np.count_nonzero(qc)),
        qc_coverage=float(np.mean(qc == 0)),
        notes={
            "per_stage_f1_is": "mean over participants in which the stage counted",
            "per_stage_precision_recall_is": "pooled over all epochs",
        },
    )


def cohens_kappa(confusion: np.ndarray) -> float:
    """Agreement corrected for what chance would give, from a confusion matrix.

    The metric automatic sleep staging is compared on. Computed from the matrix
    rather than by calling a library so that the five-class order and the
    handling of an absent class are the ones documented above.
    """
    total = confusion.sum()
    if total == 0:
        return 0.0
    observed = np.trace(confusion) / total
    expected = float(
        np.sum(confusion.sum(axis=0) * confusion.sum(axis=1)) / (total * total)
    )
    if expected >= 1.0:
        return 0.0
    return float((observed - expected) / (1 - expected))


def evaluate_predictions(
    path: Path | str, *, model: str | None = None, split_part: str | None = None
) -> list[EvaluationResult]:
    """Read a prediction file and score each model in it separately."""
    from sleepstatelab.evaluation.predictions import read_predictions

    rows = read_predictions(path)
    if model is not None:
        rows = [row for row in rows if row["model"] == model]
    if split_part is not None:
        rows = [row for row in rows if row["split_part"] == split_part]
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["run_id"], row["split_part"], row["seed"])].append(row)
    return [evaluate_rows(group) for _, group in sorted(grouped.items())]
