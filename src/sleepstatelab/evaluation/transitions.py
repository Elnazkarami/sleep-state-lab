"""Scoring by distance from a stage change, using annotations for evaluation only.

Stage boundaries are where sleep scoring is hard, for people as well as for
models: the epoch in which N1 becomes N2 is genuinely ambiguous, and two expert
scorers disagree there far more often than they do in the middle of a long
stretch of slow-wave sleep. A model's mean score mixes the two situations
together, and a claim that a temporal model "helps at transitions" is exactly
the sort of thing that needs measuring rather than assuming.

**Annotations are read here only to decide which epochs to report separately.**
Distance from a transition never enters training, sampling, class weighting, or
model selection -- if it did, the model would have been told where the
boundaries are, and the resulting table would be circular. It is computed after
the fact, from the same saved predictions everything else is computed from.

**Gaps are not transitions.** A stage change is a change between two epochs whose
original indices differ by exactly one. Two epochs either side of an excluded
epoch may well carry different stages, and nothing is known about what happened
in between; counting that as a boundary would invent a transition and then
score a model on how well it found it.

**What this cannot show.** That stages are attractors, that transitions are
bifurcations, or that a difference between bands reflects anything about
neural dynamics. It shows where a classifier's errors are concentrated, which is
a fact about the classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from sleepstatelab.labels import STAGES

UNREACHABLE = np.iinfo(np.int32).max
"""Distance for an epoch in a run that contains no stage change at all."""

DEFAULT_BANDS: tuple[tuple[str, int, int], ...] = (
    ("at a change", 0, 0),
    ("1 epoch away", 1, 1),
    ("2 epochs away", 2, 2),
    ("3-5 epochs away", 3, 5),
    ("6+ epochs away", 6, UNREACHABLE - 1),
    ("no change in the run", UNREACHABLE, UNREACHABLE),
)
"""Bands in epochs. Thirty seconds each, so "2 epochs away" is a minute from the
nearest boundary."""


def distance_to_change(epoch_index: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Epochs to the nearest stage change, per epoch.

    An epoch flanking a change has distance 0 -- both sides of a boundary are at
    it. Distance is measured only within runs of consecutive epoch indices, so
    it never counts across a gap, and an epoch in a run with no change at all
    gets ``UNREACHABLE`` rather than a number that would read as "very far".
    """
    index = np.asarray(epoch_index)
    labels = np.asarray(labels)
    if index.size != labels.size:
        raise ValueError("epoch_index and labels must be the same length")
    distance = np.full(index.size, UNREACHABLE, dtype=np.int64)
    if index.size == 0:
        return distance

    breaks = np.flatnonzero(np.diff(index) != 1) + 1
    bounds = [0, *breaks.tolist(), int(index.size)]
    for start, stop in pairwise(bounds):
        run = labels[start:stop]
        if run.size < 2:
            continue
        changes = np.flatnonzero(run[:-1] != run[1:])
        if changes.size == 0:
            continue
        # Both epochs flanking a change are at distance zero.
        at_change = np.zeros(run.size, dtype=bool)
        at_change[changes] = True
        at_change[changes + 1] = True
        positions = np.flatnonzero(at_change)
        # Distance from each epoch in the run to the nearest such position.
        run_distance = np.abs(
            np.arange(run.size)[:, None] - positions[None, :]
        ).min(axis=1)
        distance[start:stop] = run_distance
    return distance


def band_of(distance: np.ndarray, bands: tuple[tuple[str, int, int], ...] = DEFAULT_BANDS) -> np.ndarray:
    """The band name for each distance."""
    out = np.empty(distance.size, dtype=object)
    for name, low, high in bands:
        out[(distance >= low) & (distance <= high)] = name
    return out


@dataclass(frozen=True, slots=True)
class BandResult:
    """One band's scores, and how much of the test set it is."""

    band: str
    n_epochs: int
    share: float
    participant_macro_f1_mean: float
    participant_macro_f1_sd: float
    n_participants: int
    accuracy: float
    per_stage_support: dict[str, int]


def by_distance(
    rows: list[dict[str, Any]],
    *,
    bands: tuple[tuple[str, int, int], ...] = DEFAULT_BANDS,
) -> list[BandResult]:
    """Score one model's saved predictions separately in each distance band.

    The rows must be one model's, on one split part. Distances are computed per
    recording from the true labels, which is why this needs nothing but the
    prediction file.
    """
    from sleepstatelab.evaluation.metrics import participant_macro_f1

    index_of = {name: i for i, name in enumerate(STAGES)}
    by_recording: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_recording.setdefault(row["recording_id"], []).append(row)

    tagged: list[tuple[str, str, int, int]] = []
    """(band, participant, true index, predicted index)"""
    for block in by_recording.values():
        block.sort(key=lambda row: row["epoch_index"])
        epoch_index = np.array([row["epoch_index"] for row in block])
        truth = np.array([index_of[row["true_label"]] for row in block])
        predicted = np.array([index_of[row["pred_label"]] for row in block])
        names = band_of(distance_to_change(epoch_index, truth), bands)
        for position, row in enumerate(block):
            tagged.append(
                (names[position], row["participant_id"], int(truth[position]), int(predicted[position]))
            )

    total = len(tagged)
    results: list[BandResult] = []
    for name, _, _ in bands:
        rows_here = [t for t in tagged if t[0] == name]
        if not rows_here:
            continue
        people = sorted({t[1] for t in rows_here})
        scores = []
        for person in people:
            mine = [t for t in rows_here if t[1] == person]
            score, _, _ = participant_macro_f1(
                np.array([t[2] for t in mine]), np.array([t[3] for t in mine])
            )
            if not np.isnan(score):
                scores.append(score)
        truth = np.array([t[2] for t in rows_here])
        predicted = np.array([t[3] for t in rows_here])
        results.append(
            BandResult(
                band=name,
                n_epochs=len(rows_here),
                share=len(rows_here) / total if total else 0.0,
                participant_macro_f1_mean=float(np.mean(scores)) if scores else float("nan"),
                participant_macro_f1_sd=float(np.std(scores)) if scores else float("nan"),
                n_participants=len(scores),
                accuracy=float(np.mean(truth == predicted)),
                per_stage_support={
                    stage: int(np.count_nonzero(truth == i)) for i, stage in enumerate(STAGES)
                },
            )
        )
    return results


def table(results: list[BandResult]) -> str:
    """Markdown for one model's bands."""
    lines = [
        "| distance from a stage change | epochs | share | participant macro-F1 | accuracy | participants |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        mean = (
            "n/a"
            if r.participant_macro_f1_mean != r.participant_macro_f1_mean
            else f"{r.participant_macro_f1_mean:.3f} ± {r.participant_macro_f1_sd:.3f}"
        )
        lines.append(
            f"| {r.band} | {r.n_epochs} | {r.share:.1%} | {mean} | "
            f"{r.accuracy:.3f} | {r.n_participants} |"
        )
    return "\n".join(lines)
