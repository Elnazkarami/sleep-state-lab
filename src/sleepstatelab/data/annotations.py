"""The hypnogram, expanded to one label per 30-second epoch.

A Sleep-EDF hypnogram stores runs, not epochs: a single annotation says "stage 2
for 3,600 seconds". Expanding a run to the epochs it covers is the whole job,
and there are three ways to get it wrong that this module checks for rather than
assumes away.

**Alignment.** An annotation whose onset or duration is not a whole number of
30-second epochs cannot be expanded without deciding what to do with a partial
epoch. The decision here is to record the misalignment and to label only the
epochs the annotation wholly covers.

**Overlap.** Two annotations covering the same epoch is a contradiction in the
scoring. It is recorded, and the later annotation does not silently overwrite
the earlier one without being counted.

**Non-stages.** ``Movement time`` and ``Sleep stage ?`` are annotations, not
stages. They mark an epoch excluded with a reason, and they never become wake.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sleepstatelab.labels import NON_STAGE_ANNOTATIONS, STAGE_INDEX, stage_of

UNLABELLED = -1
"""The label of an epoch that has no stage. Never a class index."""


@dataclass(frozen=True, slots=True)
class Annotation:
    """One scored run, as the file writes it."""

    onset_seconds: float
    duration_seconds: float
    text: str


@dataclass(frozen=True, slots=True)
class Hypnogram:
    """One recording's scoring, expanded to epochs and audited."""

    path: Path
    epoch_seconds: float
    labels: np.ndarray
    """``int8`` per epoch: 0-4 for the five stages, -1 for no stage."""

    raw_text: tuple[str, ...]
    """The original annotation text for each epoch, so the R&K stage 3/4 merge
    and every exclusion can be reconstructed."""

    exclusions: dict[str, int]
    annotations: tuple[Annotation, ...]
    misaligned: tuple[str, ...]
    overlapping_epochs: int
    covered_seconds: float

    @property
    def n_epochs(self) -> int:
        return int(self.labels.size)

    @property
    def n_labelled(self) -> int:
        return int(np.count_nonzero(self.labels >= 0))

    def counts(self) -> dict[str, int]:
        from sleepstatelab.labels import STAGES

        found = dict.fromkeys(STAGES, 0)
        for index, name in enumerate(STAGES):
            found[name] = int(np.count_nonzero(self.labels == index))
        return found


def read_annotations(path: Path | str) -> tuple[Annotation, ...]:
    """Every annotation in an EDF+ hypnogram, via MNE's reader.

    MNE is used rather than a hand-written TAL parser because annotation parsing
    is exactly the kind of thing an established reader has already had the edge
    cases beaten out of. What is *not* delegated is what the annotations mean --
    that is the module above.
    """
    import mne

    with _quiet_mne():
        found = mne.read_annotations(str(path))
    return tuple(
        Annotation(
            onset_seconds=float(onset),
            duration_seconds=float(duration),
            text=str(description),
        )
        for onset, duration, description in zip(
            found.onset, found.duration, found.description, strict=True
        )
    )


def expand(
    annotations: tuple[Annotation, ...],
    *,
    epoch_seconds: float = 30.0,
    n_epochs: int | None = None,
    path: Path | str = "",
) -> Hypnogram:
    """One label per epoch, plus every exclusion and inconsistency found."""
    if not annotations:
        raise ValueError(f"{path or 'hypnogram'} carries no annotations")

    end = max(a.onset_seconds + a.duration_seconds for a in annotations)
    total = n_epochs if n_epochs is not None else round(end / epoch_seconds)
    labels = np.full(total, UNLABELLED, dtype=np.int8)
    raw = [""] * total
    written = np.zeros(total, dtype=bool)

    exclusions = dict.fromkeys(("movement_time", "unscored", "unknown_annotation"), 0)
    misaligned: list[str] = []
    overlaps = 0
    covered = 0.0

    for note in annotations:
        onset_epochs = note.onset_seconds / epoch_seconds
        duration_epochs = note.duration_seconds / epoch_seconds
        if abs(onset_epochs - round(onset_epochs)) > 1e-6 or (
            abs(duration_epochs - round(duration_epochs)) > 1e-6
        ):
            misaligned.append(
                f"{note.text!r} at {note.onset_seconds:.3f}s "
                f"for {note.duration_seconds:.3f}s"
            )
        first = int(np.ceil(onset_epochs - 1e-6))
        last = int(np.floor(onset_epochs + duration_epochs + 1e-6))
        first = max(first, 0)
        last = min(last, total)
        if last <= first:
            continue
        covered += (last - first) * epoch_seconds

        span = slice(first, last)
        overlaps += int(np.count_nonzero(written[span]))
        written[span] = True
        for index in range(first, last):
            raw[index] = note.text

        stage = stage_of(note.text)
        if stage is not None:
            labels[span] = STAGE_INDEX[stage]
            continue
        n = last - first
        if note.text.strip() == "Movement time":
            exclusions["movement_time"] += n
        elif note.text.strip() in NON_STAGE_ANNOTATIONS:
            exclusions["unscored"] += n
        else:
            exclusions["unknown_annotation"] += n

    exclusions["no_annotation"] = int(np.count_nonzero(~written))

    return Hypnogram(
        path=Path(path),
        epoch_seconds=epoch_seconds,
        labels=labels,
        raw_text=tuple(raw),
        exclusions=exclusions,
        annotations=annotations,
        misaligned=tuple(misaligned),
        overlapping_epochs=overlaps,
        covered_seconds=covered,
    )


def read_hypnogram(
    path: Path | str, *, epoch_seconds: float = 30.0, n_epochs: int | None = None
) -> Hypnogram:
    """Read and expand one hypnogram file."""
    return expand(
        read_annotations(path),
        epoch_seconds=epoch_seconds,
        n_epochs=n_epochs,
        path=path,
    )


class _quiet_mne:
    """Silence MNE's per-file logging without silencing warnings that matter.

    MNE prints a paragraph per file at INFO level. Over 153 recordings that
    buries the audit output, which is the thing anyone actually reads.
    """

    def __enter__(self) -> None:
        import mne

        self._previous = mne.set_log_level("ERROR", return_old_level=True)

    def __exit__(self, *_: object) -> None:
        import mne

        mne.set_log_level(self._previous)
