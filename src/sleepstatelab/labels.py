"""The five stages, in one fixed order, defined once.

Everything downstream -- the class weights, the confusion matrix, the columns of
the saved probabilities, the checkpoint metadata -- is indexed by this order. It
is defined in its own module with no imports so that nothing can end up with a
second copy of it.

Sleep-EDF Expanded is scored under Rechtschaffen and Kales, which separates
stages 3 and 4. They are merged into N3, as current AASM practice does, and the
original annotation text is carried alongside every epoch so the merge can be
undone by anyone who wants stage 4 back.

``Movement time`` and ``Sleep stage ?`` are deliberately absent from the map.
They are not a sixth stage and they are not wake; an epoch carrying one is
excluded from the labelled set and recorded as an exclusion.
"""

from __future__ import annotations

STAGES: tuple[str, ...] = ("Wake", "N1", "N2", "N3", "REM")
"""The label order used everywhere. Do not reorder: saved predictions,
checkpoints, and class weights are all positional against this tuple."""

STAGE_INDEX: dict[str, int] = {name: i for i, name in enumerate(STAGES)}

ANNOTATION_TO_STAGE: dict[str, str] = {
    "Sleep stage W": "Wake",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}
"""Rechtschaffen and Kales as the files write it, mapped to the five reported
stages. Stage 4 folds into N3; nothing else folds into anything."""

NON_STAGE_ANNOTATIONS: frozenset[str] = frozenset(
    {"Movement time", "Sleep stage ?", "Sleep stage e"}
)
"""Annotations that exist in the hypnograms and are not stages. They become
exclusions with a reason, never labels."""

EXCLUSION_REASONS: tuple[str, ...] = (
    "movement_time",
    "unscored",
    "unknown_annotation",
    "no_annotation",
    "beyond_signal",
    "short_epoch",
    "qc_flatline",
    "qc_clipped",
    "qc_high_amplitude",
)
"""Every reason an epoch can fail to become a labelled example. Closed set, so
a count of exclusions can be checked against the count of epochs."""


def stage_of(annotation: str) -> str | None:
    """The reported stage for one annotation string, or ``None`` if it is not one."""
    return ANNOTATION_TO_STAGE.get(annotation.strip())
