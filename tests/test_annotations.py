"""Labels mean what the hypnogram says, and non-stages never become stages."""

from __future__ import annotations

import numpy as np
import pytest

from sleepstatelab.data.annotations import Annotation, expand, read_hypnogram
from sleepstatelab.labels import STAGE_INDEX, STAGES, stage_of

pytestmark = pytest.mark.synthetic


def test_r_and_k_stage_four_folds_into_n3():
    assert stage_of("Sleep stage 3") == "N3"
    assert stage_of("Sleep stage 4") == "N3"
    assert stage_of("Sleep stage R") == "REM"
    assert stage_of("Sleep stage W") == "Wake"


def test_movement_and_unknown_are_not_labels():
    assert stage_of("Movement time") is None
    assert stage_of("Sleep stage ?") is None
    for name in STAGES:
        assert name not in {"Movement", "Unknown"}


def test_runs_expand_to_the_epochs_they_cover():
    hypnogram = expand(
        (
            Annotation(0.0, 60.0, "Sleep stage W"),
            Annotation(60.0, 30.0, "Movement time"),
            Annotation(90.0, 90.0, "Sleep stage 2"),
        ),
        epoch_seconds=30.0,
    )
    assert hypnogram.labels.tolist() == [
        STAGE_INDEX["Wake"],
        STAGE_INDEX["Wake"],
        -1,
        STAGE_INDEX["N2"],
        STAGE_INDEX["N2"],
        STAGE_INDEX["N2"],
    ]
    assert hypnogram.exclusions["movement_time"] == 1


def test_misaligned_annotations_are_recorded_not_rounded_away():
    # 7.5 s to 67.5 s: it wholly covers epoch 1 ([30, 60)) and only part of
    # epochs 0 and 2, so only epoch 1 is labelled and the misalignment is named.
    hypnogram = expand(
        (Annotation(7.5, 60.0, "Sleep stage 2"),), epoch_seconds=30.0, n_epochs=3
    )
    assert hypnogram.misaligned, "a 7.5 s onset should be reported as misaligned"
    assert hypnogram.labels.tolist() == [-1, STAGE_INDEX["N2"], -1]


def test_overlapping_scoring_is_counted():
    hypnogram = expand(
        (
            Annotation(0.0, 60.0, "Sleep stage W"),
            Annotation(30.0, 30.0, "Sleep stage 2"),
        ),
        epoch_seconds=30.0,
    )
    assert hypnogram.overlapping_epochs == 1


def test_reading_a_generated_hypnogram_recovers_its_stages(synthetic_night):
    hypnogram = read_hypnogram(synthetic_night.hypnogram_path)
    expected = [
        -1 if index in {7, 12, 13} else STAGE_INDEX[stage]
        for index, stage in enumerate(synthetic_night.stages)
    ]
    assert hypnogram.labels[: len(expected)].tolist() == expected
    assert hypnogram.exclusions["movement_time"] == 1
    assert hypnogram.exclusions["unscored"] == 2
    assert np.count_nonzero(hypnogram.labels < 0) == 3
