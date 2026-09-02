"""Epoching keeps its gaps, its indices and its exclusions."""

from __future__ import annotations

import numpy as np
import pytest

from sleepstatelab.data.discovery import discover
from sleepstatelab.data.epochs import (
    QC_FLATLINE,
    EpochedRecording,
    QCPolicy,
    epoch_recording,
    quality_flags,
    sleep_window_mask,
)
from sleepstatelab.labels import STAGE_INDEX

pytestmark = pytest.mark.synthetic


def _record(night):
    found = discover(night.psg_path.parent)
    return epoch_recording(found.pairs[0], channels=("EEG Fpz-Cz", "EEG Pz-Oz"))


def test_epoch_shape_is_the_documented_one(synthetic_night):
    record = _record(synthetic_night)
    assert record.signals.shape[1:] == (2, 3000)
    assert record.signals.dtype == np.float32


def test_excluded_epochs_leave_a_visible_hole(synthetic_night):
    """The movement epoch and the two unscored epochs must not close up."""
    record = _record(synthetic_night)
    assert 7 not in record.epoch_index.tolist()
    assert 12 not in record.epoch_index.tolist()
    assert 13 not in record.epoch_index.tolist()
    assert np.count_nonzero(np.diff(record.epoch_index) > 1) == 2


def test_no_epoch_is_concatenated_across_a_gap(synthetic_night):
    """Neighbouring rows either have adjacent indices or they do not, and code
    that needs neighbours must be able to tell. This asserts the record makes
    that decidable, which is what D2's masking will depend on."""
    record = _record(synthetic_night)
    steps = np.diff(record.epoch_index)
    assert steps.min() >= 1
    assert (steps > 1).any(), "the fixture is supposed to contain gaps"


def test_labels_line_up_with_the_generated_truth(synthetic_night):
    record = _record(synthetic_night)
    expected = [
        STAGE_INDEX[synthetic_night.stages[index]] for index in record.epoch_index.tolist()
    ]
    assert record.labels.tolist() == expected


def test_flatline_is_flagged():
    policy = QCPolicy()
    flat = np.zeros((2, 3000), dtype=np.float32)
    assert quality_flags(flat, 100.0, policy) & QC_FLATLINE


def test_cache_round_trip_preserves_everything(synthetic_night, tmp_path):
    record = _record(synthetic_night)
    record.save(tmp_path / "one.npz")
    again = EpochedRecording.load(tmp_path / "one.npz")
    assert np.array_equal(record.signals, again.signals)
    assert np.array_equal(record.epoch_index, again.epoch_index)
    assert np.array_equal(record.labels, again.labels)
    assert again.channels == record.channels
    assert again.exclusions == record.exclusions


def test_sleep_window_is_an_option_not_the_default(synthetic_night):
    """The primary preparation keeps every scored epoch; the crop is opt-in."""
    labels = np.array([0, 0, 0, 2, 2, 0, 0, 0])
    inside = sleep_window_mask(labels, margin_epochs=1)
    assert inside.tolist() == [False, False, True, True, True, True, False, False]

    record = _record(synthetic_night)
    assert record.source["sleep_window"] is False
