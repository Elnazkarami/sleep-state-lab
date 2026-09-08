"""Distance from a stage change: the arithmetic, and what must not count as one.

The analysis reads annotations only to decide which epochs to report separately.
Two ways of getting it wrong would both produce a plausible table: counting a
gap as a transition, and letting distance leak into anything upstream of the
predictions.
"""

from __future__ import annotations

import numpy as np
import pytest

from sleepstatelab.evaluation.transitions import (
    UNREACHABLE,
    band_of,
    by_distance,
    distance_to_change,
    table,
)
from sleepstatelab.labels import STAGES

pytestmark = pytest.mark.synthetic


def test_both_epochs_flanking_a_change_are_at_it():
    """A boundary sits between two epochs, and both of them are at it."""
    index = np.arange(6)
    labels = np.array([0, 0, 0, 2, 2, 2])
    assert distance_to_change(index, labels).tolist() == [2, 1, 0, 0, 1, 2]


def test_distance_is_to_the_nearest_change():
    index = np.arange(7)
    labels = np.array([0, 0, 2, 2, 2, 4, 4])
    assert distance_to_change(index, labels).tolist() == [1, 0, 0, 1, 0, 0, 1]


def test_a_gap_is_not_a_transition():
    """Two epochs either side of an excluded one may carry different stages, and
    nothing is known about what happened between them. Counting it would invent
    a transition and then score a model on finding it."""
    index = np.array([0, 1, 2, 10, 11, 12])
    labels = np.array([0, 0, 0, 4, 4, 4])
    distance = distance_to_change(index, labels)
    assert (distance == UNREACHABLE).all()
    assert set(band_of(distance)) == {"no change in the run"}


def test_a_run_with_no_change_is_named_rather_than_called_far():
    index = np.arange(4)
    labels = np.zeros(4, dtype=int)
    distance = distance_to_change(index, labels)
    assert (distance == UNREACHABLE).all()
    assert band_of(distance)[0] == "no change in the run"


def test_changes_within_each_run_are_found_independently():
    index = np.array([0, 1, 2, 3, 20, 21, 22, 23])
    labels = np.array([0, 0, 2, 2, 4, 4, 1, 1])
    assert distance_to_change(index, labels).tolist() == [1, 0, 0, 1, 1, 0, 0, 1]


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="same length"):
        distance_to_change(np.arange(3), np.zeros(4, dtype=int))


def test_an_empty_recording_is_handled():
    assert distance_to_change(np.array([]), np.array([])).size == 0


def _rows(labels, predictions, participant="SC400", recording="SC400-n1"):
    out = []
    for position, (truth, guess) in enumerate(zip(labels, predictions, strict=True)):
        probabilities = {f"p_{name}": 0.0 for name in STAGES}
        probabilities[f"p_{STAGES[guess]}"] = 1.0
        out.append(
            {
                "run_id": "r", "model": "m", "split_id": "s", "split_part": "test",
                "seed": 0, "participant_id": participant, "recording_id": recording,
                "epoch_index": position, "true_label": STAGES[truth],
                "pred_label": STAGES[guess], "qc_flags": 0, **probabilities,
            }
        )
    return out


def test_bands_partition_the_epochs():
    labels = [0, 0, 0, 2, 2, 2, 2, 2, 2, 2]
    results = by_distance(_rows(labels, labels))
    assert sum(r.n_epochs for r in results) == len(labels)
    assert sum(r.share for r in results) == pytest.approx(1.0)


def test_errors_concentrated_at_a_boundary_show_up_there():
    """The table has to be able to say that a model fails at transitions."""
    labels = [0, 0, 0, 0, 2, 2, 2, 2]
    predictions = [0, 0, 0, 2, 0, 2, 2, 2]  # both flanking epochs wrong
    results = {r.band: r for r in by_distance(_rows(labels, predictions))}
    assert results["at a change"].accuracy == 0.0
    assert results["3-5 epochs away"].accuracy == 1.0


def test_the_table_renders_every_band_present():
    labels = [0, 0, 0, 2, 2, 2]
    rendered = table(by_distance(_rows(labels, labels)))
    assert "at a change" in rendered
    assert rendered.startswith("| distance from a stage change")
