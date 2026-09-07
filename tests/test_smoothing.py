"""The temporal smoothing control: transitions, gaps, and the prior correction.

This is the control that asks how much of a temporal model's advantage a table
of transition counts already provides. It has to be right in three ways, and
each has a way of being wrong that produces a plausible number.
"""

from __future__ import annotations

import numpy as np
import pytest

from sleepstatelab.baselines.smoothing import (
    consecutive_runs,
    fit_transitions,
    smooth_probabilities,
    viterbi,
)
from sleepstatelab.labels import STAGE_INDEX, STAGES

pytestmark = pytest.mark.synthetic


def test_transitions_are_counted_only_between_adjacent_epochs():
    """Epoch 3 was excluded. The pair either side of the hole is not a
    transition, and counting it would invent evidence for a stage change."""
    index = np.array([0, 1, 2, 4, 5])
    labels = np.array([0, 0, 0, 2, 2])  # Wake, Wake, Wake, | N2, N2
    model = fit_transitions([(index, labels)], ["SC400"], smoothing=0.0)

    assert model.n_transitions == 3  # 0->1, 1->2, 4->5; never 2->4
    assert model.transition[STAGE_INDEX["Wake"]][STAGE_INDEX["N2"]] == 0.0
    assert model.transition[STAGE_INDEX["Wake"]][STAGE_INDEX["Wake"]] == pytest.approx(1.0)


def test_smoothing_keeps_every_transition_possible():
    """A zero would let one unobserved pairing veto a path through the night."""
    index = np.arange(4)
    labels = np.zeros(4, dtype=int)
    model = fit_transitions([(index, labels)], ["SC400"], smoothing=1.0)
    assert all(value > 0 for row in model.transition for value in row)
    assert all(sum(row) == pytest.approx(1.0) for row in model.transition)


def test_the_model_records_who_it_was_fitted_on():
    model = fit_transitions(
        [(np.arange(3), np.zeros(3, dtype=int))], ["SC402", "SC400"]
    )
    assert model.fitted_on == ("SC400", "SC402")


def test_consecutive_runs_split_at_gaps():
    assert consecutive_runs(np.array([0, 1, 2, 5, 6, 9])) == [(0, 3), (3, 5), (5, 6)]
    assert consecutive_runs(np.array([])) == []
    assert consecutive_runs(np.array([7])) == [(0, 1)]


def test_viterbi_prefers_persistence_when_the_evidence_is_weak():
    """A single dissenting epoch inside a stable stretch is what smoothing is
    for: with a strongly persistent transition table it should be overruled."""
    index = np.arange(5)
    labels = np.zeros(5, dtype=int)
    persistent = fit_transitions(
        [(index, labels)] + [(index, np.full(5, i)) for i in range(1, len(STAGES))],
        ["SC400"],
        smoothing=0.1,
    )
    probabilities = np.full((5, len(STAGES)), 0.02)
    probabilities[:, 0] = 0.92
    probabilities[2, 0] = 0.45  # a wobble, not a stage change
    probabilities[2, 2] = 0.49
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    unsmoothed = probabilities.argmax(axis=1)
    smoothed = smooth_probabilities(probabilities, index, persistent).argmax(axis=1)
    assert unsmoothed[2] == STAGE_INDEX["N2"]
    assert smoothed.tolist() == [0, 0, 0, 0, 0]


def test_smoothing_never_decodes_across_a_gap():
    """Two epochs either side of a hole must be decoded as separate sequences."""
    index = np.array([0, 1, 50, 51])
    labels = np.array([0, 0, 2, 2])
    model = fit_transitions([(np.arange(4), labels)], ["SC400"], smoothing=0.1)
    probabilities = np.full((4, len(STAGES)), 0.02)
    probabilities[:2, 0] = 0.92
    probabilities[2:, 2] = 0.92
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    decoded = smooth_probabilities(probabilities, index, model).argmax(axis=1)
    assert decoded.tolist() == [0, 0, 2, 2]


def test_the_prior_correction_stops_the_prior_being_applied_twice():
    """A classifier's softmax already contains the class prior. Multiplying by
    it again makes the decoder far too fond of whatever is commonest."""
    # A training set that is overwhelmingly Wake.
    labels = np.array([0] * 30 + [2] * 2)
    model = fit_transitions([(np.arange(labels.size), labels)], ["SC400"], smoothing=1.0)

    emission = np.log(np.array([[0.3, 0.1, 0.4, 0.1, 0.1]] * 3))
    with_correction = viterbi(emission, model, prior_correction=True)
    without = viterbi(emission, model, prior_correction=False)

    assert with_correction.tolist() == [STAGE_INDEX["N2"]] * 3
    assert without.tolist() == [STAGE_INDEX["Wake"]] * 3


def test_the_output_is_one_hot_not_a_fake_probability():
    """Viterbi returns a sequence, not a calibrated distribution. Emitting
    something that looked like a probability would invite it to be read as one."""
    index = np.arange(4)
    model = fit_transitions([(index, np.zeros(4, dtype=int))], ["SC400"])
    probabilities = np.full((4, len(STAGES)), 0.2)
    out = smooth_probabilities(probabilities, index, model)
    assert set(np.unique(out).tolist()) <= {0.0, 1.0}
    assert (out.sum(axis=1) == 1.0).all()


def test_a_wrong_class_count_is_refused():
    model = fit_transitions([(np.arange(2), np.zeros(2, dtype=int))], ["SC400"])
    with pytest.raises(ValueError, match="expected 5 classes"):
        viterbi(np.zeros((3, 4)), model)
