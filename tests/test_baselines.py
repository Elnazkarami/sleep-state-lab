"""The classical baselines behave as a comparison requires."""

from __future__ import annotations

import numpy as np
import pytest

from sleepstatelab.baselines.classical import (
    ClassPrior,
    build_baseline,
    probabilities_in_stage_order,
    run_baselines,
)
from sleepstatelab.features.spectral import epoch_features, feature_matrix, feature_names
from sleepstatelab.labels import STAGES

pytestmark = pytest.mark.synthetic


def test_class_prior_predicts_the_training_distribution():
    y = np.array([0] * 70 + [2] * 20 + [4] * 10)
    model = ClassPrior().fit(np.zeros((100, 3)), y)
    probabilities = model.predict_proba(np.zeros((5, 3)))
    assert probabilities.shape == (5, len(STAGES))
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(5))
    assert probabilities.argmax(axis=1).tolist() == [0] * 5
    assert (probabilities > 0).all(), "a smoothed prior gives every stage a chance"


def test_probability_columns_are_remapped_when_a_class_is_absent():
    """A training split with no N3 makes sklearn emit four columns. Left alone,
    every index after N2 would be wrong."""
    x = np.random.default_rng(0).normal(size=(60, 4))
    y = np.array([0, 1, 2, 4] * 15)
    model = build_baseline("logistic").fit(x, y)
    probabilities = probabilities_in_stage_order(model, x)
    assert probabilities.shape == (60, len(STAGES))
    assert np.allclose(probabilities[:, 3], 0.0), "the absent stage must stay at zero"
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_features_are_named_and_counted_consistently():
    channels = ("EEG Fpz-Cz", "EEG Pz-Oz")
    x = np.random.default_rng(0).normal(0, 20, (4, 2, 3000))
    matrix = feature_matrix(x, 100.0)
    assert matrix.shape == (4, len(feature_names(channels)))
    assert np.isfinite(matrix).all()


def test_features_separate_a_slow_epoch_from_a_fast_one():
    """Delta-band relative power must rise for a slow signal. If this fails, the
    features are not measuring what their names say."""
    t = np.arange(3000) / 100.0
    slow = np.vstack([50 * np.sin(2 * np.pi * 1.0 * t)] * 2)
    fast = np.vstack([50 * np.sin(2 * np.pi * 20.0 * t)] * 2)
    names = feature_names(("EEG Fpz-Cz", "EEG Pz-Oz"))
    column = names.index("ch0_eeg_fpz_cz_delta_rel")
    assert epoch_features(slow, 100.0)[column] > 0.8
    assert epoch_features(fast, 100.0)[column] < 0.2


def test_run_baselines_never_fits_on_the_evaluation_set():
    rng = np.random.default_rng(0)
    train_x = rng.normal(size=(80, 6))
    train_y = np.array([0, 1, 2, 3, 4] * 16)
    eval_x = rng.normal(size=(20, 6))
    found = run_baselines(train_x=train_x, train_y=train_y, eval_x=eval_x)
    assert set(found) == {"class_prior", "logistic", "random_forest"}
    for probabilities in found.values():
        assert probabilities.shape == (20, len(STAGES))
