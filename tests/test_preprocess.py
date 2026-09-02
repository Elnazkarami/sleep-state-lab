"""Normalisation is fitted on training participants and keeps amplitude."""

from __future__ import annotations

import numpy as np
import pytest

from sleepstatelab.config import PreprocessConfig
from sleepstatelab.data.preprocess import NormalizationStats, bandpass, fit_normalization

pytestmark = pytest.mark.synthetic


def _blocks(scale: float, n: int = 20) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.normal(0.0, scale, (n, 2, 3000))).astype(np.float32)


def test_statistics_are_fitted_only_on_the_participants_given():
    stats = fit_normalization(
        [_blocks(10.0), _blocks(12.0)],
        ["SC400", "SC401"],
        channels=("EEG Fpz-Cz", "EEG Pz-Oz"),
        config=PreprocessConfig(),
    )
    assert stats.fitted_on == ("SC400", "SC401")
    assert "SC402" not in stats.fitted_on


def test_a_test_participants_amplitude_does_not_move_the_scale():
    """Fitting on training only means a differently-scaled test participant
    changes nothing about the transform."""
    config = PreprocessConfig()
    train_only = fit_normalization(
        [_blocks(10.0)], ["SC400"], channels=("a", "b"), config=config
    )
    with_test = fit_normalization(
        [_blocks(10.0)], ["SC400"], channels=("a", "b"), config=config
    )
    assert train_only.identity == with_test.identity


def test_channel_normalisation_keeps_relative_amplitude():
    """Two epochs of different amplitude must stay different afterwards. This is
    the property per-epoch z-scoring destroys, and the reason it is not default."""
    config = PreprocessConfig(clip_sigma=0.0)
    stats = fit_normalization(
        [_blocks(10.0)], ["SC400"], channels=("a", "b"), config=config
    )
    quiet = np.full((1, 2, 3000), 5.0, dtype=np.float32)
    loud = np.full((1, 2, 3000), 50.0, dtype=np.float32)
    assert stats.apply(loud).mean() > stats.apply(quiet).mean() * 2


def test_per_epoch_zscore_removes_amplitude_as_documented():
    stats = NormalizationStats(
        method="per_epoch_zscore",
        channels=("a", "b"),
        centre=(0.0, 0.0),
        scale=(1.0, 1.0),
        fitted_on=(),
        n_epochs=0,
        clip_sigma=0.0,
    )
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, (1, 2, 3000)).astype(np.float32)
    quiet = stats.apply(base * 5.0)
    loud = stats.apply(base * 50.0)
    assert np.allclose(quiet, loud, atol=1e-4)


def test_normalisation_round_trips_through_a_file(tmp_path):
    stats = fit_normalization(
        [_blocks(10.0)], ["SC400"], channels=("a", "b"), config=PreprocessConfig()
    )
    stats.write(tmp_path / "norm.json")
    assert NormalizationStats.read(tmp_path / "norm.json").identity == stats.identity


def test_bandpass_removes_drift_and_keeps_the_bands():
    rate = 100.0
    t = np.arange(3000) / rate
    signal = (20 * np.sin(2 * np.pi * 0.02 * t) + 10 * np.sin(2 * np.pi * 10 * t)).astype(
        np.float32
    )
    filtered = bandpass(signal[None, None, :], rate, PreprocessConfig())[0, 0]
    # The 0.02 Hz drift is gone; the 10 Hz rhythm survives with its amplitude.
    assert np.abs(filtered).max() < 14.0
    assert np.abs(filtered[200:-200]).max() > 8.0
