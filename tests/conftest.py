"""Fixtures shared by the tests. Everything here is generated, never a recording."""

from __future__ import annotations

from pathlib import Path

import pytest

from sleepstatelab.config import Config, DataConfig, ModelConfig, TrainConfig
from sleepstatelab.synthetic import make_cohort, make_night


@pytest.fixture(scope="session")
def synthetic_night(tmp_path_factory: pytest.TempPathFactory):
    """One generated night on disk, with a movement epoch and an unscored gap."""
    directory = tmp_path_factory.mktemp("one-night")
    return make_night(directory, participant=1, n_epochs=24, seed=7)


@pytest.fixture(scope="session")
def synthetic_cohort(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Six generated participants, one night each."""
    directory = tmp_path_factory.mktemp("cohort")
    make_cohort(directory, n_participants=6, n_epochs=24, seed=3)
    return directory


@pytest.fixture
def small_config(synthetic_cohort: Path, tmp_path: Path) -> Config:
    """A configuration pointing at the generated cohort, sized for a fast test."""
    return Config(
        name="test",
        data=DataConfig(root=str(synthetic_cohort), cache_dir=str(tmp_path / "cache")),
        model=ModelConfig(embedding_dim=32, block_channels=(16, 24, 32)),
        train=TrainConfig(device="cpu", epochs=2, batch_size=16, early_stopping_patience=2),
    )
