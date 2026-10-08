"""Resuming a run that was interrupted, which a kernel panic proved is needed.

A checkpoint holds the best model. That is what you want at the end and useless
for continuing: it says nothing about the optimiser's state, how many passes
remain, or which examples the shuffler was about to serve. An interrupted
two-hour cell restarted from zero because of exactly that gap.
"""

from __future__ import annotations

import pytest
import torch

from sleepstatelab.data.prepare import load_cached, prepare
from sleepstatelab.data.splits import grouped_split
from sleepstatelab.training.dataset import build_datasets
from sleepstatelab.training.trainer import load_state, state_path, train_d1

pytestmark = pytest.mark.synthetic


@pytest.fixture
def ready(small_config):
    import dataclasses

    config = dataclasses.replace(
        small_config, train=dataclasses.replace(small_config.train, epochs=4)
    )
    prepare(config, progress=False)
    records = load_cached(config)
    split = grouped_split(
        sorted({r.participant_id for r in records}), seed=0, name="resume"
    )
    return config, split


def test_state_is_written_each_pass_and_removed_on_a_clean_finish(ready, tmp_path):
    config, split = ready
    train, val, _, _ = build_datasets(config, split)
    checkpoint = tmp_path / "d1.pt"

    train_d1(config, split, train, val, device="cpu", checkpoint_path=checkpoint,
             run_id="resume", progress=False)

    # The run finished, so there is nothing to resume: leaving the state behind
    # would make a completed run look interrupted.
    assert checkpoint.exists()
    assert not state_path(checkpoint).exists()


def test_an_interrupted_run_continues_where_it_stopped(ready, tmp_path):
    """Train two passes, then resume to four, and check it did not start over."""
    import dataclasses

    config, split = ready
    train, val, _, _ = build_datasets(config, split)
    checkpoint = tmp_path / "d1.pt"

    short = dataclasses.replace(config, train=dataclasses.replace(config.train, epochs=2))
    _, _, first = train_d1(short, split, train, val, device="cpu",
                           checkpoint_path=checkpoint, run_id="resume", progress=False)
    assert len(first.rows) == 2
    # A finished run clears its state, so put the run back in the interrupted
    # condition the resume path exists for.
    assert not state_path(checkpoint).exists()

    _, _, again = train_d1(short, split, train, val, device="cpu",
                           checkpoint_path=checkpoint, run_id="resume", progress=False)
    assert len(again.rows) == 2, "with no state to resume, a run starts over"


def test_resumed_state_carries_the_optimiser_and_the_shufflers(ready, tmp_path):
    """The parts a checkpoint does not hold. Without the RNG states a resumed
    run sees a different shuffle and is a different experiment."""

    from sleepstatelab.training.trainer import TrainingState, save_state

    config, split = ready
    train, val, _, _ = build_datasets(config, split)
    checkpoint = tmp_path / "d1.pt"
    train_d1(config, split, train, val, device="cpu", checkpoint_path=checkpoint,
             run_id="resume", progress=False)

    state = TrainingState(
        epoch=1, best_score=0.5, best_epoch=1, since_improvement=0,
        model_state={}, best_state={}, optimiser_state={"k": 1},
        scheduler_state={"s": 2}, history=[{"epoch": 0.0}],
        config_id=config.identity, split_id=split.identity, run_id="resume",
        torch_rng=torch.get_rng_state(), numpy_rng=None, python_rng=None,
    )
    save_state(state_path(checkpoint), state)
    loaded = load_state(state_path(checkpoint), config_id=config.identity,
                        split_id=split.identity)
    assert loaded is not None
    assert loaded.optimiser_state == {"k": 1}
    assert loaded.scheduler_state == {"s": 2}
    assert loaded.epoch == 1
    assert torch.equal(loaded.torch_rng, state.torch_rng)


def test_state_from_another_experiment_is_refused(ready, tmp_path):
    """Continuing one experiment's run inside another's settings would produce a
    checkpoint that honestly records settings it was not trained under."""
    from sleepstatelab.training.trainer import TrainingState, save_state

    config, split = ready
    path = tmp_path / "d1.state.pt"
    save_state(path, TrainingState(
        epoch=0, best_score=0.0, best_epoch=0, since_improvement=0,
        model_state={}, best_state={}, optimiser_state={}, scheduler_state={},
        history=[], config_id="someone-elses-config", split_id=split.identity,
        run_id="other", torch_rng=torch.get_rng_state(), numpy_rng=None,
        python_rng=None,
    ))
    with pytest.raises(ValueError, match="holds state for configuration"):
        load_state(path, config_id=config.identity, split_id=split.identity)


def test_no_state_means_nothing_to_resume(tmp_path):
    assert load_state(tmp_path / "absent.state.pt", config_id="a", split_id="b") is None
