"""End to end on generated data: prepare, split, train, checkpoint, predict, score.

Marked synthetic throughout. These tests assert that the pipeline is connected
and that its guarantees hold; they assert nothing about sleep.
"""

from __future__ import annotations

import numpy as np
import pytest

from sleepstatelab.data.prepare import load_cached, prepare
from sleepstatelab.data.splits import grouped_split
from sleepstatelab.evaluation.metrics import evaluate_predictions
from sleepstatelab.evaluation.predictions import PredictionWriter, read_predictions
from sleepstatelab.training.checkpoint import load_checkpoint
from sleepstatelab.training.dataset import build_datasets
from sleepstatelab.training.trainer import predict, train_d1

pytestmark = pytest.mark.synthetic


@pytest.fixture
def prepared(small_config):
    prepare(small_config, progress=False)
    records = load_cached(small_config)
    participants = sorted({r.participant_id for r in records})
    split = grouped_split(participants, seed=0, name="test")
    return small_config, split


def test_prepare_then_load_gives_the_same_epochs(small_config):
    report = prepare(small_config, progress=False)
    records = load_cached(small_config)
    assert sum(r.n_epochs for r in records) == report.stored_epochs


def test_datasets_are_participant_disjoint(prepared):
    config, split = prepared
    train, val, test, stats = build_datasets(config, split)
    assert not set(train.participants) & set(test.participants)
    assert not set(val.participants) & set(test.participants)
    assert not set(stats.fitted_on) & set(split.test)
    assert not set(stats.fitted_on) & set(split.val)


def test_class_weights_come_from_training_only(prepared):
    config, split = prepared
    train, _, test, _ = build_datasets(config, split)
    weights = train.class_weights("inverse_frequency")
    assert weights.shape == (5,)
    assert np.isfinite(weights).all()
    # Recomputing from the test set would give different numbers; the point is
    # that the trainer is only ever handed the training dataset's.
    assert not np.allclose(weights, test.class_weights("inverse_frequency"))


@pytest.mark.slow
def test_train_checkpoint_reload_predict_and_score(prepared, tmp_path):
    config, split = prepared
    train, val, test, _ = build_datasets(config, split)

    model, checkpoint, history = train_d1(
        config,
        split,
        train,
        val,
        device="cpu",
        checkpoint_path=tmp_path / "d1.pt",
        run_id="unit",
        progress=False,
    )
    assert history.rows
    assert checkpoint.split_id == split.identity
    assert checkpoint.channels == tuple(config.data.channels)
    assert checkpoint.train_participants == train.participants
    assert set(checkpoint.train_participants) & set(split.test) == set()

    reloaded, loaded = load_checkpoint(
        tmp_path / "d1.pt",
        expect_channels=tuple(config.data.channels),
        expect_preprocessing_id=config.preprocessing_identity,
    )
    assert loaded.config_id == checkpoint.config_id
    before = predict(model, test, device="cpu")
    after = predict(reloaded, test, device="cpu")
    assert np.allclose(before, after, atol=1e-6), "a reloaded model must predict the same"

    path = tmp_path / "predictions.csv"
    with PredictionWriter(path) as writer:
        writer.write(
            run_id="unit",
            model="D1",
            split_id=split.identity,
            split_part="test",
            seed=0,
            participant_ids=[e.participant_id for e in test.entries],
            recording_ids=[e.recording_id for e in test.entries],
            epoch_indices=np.array([e.epoch_index for e in test.entries]),
            true_labels=test.y,
            probabilities=after,
            qc_flags=np.array([e.qc_flags for e in test.entries]),
        )
    rows = read_predictions(path)
    assert len(rows) == len(test)
    assert {row["participant_id"] for row in rows} == set(test.participants)

    results = evaluate_predictions(path, split_part="test")
    assert len(results) == 1
    assert 0.0 <= results[0].participant_macro_f1_mean <= 1.0


def test_checkpoint_refuses_a_different_channel_order(prepared, tmp_path):
    config, split = prepared
    train, val, _, _ = build_datasets(config, split)
    train_d1(
        config,
        split,
        train,
        val,
        device="cpu",
        checkpoint_path=tmp_path / "d1.pt",
        run_id="unit",
        progress=False,
    )
    with pytest.raises(ValueError, match="channels"):
        load_checkpoint(tmp_path / "d1.pt", expect_channels=("EEG Pz-Oz", "EEG Fpz-Cz"))
    with pytest.raises(ValueError, match="preprocessing"):
        load_checkpoint(tmp_path / "d1.pt", expect_preprocessing_id="not-the-one")


@pytest.mark.slow
def test_prediction_uses_the_checkpoints_own_normalisation(prepared, tmp_path):
    """A label-budget run fits its statistics on a subset. Re-fitting them at
    prediction time would show the model signals it was never trained on."""
    from sleepstatelab.data.preprocess import NormalizationStats

    config, split = prepared
    subset = split.train[:1]
    train, val, _, stats = build_datasets(config, split, train_participants=subset)
    assert stats.fitted_on == subset

    _, checkpoint, _ = train_d1(
        config,
        split,
        train,
        val,
        device="cpu",
        checkpoint_path=tmp_path / "budget.pt",
        run_id="budget",
        progress=False,
    )
    stored = NormalizationStats(
        method=checkpoint.normalization["method"],
        channels=tuple(checkpoint.normalization["channels"]),
        centre=tuple(checkpoint.normalization["centre"]),
        scale=tuple(checkpoint.normalization["scale"]),
        fitted_on=tuple(checkpoint.normalization["fitted_on"]),
        n_epochs=checkpoint.normalization["n_epochs"],
        clip_sigma=checkpoint.normalization["clip_sigma"],
    )
    assert stored.identity == stats.identity

    _, _, test_with_stored, used = build_datasets(config, split, stats=stored)
    assert used.identity == stats.identity
    _, _, test_refitted, refitted = build_datasets(config, split)
    assert refitted.identity != stats.identity, "the fixture must make the two differ"
    assert not np.allclose(test_with_stored.x, test_refitted.x)


def test_prepare_stops_before_filling_the_disk(small_config, monkeypatch):
    """A truncated cache entry is worse than a missing one: nothing downstream
    checks it the way a published checksum checks a download."""
    import sleepstatelab.data.prepare as prepare_module

    monkeypatch.setattr(prepare_module, "free_bytes", lambda path: int(0.2e9))
    report = prepare_module.prepare(small_config, progress=False, min_free_gb=1.0)

    assert report.recordings == 0
    assert report.recordings_discovered > 0
    assert "below the" in report.stopped_early
    assert "INCOMPLETE" in report.summary()


def test_prepare_reports_completion_when_there_is_room(small_config):
    report = prepare(small_config, progress=False, min_free_gb=0.0)
    assert report.stopped_early == ""
    assert report.recordings == report.recordings_discovered
    assert "INCOMPLETE" not in report.summary()


def test_load_cached_honours_the_configured_nights(small_config, tmp_path):
    """The cache accumulates -- it is keyed by preprocessing, not by cohort --
    so an experiment pinned to first nights must not pick up second nights that
    were prepared later. This silently doubled an evaluation set once."""
    import dataclasses

    from sleepstatelab.data.prepare import load_cached, prepare
    from sleepstatelab.synthetic import make_cohort

    root = tmp_path / "edf"
    make_cohort(root, n_participants=3, nights=(1, 2), n_epochs=24, seed=11)
    config = dataclasses.replace(
        small_config,
        data=dataclasses.replace(
            small_config.data, root=str(root), cache_dir=str(tmp_path / "cache")
        ),
    )
    prepare(config, progress=False)
    assert len(load_cached(config)) == 6

    first_nights = dataclasses.replace(
        config, data=dataclasses.replace(config.data, nights=(1,))
    )
    loaded = load_cached(first_nights)
    assert len(loaded) == 3
    assert all(record.night == 1 for record in loaded)


def test_load_cached_honours_the_configured_participants(small_config, tmp_path):
    import dataclasses

    from sleepstatelab.data.prepare import load_cached, prepare
    from sleepstatelab.synthetic import make_cohort

    root = tmp_path / "edf"
    make_cohort(root, n_participants=4, n_epochs=24, seed=12)
    config = dataclasses.replace(
        small_config,
        data=dataclasses.replace(
            small_config.data, root=str(root), cache_dir=str(tmp_path / "cache")
        ),
    )
    prepare(config, progress=False)
    assert len(load_cached(config)) == 4

    two = dataclasses.replace(
        config, data=dataclasses.replace(config.data, participants=("SC400", "SC402"))
    )
    assert sorted(r.participant_id for r in load_cached(two)) == ["SC400", "SC402"]


def test_materialised_and_in_memory_datasets_agree(small_config, tmp_path):
    """The memory-mapped store must be an implementation detail, not a change.

    It exists because the cohort's epochs do not fit in RAM beside the model;
    if it altered a single value, every result before and after it would be
    incomparable.
    """
    import numpy as np

    from sleepstatelab.data.prepare import load_cached, prepare, reject_mask_flags
    from sleepstatelab.data.preprocess import bandpass, fit_normalization
    from sleepstatelab.training.dataset import EpochDataset

    prepare(small_config, progress=False)
    records = load_cached(small_config)
    reject = reject_mask_flags(tuple(small_config.preprocess.qc_reject))
    stats = fit_normalization(
        [
            bandpass(r.signals[r.eligible(reject)], r.sampling_rate_hz, small_config.preprocess)
            for r in records
        ],
        [r.participant_id for r in records],
        channels=tuple(small_config.data.channels),
        config=small_config.preprocess,
    )

    in_memory = EpochDataset(
        records, config=small_config, stats=stats, reject_flags=reject
    )
    mapped = EpochDataset(
        records,
        config=small_config,
        stats=stats,
        reject_flags=reject,
        store_dir=tmp_path / "store",
    )
    assert np.array_equal(in_memory.y, mapped.y)
    assert np.allclose(np.asarray(in_memory.x), np.asarray(mapped.x))
    assert mapped.x.shape == in_memory.x.shape

    first_x, first_y = in_memory[3]
    second_x, second_y = mapped[3]
    assert first_y == second_y
    assert np.allclose(first_x.numpy(), second_x.numpy())


def test_the_store_is_reused_rather_than_rewritten(small_config, tmp_path):
    """The second model trained on a split must start immediately."""
    from sleepstatelab.data.prepare import load_cached, prepare, reject_mask_flags
    from sleepstatelab.data.preprocess import fit_normalization
    from sleepstatelab.training.dataset import EpochDataset

    prepare(small_config, progress=False)
    records = load_cached(small_config)
    reject = reject_mask_flags(tuple(small_config.preprocess.qc_reject))
    stats = fit_normalization(
        [r.signals[r.eligible(reject)] for r in records],
        [r.participant_id for r in records],
        channels=tuple(small_config.data.channels),
        config=small_config.preprocess,
    )
    store = tmp_path / "store"
    EpochDataset(records, config=small_config, stats=stats, reject_flags=reject, store_dir=store)
    written = sorted(store.glob("epochs-*.npy"))
    assert len(written) == 1
    stamp = written[0].stat().st_mtime_ns

    EpochDataset(records, config=small_config, stats=stats, reject_flags=reject, store_dir=store)
    assert sorted(store.glob("epochs-*.npy")) == written
    assert written[0].stat().st_mtime_ns == stamp
