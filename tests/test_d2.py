"""D2: the temporal model, its masking, and the guarantees the comparison needs.

The rules asserted here are the ones that decide whether a D2-minus-D1 number
means anything: that a window never crosses a recording or a gap, that an absent
neighbour is stated rather than fabricated, that only the central epoch is
predicted, and that D2 is trained with exactly D1's settings.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sleepstatelab.data.prepare import load_cached, prepare, reject_mask_flags
from sleepstatelab.data.splits import grouped_split
from sleepstatelab.labels import STAGES
from sleepstatelab.models.d1 import D1Classifier
from sleepstatelab.models.d2 import D2Classifier
from sleepstatelab.models.encoder import EpochEncoder
from sleepstatelab.training.windows import (
    build_window_datasets,
    plan_windows,
    shuffle_context,
)

pytestmark = pytest.mark.synthetic


# --------------------------------------------------------------------------
# The window plan: where the masking rules actually live.
# --------------------------------------------------------------------------


def test_a_gap_is_masked_not_bridged():
    """Epoch 4 was excluded. Every window that would have contained it must show
    an absence there, not the nearest surviving epoch."""
    index = np.array([0, 1, 2, 3, 5, 6, 7, 8])
    rows, mask = plan_windows(index, 5)

    centred_on_3 = index.tolist().index(3)
    assert mask[centred_on_3].tolist() == [True, True, True, False, True]
    assert rows[centred_on_3][3] == -1
    # Position 4 of that window is epoch 5, which is genuinely at centre+2.
    assert index[rows[centred_on_3][4]] == 5


def test_a_boundary_is_masked_the_same_way_as_a_gap():
    index = np.arange(6)
    rows, mask = plan_windows(index, 5)
    assert mask[0].tolist() == [False, False, True, True, True]
    assert mask[-1].tolist() == [True, True, True, False, False]
    assert (rows[0][:2] == -1).all()


def test_every_present_position_is_at_exactly_the_expected_distance():
    index = np.array([0, 1, 5, 6, 7, 20, 21])
    rows, mask = plan_windows(index, 11)
    half = 5
    for row, centre in enumerate(index):
        for position in range(11):
            if not mask[row, position]:
                continue
            assert index[rows[row, position]] == centre + position - half


def test_the_centre_is_always_present():
    index = np.array([0, 7, 8, 30])
    _, mask = plan_windows(index, 11)
    assert mask[:, 5].all()


def test_even_context_is_refused():
    with pytest.raises(ValueError, match="odd"):
        plan_windows(np.arange(4), 4)


# --------------------------------------------------------------------------
# The model.
# --------------------------------------------------------------------------


def test_shapes_and_centre_only_prediction():
    model = D2Classifier(in_channels=2, context=11)
    x = torch.randn(3, 11, 2, 3000)
    mask = torch.ones(3, 11, dtype=torch.bool)
    assert model(x, mask).shape == (3, len(STAGES))


def test_only_the_central_epoch_decides_when_context_is_absent():
    """With every neighbour masked, D2 is a function of the central epoch alone.
    Changing a masked position must not move the logits at all."""
    torch.manual_seed(0)
    model = D2Classifier(in_channels=2, context=5).eval()
    x = torch.randn(2, 5, 2, 3000)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[:, 2] = True

    with torch.no_grad():
        before = model(x, mask)
        changed = x.clone()
        changed[:, 0] = torch.randn(2, 2, 3000)
        changed[:, 4] = torch.randn(2, 2, 3000)
        after = model(changed, mask)
    assert torch.allclose(before, after, atol=1e-5)


def test_a_present_neighbour_does_change_the_answer():
    """The other half of the previous test: unmasked context is actually used."""
    torch.manual_seed(0)
    model = D2Classifier(in_channels=2, context=5).eval()
    x = torch.randn(2, 5, 2, 3000)
    mask = torch.ones(2, 5, dtype=torch.bool)
    with torch.no_grad():
        before = model(x, mask)
        changed = x.clone()
        changed[:, 0] = torch.randn(2, 2, 3000)
        after = model(changed, mask)
    assert not torch.allclose(before, after, atol=1e-5)


def test_a_masked_centre_is_refused():
    model = D2Classifier(in_channels=2, context=5)
    mask = torch.ones(1, 5, dtype=torch.bool)
    mask[0, 2] = False
    with pytest.raises(ValueError, match="central position"):
        model(torch.randn(1, 5, 2, 3000), mask)


def test_even_context_model_is_refused():
    with pytest.raises(ValueError, match="odd"):
        D2Classifier(in_channels=2, context=10)


def test_d2_reuses_the_d1_encoder_unchanged():
    """The seam the whole comparison rests on: the same backbone object, with
    the same parameter count, under a different temporal stack."""
    encoder = EpochEncoder(in_channels=2)
    d1 = D1Classifier.from_encoder(encoder)
    d2 = D2Classifier.from_encoder(encoder, context=11)
    assert d1.encoder is d2.encoder
    assert d2.n_parameters()["encoder"] == d1.n_parameters()["encoder"] == 488_832

    x = torch.randn(2, 2, 3000)
    with torch.no_grad():
        through_d1 = d1.encoder(x).embedding
        through_d2 = d2.encoder(x).embedding
    assert torch.allclose(through_d1, through_d2)


def test_the_two_inference_paths_agree():
    """`classify` on pre-computed embeddings must equal `forward` on the window.

    This is what makes it safe to encode each epoch of a recording once instead
    of eleven times.
    """
    torch.manual_seed(0)
    model = D2Classifier(in_channels=2, context=5).eval()
    x = torch.randn(2, 5, 2, 3000)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[1, 0] = False
    with torch.no_grad():
        direct = model(x, mask)
        staged = model.classify(model.embed_window(x), mask)
    assert torch.allclose(direct, staged, atol=1e-6)


# --------------------------------------------------------------------------
# The dataset, on generated recordings.
# --------------------------------------------------------------------------


@pytest.fixture
def windows(small_config):
    prepare(small_config, progress=False)
    records = load_cached(small_config)
    split = grouped_split(sorted({r.participant_id for r in records}), seed=0, name="test")
    return small_config, split


def test_windows_never_cross_a_recording_or_participant(windows):
    config, split = windows
    train, _, test, _ = build_window_datasets(config, split, context=11)
    assert not set(train.participants) & set(test.participants)
    # Every window's rows index into exactly one recording's block.
    for recording, row in train.index:
        rows = train.rows[recording][row]
        present = rows[rows >= 0]
        assert present.max() < train.blocks[recording].shape[0]


def test_dataset_masks_match_the_stored_epoch_indices(windows):
    config, split = windows
    _, _, test, _ = build_window_datasets(config, split, context=11)
    reject = reject_mask_flags(tuple(config.preprocess.qc_reject))
    records = {r.recording_id: r for r in load_cached(config, split.test)}

    for item in range(len(test)):
        recording, row = test.index[item]
        entry = test.entries[item]
        stored = records[entry.recording_id]
        index = stored.epoch_index[stored.eligible(reject)]
        rows = test.rows[recording][row]
        for position in range(test.context):
            wanted = entry.epoch_index + position - test.context // 2
            if rows[position] < 0:
                assert wanted not in index.tolist()
            else:
                assert index[rows[position]] == wanted


def test_absent_positions_are_zero_and_flagged(windows):
    config, split = windows
    _, _, test, _ = build_window_datasets(config, split, context=11)
    # The first window of a recording necessarily has absences at the front.
    x, mask, _ = test[0]
    assert not bool(mask[0])
    assert torch.count_nonzero(x[0]) == 0
    assert bool(mask[test.context // 2])


def test_context_coverage_is_reported_and_sane(windows):
    config, split = windows
    train, _, _, _ = build_window_datasets(config, split, context=11)
    coverage = train.context_coverage()
    assert 0.0 < coverage <= 1.0


def test_class_weights_match_the_epoch_dataset(windows):
    """D1 and D2 must be weighted identically or their difference is not context."""
    from sleepstatelab.training.dataset import build_datasets

    config, split = windows
    epochs_train, _, _, _ = build_datasets(config, split)
    windows_train, _, _, _ = build_window_datasets(config, split, context=11)
    assert np.allclose(
        epochs_train.class_weights("inverse_frequency"),
        windows_train.class_weights("inverse_frequency"),
    )
    assert np.array_equal(epochs_train.class_counts(), windows_train.class_counts())


def test_shuffling_context_keeps_the_centre_and_the_amount_of_context(windows):
    config, split = windows
    _, _, test, _ = build_window_datasets(config, split, context=11)
    centre = test.context // 2
    before_centre = [rows[:, centre].copy() for rows in test.rows]
    before_coverage = test.context_coverage()

    shuffle_context(test, seed=1)

    assert test.context_coverage() == pytest.approx(before_coverage)
    for rows, expected in zip(test.rows, before_centre, strict=True):
        assert np.array_equal(rows[:, centre], expected)


@pytest.mark.slow
def test_d2_can_overfit_a_tiny_batch():
    """The optimisation check for the temporal stack, as D1 has for its own.

    Dropout is switched off here. The question is whether gradients reach every
    part of the model -- encoder, positional embeddings, transformer, head --
    and dropout only makes a memorisation target take longer to hit for reasons
    that have nothing to do with that. With dropout at its trained values the
    same batch is memorised in about 400 steps; the check is run without it so
    the suite does not spend a minute proving something it already knows.
    """
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n, context = 8, 5
    t = np.arange(3000) / 100.0
    y = np.array([i % len(STAGES) for i in range(n)], dtype=np.int64)
    x = np.zeros((n, context, 2, 3000), dtype=np.float32)
    for row in range(n):
        for position in range(context):
            frequency = 1.0 + 3.0 * y[row]
            x[row, position] = np.sin(2 * np.pi * frequency * t + rng.uniform(0, 6.28))

    model = D2Classifier(
        in_channels=2,
        context=context,
        embedding_dim=64,
        dropout=0.0,
        transformer_dropout=0.0,
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = torch.nn.CrossEntropyLoss()
    inputs = torch.from_numpy(x)
    mask = torch.ones(n, context, dtype=torch.bool)
    targets = torch.from_numpy(y)

    model.train()
    for _ in range(200):
        optimiser.zero_grad(set_to_none=True)
        loss = criterion(model(inputs, mask), targets)
        loss.backward()
        optimiser.step()

    model.eval()
    with torch.no_grad():
        accuracy = float((model(inputs, mask).argmax(dim=1) == targets).float().mean())
    assert accuracy == 1.0
    assert float(loss.item()) < 0.05


# --------------------------------------------------------------------------
# The segment path: the same model, with shared encodings.
# --------------------------------------------------------------------------


def test_segment_forward_equals_window_forward():
    """The claim the fast path rests on, checked directly on the model.

    Overlapping windows share ten of their eleven epochs. Encoding a stretch
    once and gathering windows out of it must give exactly what encoding each
    window separately gives, or the speed-up is a different model.
    """
    torch.manual_seed(0)
    model = D2Classifier(in_channels=2, context=5, embedding_dim=32).eval()
    rows, centres = 12, 6
    signals = torch.randn(2, rows, 2, 3000)
    gather = (
        torch.stack([torch.arange(c, c + 5) for c in range(centres)])
        .unsqueeze(0)
        .expand(2, -1, -1)
        .contiguous()
    )
    mask = torch.ones(2, centres, 5, dtype=torch.bool)
    mask[0, 0, 0] = False

    with torch.no_grad():
        fast = model.forward_segment(signals, gather, mask)
        slow = torch.stack(
            [
                torch.stack(
                    [
                        model(
                            signals[b, gather[b, c]].unsqueeze(0),
                            mask[b, c].unsqueeze(0),
                        )[0]
                        for c in range(centres)
                    ]
                )
                for b in range(2)
            ]
        )
    assert torch.allclose(fast, slow, atol=1e-5)


def test_segments_cover_every_centre_exactly_once(windows):
    """No epoch trained on twice in a pass, and none skipped."""
    from sleepstatelab.training.windows import SegmentDataset

    config, split = windows
    window_train, _, _, stats = build_window_datasets(config, split, context=11)
    segment_train, _, _, _ = build_window_datasets(
        config, split, context=11, segments=True, centres_per_segment=8, stats=stats
    )
    assert isinstance(segment_train, SegmentDataset)

    ids = np.concatenate(
        [segment_train.centre_ids_of(i) for i in range(len(segment_train))]
    )
    live = np.sort(ids[ids >= 0])
    assert np.array_equal(live, np.arange(len(window_train)))
    assert segment_train.n_centres() == len(window_train)


def test_segments_cost_far_fewer_encodings(windows):
    config, split = windows
    segment_train, _, _, _ = build_window_datasets(
        config, split, context=11, segments=True, centres_per_segment=16
    )
    naive = segment_train.n_centres() * 11
    assert segment_train.encodings_per_pass() < naive / 2


def test_segment_and_window_predictions_agree(windows):
    """End to end: the two datasets, one model, the same probabilities.

    This is what makes the fast path usable for a result rather than only for a
    demonstration -- the validation score a run selects on is the same number
    either way.
    """
    from sleepstatelab.training.trainer import build_d2, predict

    config, split = windows
    _, _, window_test, stats = build_window_datasets(config, split, context=11)
    _, _, segment_test, _ = build_window_datasets(
        config, split, context=11, segments=True, centres_per_segment=8, stats=stats
    )
    torch.manual_seed(0)
    model = build_d2(config).eval()

    slow = predict(model, window_test, device="cpu", batch_size=32)
    fast = predict(model, segment_test, device="cpu", batch_size=4)
    assert fast.shape == slow.shape
    assert np.allclose(fast, slow, atol=1e-4)


def test_segment_dataset_reports_the_same_class_counts(windows):
    config, split = windows
    window_train, _, _, stats = build_window_datasets(config, split, context=11)
    segment_train, _, _, _ = build_window_datasets(
        config, split, context=11, segments=True, centres_per_segment=8, stats=stats
    )
    assert np.array_equal(window_train.class_counts(), segment_train.class_counts())
    assert np.allclose(
        window_train.class_weights("inverse_frequency"),
        segment_train.class_weights("inverse_frequency"),
    )


def test_forward_accepts_both_dataset_layouts():
    """The training loop calls `model(*inputs)`; both datasets must fit that."""
    torch.manual_seed(0)
    model = D2Classifier(in_channels=2, context=5, embedding_dim=32).eval()
    windows = model(torch.randn(2, 5, 2, 3000), torch.ones(2, 5, dtype=torch.bool))
    assert windows.shape == (2, len(STAGES))

    gather = (
        torch.stack([torch.arange(c, c + 5) for c in range(4)])
        .unsqueeze(0)
        .expand(2, -1, -1)
        .contiguous()
    )
    segments = model(
        torch.randn(2, 10, 2, 3000), gather, torch.ones(2, 4, 5, dtype=torch.bool)
    )
    assert segments.shape == (2, 4, len(STAGES))


@pytest.mark.slow
def test_d2_trains_through_the_segment_path(windows, tmp_path):
    """One pass of real training on segments, end to end.

    The gap this closes: every earlier segment test went through `predict` or
    `forward_segment` directly, so the training loop's own call into the model
    was never exercised -- and it was broken.
    """
    from sleepstatelab.training.checkpoint import load_checkpoint
    from sleepstatelab.training.trainer import train_d2

    config, split = windows
    train, val, _, _ = build_window_datasets(
        config, split, context=11, segments=True, centres_per_segment=8
    )
    _, checkpoint, history = train_d2(
        config,
        split,
        train,
        val,
        device="cpu",
        checkpoint_path=tmp_path / "d2seg.pt",
        run_id="segment-unit",
        progress=False,
        loader_batch_size=2,
    )
    assert history.rows
    assert checkpoint.model_name == "D2"
    assert checkpoint.temporal_kwargs["context"] == 11
    # The configuration keeps counting examples; the loader's own batch is
    # recorded separately so the checkpoint does not misdescribe itself.
    assert checkpoint.notes["loader_batch_size"] == 2
    assert checkpoint.notes["examples_per_step"] == config.train.batch_size

    model, _ = load_checkpoint(
        tmp_path / "d2seg.pt",
        expect_channels=tuple(config.data.channels),
        expect_preprocessing_id=config.preprocessing_identity,
    )
    assert model.context == 11
