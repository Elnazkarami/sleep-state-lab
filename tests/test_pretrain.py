"""Masked reconstruction, and the leak it must not have.

The single property everything about D3 depends on: **a hidden value must never
reach the encoder.** A masked autoencoder that leaks trains beautifully and
learns nothing, and the leak is invisible in the loss curve -- it looks like a
model that is good at its task. So it is asserted directly, several ways.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sleepstatelab.config import Config, PretrainConfig
from sleepstatelab.models.encoder import EpochEncoder
from sleepstatelab.models.pretrain import MaskedReconstruction, patch_mask

pytestmark = pytest.mark.synthetic


def _model(**kwargs) -> MaskedReconstruction:
    torch.manual_seed(0)
    return MaskedReconstruction(n_channels=2, n_samples=3000, **kwargs)


# --------------------------------------------------------------------------
# The mask itself.
# --------------------------------------------------------------------------


def test_the_mask_hides_whole_patches_and_the_right_number_of_them():
    mask = patch_mask(3000, 15, 100, batch=8, generator=torch.Generator().manual_seed(0))
    assert mask.shape == (8, 3000)
    assert mask.sum(dim=1).unique().tolist() == [1500]
    # Every hidden region is a whole patch: within a patch the mask is constant.
    patched = mask.reshape(8, 30, 100)
    assert ((patched.all(dim=2)) | (~patched.any(dim=2))).all()


def test_the_mask_differs_between_epochs_in_a_batch():
    mask = patch_mask(3000, 15, 100, batch=4, generator=torch.Generator().manual_seed(0))
    assert len({tuple(row.tolist()) for row in mask.reshape(4, 30, 100)[:, :, 0]}) > 1


def test_an_impossible_mask_is_refused():
    with pytest.raises(ValueError, match="no whole patch"):
        patch_mask(50, 1, 100, batch=1)
    with pytest.raises(ValueError, match="cannot mask"):
        patch_mask(3000, 31, 100, batch=1)


# --------------------------------------------------------------------------
# The leak. These are the tests this file exists for.
# --------------------------------------------------------------------------


def test_hidden_values_never_reach_the_encoder():
    """Change the signal inside the hidden patches; the encoder's input must be
    identical, element for element."""
    model = _model()
    x = torch.randn(4, 2, 3000)

    first = model.apply_mask(x, torch.Generator().manual_seed(7))
    changed = x.clone()
    changed[first.masked] = torch.randn(int(first.masked.sum())) * 100.0
    second = model.apply_mask(changed, torch.Generator().manual_seed(7))

    assert torch.equal(first.masked, second.masked), "the fixture must hide the same patches"
    assert torch.equal(first.visible, second.visible)
    assert not torch.equal(first.target, second.target), "the targets should differ"


def test_hidden_values_do_not_move_the_embedding():
    """The same claim one level up: the representation cannot depend on them."""
    model = _model().eval()
    x = torch.randn(2, 2, 3000)
    first = model.apply_mask(x, torch.Generator().manual_seed(3))
    changed = x.clone()
    changed[first.masked] = torch.randn(int(first.masked.sum())) * 50.0
    second = model.apply_mask(changed, torch.Generator().manual_seed(3))

    with torch.no_grad():
        assert torch.allclose(
            model.encoder(first.visible).embedding,
            model.encoder(second.visible).embedding,
            atol=1e-6,
        )


def test_hidden_samples_are_replaced_not_attenuated():
    """Every hidden sample equals the learned mask value for its channel. A
    model that multiplied by zero would pass a weaker test than this one, and a
    model that scaled the original would pass neither."""
    model = _model()
    x = torch.randn(3, 2, 3000) * 20.0
    batch = model.apply_mask(x, torch.Generator().manual_seed(1))
    for channel in range(2):
        hidden = batch.visible[:, channel][batch.masked[:, channel]]
        assert torch.allclose(
            hidden, model.mask_value[0, channel, 0].expand_as(hidden), atol=1e-7
        )


def test_visible_samples_are_untouched():
    model = _model()
    x = torch.randn(3, 2, 3000)
    batch = model.apply_mask(x, torch.Generator().manual_seed(2))
    assert torch.equal(batch.visible[~batch.masked], x[~batch.masked])


# --------------------------------------------------------------------------
# The loss.
# --------------------------------------------------------------------------


def test_the_loss_ignores_visible_samples():
    """Scoring the visible ones would reward copying."""
    model = _model()
    x = torch.randn(2, 2, 3000)
    batch = model.apply_mask(x, torch.Generator().manual_seed(4))
    prediction = torch.zeros_like(x)

    baseline = model.loss(prediction, batch)
    moved = prediction.clone()
    moved[~batch.masked] += 1000.0
    assert model.loss(moved, batch) == pytest.approx(float(baseline), rel=1e-6)


def test_the_loss_counts_masked_samples():
    model = _model()
    x = torch.randn(2, 2, 3000)
    batch = model.apply_mask(x, torch.Generator().manual_seed(5))
    prediction = batch.target.clone()
    assert model.loss(prediction, batch) == pytest.approx(0.0, abs=1e-9)
    prediction[batch.masked] += 2.0
    assert model.loss(prediction, batch) == pytest.approx(4.0, rel=1e-5)


def test_the_loss_can_exclude_invalid_samples():
    """A reconstruction loss that trains on padding produces a plausible curve
    and a meaningless encoder."""
    model = _model()
    x = torch.randn(1, 2, 3000)
    batch = model.apply_mask(x, torch.Generator().manual_seed(6))
    prediction = batch.target.clone()
    prediction[batch.masked] += 2.0
    valid = torch.zeros_like(batch.masked)
    assert model.loss(prediction, batch, valid=valid) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# The encoder that comes out.
# --------------------------------------------------------------------------


def test_pretraining_does_not_change_the_encoder_architecture():
    """D3's claim rests on the backbone being the one D2 trains."""
    plain = EpochEncoder(in_channels=2)
    model = _model()
    assert model.encoder.n_parameters() == plain.n_parameters() == 488_832
    assert type(model.encoder) is type(plain)


def test_a_supplied_encoder_is_used_as_is():
    encoder = EpochEncoder(in_channels=2)
    model = MaskedReconstruction(encoder=encoder, n_channels=2, n_samples=3000)
    assert model.encoder is encoder


def test_the_decoder_is_small_relative_to_the_encoder():
    """A strong decoder moves the work out of the part that is kept."""
    counts = _model().n_parameters()
    assert counts["decoder"] < counts["encoder"] / 10


def test_the_patch_is_aligned_to_the_encoders_tokens():
    """One token per patch. A decoder that had to smooth across tokens could not
    represent a spindle, and the encoder would learn to discard one."""
    model = _model()
    assert model.patch_samples == 3000 // model.n_tokens
    assert model.n_patches == model.n_tokens
    assert model.covered_samples == model.n_tokens * model.patch_samples


def test_the_uncovered_tail_is_never_masked_and_never_scored():
    """The tokens cover 2,976 of 3,000 samples; the remainder must not appear in
    the loss, or the model is scored on something it cannot draw."""
    model = _model()
    x = torch.randn(4, 2, 3000)
    batch = model.apply_mask(x, torch.Generator().manual_seed(0))
    assert not bool(batch.masked[..., model.covered_samples :].any())

    prediction = batch.target.clone()
    prediction[..., model.covered_samples :] += 1000.0
    assert model.loss(prediction, batch) == pytest.approx(0.0, abs=1e-9)


def test_the_reconstruction_can_represent_a_fast_rhythm():
    """A decoder that interpolated between tokens would be limited to about half
    a hertz, and sleep spindles are at 12-16 Hz. This asserts the decoder can at
    least produce such a rhythm: fit one epoch and check the residual."""
    torch.manual_seed(0)
    model = _model()
    t = torch.arange(3000, dtype=torch.float32) / 100.0
    target = torch.stack([torch.sin(2 * torch.pi * 13.0 * t)] * 2).unsqueeze(0)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3)
    visible = target.clone()
    for _ in range(200):
        optimiser.zero_grad(set_to_none=True)
        loss = ((model.reconstruct(visible) - target) ** 2).mean()
        loss.backward()
        optimiser.step()
    assert float(loss.item()) < 0.1, f"13 Hz residual {float(loss.item()):.4f}"


@pytest.mark.slow
def test_reconstruction_learns_something_on_a_structured_signal():
    """An optimisation check: on sine waves the masked patches are predictable,
    so the loss must fall well below predicting the mean. This says the gradient
    path works; it says nothing about EEG."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    t = np.arange(3000) / 100.0
    x = np.stack(
        [
            np.stack([np.sin(2 * np.pi * f * t + p) for f in (1.0, 10.0)])
            for p in rng.uniform(0, 6.28, 16)
        ]
    ).astype(np.float32)
    data = torch.from_numpy(x)

    model = _model(decoder_width=64)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(0)

    model.train()
    first = last = None
    for step in range(300):
        batch = model.apply_mask(data, generator)
        optimiser.zero_grad(set_to_none=True)
        loss = model.loss(model.reconstruct(batch.visible), batch)
        loss.backward()
        optimiser.step()
        if step == 0:
            first = float(loss.item())
        last = float(loss.item())
    assert last < first / 2.0, f"reconstruction loss went {first:.4f} -> {last:.4f}"


def test_config_carries_the_pretraining_settings():
    config = Config()
    # 0 means "derive the patch from the encoder", which is what keeps the patch
    # aligned to the tokens that have to reconstruct it.
    assert config.pretrain.patch_samples == 0
    assert 0 < config.pretrain.mask_ratio < 1
    # Pretraining settings must not change the epoch cache's identity.
    other = Config(pretrain=PretrainConfig(mask_ratio=0.75))
    assert other.preprocessing_identity == config.preprocessing_identity
    assert other.identity != config.identity
