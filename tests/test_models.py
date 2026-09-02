"""The encoder's contract, and that D1 can be optimised at all."""

from __future__ import annotations

import pytest
import torch

from sleepstatelab.labels import STAGES
from sleepstatelab.models.d1 import D1Classifier
from sleepstatelab.models.encoder import EpochEncoder

pytestmark = pytest.mark.synthetic


def test_shapes_are_the_documented_ones():
    model = D1Classifier(in_channels=2)
    x = torch.randn(3, 2, 3000)
    assert model(x).shape == (3, len(STAGES))
    out = model.represent(x)
    assert out.embedding.shape == (3, 128)
    assert out.tokens.shape[0] == 3 and out.tokens.shape[1] == 128


def test_logits_are_unnormalised():
    """Five raw logits, not probabilities: the loss applies the softmax."""
    model = D1Classifier(in_channels=2)
    logits = model(torch.randn(2, 2, 3000))
    assert not torch.allclose(logits.sum(dim=1), torch.ones(2), atol=1e-3)


def test_wrong_channel_count_is_refused():
    model = D1Classifier(in_channels=2)
    with pytest.raises(ValueError, match="channel"):
        model(torch.randn(2, 1, 3000))


def test_the_head_is_separable_from_the_backbone():
    """The property D2 and D3 depend on: the same encoder, a different head."""
    encoder = EpochEncoder(in_channels=2)
    first = D1Classifier.from_encoder(encoder)
    second = D1Classifier.from_encoder(encoder, n_classes=5, dropout=0.1)
    assert first.encoder is second.encoder
    assert first.head is not second.head


def test_parameter_count_is_stable():
    """A silent architecture change would move this. The number is documented."""
    model = D1Classifier(in_channels=2)
    counts = model.n_parameters()
    assert counts["encoder"] == 488_832
    assert counts["total"] == 489_477


def test_batchnorm_is_not_used():
    """Batch statistics would make inference depend on who else is in the batch."""
    model = D1Classifier(in_channels=2)
    assert not any(
        isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        for module in model.modules()
    )


@pytest.mark.slow
def test_d1_can_overfit_a_tiny_batch():
    """An optimisation check, not an experiment: a network that cannot memorise
    sixteen examples has a broken gradient path."""
    from sleepstatelab.smoke import overfit_check

    found = overfit_check(n=16, steps=250, device="cpu", seed=0)
    assert found["train_accuracy"] == 1.0
    assert found["final_loss"] < 0.05
