"""The probe controls: what a frozen encoder is, and that freezing works.

Two of the six controls the D3 comparison requires live here. Both are about
telling apart claims that are easy to confuse:

* a **pretrained encoder without temporal context** separates "the
  representation is better" from "a transformer helps";
* a **frozen encoder** separates what a representation already carries from what
  fine-tuning can find in it. Frozen random against frozen pretrained, same head
  on each, is the comparison that answers it.
"""

from __future__ import annotations

import pytest
import torch

from sleepstatelab.models.encoder import EpochEncoder
from sleepstatelab.training.trainer import build_d2, build_model, freeze

pytestmark = pytest.mark.synthetic


def test_freezing_stops_the_gradients_and_counts_what_it_froze():
    encoder = EpochEncoder(in_channels=2)
    frozen = freeze(encoder)
    assert frozen == 488_832
    assert not any(p.requires_grad for p in encoder.parameters())
    # Freezing twice must not double-count what was already frozen.
    assert freeze(encoder) == 0


def test_a_frozen_encoder_does_not_move_while_the_head_does(small_config):
    """The property the probe control needs: identical encoder weights before
    and after, and a head that actually trained."""
    torch.manual_seed(0)
    model = build_model(small_config)
    freeze(model.encoder)
    before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    head_before = model.head.linear.weight.clone()

    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.1
    )
    criterion = torch.nn.CrossEntropyLoss()
    x = torch.randn(8, 2, 3000)
    y = torch.randint(0, 5, (8,))
    for _ in range(5):
        optimiser.zero_grad(set_to_none=True)
        criterion(model(x), y).backward()
        optimiser.step()

    for key, value in model.encoder.state_dict().items():
        assert torch.equal(value, before[key]), f"{key} moved while frozen"
    assert not torch.equal(model.head.linear.weight, head_before)


def test_weight_decay_cannot_reach_a_frozen_encoder(small_config):
    """Passing every parameter to the optimiser would let weight decay shrink a
    frozen encoder even with no gradient, which is a slow and invisible way to
    destroy the thing being probed."""
    torch.manual_seed(0)
    model = build_model(small_config)
    freeze(model.encoder)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert len(trainable) == len(list(model.head.parameters()))


def test_a_supplied_encoder_reaches_both_models(small_config):
    """The pretrained backbone must arrive intact in D1 and in D2."""
    # The same construction the trainer uses, so an encoder built here is one
    # the models will accept.
    from sleepstatelab.training.trainer import encoder_kwargs

    encoder = EpochEncoder(**encoder_kwargs(small_config))
    marker = torch.full_like(encoder.stem[0].weight, 0.123)
    with torch.no_grad():
        encoder.stem[0].weight.copy_(marker)

    d1 = build_model(small_config, encoder=encoder)
    d2 = build_d2(small_config, encoder=encoder)
    assert d1.encoder is encoder
    assert d2.encoder is encoder
    assert torch.equal(d1.encoder.stem[0].weight, marker)
    assert torch.equal(d2.encoder.stem[0].weight, marker)


def test_an_encoder_of_the_wrong_width_is_refused(small_config):
    with pytest.raises(ValueError, match="embedding dimension"):
        build_d2(small_config, encoder=EpochEncoder(in_channels=2, embedding_dim=999))


@pytest.mark.slow
def test_training_records_that_the_encoder_was_frozen(small_config, tmp_path):
    """A checkpoint has to say it was a probe, or nobody can tell later."""
    from sleepstatelab.data.prepare import prepare
    from sleepstatelab.data.splits import grouped_split
    from sleepstatelab.training.dataset import build_datasets
    from sleepstatelab.training.trainer import train_d1

    prepare(small_config, progress=False)
    from sleepstatelab.data.prepare import load_cached

    records = load_cached(small_config)
    split = grouped_split(
        sorted({r.participant_id for r in records}), seed=0, name="probe"
    )
    train, val, _, _ = build_datasets(small_config, split)

    _, checkpoint, _ = train_d1(
        small_config,
        split,
        train,
        val,
        freeze_encoder=True,
        device="cpu",
        checkpoint_path=tmp_path / "probe.pt",
        run_id="probe",
        progress=False,
    )
    assert checkpoint.notes["frozen_encoder_parameters"] > 0
