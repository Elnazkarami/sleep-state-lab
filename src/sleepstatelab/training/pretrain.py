"""Self-supervised pretraining: the loop that produces D3's encoder.

Three rules, each of which the corresponding experiment is void without.

**Training participants only.** Not "mostly", not "the unlabelled ones". The
participants the encoder saw are written into the checkpoint and checked against
the split when it is loaded, so a pretrained encoder cannot quietly be used on a
run that holds out someone it has already seen.

**Selection on a held-out reconstruction loss, never on a label.** The
pretraining stage has no access to stages at all -- not for its objective, not
for its stopping rule. The validation participants supply a reconstruction loss
only, and the number it selects on is that.

**No labels are read.** The dataset is built from the same cache with the same
eligibility rules, and its labels are ignored. That is a property of this file
and a test asserts it.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from sleepstatelab.config import Config
from sleepstatelab.data.splits import Split
from sleepstatelab.models.encoder import EpochEncoder
from sleepstatelab.models.pretrain import MaskedReconstruction
from sleepstatelab.provenance import code_revision
from sleepstatelab.training.checkpoint import (
    ENCODER_FORMAT,
    EncoderCheckpoint,
    save_encoder_checkpoint,
)
from sleepstatelab.training.dataset import EpochDataset
from sleepstatelab.training.trainer import TrainingHistory, encoder_kwargs, seed_everything


def build_pretrainer(config: Config, *, encoder: EpochEncoder | None = None) -> MaskedReconstruction:
    """The reconstruction model, wrapped around a fresh or supplied encoder."""
    return MaskedReconstruction(
        encoder=encoder if encoder is not None else EpochEncoder(**encoder_kwargs(config)),
        n_channels=len(config.data.channels),
        n_samples=config.samples_per_epoch,
        patch_samples=config.pretrain.patch_samples,
        mask_ratio=config.pretrain.mask_ratio,
        decoder_width=config.pretrain.decoder_width,
    )


@torch.no_grad()
def reconstruction_loss(
    model: MaskedReconstruction,
    dataset: EpochDataset,
    *,
    device: str,
    batch_size: int = 64,
    seed: int = 0,
) -> float:
    """Mean masked reconstruction error over a dataset, with a fixed mask.

    The generator is re-seeded for every evaluation, so two validation passes
    hide the same patches and the numbers they produce can be compared. A moving
    mask would make the validation curve a measure of which patches were drawn.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total = 0.0
    seen = 0
    for x, _ in loader:
        x = x.to(device)
        masked = model.apply_mask(x, generator)
        prediction = model.reconstruct(masked.visible)
        total += float(model.loss(prediction, masked).item()) * x.shape[0]
        seen += int(x.shape[0])
    return total / max(seen, 1)


def pretrain_encoder(
    config: Config,
    split: Split,
    train: EpochDataset,
    val: EpochDataset,
    *,
    device: str = "cpu",
    checkpoint_path: Path | str | None = None,
    run_id: str = "pretrain",
    progress: bool = True,
) -> tuple[EpochEncoder, EncoderCheckpoint, TrainingHistory]:
    """Pretrain by masked reconstruction and keep the best encoder."""
    forbidden = set(split.val) | set(split.test)
    saw = set(train.participants)
    leaked = sorted(saw & forbidden)
    if leaked:
        raise ValueError(
            f"pretraining was handed {leaked}, who are held out of split "
            f"{split.name!r}. Pretraining sees training participants only."
        )

    seed_everything(config.train.seed)
    model = build_pretrainer(config).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config.pretrain.learning_rate,
        weight_decay=config.pretrain.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(config.pretrain.epochs, 1)
    )
    loader = DataLoader(
        train,
        batch_size=config.pretrain.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.train.seed)

    history = TrainingHistory()
    best = float("inf")
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    since = 0

    for epoch in range(config.pretrain.epochs):
        model.train()
        started = time.time()
        total = 0.0
        seen = 0
        for batch, (x, _) in enumerate(loader):
            if config.pretrain.max_batches and batch >= config.pretrain.max_batches:
                break
            x = x.to(device)
            masked = model.apply_mask(x, generator)
            optimiser.zero_grad(set_to_none=True)
            loss = model.loss(model.reconstruct(masked.visible), masked)
            loss.backward()
            if config.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimiser.step()
            total += float(loss.item()) * x.shape[0]
            seen += int(x.shape[0])
        scheduler.step()

        validation = reconstruction_loss(
            model, val, device=device, seed=config.split.seed
        )
        history.add(
            epoch=epoch,
            train_loss=total / max(seen, 1),
            val_reconstruction_loss=validation,
            seconds=time.time() - started,
        )
        if progress:
            print(
                f"  epoch {epoch + 1}/{config.pretrain.epochs}  "
                f"train {total / max(seen, 1):.4f}  "
                f"val reconstruction {validation:.4f}  "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )

        if validation < best:
            best = validation
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()
            }
            best_epoch = epoch
            since = 0
        else:
            since += 1
            if since >= config.pretrain.early_stopping_patience:
                if progress:
                    print(
                        f"  stopping: no improvement in "
                        f"{config.pretrain.early_stopping_patience} epochs",
                        flush=True,
                    )
                break

    if best_state is None:
        raise RuntimeError("pretraining produced no encoder; did it run zero epochs?")
    encoder = EpochEncoder(**encoder_kwargs(config))
    encoder.load_state_dict(best_state)
    encoder.eval()

    checkpoint = EncoderCheckpoint(
        format=ENCODER_FORMAT,
        state_dict=best_state,
        encoder_kwargs=encoder_kwargs(config),
        channels=tuple(config.data.channels),
        preprocessing_id=config.preprocessing_identity,
        normalization=asdict(train.stats),
        split_id=split.identity,
        split_name=split.name,
        pretrain_participants=train.participants,
        seed=config.train.seed,
        code_revision=code_revision(),
        config_id=config.identity,
        config=config.to_dict(),
        objective="masked_patch_reconstruction",
        epoch_selected=best_epoch,
        metric_name="val_reconstruction_loss",
        metric_value=best,
        history=history.rows,
        notes={
            "run_id": run_id,
            "device": device,
            "n_parameters": build_pretrainer(config).n_parameters(),
            "patch_samples": config.pretrain.patch_samples,
            "mask_ratio": config.pretrain.mask_ratio,
            "patches_masked": model.n_patches_masked,
            "patches_total": model.n_patches,
            "train_epochs_available": len(train),
            "labels_read": False,
        },
    )
    if checkpoint_path is not None:
        save_encoder_checkpoint(checkpoint_path, checkpoint)
    _ = np  # numpy is imported for the type of the arrays this module never mutates
    return encoder, checkpoint, history
