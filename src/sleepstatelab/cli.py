"""The supervised loop for D1, and the rule for stopping it.

**The stopping rule is validation participant macro-F1.** Not loss, because a
class-weighted cross-entropy on a 70%-wake problem improves for a while by
becoming better at wake; not test anything, ever. The checkpoint that is kept is
the epoch with the best validation score, and the epoch number is recorded so a
run that stopped at its first epoch is visible as such.

**Seeding is explicit and complete.** Python, NumPy and torch are all seeded
from the configured value, and the seed is recorded in the checkpoint. This does
not make a CUDA run bit-identical -- it is not claimed to; on CPU it does.
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sleepstatelab.config import Config
from sleepstatelab.data.splits import Split
from sleepstatelab.evaluation.metrics import participant_macro_f1
from sleepstatelab.labels import STAGES
from sleepstatelab.models.d1 import D1Classifier
from sleepstatelab.models.encoder import EpochEncoder
from sleepstatelab.provenance import code_revision
from sleepstatelab.training.checkpoint import CHECKPOINT_FORMAT, Checkpoint, save_checkpoint
from sleepstatelab.training.windows import IGNORE_LABEL


def seed_everything(seed: int) -> None:
    """Seed every generator this package draws from."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainingHistory:
    """What happened, per pass over the training set."""

    rows: list[dict[str, float]] = field(default_factory=list)

    def add(self, **values: float) -> None:
        self.rows.append({k: float(v) for k, v in values.items()})

    def best(self, key: str = "val_participant_macro_f1") -> dict[str, float]:
        return max(self.rows, key=lambda row: row.get(key, float("-inf")))


def encoder_kwargs(config: Config) -> dict[str, Any]:
    """The encoder's constructor arguments, derived from the configuration once."""
    return {
        "in_channels": len(config.data.channels),
        "stem_channels": config.model.stem_channels,
        "stem_kernel": config.model.stem_kernel,
        "stem_stride": config.model.stem_stride,
        "block_channels": tuple(config.model.block_channels),
        "block_kernel": config.model.block_kernel,
        "embedding_dim": config.model.embedding_dim,
    }


def build_model(config: Config) -> D1Classifier:
    return D1Classifier.from_encoder(
        EpochEncoder(**encoder_kwargs(config)),
        n_classes=config.model.n_classes,
        dropout=config.model.dropout,
    )


def split_batch(batch: Sequence[Any], device: str) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Separate a batch into the model's inputs and its targets.

    D1's dataset yields ``(x, y)`` and D2's yields ``(x, mask, y)``. Rather than
    two training loops that can drift apart -- and they would, which is exactly
    what would make a D2-minus-D1 comparison meaningless -- the loop takes the
    last element as the target and passes the rest to the model.
    """
    *inputs, targets = batch
    return [tensor.to(device) for tensor in inputs], targets.to(device)


def flatten_logits(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold a per-segment prediction into the flat shape the loss expects.

    A segment model returns ``[batch, centres, classes]`` against
    ``[batch, centres]`` targets; everything else returns ``[batch, classes]``
    against ``[batch]``. Padded centres carry ``IGNORE_LABEL``, which the loss
    is configured to skip, so folding them in costs nothing.
    """
    if logits.dim() == 3:
        return logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    return logits, targets


@torch.no_grad()
def predict(
    model: nn.Module,
    dataset: Dataset,
    *,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    """Probabilities ``[n, 5]`` in stage order, one row per example.

    For a segment dataset the examples are the centres, not the segments, so the
    result is written back through each segment's centre ids and comes out in
    the same order -- and with the same length -- as the window dataset would
    have produced. That is what lets the two paths be compared, and what lets
    the validation metric be computed the same way for every model.
    """
    from sleepstatelab.training.windows import SegmentDataset

    model.eval()
    if isinstance(dataset, SegmentDataset):
        return _predict_segments(model, dataset, device=device, batch_size=batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    blocks: list[np.ndarray] = []
    for batch in loader:
        inputs, _ = split_batch(batch, device)
        logits = model(*inputs)
        blocks.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(blocks, axis=0) if blocks else np.empty((0, len(STAGES)))


@torch.no_grad()
def _predict_segments(
    model: nn.Module, dataset: Any, *, device: str, batch_size: int = 8
) -> np.ndarray:
    """Centre probabilities from a segment dataset, in entry order."""
    out = np.zeros((len(dataset.entries), len(STAGES)), dtype=np.float64)
    written = np.zeros(len(dataset.entries), dtype=bool)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    segment = 0
    for signals, gather, mask, _ in loader:
        logits = model.forward_segment(
            signals.to(device), gather.to(device), mask.to(device)
        )
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        for row in range(probabilities.shape[0]):
            ids = dataset.centre_ids_of(segment)
            live = ids >= 0
            out[ids[live]] = probabilities[row][live]
            written[ids[live]] = True
            segment += 1
    if not written.all():
        raise RuntimeError(
            f"{int((~written).sum())} centre(s) were never predicted; the segment "
            "plan does not cover the dataset"
        )
    return out


def participant_mean_macro_f1(dataset: Any, probabilities: np.ndarray) -> float:
    """The primary metric, computed on a dataset's own participant labels."""
    predicted = probabilities.argmax(axis=1)
    truth = dataset.y
    people = np.array([entry.participant_id for entry in dataset.entries])
    scores = []
    for person in sorted(set(people.tolist())):
        mask = people == person
        score, _, _ = participant_macro_f1(truth[mask], predicted[mask])
        if not np.isnan(score):
            scores.append(score)
    return float(np.mean(scores)) if scores else float("nan")


def train_d1(
    config: Config,
    split: Split,
    train: Any,
    val: Any,
    *,
    device: str = "cpu",
    checkpoint_path: Path | str | None = None,
    run_id: str = "d1",
    progress: bool = True,
) -> tuple[nn.Module, Checkpoint, TrainingHistory]:
    """Train D1, selecting on validation participant macro-F1."""
    return train_supervised(
        config,
        split,
        train,
        val,
        model=build_model(config),
        model_name="D1",
        temporal_kwargs={},
        device=device,
        checkpoint_path=checkpoint_path,
        run_id=run_id,
        progress=progress,
    )


def build_d2(config: Config, *, encoder: EpochEncoder | None = None) -> Any:
    """D2 with a fresh or supplied encoder.

    ``encoder`` is how a pretrained backbone enters the model, which is the
    route D3 will take. Passing one that was built with different arguments than
    the configuration asks for is refused, because the checkpoint would then
    record a shape the weights do not have.
    """
    from sleepstatelab.models.d2 import D2Classifier

    if encoder is None:
        encoder = EpochEncoder(**encoder_kwargs(config))
    elif encoder.embedding_dim != config.model.embedding_dim:
        raise ValueError(
            f"the supplied encoder has embedding dimension {encoder.embedding_dim}, "
            f"the configuration asks for {config.model.embedding_dim}"
        )
    return D2Classifier.from_encoder(
        encoder,
        context=config.model.context_epochs,
        n_layers=config.model.temporal_layers,
        n_heads=config.model.temporal_heads,
        n_classes=config.model.n_classes,
        dropout=config.model.dropout,
        transformer_dropout=config.model.temporal_dropout,
    )


def temporal_kwargs_of(config: Config) -> dict[str, Any]:
    """What a D2 checkpoint has to record to be rebuildable."""
    return {
        "context": config.model.context_epochs,
        "n_layers": config.model.temporal_layers,
        "n_heads": config.model.temporal_heads,
        "transformer_dropout": config.model.temporal_dropout,
    }


def train_d2(
    config: Config,
    split: Split,
    train: Any,
    val: Any,
    *,
    encoder: EpochEncoder | None = None,
    device: str = "cpu",
    checkpoint_path: Path | str | None = None,
    run_id: str = "d2",
    progress: bool = True,
    loader_batch_size: int | None = None,
) -> tuple[nn.Module, Checkpoint, TrainingHistory]:
    """Train D2 under settings identical to D1's, which is the whole point."""
    return train_supervised(
        config,
        split,
        train,
        val,
        model=build_d2(config, encoder=encoder),
        model_name="D2",
        temporal_kwargs=temporal_kwargs_of(config),
        device=device,
        checkpoint_path=checkpoint_path,
        run_id=run_id,
        progress=progress,
        loader_batch_size=loader_batch_size,
    )


def train_supervised(
    config: Config,
    split: Split,
    train: Any,
    val: Any,
    *,
    model: nn.Module,
    model_name: str,
    temporal_kwargs: dict[str, Any],
    device: str = "cpu",
    checkpoint_path: Path | str | None = None,
    run_id: str = "run",
    progress: bool = True,
    loader_batch_size: int | None = None,
) -> tuple[nn.Module, Checkpoint, TrainingHistory]:
    """The supervised loop, shared by every model this repository trains.

    ``loader_batch_size`` is how many *dataset items* go into a step, which is
    only different from ``config.train.batch_size`` when an item holds more than
    one example -- a segment of centres. The configuration keeps counting
    examples, so a checkpoint says what it means.

    Shared deliberately. If D1 and D2 had a loop each, the two would drift --
    a different schedule here, a different clipping threshold there -- and the
    difference between their scores would stop being temporal context.
    """
    seed_everything(config.train.seed)
    model = model.to(device)

    weights = torch.tensor(
        train.class_weights(config.train.class_weighting), dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_LABEL)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(config.train.epochs, 1)
    )
    loader = DataLoader(
        train,
        batch_size=loader_batch_size or config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        drop_last=False,
    )

    history = TrainingHistory()
    best_score = float("-inf")
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    since_improvement = 0

    for epoch in range(config.train.epochs):
        model.train()
        started = time.time()
        total_loss = 0.0
        seen = 0
        for batch, items in enumerate(loader):
            if config.train.max_train_batches and batch >= config.train.max_train_batches:
                break
            inputs, targets = split_batch(items, device)
            optimiser.zero_grad(set_to_none=True)
            logits, flat_targets = flatten_logits(model(*inputs), targets)
            loss = criterion(logits, flat_targets)
            loss.backward()
            if config.train.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimiser.step()
            counted = int((flat_targets != IGNORE_LABEL).sum().item())
            total_loss += float(loss.item()) * counted
            seen += counted
        scheduler.step()

        val_probabilities = predict(model, val, device=device)
        val_score = participant_mean_macro_f1(val, val_probabilities)
        history.add(
            epoch=epoch,
            train_loss=total_loss / max(seen, 1),
            val_participant_macro_f1=val_score,
            seconds=time.time() - started,
        )
        if progress:
            print(
                f"  epoch {epoch + 1}/{config.train.epochs}  "
                f"loss {total_loss / max(seen, 1):.4f}  "
                f"val participant macro-F1 {val_score:.4f}  "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )

        if val_score > best_score:
            best_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= config.train.early_stopping_patience:
                if progress:
                    print(
                        f"  stopping: no improvement in "
                        f"{config.train.early_stopping_patience} epochs",
                        flush=True,
                    )
                break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint; did it run zero epochs?")
    model.load_state_dict(best_state)
    model.eval()

    checkpoint = Checkpoint(
        format=CHECKPOINT_FORMAT,
        model_name=model_name,
        state_dict=dict(best_state),
        encoder_kwargs=encoder_kwargs(config),
        temporal_kwargs=dict(temporal_kwargs),
        n_classes=config.model.n_classes,
        dropout=config.model.dropout,
        label_order=STAGES,
        channels=tuple(config.data.channels),
        preprocessing_id=config.preprocessing_identity,
        normalization=asdict(train.stats),
        split_id=split.identity,
        split_name=split.name,
        train_participants=train.participants,
        val_participants=val.participants,
        test_participants=tuple(split.test),
        seed=config.train.seed,
        code_revision=code_revision(),
        config_id=config.identity,
        config=config.to_dict(),
        epoch_selected=best_epoch,
        val_metric_name="participant_macro_f1",
        val_metric_value=best_score,
        history=history.rows,
        notes={
            "run_id": run_id,
            "device": device,
            "n_parameters": model.n_parameters(),
            "context_epochs": temporal_kwargs.get("context"),
            "context_coverage": (
                train.context_coverage() if hasattr(train, "context_coverage") else None
            ),
            "train_epochs_available": len(train),
            "truncated_batches_per_epoch": config.train.max_train_batches or None,
            "loader_batch_size": loader_batch_size or config.train.batch_size,
            "examples_per_step": config.train.batch_size,
        },
    )
    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, checkpoint)
    return model, checkpoint, history
