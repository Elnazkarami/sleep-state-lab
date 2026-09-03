"""Checkpoints that say what they are.

A state dict alone is not a model: loaded against the wrong channel order it
runs and produces confident nonsense, and loaded against a different
normalisation it is quietly measuring something else. So every checkpoint here
carries the split it was trained on, the channel order, the preprocessing
identity, the label order, the seed, the code revision and the configuration
hash -- and the loader refuses one whose contract does not match what it is being
loaded into.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sleepstatelab.labels import STAGES
from sleepstatelab.models.d1 import D1Classifier
from sleepstatelab.models.encoder import EpochEncoder

CHECKPOINT_FORMAT = "sleepstatelab-checkpoint-1.1"
"""Version 1.1 adds ``temporal_kwargs`` and makes ``model_name`` load-bearing, so
a file says which architecture to rebuild. Version 1.0 files still load: they
predate D2 and are read as D1 with no temporal stack."""

READABLE_FORMATS = ("sleepstatelab-checkpoint-1.0", CHECKPOINT_FORMAT)


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A trained model and the contract it was trained under."""

    format: str
    model_name: str
    state_dict: dict[str, Any]
    encoder_kwargs: dict[str, Any]
    temporal_kwargs: dict[str, Any]
    """D2's context length and transformer shape. Empty for a model that has no
    temporal stack, which is how a loader tells the two apart."""

    n_classes: int
    dropout: float
    label_order: tuple[str, ...]
    channels: tuple[str, ...]
    preprocessing_id: str
    normalization: dict[str, Any]
    split_id: str
    split_name: str
    train_participants: tuple[str, ...]
    val_participants: tuple[str, ...]
    test_participants: tuple[str, ...]
    seed: int
    code_revision: str
    config_id: str
    config: dict[str, Any]
    epoch_selected: int
    val_metric_name: str
    val_metric_value: float
    history: list[dict[str, float]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def save_checkpoint(path: Path | str, checkpoint: Checkpoint) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint.to_payload(), target)
    return target


def load_checkpoint(
    path: Path | str,
    *,
    expect_channels: tuple[str, ...] | None = None,
    expect_preprocessing_id: str | None = None,
) -> tuple[nn.Module, Checkpoint]:
    """Rebuild a model from a checkpoint, refusing a mismatched contract."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    missing = [
        field_name
        for field_name in (
            "format",
            "state_dict",
            "encoder_kwargs",
            "label_order",
            "channels",
            "preprocessing_id",
            "split_id",
        )
        if field_name not in payload
    ]
    if missing:
        raise ValueError(f"{path} is not a complete checkpoint; it is missing {missing}")
    if payload["format"] not in READABLE_FORMATS:
        raise ValueError(
            f"{path} is format {payload['format']!r}, this package reads "
            f"{READABLE_FORMATS}"
        )
    if tuple(payload["label_order"]) != STAGES:
        raise ValueError(
            f"{path} was trained with label order {tuple(payload['label_order'])}, "
            f"this package uses {STAGES}"
        )
    if expect_channels is not None and tuple(payload["channels"]) != tuple(expect_channels):
        raise ValueError(
            f"{path} was trained on channels {tuple(payload['channels'])}, "
            f"but {tuple(expect_channels)} were asked for. Channel order is part of "
            "the model's contract."
        )
    if (
        expect_preprocessing_id is not None
        and payload["preprocessing_id"] != expect_preprocessing_id
    ):
        raise ValueError(
            f"{path} was trained under preprocessing {payload['preprocessing_id']}, "
            f"but {expect_preprocessing_id} is configured. The model would be seeing "
            "differently scaled signals than it was trained on."
        )

    encoder = EpochEncoder(**payload["encoder_kwargs"])
    temporal = dict(payload.get("temporal_kwargs") or {})
    model_name = payload.get("model_name", "D1")
    if temporal:
        from sleepstatelab.models.d2 import D2Classifier

        model: nn.Module = D2Classifier.from_encoder(
            encoder,
            n_classes=payload["n_classes"],
            dropout=payload["dropout"],
            **temporal,
        )
    else:
        model = D1Classifier.from_encoder(
            encoder, n_classes=payload["n_classes"], dropout=payload["dropout"]
        )
    model.load_state_dict(payload["state_dict"])
    model.eval()

    checkpoint = Checkpoint(
        format=payload["format"],
        model_name=model_name,
        state_dict=payload["state_dict"],
        encoder_kwargs=payload["encoder_kwargs"],
        temporal_kwargs=temporal,
        n_classes=payload["n_classes"],
        dropout=payload["dropout"],
        label_order=tuple(payload["label_order"]),
        channels=tuple(payload["channels"]),
        preprocessing_id=payload["preprocessing_id"],
        normalization=payload["normalization"],
        split_id=payload["split_id"],
        split_name=payload["split_name"],
        train_participants=tuple(payload["train_participants"]),
        val_participants=tuple(payload["val_participants"]),
        test_participants=tuple(payload["test_participants"]),
        seed=payload["seed"],
        code_revision=payload["code_revision"],
        config_id=payload["config_id"],
        config=payload["config"],
        epoch_selected=payload["epoch_selected"],
        val_metric_name=payload["val_metric_name"],
        val_metric_value=payload["val_metric_value"],
        history=payload.get("history", []),
        notes=payload.get("notes", {}),
    )
    return model, checkpoint


ENCODER_FORMAT = "sleepstatelab-encoder-1.0"
"""A pretrained backbone on its own, with no head and no classes.

Kept as its own format rather than as a classifier with a random head, because
what the file contains is exactly what D3 inherits and nothing else. A file that
claimed to be a classifier would invite someone to evaluate it as one.
"""


@dataclass(frozen=True, slots=True)
class EncoderCheckpoint:
    """A self-supervised encoder and the record of what produced it."""

    format: str
    state_dict: dict[str, Any]
    encoder_kwargs: dict[str, Any]
    channels: tuple[str, ...]
    preprocessing_id: str
    normalization: dict[str, Any]
    split_id: str
    split_name: str
    pretrain_participants: tuple[str, ...]
    """Exactly whose data the encoder saw. The whole limited-label claim rests on
    this list excluding the validation and test participants, so it is recorded
    rather than assumed, and the loader checks it against the split."""

    seed: int
    code_revision: str
    config_id: str
    config: dict[str, Any]
    objective: str
    epoch_selected: int
    metric_name: str
    metric_value: float
    history: list[dict[str, float]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def save_encoder_checkpoint(path: Path | str, checkpoint: EncoderCheckpoint) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint.to_payload(), target)
    return target


def load_encoder_checkpoint(
    path: Path | str,
    *,
    expect_channels: tuple[str, ...] | None = None,
    expect_preprocessing_id: str | None = None,
    forbid_participants: tuple[str, ...] = (),
) -> tuple[EpochEncoder, EncoderCheckpoint]:
    """Rebuild a pretrained encoder, refusing one that saw the wrong people.

    ``forbid_participants`` is the caller's validation and test set. A pretrained
    encoder that has seen them is not a limited-label result, and the failure is
    silent in every other respect -- the run trains, the numbers look good, and
    the claim is void. So it raises here.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") != ENCODER_FORMAT:
        raise ValueError(
            f"{path} is format {payload.get('format')!r}, not an encoder checkpoint "
            f"({ENCODER_FORMAT!r})"
        )
    if expect_channels is not None and tuple(payload["channels"]) != tuple(expect_channels):
        raise ValueError(
            f"{path} was pretrained on channels {tuple(payload['channels'])}, "
            f"but {tuple(expect_channels)} were asked for"
        )
    if (
        expect_preprocessing_id is not None
        and payload["preprocessing_id"] != expect_preprocessing_id
    ):
        raise ValueError(
            f"{path} was pretrained under preprocessing {payload['preprocessing_id']}, "
            f"but {expect_preprocessing_id} is configured"
        )
    seen = set(payload["pretrain_participants"])
    leaked = sorted(seen & set(forbid_participants))
    if leaked:
        raise ValueError(
            f"{path} was pretrained on {leaked}, which this run holds out. "
            "A pretrained encoder that has seen the held-out participants makes "
            "the limited-label comparison meaningless."
        )

    encoder = EpochEncoder(**payload["encoder_kwargs"])
    encoder.load_state_dict(payload["state_dict"])
    encoder.eval()
    checkpoint = EncoderCheckpoint(
        format=payload["format"],
        state_dict=payload["state_dict"],
        encoder_kwargs=payload["encoder_kwargs"],
        channels=tuple(payload["channels"]),
        preprocessing_id=payload["preprocessing_id"],
        normalization=payload["normalization"],
        split_id=payload["split_id"],
        split_name=payload["split_name"],
        pretrain_participants=tuple(payload["pretrain_participants"]),
        seed=payload["seed"],
        code_revision=payload["code_revision"],
        config_id=payload["config_id"],
        config=payload["config"],
        objective=payload["objective"],
        epoch_selected=payload["epoch_selected"],
        metric_name=payload["metric_name"],
        metric_value=payload["metric_value"],
        history=payload.get("history", []),
        notes=payload.get("notes", {}),
    )
    return encoder, checkpoint


def is_encoder_checkpoint(path: Path | str) -> bool:
    """Whether a file is a bare pretrained encoder rather than a whole model."""
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except Exception:
        # An unreadable file is simply not one of ours; the caller falls back to
        # treating it as a model checkpoint, which will raise with a better message.
        return False
    return isinstance(payload, dict) and payload.get("format") == ENCODER_FORMAT
