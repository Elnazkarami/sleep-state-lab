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

from sleepstatelab.labels import STAGES
from sleepstatelab.models.d1 import D1Classifier
from sleepstatelab.models.encoder import EpochEncoder

CHECKPOINT_FORMAT = "sleepstatelab-checkpoint-1.0"


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A trained model and the contract it was trained under."""

    format: str
    model_name: str
    state_dict: dict[str, Any]
    encoder_kwargs: dict[str, Any]
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
) -> tuple[D1Classifier, Checkpoint]:
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
    if payload["format"] != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path} is format {payload['format']!r}, this package reads "
            f"{CHECKPOINT_FORMAT!r}"
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
    model = D1Classifier.from_encoder(
        encoder, n_classes=payload["n_classes"], dropout=payload["dropout"]
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()

    checkpoint = Checkpoint(
        format=payload["format"],
        model_name=payload["model_name"],
        state_dict=payload["state_dict"],
        encoder_kwargs=payload["encoder_kwargs"],
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
