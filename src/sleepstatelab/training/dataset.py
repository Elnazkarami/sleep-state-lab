"""The dataset a model iterates, assembled from the epoch cache.

Everything is held in memory as float32. A Sleep Cassette night is about 2,700
epochs of 2x3000 float32, which is 65 MB; twenty nights is 1.3 GB, which fits,
and the alternative -- re-reading EDF per batch -- makes every experiment slower
for no benefit at this scale. When the full 153-recording cohort is run this is
the first thing that will need a memory-mapped store, and that is noted rather
than pre-built.

The row identity travels with the row. Every example carries its participant,
recording and *original* epoch index, so a prediction can be written back
against the epoch it was made for, and so a temporal model can tell a genuine
neighbour from the epoch on the other side of a gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from sleepstatelab.config import Config
from sleepstatelab.data.epochs import EpochedRecording
from sleepstatelab.data.prepare import load_cached, reject_mask_flags
from sleepstatelab.data.preprocess import NormalizationStats, bandpass, fit_normalization
from sleepstatelab.data.splits import Split
from sleepstatelab.labels import STAGES


@dataclass(frozen=True, slots=True)
class EpochIndexEntry:
    """Where one example came from."""

    participant_id: str
    recording_id: str
    epoch_index: int
    qc_flags: int


def class_weights_from_counts(counts: np.ndarray, scheme: str) -> np.ndarray:
    """Inverse-frequency loss weights, normalised to a mean of one.

    Shared by the epoch dataset and the context-window dataset so that D1 and D2
    are weighted identically -- if they were not, the difference between them
    would include the loss, and the comparison would be about something else.

    Normalised to a mean of one so the loss magnitude does not move with the
    class balance, and a class absent from training gets a weight of one rather
    than an infinity.
    """
    if scheme == "none":
        return np.ones(len(STAGES), dtype=np.float64)
    if scheme != "inverse_frequency":
        raise ValueError(f"unknown class weighting {scheme!r}")
    counts = np.asarray(counts, dtype=np.float64)
    weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1.0), 1.0)
    present = counts > 0
    if present.any():
        weights[present] /= weights[present].mean()
    weights[~present] = 1.0
    return weights


class EpochDataset(Dataset):
    """Eligible epochs from a set of recordings, normalised and ready for a model."""

    def __init__(
        self,
        recordings: list[EpochedRecording],
        *,
        config: Config,
        stats: NormalizationStats,
        reject_flags: int,
    ) -> None:
        self.config = config
        self.stats = stats
        blocks: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        entries: list[EpochIndexEntry] = []
        for record in recordings:
            keep = record.eligible(reject_flags)
            if not keep.any():
                continue
            signals = record.signals[keep]
            signals = bandpass(signals, record.sampling_rate_hz, config.preprocess)
            blocks.append(stats.apply(signals))
            labels.append(record.labels[keep].astype(np.int64))
            entries.extend(
                EpochIndexEntry(
                    participant_id=record.participant_id,
                    recording_id=record.recording_id,
                    epoch_index=int(index),
                    qc_flags=int(flag),
                )
                for index, flag in zip(
                    record.epoch_index[keep], record.qc[keep], strict=True
                )
            )
        if not blocks:
            raise ValueError("no eligible epochs in these recordings")
        self.x = np.concatenate(blocks, axis=0)
        self.y = np.concatenate(labels, axis=0)
        self.entries = entries

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(self.x[index]), int(self.y[index])

    @property
    def participants(self) -> tuple[str, ...]:
        return tuple(sorted({e.participant_id for e in self.entries}))

    def class_counts(self) -> np.ndarray:
        return np.array(
            [np.count_nonzero(self.y == index) for index in range(len(STAGES))], dtype=np.int64
        )

    def class_weights(self, scheme: str = "inverse_frequency") -> np.ndarray:
        """Loss weights from *this* dataset's class counts. Training only."""
        return class_weights_from_counts(self.class_counts(), scheme)


def build_datasets(
    config: Config,
    split: Split,
    *,
    train_participants: tuple[str, ...] | None = None,
    stats: NormalizationStats | None = None,
) -> tuple[EpochDataset, EpochDataset, EpochDataset, NormalizationStats]:
    """Train, validation and test datasets, with statistics fitted on training only.

    ``train_participants`` narrows the training side for a label-budget run
    without touching validation or test. The normalisation is then fitted on the
    narrowed set, because that is what a run at that budget actually has.

    ``stats`` supplies normalisation instead of fitting it, which is what
    inference must do: a checkpoint carries the statistics it was trained under,
    and re-fitting them at prediction time would show the model differently
    scaled signals than it ever saw -- silently, and worst of all for a
    label-budget run, where the training side is not the whole training split.
    """
    reject = reject_mask_flags(tuple(config.preprocess.qc_reject))
    train_ids = tuple(train_participants) if train_participants else split.train
    unexpected = set(train_ids) - set(split.train)
    if unexpected:
        raise ValueError(
            f"{sorted(unexpected)} are not training participants of split {split.name!r}"
        )

    train_records = load_cached(config, train_ids)
    val_records = load_cached(config, split.val) if split.val else []
    test_records = load_cached(config, split.test)

    if stats is None:
        filtered = []
        participants = []
        for record in train_records:
            keep = record.eligible(reject)
            if keep.any():
                filtered.append(
                    bandpass(record.signals[keep], record.sampling_rate_hz, config.preprocess)
                )
                participants.append(record.participant_id)
        stats = fit_normalization(
            filtered,
            participants,
            channels=tuple(config.data.channels),
            config=config.preprocess,
            seed=config.split.seed,
        )
    elif tuple(stats.channels) != tuple(config.data.channels):
        raise ValueError(
            f"the supplied normalization was fitted for channels {tuple(stats.channels)}, "
            f"the configuration asks for {tuple(config.data.channels)}"
        )

    train = EpochDataset(train_records, config=config, stats=stats, reject_flags=reject)
    if not val_records:
        # Falling back to the training set would make early stopping select on
        # data the model is fitting, which is not a validation score at all.
        raise ValueError(
            f"split {split.name!r} has no validation participants with eligible "
            "epochs; checkpoint selection would have nothing to select on"
        )
    val = EpochDataset(val_records, config=config, stats=stats, reject_flags=reject)
    test = EpochDataset(test_records, config=config, stats=stats, reject_flags=reject)
    return train, val, test, stats
