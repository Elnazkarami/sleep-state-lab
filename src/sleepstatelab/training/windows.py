"""Context windows: eleven epochs around one labelled centre, and what is missing.

Every rule that keeps D2 honest is enforced here rather than in the model, so
that a window handed to a model is already correct by construction.

* **A window never leaves its recording.** Offsets are resolved inside one
  recording's epoch index, so a window cannot reach into the next night, and
  certainly not into another participant.
* **A neighbour must be at exactly the expected index.** Position ``k`` of a
  window centred on epoch ``i`` is the epoch whose *original* index is
  ``i + k - 5``. If that epoch was excluded -- unscored, movement time, rejected
  by quality control -- there is no neighbour there, and the position is marked
  absent. The nearest surviving epoch is never substituted: it is a different
  point in the night, and using it would fabricate continuity across the gap.
* **A boundary is an absence, not a zero.** The first epoch of a recording has
  five absent positions before it, marked the same way as a gap.
* **The centre is always real.** Windows are built only around eligible,
  labelled epochs.

The signal for an absent position is left as zeros in the returned tensor and
the model is told, through the mask, not to look at it: the model replaces those
positions with a learned token before attention, so nothing downstream ever
treats a zero epoch as a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from sleepstatelab.config import Config
from sleepstatelab.data.epochs import EpochedRecording
from sleepstatelab.data.prepare import load_cached, reject_mask_flags
from sleepstatelab.data.preprocess import NormalizationStats, bandpass, fit_normalization
from sleepstatelab.data.splits import Split
from sleepstatelab.labels import STAGES
from sleepstatelab.training.dataset import EpochIndexEntry, class_weights_from_counts


@dataclass(frozen=True, slots=True)
class WindowPlan:
    """Which stored rows fill a window, and which positions have nobody.

    ``rows`` holds an index into the recording's epoch array for every present
    position and ``-1`` for an absent one; ``mask`` is the same information as a
    boolean. Both are kept because the first is what gathers the signal and the
    second is what the model is told.
    """

    recording: int
    centre_row: int
    rows: np.ndarray
    mask: np.ndarray


def plan_windows(epoch_index: np.ndarray, context: int) -> tuple[np.ndarray, np.ndarray]:
    """For every stored epoch, the rows of its neighbours and which exist.

    ``epoch_index`` is one recording's original epoch indices, ascending and not
    necessarily contiguous. Returns ``rows`` and ``mask``, both
    ``[n_epochs, context]``.

    Resolution is by index lookup rather than by walking the array, so an epoch
    that is *stored* but is not at the expected distance is correctly treated as
    no neighbour at all.
    """
    if context % 2 == 0:
        raise ValueError(f"context must be odd, got {context}")
    half = context // 2
    n = int(epoch_index.size)
    rows = np.full((n, context), -1, dtype=np.int64)
    mask = np.zeros((n, context), dtype=bool)

    # A map from original epoch index to the row that holds it. Built once per
    # recording; the alternative, searching the index array per window, is the
    # same answer computed n times.
    where = {int(value): row for row, value in enumerate(epoch_index)}
    for row, centre in enumerate(epoch_index.tolist()):
        for position in range(context):
            wanted = int(centre) + position - half
            found = where.get(wanted)
            if found is None:
                continue
            rows[row, position] = found
            mask[row, position] = True
    return rows, mask


class ContextWindowDataset(Dataset):
    """Windows of ``context`` epochs, labelled by the stage of the centre."""

    def __init__(
        self,
        recordings: list[EpochedRecording],
        *,
        config: Config,
        stats: NormalizationStats,
        reject_flags: int,
        context: int = 11,
    ) -> None:
        self.config = config
        self.stats = stats
        self.context = context
        self.blocks: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.rows: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.entries: list[EpochIndexEntry] = []
        self.index: list[tuple[int, int]] = []
        """One entry per window: which recording, and which row is its centre."""

        for record in recordings:
            keep = record.eligible(reject_flags)
            if not keep.any():
                continue
            signals = bandpass(record.signals[keep], record.sampling_rate_hz, config.preprocess)
            block = stats.apply(signals)
            epoch_index = record.epoch_index[keep]
            qc = record.qc[keep]
            rows, mask = plan_windows(epoch_index, context)

            recording_id = len(self.blocks)
            self.blocks.append(block)
            self.labels.append(record.labels[keep].astype(np.int64))
            self.rows.append(rows)
            self.masks.append(mask)
            for row in range(int(epoch_index.size)):
                self.index.append((recording_id, row))
                self.entries.append(
                    EpochIndexEntry(
                        participant_id=record.participant_id,
                        recording_id=record.recording_id,
                        epoch_index=int(epoch_index[row]),
                        qc_flags=int(qc[row]),
                    )
                )
        if not self.index:
            raise ValueError("no eligible epochs in these recordings")

        # Ordered by recording and then by row, which is exactly the order of
        # ``self.index``: a window's label is ``self.y[i]``.
        self.y = np.concatenate(self.labels)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        recording, row = self.index[item]
        block = self.blocks[recording]
        rows = self.rows[recording][row]
        mask = self.masks[recording][row]

        window = np.zeros((self.context, *block.shape[1:]), dtype=np.float32)
        present = rows >= 0
        window[present] = block[rows[present]]
        return (
            torch.from_numpy(window),
            torch.from_numpy(mask.copy()),
            int(self.labels[recording][row]),
        )

    @property
    def participants(self) -> tuple[str, ...]:
        return tuple(sorted({e.participant_id for e in self.entries}))

    def class_counts(self) -> np.ndarray:
        return np.array(
            [np.count_nonzero(self.y == index) for index in range(len(STAGES))],
            dtype=np.int64,
        )

    def class_weights(self, scheme: str = "inverse_frequency") -> np.ndarray:
        """Identical to the epoch dataset's, so D1 and D2 are weighted the same."""
        return class_weights_from_counts(self.class_counts(), scheme)

    def context_coverage(self) -> float:
        """Share of context positions that are genuine neighbours.

        Reported with any D2 result: a dataset where a third of the context is
        absent is a different experiment from one where almost none is, and the
        number says which.
        """
        total = sum(int(mask.size) for mask in self.masks)
        present = sum(int(np.count_nonzero(mask)) for mask in self.masks)
        return present / total if total else 0.0


def build_window_datasets(
    config: Config,
    split: Split,
    *,
    context: int = 11,
    train_participants: tuple[str, ...] | None = None,
    stats: NormalizationStats | None = None,
    segments: bool = False,
    centres_per_segment: int = 32,
) -> tuple[Any, Any, Any, NormalizationStats]:
    """Train, validation and test window datasets, fitted on training only.

    The same contract as ``build_datasets``: statistics come from the training
    participants a run actually has, and inference supplies the statistics the
    checkpoint was trained under rather than fitting new ones.
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
    if not val_records:
        raise ValueError(
            f"split {split.name!r} has no validation participants; checkpoint "
            "selection would have nothing to select on"
        )

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

    def build(records: list[EpochedRecording]) -> Any:
        if segments:
            return SegmentDataset(
                records,
                config=config,
                stats=stats,
                reject_flags=reject,
                context=context,
                centres=centres_per_segment,
            )
        return ContextWindowDataset(
            records,
            config=config,
            stats=stats,
            reject_flags=reject,
            context=context,
        )

    return build(train_records), build(val_records), build(test_records), stats


def _windows_of(dataset: Any) -> ContextWindowDataset:
    """The window dataset underneath, whichever wrapper is holding it."""
    if isinstance(dataset, ContextWindowDataset):
        return dataset
    inner = getattr(dataset, "windows", None)
    if isinstance(inner, ContextWindowDataset):
        return inner
    raise TypeError(f"{type(dataset).__name__} holds no context windows to alter")


def shuffle_context(dataset: Any, *, seed: int = 0) -> None:
    """Shuffle the non-central positions of every window, in place.

    A control, not an augmentation. The central position is left exactly where
    it is, so the model is still being asked about the same epoch, and presence
    is shuffled with the signal, so the *amount* of real context is unchanged
    and only its order is destroyed.

    **What this control can and cannot show.** If a model scores as well with
    its neighbours shuffled, it is not using their *order*. That is not the same
    as not using them: a model that averaged its neighbours, or counted how many
    of them looked like slow-wave sleep, would be entirely unaffected by
    shuffling while still depending on context. Use :func:`mask_context` to ask
    the other question.
    """
    windows = _windows_of(dataset)
    rng = np.random.default_rng(seed)
    centre = windows.context // 2
    others = np.array([i for i in range(windows.context) if i != centre])
    for rows, mask in zip(windows.rows, windows.masks, strict=True):
        for window in range(rows.shape[0]):
            order = rng.permutation(others)
            rows[window, others] = rows[window, order]
            mask[window, others] = mask[window, order]


def mask_context(dataset: Any) -> None:
    """Mark every non-central position absent, in place.

    The sharper of the two controls, and the one that answers the question
    shuffling cannot: with all context removed, a temporal model is reduced to
    its encoder on the central epoch. If the score does not move, the context
    was contributing nothing at all -- not merely nothing that depended on order.

    The absences are exactly the ones the model already knows how to handle: a
    boundary window looks like this, so nothing about the input is out of
    distribution.
    """
    windows = _windows_of(dataset)
    centre = windows.context // 2
    for mask in windows.masks:
        keep = mask[:, centre].copy()
        mask[:, :] = False
        mask[:, centre] = keep


IGNORE_LABEL = -100
"""Torch's cross-entropy ignore index. Padded centres carry it, so they cost a
forward pass and contribute nothing to the loss."""


class SegmentDataset(Dataset):
    """Contiguous stretches of one recording, predicting every centre in them.

    The same windows as :class:`ContextWindowDataset` and the same masking, laid
    out so an epoch shared by overlapping windows is encoded once instead of
    eleven times. A stretch of ``centres`` predictions needs
    ``centres + context - 1`` epochs encoded, so with 32 centres and an
    eleven-epoch context that is 42 encodings for 32 predictions rather than
    352 -- about eight times less work per pass.

    This is an implementation of the same model, not a different one. The test
    ``test_segment_and_window_paths_agree`` asserts that a model produces the
    same logits either way.

    What it does change is what a batch is. A batch here is several stretches
    rather than several independent windows, so the examples in it are
    correlated in time. The encoder carries no batch statistics -- it is
    GroupNorm throughout, for this reason among others -- so nothing in the model
    depends on the composition of a batch; what remains is that the gradient
    estimate at each step is a little less diverse than a fully shuffled one.
    Stretches are drawn from random recordings in random order, which is what
    keeps that manageable.
    """

    def __init__(
        self,
        recordings: list[EpochedRecording],
        *,
        config: Config,
        stats: NormalizationStats,
        reject_flags: int,
        context: int = 11,
        centres: int = 32,
    ) -> None:
        self.windows = ContextWindowDataset(
            recordings,
            config=config,
            stats=stats,
            reject_flags=reject_flags,
            context=context,
        )
        self.context = context
        self.centres = centres
        self.half = context // 2
        self.rows_per_segment = centres + 2 * self.half
        self.entries = self.windows.entries
        self.y = self.windows.y

        # Where each recording's entries start in the global ordering, so a
        # centre in a segment can be written back to the row it came from.
        self.offsets: list[int] = []
        seen = 0
        for labels in self.windows.labels:
            self.offsets.append(seen)
            seen += int(labels.size)

        self.segments: list[tuple[int, int, int]] = []
        """(recording, first centre row, number of centres that are not duplicates)"""
        for recording, labels in enumerate(self.windows.labels):
            n = int(labels.size)
            if n <= centres:
                self.segments.append((recording, 0, n))
                continue
            start = 0
            while start < n:
                if start + centres <= n:
                    self.segments.append((recording, start, centres))
                    start += centres
                else:
                    # The tail: step back so the segment is full-length, and
                    # mark only the centres not already covered as live, so no
                    # epoch is trained on twice in one pass.
                    begin = n - centres
                    self.segments.append((recording, begin, n - start))
                    break

    def __len__(self) -> int:
        return len(self.segments)

    def centre_ids_of(self, item: int) -> np.ndarray:
        """Global entry index for each centre of a segment, ``-1`` where there is
        none -- padding, or a tail centre the previous segment already covered.

        Kept out of the batch on purpose: a batch is ``(inputs..., target)``
        everywhere in this package, and prediction walks the segments in order,
        so the mapping can be recomputed rather than carried.
        """
        recording, first, live = self.segments[item]
        n = int(self.windows.labels[recording].size)
        ids = np.full(self.centres, -1, dtype=np.int64)
        first_live = first + self.centres - live
        for position in range(self.centres):
            row = first + position
            if row < n and row >= first_live:
                ids[position] = self.offsets[recording] + row
        return ids

    def __getitem__(
        self, item: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        recording, first, live = self.segments[item]
        block = self.windows.blocks[recording]
        rows = self.windows.rows[recording]
        masks = self.windows.masks[recording]
        labels = self.windows.labels[recording]
        n = int(labels.size)

        base = first - self.half
        signals = np.zeros((self.rows_per_segment, *block.shape[1:]), dtype=np.float32)
        low = max(base, 0)
        high = min(base + self.rows_per_segment, n)
        if high > low:
            signals[low - base : high - base] = block[low:high]

        gather = np.zeros((self.centres, self.context), dtype=np.int64)
        mask = np.zeros((self.centres, self.context), dtype=bool)
        y = np.full(self.centres, IGNORE_LABEL, dtype=np.int64)
        first_live = first + self.centres - live

        for position in range(self.centres):
            row = first + position
            if row >= n:
                # Padding, only possible in a recording shorter than one
                # segment. Point the window at something valid and let
                # IGNORE_LABEL discard the result.
                gather[position, self.half] = 0
                mask[position, self.half] = True
                continue
            window_rows = rows[row]
            # The stored mask is the authority on presence; the row array agrees
            # with it by construction and stays in step through a shuffle,
            # because the control permutes both together.
            present = masks[row]
            local = np.where(present, window_rows - base, 0)
            gather[position] = np.clip(local, 0, self.rows_per_segment - 1)
            mask[position] = present
            if row >= first_live:
                y[position] = labels[row]
            else:
                # An overlapping tail centre, already covered by the previous
                # segment. Predicted and thrown away rather than trained twice.
                mask[position, self.half] = True

        return (
            torch.from_numpy(signals),
            torch.from_numpy(gather),
            torch.from_numpy(mask),
            torch.from_numpy(y),
        )

    @property
    def participants(self) -> tuple[str, ...]:
        return self.windows.participants

    @property
    def stats(self) -> NormalizationStats:
        """The normalisation the underlying epochs were scaled by, so a
        checkpoint records the statistics this dataset actually applied."""
        return self.windows.stats

    def class_counts(self) -> np.ndarray:
        return self.windows.class_counts()

    def class_weights(self, scheme: str = "inverse_frequency") -> np.ndarray:
        return self.windows.class_weights(scheme)

    def context_coverage(self) -> float:
        return self.windows.context_coverage()

    def n_centres(self) -> int:
        """How many real predictions one pass over this dataset makes."""
        return int(sum(live for _, _, live in self.segments))

    def encodings_per_pass(self) -> int:
        """Epoch encodings one pass costs, for comparison with the window path."""
        return len(self.segments) * self.rows_per_segment
