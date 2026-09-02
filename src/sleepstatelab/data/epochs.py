"""Cutting a night into the 30-second epochs the scorer scored.

The arrays this produces are what every model in the repository sees, so the
rules are strict and they are all here.

**Epoch boundaries come from the recording start.** Epoch *i* is samples
``[i*3000, (i+1)*3000)`` of the PSG. That is the same grid the hypnogram's
onsets are on, so alignment is a property of the format rather than something
this code negotiates.

**Original indices survive.** Every epoch carries the index it had in its
recording. An epoch dropped for being unscored leaves a hole, and the hole is
visible: ``epoch_index`` jumps. Nothing in this package treats two rows as
neighbours because they are adjacent in an array -- D2's context windows check
the index difference, which is the only reason they can be trusted at a gap.

**Nothing is normalised here.** The arrays are microvolts as the header
declares them. Normalisation is fitted on training participants and applied
later; doing it at this stage would bake a training set into a cache.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sleepstatelab.data.annotations import UNLABELLED, read_hypnogram
from sleepstatelab.data.discovery import RecordingPair
from sleepstatelab.data.edf_header import read_header, to_microvolt_factor
from sleepstatelab.labels import STAGES
from sleepstatelab.numerics import trapezoid

QC_OK = 0
QC_FLATLINE = 1
QC_CLIPPED = 2
QC_HIGH_AMPLITUDE = 4
QC_MUSCLE = 8
"""Quality-control codes, as bit flags so one epoch can carry several. Muscle
contamination is recorded and, by default, does not reject: wake epochs
legitimately carry it, and rejecting on it deletes part of the class."""

QC_NAMES: dict[int, str] = {
    QC_FLATLINE: "qc_flatline",
    QC_CLIPPED: "qc_clipped",
    QC_HIGH_AMPLITUDE: "qc_high_amplitude",
    QC_MUSCLE: "qc_muscle",
}

CACHE_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class QCPolicy:
    """Thresholds for judging an epoch of scalp EEG, in microvolts.

    Taken from the PhysioML EEG quality policy (``sleep-eeg-1.0``) so that the
    classical baselines here and the published PhysioML numbers reject on the
    same grounds; see docs/physioml_reuse.md.
    """

    flat_sd_uv: float = 0.1
    """Below this the electrode is not connected to anything. A real scalp
    signal is a few microvolts of standard deviation even in the quietest
    stage."""

    clip_fraction: float = 0.02
    """Share of an epoch's samples sitting at the amplifier's rail before the
    epoch is judged saturated. The rail is read from the file's declared
    physical range -- about 192 microvolts on Sleep Cassette -- rather than
    assumed, because a saturated epoch is one that hit *this* amplifier's
    limit."""

    rail_tolerance: float = 0.995
    """How close to the declared rail counts as being at it."""

    max_amplitude_uv: float = 500.0
    """An absolute ceiling, for montages whose range is wide enough for it to
    mean something. It cannot fire on Sleep Cassette, where the declared
    physical range is about +/-192 microvolts, and that is recorded here rather
    than left for someone to discover from a column of zeros."""

    max_muscle_share: float = 0.5
    version: str = "sleepstatelab-eeg-qc-1.0"


@dataclass(frozen=True, slots=True)
class EpochedRecording:
    """One night, epoched. Arrays are parallel and ordered by epoch index."""

    participant_id: str
    recording_id: str
    night: int
    channels: tuple[str, ...]
    sampling_rate_hz: float
    epoch_seconds: float
    signals: np.ndarray
    """``float32 [n_epochs, n_channels, samples]``, microvolts."""

    epoch_index: np.ndarray
    """``int32``: the epoch's position in the original recording. Not contiguous."""

    labels: np.ndarray
    """``int8``: 0-4, or -1 for an epoch with no stage."""

    raw_labels: tuple[str, ...]
    qc: np.ndarray
    """``int16`` bit flags per epoch, ``QC_OK`` when nothing was found."""

    exclusions: dict[str, int]
    counts_by_stage: dict[str, int]
    source: dict[str, Any]

    @property
    def n_epochs(self) -> int:
        return int(self.signals.shape[0])

    def eligible(self, reject_flags: int) -> np.ndarray:
        """Boolean mask of epochs that are labelled and pass quality control."""
        return (self.labels >= 0) & ((self.qc & reject_flags) == 0)

    def save(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            signals=self.signals,
            epoch_index=self.epoch_index,
            labels=self.labels,
            qc=self.qc,
            raw_labels=np.array(self.raw_labels, dtype=object),
            meta=json.dumps(
                {
                    "cache_version": CACHE_VERSION,
                    "participant_id": self.participant_id,
                    "recording_id": self.recording_id,
                    "night": self.night,
                    "channels": list(self.channels),
                    "sampling_rate_hz": self.sampling_rate_hz,
                    "epoch_seconds": self.epoch_seconds,
                    "exclusions": self.exclusions,
                    "counts_by_stage": self.counts_by_stage,
                    "source": self.source,
                }
            ),
        )

    @classmethod
    def load(cls, path: Path | str) -> EpochedRecording:
        with np.load(path, allow_pickle=True) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("cache_version") != CACHE_VERSION:
                raise ValueError(
                    f"{path} was written by cache version "
                    f"{meta.get('cache_version')!r}, this package writes {CACHE_VERSION!r}"
                )
            return cls(
                participant_id=meta["participant_id"],
                recording_id=meta["recording_id"],
                night=meta["night"],
                channels=tuple(meta["channels"]),
                sampling_rate_hz=meta["sampling_rate_hz"],
                epoch_seconds=meta["epoch_seconds"],
                signals=data["signals"],
                epoch_index=data["epoch_index"],
                labels=data["labels"],
                raw_labels=tuple(str(x) for x in data["raw_labels"]),
                qc=data["qc"],
                exclusions=meta["exclusions"],
                counts_by_stage=meta["counts_by_stage"],
                source=meta["source"],
            )


def read_channels_microvolts(
    psg_path: Path | str, channels: tuple[str, ...], *, expected_rate_hz: float
) -> tuple[np.ndarray, float]:
    """The requested channels, in the requested order, in microvolts.

    MNE reads the samples and applies the header's calibration, returning volts
    for anything it recognises as EEG. The conversion back to microvolts is done
    against the *declared* physical dimension rather than by assuming MNE's
    convention, so a file that declares millivolts is converted correctly and a
    file that declares something unexpected raises instead of being scaled by a
    guess.
    """
    import mne

    from sleepstatelab.data.annotations import _quiet_mne

    header = read_header(psg_path)
    missing = [c for c in channels if c not in header.labels]
    if missing:
        raise ValueError(
            f"{Path(psg_path).name} is missing channel(s) {missing}; it has {header.labels}"
        )
    for label in channels:
        rate = header.sampling_rate(label)
        if abs(rate - expected_rate_hz) > 1e-6:
            raise ValueError(
                f"{Path(psg_path).name}: {label!r} is at {rate:g} Hz, "
                f"expected {expected_rate_hz:g} Hz. Resampling is not done silently."
            )

    with _quiet_mne():
        raw = mne.io.read_raw_edf(str(psg_path), preload=False, verbose="ERROR")
        data = raw.get_data(picks=list(channels))  # volts, in the order asked for

    scaled = np.empty_like(data, dtype=np.float64)
    for row, label in enumerate(channels):
        declared = header.signal(label).physical_dimension
        # MNE has already converted the declared unit to volts. Undo that to get
        # back to the declared unit, then convert the declared unit to microvolts.
        to_declared = data[row] / _MNE_VOLT_FACTOR[_unit_family(declared)]
        scaled[row] = to_declared * to_microvolt_factor(declared)
    return scaled, header.duration_seconds


def _unit_family(dimension: str) -> str:
    key = dimension.strip()
    if key in {"uV", "µV", "μV"}:
        return "uV"
    if key == "mV":
        return "mV"
    if key == "V":
        return "V"
    raise ValueError(f"unrecognised physical dimension {dimension!r}")


#: What MNE's returned volts are, per declared unit: it scales the declared
#: physical values into volts, so dividing by this recovers the declared values.
_MNE_VOLT_FACTOR = {"uV": 1e-6, "mV": 1e-3, "V": 1.0}


def quality_flags(
    epoch: np.ndarray,
    rate: float,
    policy: QCPolicy,
    rails_uv: tuple[float, ...] | None = None,
) -> int:
    """Quality-control bit flags for one epoch, across all its channels.

    ``rails_uv`` is each channel's declared physical limit in microvolts, taken
    from the EDF header. Given it, saturation is detected against the actual
    amplifier range; without it, the fallback is an epoch whose extreme value
    repeats often enough to be a rail rather than a peak.
    """
    flags = QC_OK
    for index, channel in enumerate(epoch):
        data = np.asarray(channel, dtype=np.float64)
        if float(np.std(data)) < policy.flat_sd_uv:
            flags |= QC_FLATLINE
        extreme = float(np.max(np.abs(data))) if data.size else 0.0
        rail = None
        if rails_uv is not None and index < len(rails_uv) and rails_uv[index] > 0:
            rail = rails_uv[index] * policy.rail_tolerance
        threshold = rail if rail is not None else extreme * 0.999
        if extreme > 0 and float(np.mean(np.abs(data) >= threshold)) > policy.clip_fraction:
            flags |= QC_CLIPPED
        if extreme > policy.max_amplitude_uv:
            flags |= QC_HIGH_AMPLITUDE
    muscle = _muscle_share(epoch, rate)
    if muscle > policy.max_muscle_share:
        flags |= QC_MUSCLE
    return flags


def _muscle_share(epoch: np.ndarray, rate: float) -> float:
    """Share of 0.5-45 Hz power sitting above 30 Hz, averaged over channels."""
    from scipy.signal import welch

    shares: list[float] = []
    for channel in epoch:
        data = np.asarray(channel, dtype=np.float64)
        segment = min(int(rate * 4), data.size)
        if segment < 8:
            continue
        frequencies, power = welch(data, fs=rate, nperseg=segment)
        whole = (frequencies >= 0.5) & (frequencies <= 45.0)
        high = (frequencies > 30.0) & (frequencies <= 45.0)
        total = trapezoid(power[whole], frequencies[whole])
        if total <= 0:
            continue
        above = trapezoid(power[high], frequencies[high])
        shares.append(above / total)
    return float(np.mean(shares)) if shares else 0.0


def sleep_window_mask(labels: np.ndarray, margin_epochs: int) -> np.ndarray:
    """The annotation-defined sleep interval plus a margin, as a mask.

    The sensitivity analysis, not the primary preparation. A Sleep Cassette
    night runs about twenty hours and most of it is the recorder waiting; cutting
    to the sleep period changes the class balance drastically -- wake falls from
    about 70% to about 30% -- and therefore changes what every metric means. It
    is available so the difference can be measured, and it is labelled wherever
    it is used.
    """
    asleep = np.flatnonzero((labels >= 1) & (labels <= 4))
    mask = np.zeros(labels.size, dtype=bool)
    if asleep.size == 0:
        return mask
    first = max(int(asleep[0]) - margin_epochs, 0)
    last = min(int(asleep[-1]) + margin_epochs + 1, labels.size)
    mask[first:last] = True
    return mask


def epoch_recording(
    pair: RecordingPair,
    *,
    channels: tuple[str, ...],
    sampling_rate_hz: float = 100.0,
    epoch_seconds: float = 30.0,
    policy: QCPolicy | None = None,
    sleep_window: bool = False,
    sleep_window_margin_minutes: float = 30.0,
    drop_ineligible: bool = True,
) -> EpochedRecording:
    """Read one night and cut it into scored epochs.

    ``drop_ineligible`` removes epochs with no stage from the stored arrays --
    they are 8% to 12% of a Sleep Cassette night and storing them would triple
    nothing useful. Their indices are *not* renumbered, so the gap they leave is
    the record that they were there, and the count of each exclusion reason is
    kept.
    """
    policy = policy or QCPolicy()
    samples_per_epoch = round(sampling_rate_hz * epoch_seconds)

    data, duration = read_channels_microvolts(
        pair.psg_path, channels, expected_rate_hz=sampling_rate_hz
    )
    header = read_header(pair.psg_path)
    rails = tuple(
        min(abs(header.signal(label).physical_min), abs(header.signal(label).physical_max))
        * to_microvolt_factor(header.signal(label).physical_dimension)
        for label in channels
    )
    n_signal_epochs = int(data.shape[1] // samples_per_epoch)
    hypnogram = read_hypnogram(pair.hypnogram_path, epoch_seconds=epoch_seconds)

    labels = np.full(n_signal_epochs, UNLABELLED, dtype=np.int8)
    raw = [""] * n_signal_epochs
    usable = min(n_signal_epochs, hypnogram.n_epochs)
    labels[:usable] = hypnogram.labels[:usable]
    raw[:usable] = list(hypnogram.raw_text[:usable])

    exclusions = {
        "movement_time": 0,
        "unscored": 0,
        "unknown_annotation": 0,
        "no_annotation": 0,
        "beyond_signal": max(hypnogram.n_epochs - n_signal_epochs, 0),
        "short_epoch": int(data.shape[1] % samples_per_epoch > 0),
        "qc_flatline": 0,
        "qc_clipped": 0,
        "qc_high_amplitude": 0,
        "outside_sleep_window": 0,
    }
    for index in range(n_signal_epochs):
        if labels[index] >= 0:
            continue
        text = raw[index].strip()
        if text == "Movement time":
            exclusions["movement_time"] += 1
        elif text in {"Sleep stage ?", "Sleep stage e"}:
            exclusions["unscored"] += 1
        elif text:
            exclusions["unknown_annotation"] += 1
        else:
            exclusions["no_annotation"] += 1

    trimmed = data[:, : n_signal_epochs * samples_per_epoch]
    stacked = trimmed.reshape(len(channels), n_signal_epochs, samples_per_epoch)
    stacked = np.transpose(stacked, (1, 0, 2)).astype(np.float32, copy=False)

    if sleep_window:
        margin = round(sleep_window_margin_minutes * 60.0 / epoch_seconds)
        inside = sleep_window_mask(labels, margin)
        exclusions["outside_sleep_window"] = int(np.count_nonzero(~inside & (labels >= 0)))
        labels = np.where(inside, labels, UNLABELLED).astype(np.int8)

    keep = np.flatnonzero(labels >= 0) if drop_ineligible else np.arange(n_signal_epochs)
    signals = np.ascontiguousarray(stacked[keep])
    kept_labels = labels[keep].astype(np.int8)
    kept_raw = tuple(raw[i] for i in keep)

    qc = np.zeros(keep.size, dtype=np.int16)
    for row in range(keep.size):
        qc[row] = quality_flags(signals[row], sampling_rate_hz, policy, rails)
    for flag, name in QC_NAMES.items():
        if name in exclusions:
            exclusions[name] = int(np.count_nonzero(qc & flag))

    counts = {
        name: int(np.count_nonzero(kept_labels == index)) for index, name in enumerate(STAGES)
    }

    return EpochedRecording(
        participant_id=pair.participant_id,
        recording_id=pair.recording_id,
        night=pair.night,
        channels=tuple(channels),
        sampling_rate_hz=sampling_rate_hz,
        epoch_seconds=epoch_seconds,
        signals=signals,
        epoch_index=keep.astype(np.int32),
        labels=kept_labels,
        raw_labels=kept_raw,
        qc=qc,
        exclusions=exclusions,
        counts_by_stage=counts,
        source={
            "psg_file": str(pair.psg_path),
            "hypnogram_file": str(pair.hypnogram_path),
            "psg_duration_seconds": duration,
            "n_epochs_signal": n_signal_epochs,
            "n_epochs_stored": int(keep.size),
            "qc_policy": asdict(policy),
            "channel_rails_uv": list(rails),
            "sleep_window": sleep_window,
            "sleep_window_margin_minutes": (
                sleep_window_margin_minutes if sleep_window else None
            ),
        },
    )
