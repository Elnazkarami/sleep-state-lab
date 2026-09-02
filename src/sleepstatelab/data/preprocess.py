"""What happens to a signal between the cache and a model, and where it is fitted.

Two decisions here are the ones most likely to quietly invalidate a result.

**Statistics are fitted on training participants only.** The median and
inter-quartile range that normalisation uses are estimated over the training
split, saved, and then applied unchanged to validation and test. A scaler fitted
on everything lets the test participants' distribution shape the transform
applied to training data, which inflates scores in a way no split check catches.

**Amplitude is not thrown away.** The default normalisation is one median and
one scale *per channel*, shared by every epoch. Per-epoch z-scoring -- dividing
each 30 seconds by its own standard deviation -- is the common default in EEG
code and it deletes the absolute amplitude that separates N3 from N1, which is
one of the things a human scorer reads directly. It is implemented as
``per_epoch_zscore`` so its cost can be measured, and it is not the default.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from sleepstatelab.config import PreprocessConfig
from sleepstatelab.provenance import digest


@dataclass(frozen=True, slots=True)
class NormalizationStats:
    """Per-channel centre and scale, and the participants they were fitted on."""

    method: str
    channels: tuple[str, ...]
    centre: tuple[float, ...]
    scale: tuple[float, ...]
    fitted_on: tuple[str, ...]
    n_epochs: int
    clip_sigma: float

    @property
    def identity(self) -> str:
        return digest(asdict(self))

    def apply(self, signals: np.ndarray) -> np.ndarray:
        """Normalise ``[n, channels, samples]`` in microvolts. Returns float32."""
        data = np.asarray(signals, dtype=np.float32)
        if self.method == "none":
            out = data
        elif self.method == "train_robust_channel":
            centre = np.asarray(self.centre, dtype=np.float32).reshape(1, -1, 1)
            scale = np.asarray(self.scale, dtype=np.float32).reshape(1, -1, 1)
            out = (data - centre) / scale
        elif self.method == "per_epoch_zscore":
            centre = data.mean(axis=2, keepdims=True)
            scale = data.std(axis=2, keepdims=True)
            scale = np.where(scale < 1e-6, np.float32(1.0), scale)
            out = (data - centre) / scale
        else:
            raise ValueError(f"unknown normalization {self.method!r}")
        if self.clip_sigma > 0:
            out = np.clip(out, -self.clip_sigma, self.clip_sigma)
        return np.ascontiguousarray(out, dtype=np.float32)

    def write(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def read(cls, path: Path | str) -> NormalizationStats:
        payload = json.loads(Path(path).read_text())
        return cls(
            method=payload["method"],
            channels=tuple(payload["channels"]),
            centre=tuple(payload["centre"]),
            scale=tuple(payload["scale"]),
            fitted_on=tuple(payload["fitted_on"]),
            n_epochs=payload["n_epochs"],
            clip_sigma=payload["clip_sigma"],
        )


def fit_normalization(
    epochs: list[np.ndarray],
    participants: list[str],
    *,
    channels: tuple[str, ...],
    config: PreprocessConfig,
    max_epochs_per_recording: int = 400,
    seed: int = 0,
) -> NormalizationStats:
    """Fit per-channel statistics over training recordings only.

    A subsample per recording rather than every sample: the estimate is a median
    and an inter-quartile range over tens of millions of points either way, and
    the subsample keeps the fit to seconds. The subsample is seeded, so the
    statistics are reproducible.
    """
    method = config.normalization
    if method in {"none", "per_epoch_zscore"}:
        # Nothing is fitted, but the record of what was applied and to whom is
        # still written, so a run using it is as traceable as one that is not.
        return NormalizationStats(
            method=method,
            channels=tuple(channels),
            centre=tuple(0.0 for _ in channels),
            scale=tuple(1.0 for _ in channels),
            fitted_on=tuple(sorted(set(participants))),
            n_epochs=int(sum(e.shape[0] for e in epochs)),
            clip_sigma=config.clip_sigma if method == "per_epoch_zscore" else 0.0,
        )
    if method != "train_robust_channel":
        raise ValueError(f"unknown normalization {method!r}")

    rng = np.random.default_rng(seed)
    pooled: list[np.ndarray] = []
    total = 0
    for block in epochs:
        total += block.shape[0]
        take = min(max_epochs_per_recording, block.shape[0])
        rows = rng.choice(block.shape[0], size=take, replace=False)
        pooled.append(np.asarray(block[np.sort(rows)], dtype=np.float32))
    if not pooled:
        raise ValueError("no training epochs to fit normalization on")

    stacked = np.concatenate(pooled, axis=0)
    flat = stacked.transpose(1, 0, 2).reshape(len(channels), -1)
    centre = np.median(flat, axis=1)
    q75, q25 = np.percentile(flat, [75, 25], axis=1)
    iqr = q75 - q25
    # An IQR of zero means a channel that never moved in the training data; a
    # scale of one leaves it in microvolts rather than dividing by nothing.
    scale = np.where(iqr < 1e-6, 1.0, iqr)

    return NormalizationStats(
        method=method,
        channels=tuple(channels),
        centre=tuple(float(x) for x in centre),
        scale=tuple(float(x) for x in scale),
        fitted_on=tuple(sorted(set(participants))),
        n_epochs=total,
        clip_sigma=config.clip_sigma,
    )


def bandpass(
    signals: np.ndarray, rate: float, config: PreprocessConfig
) -> np.ndarray:
    """Zero-phase Butterworth band-pass, applied along the sample axis.

    Zero-phase (``filtfilt``) because a causal filter shifts slow waves in time,
    and a stage boundary that moves by a second is a stage boundary in the wrong
    epoch. Applied per epoch, so no filter state crosses a gap -- the edge
    transient that costs is a fraction of a second at each end of 30.
    """
    if config.highpass_hz <= 0 and config.lowpass_hz <= 0:
        return np.asarray(signals, dtype=np.float32)

    from scipy.signal import butter, filtfilt, sosfiltfilt

    nyquist = rate / 2.0
    low = config.highpass_hz / nyquist if config.highpass_hz > 0 else None
    high = config.lowpass_hz / nyquist if config.lowpass_hz > 0 else None
    if high is not None and high >= 1.0:
        high = None
    if low is not None and high is not None:
        sos = butter(config.filter_order, [low, high], btype="bandpass", output="sos")
        filtered = sosfiltfilt(sos, np.asarray(signals, dtype=np.float64), axis=-1)
    elif low is not None:
        b, a = butter(config.filter_order, low, btype="highpass")
        filtered = filtfilt(b, a, np.asarray(signals, dtype=np.float64), axis=-1)
    elif high is not None:
        b, a = butter(config.filter_order, high, btype="lowpass")
        filtered = filtfilt(b, a, np.asarray(signals, dtype=np.float64), axis=-1)
    else:
        return np.asarray(signals, dtype=np.float32)
    return np.ascontiguousarray(filtered, dtype=np.float32)
