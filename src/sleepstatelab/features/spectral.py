"""Spectral and time-domain features of one 30-second epoch.

**Provenance.** The feature definitions here are an attributed extraction from
PhysioML's ``physioml.neural.features`` (feature set ``sleep-eeg`` v1.0, revision
4f18d97), by the same author, re-implemented against this package's array layout
rather than imported. PhysioML is not a dependency of this repository and is not
modified by it; see docs/physioml_reuse.md for what was taken and what changed.

What changed: the EOG and EMG features are gone, because SleepStateLab reads the
two EEG derivations only; the extractor takes an ``[channels, samples]`` array
instead of one channel at a time; and the names are prefixed by channel index as
well as by derivation, so a one-channel ablation produces a matrix whose columns
can still be traced.

Sleep scoring is a spectral judgement made by eye -- slow high-amplitude waves,
spindles, the flattening of REM -- and these are the arithmetic of the same
observations. Relative band power is emitted alongside absolute because absolute
amplitude varies several-fold between people for reasons unrelated to sleep, and
a model trained on one person's microvolts and tested on another's is being
asked to generalise across the wrong thing.
"""

from __future__ import annotations

import numpy as np

from sleepstatelab.numerics import trapezoid

FEATURE_SET = "sleepstatelab-eeg-spectral"
FEATURE_SET_VERSION = "1.0"
DERIVED_FROM = "physioml.neural.features sleep-eeg 1.0 (rev 4f18d97)"

BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}
"""The bands sleep is scored in. Sigma is kept apart from beta because sleep
spindles live there and are one of the defining features of N2; folded into a
wide beta band they are invisible."""

TOTAL_BAND = (0.5, 30.0)
"""What relative power is relative to. Not the whole spectrum: above 30 Hz a
scalp recording is largely muscle, and dividing by it would make every relative
figure a function of how tense the participant's jaw was."""

_PER_CHANNEL = (
    *(f"{band}" for band in BANDS),
    *(f"{band}_rel" for band in BANDS),
    "delta_theta_ratio",
    "alpha_beta_ratio",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "entropy",
    "edge95",
    "total_power",
    "amplitude_p95",
    "zero_crossings",
)


def feature_names(channels: tuple[str, ...]) -> tuple[str, ...]:
    """Column names, in the order ``feature_matrix`` produces them."""
    names: list[str] = []
    for index, channel in enumerate(channels):
        slug = channel.replace(" ", "_").replace("-", "_").lower()
        names.extend(f"ch{index}_{slug}_{name}" for name in _PER_CHANNEL)
    return tuple(names)


def spectrum(samples: np.ndarray, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Power spectral density by Welch, at about a quarter of a hertz.

    Four-second segments: long enough to resolve the delta band, which starts at
    0.5 Hz and would otherwise be one bin, and short enough that a 30-second
    epoch holds several to average over.
    """
    from scipy.signal import welch

    data = np.asarray(samples, dtype=np.float64).ravel()
    segment = min(int(rate * 4), data.size)
    if segment < 8:
        return np.empty(0), np.empty(0)
    return welch(data, fs=rate, nperseg=segment)


def band_power(frequencies: np.ndarray, power: np.ndarray, band: tuple[float, float]) -> float:
    """Integrated power in one band."""
    inside = (frequencies >= band[0]) & (frequencies <= band[1])
    if inside.sum() < 2:
        return 0.0
    return trapezoid(power[inside], frequencies[inside])


def hjorth(samples: np.ndarray) -> tuple[float, float, float]:
    """Activity, mobility, complexity: the shape of a signal in three numbers.

    Older than the spectral measures and still used in sleep scoring, because
    they say something the band powers do not: mobility is a mean frequency and
    complexity is how far the signal departs from a pure sine at it.
    """
    data = np.asarray(samples, dtype=np.float64).ravel()
    if data.size < 3:
        return 0.0, 0.0, 0.0
    variance = float(np.var(data))
    if variance == 0.0:
        return 0.0, 0.0, 0.0
    first = np.diff(data)
    var_first = float(np.var(first))
    if var_first == 0.0:
        return variance, 0.0, 0.0
    second = np.diff(first)
    mobility = float(np.sqrt(var_first / variance))
    complexity = float(np.sqrt(float(np.var(second)) / var_first) / mobility)
    return variance, mobility, complexity


def spectral_entropy(power: np.ndarray) -> float:
    """How evenly the power is spread, normalised to [0, 1].

    Deep sleep concentrates power at the bottom of the spectrum and scores low;
    wake and REM spread it and score high.
    """
    total = float(np.sum(power))
    if total <= 0 or power.size < 2:
        return 0.0
    share = power / total
    share = share[share > 0]
    return float(-np.sum(share * np.log(share)) / np.log(power.size))


def spectral_edge(frequencies: np.ndarray, power: np.ndarray, share: float = 0.95) -> float:
    """The frequency below which ``share`` of the power lies."""
    if frequencies.size < 2:
        return 0.0
    cumulative = np.cumsum(power)
    if cumulative[-1] <= 0:
        return 0.0
    index = int(np.searchsorted(cumulative / cumulative[-1], share))
    return float(frequencies[min(index, frequencies.size - 1)])


def channel_features(samples: np.ndarray, rate: float) -> np.ndarray:
    """One channel's features, in the order of ``_PER_CHANNEL``."""
    data = np.asarray(samples, dtype=np.float64).ravel()
    frequencies, power = spectrum(data, rate)
    if frequencies.size == 0:
        return np.zeros(len(_PER_CHANNEL), dtype=np.float64)

    total = band_power(frequencies, power, TOTAL_BAND)
    absolute = {name: band_power(frequencies, power, band) for name, band in BANDS.items()}
    relative = {name: (value / total if total > 0 else 0.0) for name, value in absolute.items()}
    theta = absolute["theta"]
    beta = absolute["beta"]
    activity, mobility, complexity = hjorth(data)
    centred = data - float(np.mean(data))

    values = [
        *absolute.values(),
        *relative.values(),
        absolute["delta"] / theta if theta > 0 else 0.0,
        absolute["alpha"] / beta if beta > 0 else 0.0,
        activity,
        mobility,
        complexity,
        spectral_entropy(power),
        spectral_edge(frequencies, power),
        total,
        float(np.percentile(np.abs(data), 95)),
        float(np.count_nonzero(np.diff(np.signbit(centred))) / (data.size / rate)),
    ]
    return np.asarray(values, dtype=np.float64)


def epoch_features(epoch: np.ndarray, rate: float) -> np.ndarray:
    """Features for one ``[channels, samples]`` epoch, channels concatenated."""
    return np.concatenate([channel_features(channel, rate) for channel in epoch])


def feature_matrix(signals: np.ndarray, rate: float, *, progress: bool = False) -> np.ndarray:
    """``[n_epochs, n_features]`` for a stack of ``[n, channels, samples]`` epochs."""
    rows = []
    for index in range(signals.shape[0]):
        rows.append(epoch_features(signals[index], rate))
        if progress and index % 500 == 0:
            print(f"  features {index}/{signals.shape[0]}", flush=True)
    return np.vstack(rows) if rows else np.empty((0, 0))
