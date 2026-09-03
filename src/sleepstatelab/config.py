"""Configuration: typed, defaulted, serialisable, and hashed.

Configuration is YAML on disk and dataclasses in memory. Two properties are
deliberate.

**Unknown keys are an error.** A misspelt ``lerning_rate`` that is silently
ignored produces a run at the default learning rate whose log says otherwise,
and there is no way to notice afterwards.

**A configuration has an identity.** ``Config.identity`` hashes the whole
resolved structure, and that hash is written into every checkpoint, prediction
file and report, so two artefacts can be compared for having been produced by
the same settings rather than by the same file name.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from sleepstatelab.labels import STAGES
from sleepstatelab.provenance import digest

EPOCH_SECONDS = 30.0
"""What a scored epoch is, in Sleep-EDF and in sleep scoring generally."""

DEFAULT_CHANNELS: tuple[str, ...] = ("EEG Fpz-Cz", "EEG Pz-Oz")
"""The two Sleep Cassette EEG derivations, in the order the arrays hold them.
Order is part of the model's contract: channel 0 is Fpz-Cz for every array this
package writes, and a checkpoint records it."""


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Where the recordings are and which parts of them are read."""

    root: str = "data/sleep-edf"
    """Directory searched recursively for ``SC4*-PSG.edf`` and its hypnograms.
    Sleep Cassette only: Sleep Telemetry files carry a different montage and a
    different protocol, and mixing them silently would make the cohort a
    different cohort."""

    cache_dir: str = "cache/epochs"
    channels: tuple[str, ...] = DEFAULT_CHANNELS
    sampling_rate_hz: float = 100.0
    epoch_seconds: float = EPOCH_SECONDS
    require_exact_rate: bool = True
    """Refuse a recording whose EEG is not at ``sampling_rate_hz``. Resampling
    is not done silently: it changes the spectrum the model sees."""

    participants: tuple[str, ...] = ()
    """Optional restriction, for pilots. Empty means every participant found."""

    nights: tuple[int, ...] = (1, 2)
    checksums: str = ""
    """Optional path to a PhysioNet ``SHA256SUMS.txt``. When given, every file
    in the manifest is checked against it and the verdict is recorded."""


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """What is done to the signal before it reaches a model.

    The default is deliberately thin. The band-limited 100 Hz Sleep Cassette EEG
    is already close to what a scorer looks at, and every extra transform is
    another thing that has to be identical between D1, D2 and D3 for their
    comparison to mean anything.
    """

    highpass_hz: float = 0.3
    """Removes electrode drift. Below the delta band, so nothing that a stage is
    defined by is touched."""

    lowpass_hz: float = 35.0
    """Above the bands sleep is scored in and below the 50 Hz the recorder
    band-limited at. Muscle above this is not stage information."""

    filter_order: int = 4
    normalization: str = "train_robust_channel"
    """``train_robust_channel``: subtract a median and divide by an inter-quartile
    range, one pair per channel, estimated over training participants only and
    then frozen.

    ``none``: raw microvolts.

    Per-epoch z-scoring is available as ``per_epoch_zscore`` and is not the
    default: it divides every epoch by its own amplitude, which deletes the
    absolute amplitude that separates N3 from N1 and hands the model a harder
    problem than the scorer had. It exists so the cost of doing it can be
    measured rather than argued about."""

    clip_sigma: float = 20.0
    """After normalisation, values beyond this are clipped. A movement artifact
    of 2000 microvolts otherwise dominates a batch's gradients."""

    qc_flat_sd_uv: float = 0.1
    qc_max_amplitude_uv: float = 500.0
    qc_clip_fraction: float = 0.02
    qc_reject: tuple[str, ...] = ("qc_flatline", "qc_clipped", "qc_high_amplitude")
    """Which quality-control codes remove an epoch from the eligible set. Muscle
    contamination is measured and recorded but does not reject: wake epochs
    legitimately carry muscle activity, and rejecting on it deletes part of the
    class it describes."""

    sleep_window: bool = False
    """The sensitivity analysis, off by default. When on, epochs outside the
    annotated sleep period plus ``sleep_window_margin_minutes`` are excluded.
    Primary preparation keeps every valid scored epoch."""

    sleep_window_margin_minutes: float = 30.0


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """How participants are divided, and by what seed."""

    seed: int = 20260901
    train_fraction: float = 0.6
    val_fraction: float = 0.2
    """The test fraction is the remainder, so the three always sum to one."""

    strategy: str = "grouped_random"
    label_budgets: tuple[float, ...] = (0.1, 0.25, 1.0)
    """Nested labelled-training-participant budgets for the D2/D3 comparison.
    Nested: the 10% participants are a subset of the 25%, which are a subset of
    the whole. Generated now, used by the benchmark that has not been run."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """D1's shape. Defaults are the ones the reported parameter count refers to."""

    stem_channels: int = 32
    stem_kernel: int = 49
    stem_stride: int = 6
    block_channels: tuple[int, ...] = (64, 96, 128)
    block_kernel: int = 9
    embedding_dim: int = 128
    dropout: float = 0.3
    n_classes: int = len(STAGES)

    context_epochs: int = 11
    """D2 only: how many epochs one prediction sees. Odd, so a centre exists.
    Eleven is five before and five after -- 2.5 minutes of past and 2.5 minutes
    of future, which is why D2 is an offline model."""

    temporal_layers: int = 2
    temporal_heads: int = 4
    temporal_dropout: float = 0.1

    centres_per_segment: int = 32
    """D2 only, and an implementation detail rather than a model choice: how many
    consecutive centres are predicted from one encoded stretch of a recording.
    Larger reuses more encodings and makes a batch more correlated in time;
    ``train.batch_size`` still sets how many centres are in an optimiser step."""


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """The supervised loop for D1."""

    device: str = "cpu"
    seed: int = 0
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    class_weighting: str = "inverse_frequency"
    """``inverse_frequency`` (normalised to mean one), or ``none``. Computed on
    training participants only."""

    early_stopping_patience: int = 6
    """Passes over the training set without an improvement in validation
    participant-mean macro-F1 before stopping. The selected checkpoint is the
    best validation score, never the last epoch and never anything measured on
    test."""

    grad_clip: float = 5.0
    num_workers: int = 0
    max_train_batches: int = 0
    """0 means the whole training set. Non-zero truncates each pass, for smoke
    runs; it is recorded in the checkpoint so a truncated run cannot be mistaken
    for a full one."""


@dataclass(frozen=True, slots=True)
class PretrainConfig:
    """Masked reconstruction, the self-supervised stage D3's encoder starts from."""

    patch_samples: int = 0
    """0 derives the patch from the encoder: one token per patch, 96 samples at
    this package's defaults, which is 0.96 s. Long enough that filling a hidden
    patch back in requires knowing what EEG does rather than interpolating
    between the samples either side of it, and aligned so the decoder can draw
    the patch its token covers at full sample resolution. Set it by hand only
    with a reason: a patch that straddles tokens asks an output block to answer
    for input it never saw."""

    mask_ratio: float = 0.5
    decoder_width: int = 64
    """Deliberately small. A strong decoder can reconstruct from a weak
    representation, which moves the work out of the encoder -- and the encoder is
    the only part that is kept."""

    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 6
    max_batches: int = 0


@dataclass(frozen=True, slots=True)
class Config:
    """The whole resolved configuration for one experiment."""

    name: str = "default"
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)

    @property
    def samples_per_epoch(self) -> int:
        """3,000 at 100 Hz for 30 seconds, computed rather than written down."""
        return round(self.data.sampling_rate_hz * self.data.epoch_seconds)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def identity(self) -> str:
        return digest(self.to_dict())

    @property
    def preprocessing_identity(self) -> str:
        """Identity of everything that changes the numbers a model sees."""
        return digest(
            {
                "preprocess": asdict(self.preprocess),
                "channels": list(self.data.channels),
                "rate": self.data.sampling_rate_hz,
                "epoch_seconds": self.data.epoch_seconds,
            }
        )

    def write(self, path: Path | str) -> None:
        import yaml

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _build(cls: type, payload: dict[str, Any], where: str) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(payload) - set(known))
    if unknown:
        raise ValueError(
            f"unknown configuration key(s) {unknown} in section {where!r}; "
            f"expected any of {sorted(known)}"
        )
    # Sections are one level deep by design, so a list is the only structure
    # that needs converting: YAML gives lists where the dataclasses declare
    # tuples, and a tuple is what makes a configuration hashable.
    kwargs: dict[str, Any] = {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in payload.items()
    }
    return cls(**kwargs)


def from_dict(payload: dict[str, Any]) -> Config:
    """Build a configuration, rejecting anything not declared above."""
    sections = {
        "data": DataConfig,
        "preprocess": PreprocessConfig,
        "split": SplitConfig,
        "model": ModelConfig,
        "train": TrainConfig,
        "pretrain": PretrainConfig,
    }
    known = {f.name for f in fields(Config)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"unknown top-level configuration key(s) {unknown}")
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        if key in sections:
            if not isinstance(value, dict):
                raise ValueError(f"section {key!r} must be a mapping")
            kwargs[key] = _build(sections[key], value, key)
        else:
            kwargs[key] = value
    return Config(**kwargs)


def load(path: Path | str | None) -> Config:
    """Read a configuration file, or return the defaults when given nothing."""
    if path is None:
        return Config()
    import yaml

    payload = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a mapping")
    return from_dict(payload)
