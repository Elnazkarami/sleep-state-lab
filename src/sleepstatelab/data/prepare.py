"""Building the epoch cache, and loading it back the same way every time.

Preparation is separated from training because it is the slow, deterministic
half: reading 153 nights takes minutes, and every model in the repository should
be looking at exactly the same arrays afterwards. The cache is keyed by the
preprocessing identity, so a changed filter or a changed epoch length writes to
a different directory rather than silently reusing the old one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from sleepstatelab.config import Config
from sleepstatelab.data.discovery import discover
from sleepstatelab.data.epochs import (
    QC_CLIPPED,
    QC_FLATLINE,
    QC_HIGH_AMPLITUDE,
    EpochedRecording,
    QCPolicy,
    epoch_recording,
)
from sleepstatelab.labels import STAGES

REJECT_FLAG = {
    "qc_flatline": QC_FLATLINE,
    "qc_clipped": QC_CLIPPED,
    "qc_high_amplitude": QC_HIGH_AMPLITUDE,
}


def reject_mask_flags(names: tuple[str, ...]) -> int:
    """Turn configured rejection reasons into the bit mask the arrays carry."""
    flags = 0
    for name in names:
        if name not in REJECT_FLAG:
            raise ValueError(
                f"{name!r} is not a rejectable quality-control code; "
                f"expected any of {sorted(REJECT_FLAG)}"
            )
        flags |= REJECT_FLAG[name]
    return flags


def cache_root(config: Config) -> Path:
    """Where this configuration's epochs live: one directory per preprocessing."""
    return Path(config.data.cache_dir) / config.preprocessing_identity


@dataclass(frozen=True, slots=True)
class PreparationReport:
    """What preparation did, in numbers that can be checked against a manifest."""

    cache_dir: str
    preprocessing_id: str
    recordings: int
    participants: int
    stored_epochs: int
    eligible_epochs: int
    counts_by_stage: dict[str, int]
    exclusions: dict[str, int]
    per_recording: dict[str, dict[str, int]]

    def summary(self) -> str:
        total = sum(self.counts_by_stage.values()) or 1
        share = "  ".join(f"{k} {v / total:.1%}" for k, v in self.counts_by_stage.items())
        return (
            f"{self.recordings} recording(s) from {self.participants} participant(s)\n"
            f"{self.stored_epochs} stored epochs, {self.eligible_epochs} eligible "
            f"after quality control\n{share}"
        )

    def write(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=str))


def prepare(config: Config, *, progress: bool = False, force: bool = False) -> PreparationReport:
    """Epoch every discovered recording into the cache, skipping what is there."""
    found = discover(
        config.data.root,
        participants=tuple(config.data.participants),
        nights=tuple(config.data.nights),
    )
    target = cache_root(config)
    target.mkdir(parents=True, exist_ok=True)
    policy = QCPolicy(
        flat_sd_uv=config.preprocess.qc_flat_sd_uv,
        max_amplitude_uv=config.preprocess.qc_max_amplitude_uv,
        clip_fraction=config.preprocess.qc_clip_fraction,
    )
    reject = reject_mask_flags(tuple(config.preprocess.qc_reject))

    totals = dict.fromkeys(STAGES, 0)
    exclusions: dict[str, int] = {}
    per_recording: dict[str, dict[str, int]] = {}
    stored = 0
    eligible = 0

    for index, pair in enumerate(found.pairs, start=1):
        path = target / f"{pair.recording_id}.npz"
        if path.exists() and not force:
            record = EpochedRecording.load(path)
            action = "cached"
        else:
            record = epoch_recording(
                pair,
                channels=tuple(config.data.channels),
                sampling_rate_hz=config.data.sampling_rate_hz,
                epoch_seconds=config.data.epoch_seconds,
                policy=policy,
                sleep_window=config.preprocess.sleep_window,
                sleep_window_margin_minutes=config.preprocess.sleep_window_margin_minutes,
            )
            record.save(path)
            action = "written"
        keep = record.eligible(reject)
        stored += record.n_epochs
        eligible += int(np.count_nonzero(keep))
        for name, count in record.counts_by_stage.items():
            totals[name] += count
        for name, count in record.exclusions.items():
            exclusions[name] = exclusions.get(name, 0) + count
        per_recording[record.recording_id] = {
            "stored": record.n_epochs,
            "eligible": int(np.count_nonzero(keep)),
            **record.counts_by_stage,
        }
        if progress:
            print(
                f"[{index}/{len(found.pairs)}] {pair.recording_id}: {action}, "
                f"{record.n_epochs} epochs, {int(np.count_nonzero(keep))} eligible",
                flush=True,
            )

    report = PreparationReport(
        cache_dir=str(target),
        preprocessing_id=config.preprocessing_identity,
        recordings=len(found.pairs),
        participants=len(found.participants),
        stored_epochs=stored,
        eligible_epochs=eligible,
        counts_by_stage=totals,
        exclusions=exclusions,
        per_recording=per_recording,
    )
    report.write(target / "preparation_report.json")
    return report


def load_cached(config: Config, participants: tuple[str, ...] | None = None) -> list[EpochedRecording]:
    """Every cached recording, optionally restricted to a set of participants."""
    target = cache_root(config)
    if not target.is_dir():
        raise FileNotFoundError(
            f"no epoch cache at {target}. Run `sleepstatelab prepare` first."
        )
    wanted = set(participants) if participants is not None else None
    found: list[EpochedRecording] = []
    for path in sorted(target.glob("*.npz")):
        record = EpochedRecording.load(path)
        if wanted is not None and record.participant_id not in wanted:
            continue
        if tuple(record.channels) != tuple(config.data.channels):
            raise ValueError(
                f"{path.name} holds channels {record.channels}, the configuration asks "
                f"for {tuple(config.data.channels)}"
            )
        found.append(record)
    if not found:
        raise FileNotFoundError(
            f"no cached recordings under {target}"
            + (f" for participants {sorted(wanted)}" if wanted else "")
        )
    return found
