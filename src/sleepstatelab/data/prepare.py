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


def free_bytes(path: Path | str) -> int:
    """Space left on the filesystem holding ``path``."""
    import shutil

    where = Path(path)
    while not where.exists() and where != where.parent:
        where = where.parent
    return int(shutil.disk_usage(where).free)


def cache_root(config: Config) -> Path:
    """Where this configuration's epochs live: one directory per preprocessing."""
    return Path(config.data.cache_dir) / config.preprocessing_identity


@dataclass(frozen=True, slots=True)
class PreparationReport:
    """What preparation did, in numbers that can be checked against a manifest."""

    cache_dir: str
    preprocessing_id: str
    recordings: int
    """How many were prepared. Equal to ``recordings_discovered`` unless the run
    stopped early."""

    recordings_discovered: int
    stopped_early: str
    """Empty when the run completed. Otherwise says why it did not, so a partial
    cache cannot be mistaken for a whole one."""

    participants: int
    stored_epochs: int
    eligible_epochs: int
    counts_by_stage: dict[str, int]
    exclusions: dict[str, int]
    per_recording: dict[str, dict[str, int]]

    def summary(self) -> str:
        total = sum(self.counts_by_stage.values()) or 1
        share = "  ".join(f"{k} {v / total:.1%}" for k, v in self.counts_by_stage.items())
        lines = [
            f"{self.recordings} of {self.recordings_discovered} recording(s) prepared, "
            f"from {self.participants} participant(s)",
            f"{self.stored_epochs} stored epochs, {self.eligible_epochs} eligible "
            f"after quality control",
            share,
        ]
        if self.stopped_early:
            lines.append(f"INCOMPLETE: {self.stopped_early}")
        return "\n".join(lines)

    def write(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=str))


def prepare(
    config: Config,
    *,
    progress: bool = False,
    force: bool = False,
    min_free_gb: float = 1.0,
) -> PreparationReport:
    """Epoch every discovered recording into the cache, skipping what is there.

    ``min_free_gb`` stops the run cleanly when the filesystem is nearly full,
    rather than letting the write that crosses the line produce a truncated
    cache file. This is not hypothetical: a full disk during an earlier run left
    a 157-byte EDF that still looked like a file, and a half-written cache entry
    is worse, because nothing downstream checks it the way a published checksum
    checks a download. Whatever was completed stays usable; the report says how
    far it got.
    """
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

    stopped_early = ""
    for index, pair in enumerate(found.pairs, start=1):
        path = target / f"{pair.recording_id}.npz"
        available = free_bytes(target) / 1e9
        if available < min_free_gb and not (path.exists() and not force):
            stopped_early = (
                f"stopped before {pair.recording_id}: {available:.2f} GB free, "
                f"below the {min_free_gb:.2f} GB floor. "
                f"{index - 1} of {len(found.pairs)} recording(s) were prepared and "
                "are usable; free space and run prepare again to continue."
            )
            if progress:
                print(f"\n{stopped_early}", flush=True)
            break
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

    prepared_participants = {
        recording_id.split("-n")[0] for recording_id in per_recording
    }
    report = PreparationReport(
        cache_dir=str(target),
        preprocessing_id=config.preprocessing_identity,
        recordings=len(per_recording),
        recordings_discovered=len(found.pairs),
        stopped_early=stopped_early,
        participants=len(prepared_participants),
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
    # The configuration decides which cohort an experiment is, and loading has
    # to honour it. The cache is keyed by preprocessing, not by cohort, so it
    # accumulates: preparing the full dataset puts every participant and every
    # night into the same directory. Filtering only by the participants a caller
    # asks for is not enough -- a pilot pinned to first nights would still pick
    # up second nights that were prepared later, and did, silently doubling an
    # evaluation set between one model's predictions and the next's.
    wanted = set(participants) if participants is not None else None
    configured = set(config.data.participants) or None
    nights = set(config.data.nights) if config.data.nights else None

    found: list[EpochedRecording] = []
    skipped = 0
    for path in sorted(target.glob("*.npz")):
        record = EpochedRecording.load(path)
        if wanted is not None and record.participant_id not in wanted:
            continue
        if configured is not None and record.participant_id not in configured:
            skipped += 1
            continue
        if nights is not None and record.night not in nights:
            skipped += 1
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
            + (
                f"; {skipped} cached recording(s) were excluded by the "
                "configuration's participants/nights"
                if skipped
                else ""
            )
        )
    return found
