"""The manifest: what was found, what it claims about itself, and whether it holds.

Written before anything is trained, read by everything that trains. It is the
only place where a file path turns into a participant, and the only record of
what the header said at the time -- so a recording that changes underneath a
result can be detected instead of suspected.

Checksums are of the files as they sit on disk. When a PhysioNet
``SHA256SUMS.txt`` is available, each is also compared against the published
digest and the verdict is recorded per file, because "I have the file" and "I
have the right file" are different claims.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sleepstatelab.data.annotations import read_hypnogram
from sleepstatelab.data.discovery import Discovery, RecordingPair, discover
from sleepstatelab.data.edf_header import EDFHeader, read_header, to_microvolt_factor
from sleepstatelab.provenance import file_sha256

MANIFEST_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class ChannelRecord:
    """One EEG derivation, as the header declares it."""

    label: str
    physical_dimension: str
    microvolt_factor: float
    physical_min: float
    physical_max: float
    digital_min: float
    digital_max: float
    sampling_rate_hz: float
    prefiltering: str
    transducer: str


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One recording, and everything known about it before any signal is read."""

    participant_id: str
    recording_id: str
    night: int
    psg_file: str
    hypnogram_file: str
    psg_sha256: str
    hypnogram_sha256: str
    published_checksum_match: str
    """``match``, ``mismatch``, or ``unchecked``."""

    psg_start: str | None
    hypnogram_start: str | None
    psg_duration_seconds: float
    hypnogram_covered_seconds: float
    n_epochs_signal: int
    n_epochs_annotated: int
    """Epochs the hypnogram covers, which routinely runs past the end of the
    signal: the scorer's final annotation extends to the end of the tape."""

    n_epochs_labelled: int
    """Epochs carrying one of the five stages, counted over the whole hypnogram
    including any part beyond the signal. What survives into training is smaller
    and is counted at preparation time."""

    n_epochs_beyond_signal: int
    annotation_coverage: float
    """Share of the *signal's* epochs carrying any annotation at all, so this
    cannot exceed one."""

    channels: tuple[ChannelRecord, ...]
    stage_counts: dict[str, int]
    exclusions: dict[str, int]
    misaligned_annotations: tuple[str, ...]
    overlapping_epochs: int
    problems: tuple[str, ...]
    """Anything that would make this recording unusable or suspect. Empty is the
    only value that means "nothing was wrong"."""

    @property
    def usable(self) -> bool:
        return not self.problems


@dataclass(frozen=True, slots=True)
class Manifest:
    """Every discovered recording, with the settings the audit was run under."""

    version: str
    root: str
    channels: tuple[str, ...]
    epoch_seconds: float
    expected_rate_hz: float
    entries: tuple[ManifestEntry, ...]
    psg_without_hypnogram: tuple[str, ...] = ()
    hypnogram_without_psg: tuple[str, ...] = ()
    ignored_files: tuple[str, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def participants(self) -> tuple[str, ...]:
        return tuple(sorted({e.participant_id for e in self.entries}))

    @property
    def usable_entries(self) -> tuple[ManifestEntry, ...]:
        return tuple(e for e in self.entries if e.usable)

    def entry(self, recording_id: str) -> ManifestEntry:
        for found in self.entries:
            if found.recording_id == recording_id:
                return found
        raise KeyError(f"no recording {recording_id!r} in this manifest")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def read(cls, path: Path | str) -> Manifest:
        payload = json.loads(Path(path).read_text())
        entries = tuple(
            ManifestEntry(
                **{
                    **entry,
                    "channels": tuple(ChannelRecord(**c) for c in entry["channels"]),
                    "misaligned_annotations": tuple(entry["misaligned_annotations"]),
                    "problems": tuple(entry["problems"]),
                }
            )
            for entry in payload["entries"]
        )
        return cls(
            version=payload["version"],
            root=payload["root"],
            channels=tuple(payload["channels"]),
            epoch_seconds=payload["epoch_seconds"],
            expected_rate_hz=payload["expected_rate_hz"],
            entries=entries,
            psg_without_hypnogram=tuple(payload.get("psg_without_hypnogram", ())),
            hypnogram_without_psg=tuple(payload.get("hypnogram_without_psg", ())),
            ignored_files=tuple(payload.get("ignored_files", ())),
            notes=payload.get("notes", {}),
        )

    def summary(self) -> str:
        usable = len(self.usable_entries)
        stages = {}
        for entry in self.usable_entries:
            for stage, count in entry.stage_counts.items():
                stages[stage] = stages.get(stage, 0) + count
        total = sum(stages.values())
        share = (
            "  ".join(
                f"{name} {count / total:.1%}" for name, count in stages.items() if total
            )
            or "no labelled epochs"
        )
        return (
            f"{len(self.entries)} recording(s), {usable} usable, "
            f"{len(self.participants)} participant(s)\n"
            f"{total} labelled epochs: {share}"
        )


def _published_checksums(path: Path | str | None) -> dict[str, str]:
    """Parse a PhysioNet ``SHA256SUMS.txt`` into ``{file name: digest}``."""
    if not path:
        return {}
    where = Path(path).expanduser()
    if not where.is_file():
        raise FileNotFoundError(f"checksum file {where} does not exist")
    found: dict[str, str] = {}
    for line in where.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            found[Path(parts[1]).name] = parts[0].lower()
    return found


def _channel_records(
    header: EDFHeader, channels: tuple[str, ...], expected_rate: float
) -> tuple[tuple[ChannelRecord, ...], list[str]]:
    problems: list[str] = []
    records: list[ChannelRecord] = []
    for label in channels:
        if label not in header.labels:
            problems.append(f"missing channel {label!r}")
            continue
        signal = header.signal(label)
        try:
            factor = to_microvolt_factor(signal.physical_dimension)
        except Exception as error:  # broad: an unreadable unit is recorded, not raised
            problems.append(str(error))
            factor = float("nan")
        rate = header.sampling_rate(label)
        if abs(rate - expected_rate) > 1e-6:
            problems.append(f"{label!r} is at {rate:g} Hz, expected {expected_rate:g} Hz")
        records.append(
            ChannelRecord(
                label=label,
                physical_dimension=signal.physical_dimension,
                microvolt_factor=factor,
                physical_min=signal.physical_min,
                physical_max=signal.physical_max,
                digital_min=signal.digital_min,
                digital_max=signal.digital_max,
                sampling_rate_hz=rate,
                prefiltering=signal.prefiltering,
                transducer=signal.transducer,
            )
        )
    return tuple(records), problems


def audit_recording(
    pair: RecordingPair,
    *,
    channels: tuple[str, ...],
    epoch_seconds: float = 30.0,
    expected_rate_hz: float = 100.0,
    published: dict[str, str] | None = None,
    checksums: bool = True,
) -> ManifestEntry:
    """Everything the manifest records for one night, without reading a sample."""
    header = read_header(pair.psg_path)
    hypnogram_header = read_header(pair.hypnogram_path)
    signal_epochs = int(header.duration_seconds // epoch_seconds)
    hypnogram = read_hypnogram(pair.hypnogram_path, epoch_seconds=epoch_seconds)

    records, problems = _channel_records(header, channels, expected_rate_hz)

    if signal_epochs <= 0:
        problems.append("the PSG declares no whole epochs")
    if hypnogram.n_labelled == 0:
        problems.append("the hypnogram labels no epochs")
    if hypnogram.n_epochs > signal_epochs:
        # Common and benign in Sleep-EDF: the final wake annotation runs past the
        # end of the signal. It becomes an exclusion, not an error, and the
        # count is recorded so a large excess is visible.
        pass
    if header.start is not None and hypnogram_header.start is not None:
        drift = abs((header.start - hypnogram_header.start).total_seconds())
        if drift > 1.0:
            problems.append(
                f"PSG and hypnogram start times differ by {drift:.0f}s "
                f"({header.start.isoformat()} vs {hypnogram_header.start.isoformat()})"
            )
    if hypnogram.overlapping_epochs:
        problems.append(f"{hypnogram.overlapping_epochs} epoch(s) scored twice")

    published = published or {}
    psg_sha = file_sha256(pair.psg_path) if checksums else ""
    hyp_sha = file_sha256(pair.hypnogram_path) if checksums else ""
    verdict = "unchecked"
    if published and checksums:
        expected_psg = published.get(pair.psg_path.name)
        expected_hyp = published.get(pair.hypnogram_path.name)
        if expected_psg is None or expected_hyp is None:
            verdict = "unchecked"
        elif expected_psg == psg_sha and expected_hyp == hyp_sha:
            verdict = "match"
        else:
            verdict = "mismatch"
            problems.append("file checksum does not match the published digest")

    within = hypnogram.raw_text[:signal_epochs]
    annotated_within = int(sum(1 for text in within if text))
    coverage = annotated_within / signal_epochs if signal_epochs else 0.0
    beyond = max(hypnogram.n_epochs - signal_epochs, 0)

    return ManifestEntry(
        participant_id=pair.participant_id,
        recording_id=pair.recording_id,
        night=pair.night,
        psg_file=str(pair.psg_path),
        hypnogram_file=str(pair.hypnogram_path),
        psg_sha256=psg_sha,
        hypnogram_sha256=hyp_sha,
        published_checksum_match=verdict,
        psg_start=header.start.isoformat() if header.start else None,
        hypnogram_start=(
            hypnogram_header.start.isoformat() if hypnogram_header.start else None
        ),
        psg_duration_seconds=header.duration_seconds,
        hypnogram_covered_seconds=hypnogram.covered_seconds,
        n_epochs_signal=signal_epochs,
        n_epochs_annotated=hypnogram.n_epochs,
        n_epochs_labelled=hypnogram.n_labelled,
        n_epochs_beyond_signal=beyond,
        annotation_coverage=coverage,
        channels=records,
        stage_counts=hypnogram.counts(),
        exclusions=hypnogram.exclusions,
        misaligned_annotations=hypnogram.misaligned[:20],
        overlapping_epochs=hypnogram.overlapping_epochs,
        problems=tuple(problems),
    )


def build_manifest(
    root: Path | str,
    *,
    channels: tuple[str, ...],
    epoch_seconds: float = 30.0,
    expected_rate_hz: float = 100.0,
    participants: tuple[str, ...] = (),
    nights: tuple[int, ...] = (1, 2),
    checksum_file: str | Path | None = None,
    checksums: bool = True,
    progress: bool = False,
) -> Manifest:
    """Audit every paired recording under ``root``."""
    found: Discovery = discover(root, participants=participants, nights=nights)
    published = _published_checksums(checksum_file)

    entries: list[ManifestEntry] = []
    for index, pair in enumerate(found.pairs, start=1):
        if progress:
            print(f"[{index}/{len(found.pairs)}] {pair.recording_id}", flush=True)
        entries.append(
            audit_recording(
                pair,
                channels=channels,
                epoch_seconds=epoch_seconds,
                expected_rate_hz=expected_rate_hz,
                published=published,
                checksums=checksums,
            )
        )

    return Manifest(
        version=MANIFEST_VERSION,
        root=str(found.root),
        channels=channels,
        epoch_seconds=epoch_seconds,
        expected_rate_hz=expected_rate_hz,
        entries=tuple(entries),
        psg_without_hypnogram=found.psg_without_hypnogram,
        hypnogram_without_psg=found.hypnogram_without_psg,
        ignored_files=found.ignored,
        notes={
            "cohort": (
                "These are the recordings present in this data root. Sleep Cassette "
                "holds 153 recordings from 78 participants; a smaller manifest is a "
                "subset, not the cohort."
            ),
            "checksum_source": str(checksum_file) if checksum_file else "",
        },
    )
