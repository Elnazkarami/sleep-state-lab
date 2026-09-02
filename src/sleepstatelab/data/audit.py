"""The single-recording audit: what one real night looks like, in numbers and pictures.

Written for the reader who has to decide whether to believe anything downstream.
It takes one recording and shows, from the actual file: what the header claims,
what the waveforms look like in each stage, what their spectra look like, where
the hypnogram sits against the signal, how many epochs survived, and what was
excluded and why.

The figure is optional -- it needs matplotlib -- and the markdown is not. A
machine without a plotting stack still produces the audit; it produces it
without pictures and says so.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sleepstatelab.config import Config
from sleepstatelab.data.discovery import discover
from sleepstatelab.data.edf_header import read_header
from sleepstatelab.data.epochs import QC_NAMES, EpochedRecording, epoch_recording
from sleepstatelab.data.manifest import audit_recording
from sleepstatelab.features.spectral import spectrum
from sleepstatelab.labels import STAGES


def _stage_examples(record: EpochedRecording) -> dict[str, int]:
    """One representative epoch index per stage: the first that passes quality control."""
    found: dict[str, int] = {}
    for index, name in enumerate(STAGES):
        rows = np.flatnonzero((record.labels == index) & (record.qc == 0))
        if rows.size:
            found[name] = int(rows[rows.size // 2])
    return found


def figure(record: EpochedRecording, path: Path | str) -> str | None:
    """Waveforms, spectra and the hypnogram, from the recording itself."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    examples = _stage_examples(record)
    rate = record.sampling_rate_hz
    fig = plt.figure(figsize=(13, 9))
    grid = fig.add_gridspec(3, len(examples) or 1, height_ratios=[1.1, 1.1, 0.9])

    for column, (stage, row_index) in enumerate(examples.items()):
        epoch = record.signals[row_index]
        axis = fig.add_subplot(grid[0, column])
        seconds = np.arange(epoch.shape[1]) / rate
        for channel_index, label in enumerate(record.channels):
            axis.plot(
                seconds,
                epoch[channel_index] + (0 if channel_index == 0 else -150),
                linewidth=0.4,
                label=label,
            )
        axis.set_title(f"{stage} (epoch {record.epoch_index[row_index]})", fontsize=9)
        axis.set_xlabel("s", fontsize=8)
        if column == 0:
            axis.set_ylabel("microvolts (Pz-Oz offset -150)", fontsize=8)
            axis.legend(fontsize=6, loc="upper right")
        axis.tick_params(labelsize=7)

        spectral = fig.add_subplot(grid[1, column])
        for channel_index in range(epoch.shape[0]):
            frequencies, power = spectrum(epoch[channel_index], rate)
            inside = frequencies <= 32
            spectral.semilogy(frequencies[inside], power[inside], linewidth=0.8)
        spectral.set_xlabel("Hz", fontsize=8)
        if column == 0:
            spectral.set_ylabel("PSD (uV^2/Hz)", fontsize=8)
        spectral.tick_params(labelsize=7)

    hypnogram = fig.add_subplot(grid[2, :])
    hypnogram.step(record.epoch_index / 120.0, record.labels, where="post", linewidth=0.8)
    hypnogram.set_yticks(range(len(STAGES)))
    hypnogram.set_yticklabels(STAGES, fontsize=8)
    hypnogram.set_xlabel("hours from the start of the recording", fontsize=8)
    hypnogram.set_title(
        f"{record.recording_id}: {record.n_epochs} retained epochs "
        "(gaps are excluded epochs, not missing time)",
        fontsize=9,
    )
    hypnogram.tick_params(labelsize=7)

    fig.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return str(target)


def audit_report(
    config: Config, recording_id: str | None = None, *, out_dir: Path | str = "outputs/audit"
) -> tuple[str, Path]:
    """Markdown, and a figure beside it, for one real recording."""
    found = discover(
        config.data.root,
        participants=tuple(config.data.participants),
        nights=tuple(config.data.nights),
    )
    pairs = {pair.recording_id: pair for pair in found.pairs}
    if recording_id is None:
        recording_id = sorted(pairs)[0]
    if recording_id not in pairs:
        raise KeyError(f"{recording_id!r} is not among {sorted(pairs)}")
    pair = pairs[recording_id]

    from sleepstatelab.data.manifest import _published_checksums

    entry = audit_recording(
        pair,
        channels=tuple(config.data.channels),
        epoch_seconds=config.data.epoch_seconds,
        expected_rate_hz=config.data.sampling_rate_hz,
        published=_published_checksums(config.data.checksums or None),
        checksums=True,
    )
    record = epoch_recording(
        pair,
        channels=tuple(config.data.channels),
        sampling_rate_hz=config.data.sampling_rate_hz,
        epoch_seconds=config.data.epoch_seconds,
    )
    header = read_header(pair.psg_path)

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    image = figure(record, target / f"{recording_id}_audit.png")

    stage_rows = "\n".join(
        f"| {name} | {record.counts_by_stage[name]} | "
        f"{record.counts_by_stage[name] / max(record.n_epochs, 1):.1%} |"
        for name in STAGES
    )
    exclusion_rows = "\n".join(
        f"| {reason} | {count} |" for reason, count in sorted(record.exclusions.items())
    )
    channel_rows = "\n".join(
        f"| {c.label} | {c.physical_dimension} | {c.sampling_rate_hz:g} | "
        f"[{c.physical_min:g}, {c.physical_max:g}] | {c.prefiltering} |"
        for c in entry.channels
    )
    qc_rows = "\n".join(
        f"| {name} | {int(np.count_nonzero(record.qc & flag))} |"
        for flag, name in sorted(QC_NAMES.items())
    )
    examples = _stage_examples(record)
    amplitude = {
        stage: float(np.percentile(np.abs(record.signals[index]), 95))
        for stage, index in examples.items()
    }

    text = f"""# Data audit: {recording_id}

One real recording, read from the file named below. Every number on this page was
computed from it by `sleepstatelab audit-report`; nothing is quoted from elsewhere.

## Files

| what | value |
| --- | --- |
| PSG | `{pair.psg_path.name}` |
| hypnogram | `{pair.hypnogram_path.name}` |
| PSG SHA-256 | `{entry.psg_sha256}` |
| hypnogram SHA-256 | `{entry.hypnogram_sha256}` |
| published checksum | {entry.published_checksum_match} |
| PSG start | {entry.psg_start} |
| hypnogram start | {entry.hypnogram_start} |
| PSG duration | {entry.psg_duration_seconds / 3600:.2f} h |
| EDF+ | {header.is_edf_plus} |

The PSG and the hypnogram start at the same wall-clock second. That is the check
that makes epoch *i* of the signal and epoch *i* of the scoring the same 30
seconds; without it, every label could be shifted and nothing downstream would
notice.

## Channels, as the header declares them

| channel | unit | Hz | physical range | prefiltering |
| --- | --- | ---: | --- | --- |
{channel_rows}

Units are read from the header and converted to microvolts explicitly. The
declared physical range is also the amplifier's rail, and saturation is detected
against it rather than against a fixed threshold.

## Epochs

| what | value |
| --- | ---: |
| epochs in the signal | {entry.n_epochs_signal} |
| epochs the hypnogram covers | {entry.n_epochs_annotated} |
| hypnogram epochs past the end of the signal | {entry.n_epochs_beyond_signal} |
| annotation coverage of the signal | {entry.annotation_coverage:.3f} |
| epochs retained | {record.n_epochs} |
| first/last retained epoch index | {int(record.epoch_index[0])} / {int(record.epoch_index[-1])} |
| discontinuities in the retained indices | {int(np.count_nonzero(np.diff(record.epoch_index) > 1))} |

Retained epochs keep their original index. A discontinuity in that index is an
excluded epoch, not missing time, and no downstream code treats the two epochs
either side of one as neighbours.

## Retained stages

| stage | epochs | share |
| --- | ---: | ---: |
{stage_rows}

This is the primary preparation: every valid scored epoch, with no
sleep-stage-based cropping. Wake dominates because a Sleep Cassette recorder ran
for roughly twenty hours around one night's sleep. The optional sleep-window
sensitivity analysis changes this distribution substantially and is reported
separately wherever it is used.

## Exclusions

| reason | epochs |
| --- | ---: |
{exclusion_rows}

## Quality control on the retained epochs

| code | epochs flagged |
| --- | ---: |
{qc_rows}

95th-percentile absolute amplitude of the example epoch shown for each stage,
in microvolts: {json.dumps({k: round(v, 1) for k, v in amplitude.items()})}.

## Figure

{f"![audit figure]({Path(image).name})" if image else "_matplotlib is not installed, so no figure was produced._"}

Top row: raw waveforms for one epoch of each stage, in microvolts. Middle row:
their power spectra, log scale, to 32 Hz. Bottom: the retained hypnogram against
time.
"""
    (target / f"{recording_id}_audit.md").write_text(text)
    return text, target / f"{recording_id}_audit.md"
