"""Generated recordings, written as real EDF files.

The synthetic path exists so that every stage of the pipeline -- header parsing,
unit conversion, annotation expansion, epoching, gap handling, training,
prediction, scoring -- can be exercised on a machine with no data and checked
against a ground truth that is known by construction. Signals with a known
spectrum per stage, hypnograms with deliberate movement epochs and unscored
gaps, and files an EDF reader has to actually parse.

**Nothing produced here is a result.** Every artefact this module writes is
labelled synthetic, and the smoke run puts its output in its own directory. A
number computed on generated sleep says the code runs; it says nothing about
sleep.

The writer is minimal but conformant: EDF for the PSG, EDF+C with an
``EDF Annotations`` signal for the hypnogram, which is exactly the shape
Sleep-EDF Expanded uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sleepstatelab.labels import STAGES

#: Rough spectral character of each stage, as (band centre Hz, amplitude uV)
#: pairs. Not a model of sleep -- a set of signals a classifier can separate, so
#: that "the training loop reduces the loss" is a statement about the loop.
STAGE_PROFILE: dict[str, tuple[tuple[float, float], ...]] = {
    "Wake": ((10.0, 12.0), (24.0, 8.0)),
    "N1": ((6.0, 14.0), (10.0, 6.0)),
    "N2": ((13.5, 18.0), (2.0, 14.0)),
    "N3": ((1.2, 55.0), (3.0, 20.0)),
    "REM": ((7.0, 12.0), (18.0, 7.0)),
}


def _ascii(text: str, width: int) -> bytes:
    return text.encode("ascii", errors="replace")[:width].ljust(width, b" ")


def _number(value: float, width: int) -> bytes:
    text = f"{value:g}"
    if len(text) > width:
        text = f"{value:.{max(width - 6, 0)}f}"[:width]
    return _ascii(text, width)


def write_edf(
    path: Path | str,
    signals: dict[str, np.ndarray],
    rate: float,
    *,
    start: str = "01.01.85",
    start_time: str = "22.00.00",
    physical_range: float = 200.0,
    unit: str = "uV",
    patient: str = "X X X X",
    recording: str = "Startdate 01-JAN-1985 X X X",
) -> Path:
    """Write signals to an EDF file with one-second data records.

    Values are digitised against the declared physical range, which is what
    gives the round-trip its meaning: a test that reads this file back and finds
    the amplitudes it wrote has verified the whole calibration path, not just
    the array plumbing.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    labels = list(signals)
    samples_per_record = round(rate)
    lengths = {len(v) for v in signals.values()}
    if len(lengths) != 1:
        raise ValueError("all signals must be the same length")
    n_records = int(lengths.pop() // samples_per_record)

    header = bytearray()
    header += _ascii("0", 8)
    header += _ascii(patient, 80)
    header += _ascii(recording, 80)
    header += _ascii(start, 8)
    header += _ascii(start_time, 8)
    header += _number(256 * (len(labels) + 1), 8)
    header += _ascii("", 44)
    header += _number(n_records, 8)
    header += _number(1, 8)
    header += _number(len(labels), 4)

    header += b"".join(_ascii(label, 16) for label in labels)
    header += b"".join(_ascii("Ag-AgCl electrodes", 80) for _ in labels)
    header += b"".join(_ascii(unit, 8) for _ in labels)
    header += b"".join(_number(-physical_range, 8) for _ in labels)
    header += b"".join(_number(physical_range, 8) for _ in labels)
    header += b"".join(_number(-2048, 8) for _ in labels)
    header += b"".join(_number(2047, 8) for _ in labels)
    header += b"".join(_ascii("HP:0.5Hz LP:100Hz", 80) for _ in labels)
    header += b"".join(_number(samples_per_record, 8) for _ in labels)
    header += b"".join(_ascii("", 32) for _ in labels)

    gain = (2 * physical_range) / (2047 - -2048)
    body = bytearray()
    for record in range(n_records):
        for label in labels:
            block = signals[label][
                record * samples_per_record : (record + 1) * samples_per_record
            ]
            digital = np.clip(np.round(np.asarray(block) / gain), -2048, 2047).astype("<i2")
            body += digital.tobytes()

    target.write_bytes(bytes(header) + bytes(body))
    return target


def _tal(onset: float, duration: float, text: str) -> bytes:
    """One time-stamped annotation list, as EDF+ writes them."""
    sign = "+" if onset >= 0 else "-"
    return (
        f"{sign}{abs(onset):g}\x15{duration:g}\x14{text}\x14\x00".encode("ascii", "replace")
    )


def write_hypnogram_edf(
    path: Path | str,
    runs: list[tuple[float, float, str]],
    *,
    start: str = "01.01.85",
    start_time: str = "22.00.00",
) -> Path:
    """Write an annotation-only EDF+C file: onset, duration, description.

    Each annotation is placed in its own data record, preceded by the record's
    own time-keeping annotation, which is what the format requires and what an
    established reader will expect to find.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    records = [_tal(onset, 0.0, "") + _tal(onset, duration, text) for onset, duration, text in runs]
    width = max((len(r) for r in records), default=16)
    width += (-width) % 2  # a data record holds whole 2-byte samples
    padded = [r.ljust(width, b"\x00") for r in records]

    header = bytearray()
    header += _ascii("0", 8)
    header += _ascii("X X X X", 80)
    header += _ascii("Startdate 01-JAN-1985 X X X", 80)
    header += _ascii(start, 8)
    header += _ascii(start_time, 8)
    header += _number(256 * 2, 8)
    header += _ascii("EDF+C", 44)
    header += _number(len(padded), 8)
    header += _number(0, 8)
    header += _number(1, 4)
    header += _ascii("EDF Annotations", 16)
    header += _ascii("", 80)
    header += _ascii("", 8)
    header += _number(-1, 8)
    header += _number(1, 8)
    header += _number(-32768, 8)
    header += _number(32767, 8)
    header += _ascii("", 80)
    header += _number(width // 2, 8)
    header += _ascii("", 32)

    target.write_bytes(bytes(header) + b"".join(padded))
    return target


@dataclass(frozen=True, slots=True)
class SyntheticNight:
    """A generated night and the truth it was generated from."""

    participant_id: str
    night: int
    psg_path: Path
    hypnogram_path: Path
    stages: tuple[str, ...]
    """The stage of every epoch, including the ones deliberately left unscored."""


def make_night(
    directory: Path | str,
    participant: int,
    night: int = 1,
    *,
    n_epochs: int = 60,
    rate: float = 100.0,
    epoch_seconds: float = 30.0,
    channels: tuple[str, ...] = ("EEG Fpz-Cz", "EEG Pz-Oz"),
    seed: int = 0,
    movement_epochs: tuple[int, ...] = (7,),
    unscored_epochs: tuple[int, ...] = (12, 13),
) -> SyntheticNight:
    """One generated night, written to disk as a PSG and a hypnogram.

    ``movement_epochs`` and ``unscored_epochs`` are the point: they put real
    holes in the middle of the scoring, so that gap handling is exercised rather
    than assumed. A per-participant amplitude factor is applied, because
    between-person amplitude variation is the thing normalisation has to survive.
    """
    rng = np.random.default_rng(seed + participant * 977 + night)
    samples = round(rate * epoch_seconds)
    order = [STAGES[(participant + i // 4) % len(STAGES)] for i in range(n_epochs)]
    person_gain = 0.7 + 0.6 * rng.random()

    data = {label: np.zeros(n_epochs * samples, dtype=np.float64) for label in channels}
    t = np.arange(samples) / rate
    for index, stage in enumerate(order):
        for channel_index, label in enumerate(channels):
            wave = rng.normal(0.0, 3.0, samples)
            for frequency, amplitude in STAGE_PROFILE[stage]:
                phase = rng.uniform(0, 2 * np.pi)
                scale = amplitude * (1.0 if channel_index == 0 else 0.75)
                wave += scale * np.sin(2 * np.pi * frequency * t + phase)
            data[label][index * samples : (index + 1) * samples] = wave * person_gain

    participant_id = f"SC4{participant:02d}"
    stem = f"SC4{participant:02d}{night}E0"
    psg = write_edf(Path(directory) / f"{stem}-PSG.edf", data, rate)

    runs: list[tuple[float, float, str]] = []
    for index, stage in enumerate(order):
        onset = index * epoch_seconds
        if index in movement_epochs:
            text = "Movement time"
        elif index in unscored_epochs:
            text = "Sleep stage ?"
        else:
            text = {
                "Wake": "Sleep stage W",
                "N1": "Sleep stage 1",
                "N2": "Sleep stage 2",
                "N3": "Sleep stage 3",
                "REM": "Sleep stage R",
            }[stage]
        runs.append((onset, epoch_seconds, text))
    hypnogram = write_hypnogram_edf(
        Path(directory) / f"SC4{participant:02d}{night}EC-Hypnogram.edf", runs
    )
    return SyntheticNight(
        participant_id=participant_id,
        night=night,
        psg_path=psg,
        hypnogram_path=hypnogram,
        stages=tuple(order),
    )


def make_cohort(
    directory: Path | str,
    n_participants: int = 6,
    *,
    nights: tuple[int, ...] = (1,),
    n_epochs: int = 60,
    seed: int = 0,
) -> list[SyntheticNight]:
    """A small generated cohort, one or two nights each."""
    return [
        make_night(directory, participant, night, n_epochs=n_epochs, seed=seed)
        for participant in range(n_participants)
        for night in nights
    ]
