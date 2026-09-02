"""The EDF header, read directly, so the file can be checked against its claims.

MNE reads the signals, and it is the right tool for that. It is not the right
tool for asking what the file *says*: it has already applied the physical
calibration and converted to volts by the time anything is visible, so a file
whose header declares millivolts and a file that declares microvolts look
identical downstream. The unit check has to happen before that conversion, and
that means parsing the header.

The format is fixed-width ASCII and fully specified, so this is short:
256 bytes of general header, then 256 bytes per signal, laid out field by field
rather than signal by signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


class EDFHeaderError(RuntimeError):
    """Raised when a file is not an EDF, or its header is internally inconsistent."""


@dataclass(frozen=True, slots=True)
class SignalHeader:
    """One signal's declared identity and calibration."""

    label: str
    transducer: str
    physical_dimension: str
    physical_min: float
    physical_max: float
    digital_min: float
    digital_max: float
    prefiltering: str
    samples_per_record: int

    @property
    def gain(self) -> float:
        """Physical units per digital unit, as the header declares them."""
        span = self.digital_max - self.digital_min
        if span == 0:
            raise EDFHeaderError(f"signal {self.label!r} declares a zero digital range")
        return (self.physical_max - self.physical_min) / span


@dataclass(frozen=True, slots=True)
class EDFHeader:
    """What an EDF file says about itself, before anything is interpreted."""

    path: Path
    version: str
    patient: str
    recording: str
    start: datetime | None
    header_bytes: int
    reserved: str
    n_records: int
    record_seconds: float
    signals: tuple[SignalHeader, ...]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(s.label for s in self.signals)

    @property
    def duration_seconds(self) -> float:
        return self.n_records * self.record_seconds

    @property
    def is_edf_plus(self) -> bool:
        return self.reserved.startswith("EDF+")

    def signal(self, label: str) -> SignalHeader:
        for found in self.signals:
            if found.label == label:
                return found
        raise EDFHeaderError(f"{self.path.name} has no signal {label!r}; it has {self.labels}")

    def sampling_rate(self, label: str) -> float:
        return self.signal(label).samples_per_record / self.record_seconds


def _text(raw: bytes) -> str:
    return raw.decode("ascii", errors="replace").strip()


def _number(raw: bytes, what: str) -> float:
    text = _text(raw)
    try:
        return float(text)
    except ValueError as error:
        raise EDFHeaderError(f"{what} is not a number: {text!r}") from error


def _start_time(date: str, time: str) -> datetime | None:
    """The recording clock, or ``None`` when the file does not state a usable one.

    EDF writes a two-digit year with the convention that 85-99 mean the 1900s
    and 00-84 the 2000s. Sleep-EDF was recorded in 1989 and 1994, so getting
    this wrong puts the recordings a century out. No timezone is attached: the
    format does not carry one, and every calculation in this package is relative
    to the start of the recording.
    """
    try:
        day, month, year = (int(p) for p in date.split("."))
        hour, minute, second = (int(p) for p in time.split("."))
    except (ValueError, AttributeError):
        return None
    year += 1900 if year >= 85 else 2000
    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None


def read_header(path: Path | str) -> EDFHeader:
    """Parse one EDF header. Nothing here reads a sample."""
    where = Path(path)
    with open(where, "rb") as handle:
        block = handle.read(256)
        if len(block) < 256:
            raise EDFHeaderError(f"{where.name} is too short to be an EDF file")
        version = _text(block[0:8])
        patient = _text(block[8:88])
        recording = _text(block[88:168])
        start = _start_time(_text(block[168:176]), _text(block[176:184]))
        header_bytes = int(_number(block[184:192], "header length"))
        reserved = _text(block[192:236])
        n_records = int(_number(block[236:244], "number of records"))
        record_seconds = _number(block[244:252], "record duration")
        n_signals = int(_number(block[252:256], "number of signals"))
        if n_signals <= 0:
            raise EDFHeaderError(f"{where.name} declares {n_signals} signals")

        body = handle.read(n_signals * 256)
        if len(body) < n_signals * 256:
            raise EDFHeaderError(f"{where.name} ends inside its signal header")

    def block_of(width: int, offset: int) -> list[bytes]:
        start_at = offset * n_signals
        return [
            body[start_at + i * width : start_at + (i + 1) * width] for i in range(n_signals)
        ]

    labels = block_of(16, 0)
    transducers = block_of(80, 16)
    dimensions = block_of(8, 96)
    physical_mins = block_of(8, 104)
    physical_maxes = block_of(8, 112)
    digital_mins = block_of(8, 120)
    digital_maxes = block_of(8, 128)
    prefilters = block_of(80, 136)
    counts = block_of(8, 216)

    signals = tuple(
        SignalHeader(
            label=_text(labels[i]),
            transducer=_text(transducers[i]),
            physical_dimension=_text(dimensions[i]),
            physical_min=_number(physical_mins[i], "physical minimum"),
            physical_max=_number(physical_maxes[i], "physical maximum"),
            digital_min=_number(digital_mins[i], "digital minimum"),
            digital_max=_number(digital_maxes[i], "digital maximum"),
            prefiltering=_text(prefilters[i]),
            samples_per_record=int(_number(counts[i], "samples per record")),
        )
        for i in range(n_signals)
    )

    # An annotation-only EDF+ file -- which is what a Sleep-EDF hypnogram is --
    # is allowed to declare a record duration of zero, because it carries no
    # sampled signal for the duration to describe. A file with real signals and
    # a zero duration is malformed, and that is worth stopping for.
    only_annotations = all(s.label == "EDF Annotations" for s in signals)
    if record_seconds < 0 or (record_seconds == 0 and not only_annotations):
        raise EDFHeaderError(f"{where.name} declares a record duration of {record_seconds}")

    return EDFHeader(
        path=where,
        version=version,
        patient=patient,
        recording=recording,
        start=start,
        header_bytes=header_bytes,
        reserved=reserved,
        n_records=n_records,
        record_seconds=record_seconds,
        signals=signals,
    )


#: Physical dimensions this package knows how to convert to microvolts.
MICROVOLT_UNITS: dict[str, float] = {"uV": 1.0, "µV": 1.0, "μV": 1.0, "mV": 1e3, "V": 1e6}


def to_microvolt_factor(dimension: str) -> float:
    """Factor converting a declared physical unit to microvolts.

    Raises on anything unrecognised rather than guessing. A recording whose EEG
    is declared in an unexpected unit is a recording whose amplitudes would be
    wrong by three orders of magnitude, and amplitude is what separates N3.
    """
    key = dimension.strip()
    if key not in MICROVOLT_UNITS:
        raise EDFHeaderError(
            f"physical dimension {dimension!r} is not a voltage this package converts; "
            f"expected one of {sorted(MICROVOLT_UNITS)}"
        )
    return MICROVOLT_UNITS[key]
