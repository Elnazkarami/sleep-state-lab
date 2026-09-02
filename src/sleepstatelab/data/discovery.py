"""Finding the recordings, and knowing who they belong to.

Three distinctions are made here that the file names make easy to lose.

**A participant is not a recording.** ``SC4001E0-PSG.edf`` and
``SC4002E0-PSG.edf`` are two nights of one person. Treating them as two people
puts the same person on both sides of a split, which is the single failure this
whole evaluation exists to prevent, arriving through a naming convention.

**A PSG is not paired by prefix.** The hypnogram's last two letters differ from
the PSG's -- ``SC4001E0-PSG.edf`` goes with ``SC4001EC-Hypnogram.edf`` -- so the
pairing key is the participant and night, not the file stem.

**What is on disk is not the cohort.** Sleep Cassette holds 153 recordings; a
directory holding six is a pilot. Discovery reports what it found and never
implies the rest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: SC4ssNxx: ss is the participant, N is the night, xx identifies the scorer and
#: the file kind. Sleep Telemetry files begin ST7 and are not matched: a
#: different montage and a different protocol are a different study.
NAME = re.compile(r"^SC4(?P<participant>\d{2})(?P<night>\d)(?P<tail>[A-Za-z0-9]{0,2})")


class DiscoveryError(RuntimeError):
    """Raised when a data root cannot be read as Sleep Cassette."""


@dataclass(frozen=True, slots=True)
class RecordingPair:
    """One night: a PSG file and the hypnogram that scores it."""

    participant_id: str
    """``SC4xx``: the person. Both of their nights carry the same value."""

    night: int
    psg_path: Path
    hypnogram_path: Path

    @property
    def recording_id(self) -> str:
        """``SC4xx-nN``: this night of this person, which is what a row is keyed by."""
        return f"{self.participant_id}-n{self.night}"


@dataclass(frozen=True, slots=True)
class Discovery:
    """What a data root turned out to hold, including what it did not."""

    root: Path
    pairs: tuple[RecordingPair, ...]
    psg_without_hypnogram: tuple[str, ...]
    hypnogram_without_psg: tuple[str, ...]
    ignored: tuple[str, ...]
    """Files that look like EDF but are not Sleep Cassette -- Sleep Telemetry,
    most often. Named rather than skipped silently."""

    @property
    def participants(self) -> tuple[str, ...]:
        return tuple(sorted({p.participant_id for p in self.pairs}))

    def summary(self) -> str:
        return (
            f"{len(self.pairs)} paired recording(s) from {len(self.participants)} "
            f"participant(s) under {self.root}"
        )


def discover(
    root: Path | str,
    *,
    participants: tuple[str, ...] = (),
    nights: tuple[int, ...] = (1, 2),
) -> Discovery:
    """Pair every Sleep Cassette PSG under ``root`` with its own hypnogram."""
    where = Path(root).expanduser()
    if not where.is_dir():
        raise DiscoveryError(f"{where} is not a directory")

    psg: dict[tuple[str, int], Path] = {}
    hypnograms: dict[tuple[str, int], Path] = {}
    ignored: list[str] = []
    # A case-insensitive filesystem returns the same file for both patterns, so
    # the two globs are unioned rather than concatenated: a duplicate would look
    # like two files claiming one night.
    candidates = sorted(set(where.rglob("*.edf")) | set(where.rglob("*.EDF")))
    for path in candidates:
        found = NAME.match(path.name)
        if found is None:
            ignored.append(path.name)
            continue
        key = (f"SC4{found['participant']}", int(found["night"]))
        upper = path.name.upper()
        if upper.endswith("-PSG.EDF"):
            existing = psg.get(key)
            if existing is not None and existing != path:
                raise DiscoveryError(
                    f"two PSG files claim {key[0]} night {key[1]}: "
                    f"{existing.name} and {path.name}"
                )
            psg[key] = path
        elif "HYPNOGRAM" in upper:
            existing = hypnograms.get(key)
            if existing is not None and existing != path:
                raise DiscoveryError(
                    f"two hypnograms claim {key[0]} night {key[1]}: "
                    f"{existing.name} and {path.name}"
                )
            hypnograms[key] = path
        else:
            ignored.append(path.name)

    wanted = set(participants)
    pairs = tuple(
        RecordingPair(
            participant_id=key[0],
            night=key[1],
            psg_path=psg[key],
            hypnogram_path=hypnograms[key],
        )
        for key in sorted(psg)
        if key in hypnograms
        and key[1] in nights
        and (not wanted or key[0] in wanted)
    )
    lonely_psg = tuple(psg[k].name for k in sorted(psg) if k not in hypnograms)
    lonely_hyp = tuple(hypnograms[k].name for k in sorted(hypnograms) if k not in psg)

    if not pairs:
        raise DiscoveryError(
            f"no paired Sleep Cassette recordings under {where}. "
            f"Found {len(psg)} PSG and {len(hypnograms)} hypnogram file(s); "
            "expected names of the form SC4ssN*-PSG.edf and SC4ssN*-Hypnogram.edf"
        )
    return Discovery(
        root=where,
        pairs=pairs,
        psg_without_hypnogram=lonely_psg,
        hypnogram_without_psg=lonely_hyp,
        ignored=tuple(sorted(ignored)),
    )
