"""What has to be recorded for a saved artefact to mean anything later.

A checkpoint that does not say which split it was trained on, which channel came
first, or which preprocessing statistics were applied is a file that can be
loaded and cannot be trusted. Every artefact this package writes carries the
record below, and the loader refuses a checkpoint that is missing one.

The identities are hashes of the things themselves, not names. A split called
``dev`` in two runs is not the same split; a split whose sorted participant
lists hash to the same digest is.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def digest(payload: Any) -> str:
    """A short stable hash of any JSON-serialisable structure."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def file_sha256(path: Path | str, chunk: int = 1 << 20) -> str:
    """The checksum of a file on disk, read in chunks so a 50 MB EDF is fine."""
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            sha.update(block)
    return sha.hexdigest()


def code_revision(root: Path | str | None = None) -> str:
    """The git revision of the working tree, marked when it is dirty.

    Returns ``"unknown"`` rather than raising when the code is not in a
    repository, because a run outside version control should still produce a
    usable artefact -- it should simply not claim a revision it does not have.
    """
    where = Path(root) if root is not None else Path(__file__).resolve().parent
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=where,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if head.returncode != 0:
            return "unknown"
        revision = head.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=where,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            revision += "-dirty"
        return revision
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@dataclass(frozen=True, slots=True)
class RunProvenance:
    """Everything needed to say what a run was, saved beside what it produced."""

    run_id: str
    created_utc: str
    code_revision: str
    python_version: str
    platform: str
    device: str
    seed: int
    config: dict[str, Any]
    config_id: str
    split_id: str
    channels: tuple[str, ...]
    label_order: tuple[str, ...]
    preprocessing_id: str
    package_version: str
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))


def make_run_provenance(
    *,
    run_id: str,
    device: str,
    seed: int,
    config: dict[str, Any],
    split_id: str,
    channels: tuple[str, ...],
    label_order: tuple[str, ...],
    preprocessing_id: str,
    notes: str = "",
    extra: dict[str, Any] | None = None,
) -> RunProvenance:
    from sleepstatelab import __version__

    return RunProvenance(
        run_id=run_id,
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        code_revision=code_revision(),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        device=device,
        seed=seed,
        config=config,
        config_id=digest(config),
        split_id=split_id,
        channels=tuple(channels),
        label_order=tuple(label_order),
        preprocessing_id=preprocessing_id,
        package_version=__version__,
        notes=notes,
        extra=dict(extra or {}),
    )
