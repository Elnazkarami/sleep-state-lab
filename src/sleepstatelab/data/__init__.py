"""Reading Sleep-EDF Expanded, and turning it into scored epochs.

The order of operations, and why each step exists, is in
``docs/data_audit.md``. The short version: discover files, pair each PSG with
its own hypnogram, read the header and check what it claims, expand the
annotations to one label per 30-second epoch, cut the signal at the epoch
boundaries the scorer used, and keep every exclusion with its reason.
"""

from sleepstatelab.data.discovery import (
    DiscoveryError,
    RecordingPair,
    discover,
)
from sleepstatelab.data.manifest import Manifest, ManifestEntry, build_manifest

__all__ = [
    "DiscoveryError",
    "Manifest",
    "ManifestEntry",
    "RecordingPair",
    "build_manifest",
    "discover",
]
