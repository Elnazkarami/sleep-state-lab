"""Preprocessed epochs on disk, memory-mapped, so a cohort need not fit in RAM.

The pilot's six recordings were 1.4 GB of float32 and holding them in memory was
the obvious thing to do. The cohort is 153 recordings: 249,041 training epochs,
about 6 GB, beside another 4 GB of validation and test — on a 16 GB machine,
with a model on the accelerator, that is the difference between a run and a
swap storm.

Three ways out, and why this one:

* **Filter and normalise lazily, per item.** Measured at 0.54 ms per epoch,
  which is 135 s added to a 200 s pass. Rejected: it makes every pass pay for
  work whose answer never changes.
* **Keep the compressed cache and decompress per batch.** Same objection, plus
  the decompression is not cheap.
* **Preprocess once, write the result to disk, and memory-map it.** The work
  happens a single time per (preprocessing, normalisation, cohort), the file is
  reused by every model trained on that split, and the operating system's page
  cache decides what stays resident. That is this module.

The file is keyed by everything that determines its contents, so a changed
filter, a changed normalisation, or a different set of recordings writes a
different file rather than silently reusing the wrong one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sleepstatelab.provenance import digest

STORE_VERSION = "1"


def store_key(
    *,
    preprocessing_id: str,
    normalization_id: str,
    recordings: list[str],
    reject_flags: int,
    n_epochs: int,
) -> str:
    """An identity for one materialised array: everything that shapes it."""
    return digest(
        {
            "version": STORE_VERSION,
            "preprocessing": preprocessing_id,
            "normalization": normalization_id,
            "recordings": sorted(recordings),
            "reject_flags": reject_flags,
            "n_epochs": n_epochs,
        }
    )


def materialise(
    blocks: list[np.ndarray],
    *,
    directory: Path | str,
    key: str,
    progress: bool = False,
) -> np.ndarray:
    """Concatenate preprocessed blocks into a memory-mapped array on disk.

    Returns a read-only memmap. An existing file of the right name and shape is
    reused without rewriting, which is what makes the second model trained on a
    split start immediately.

    Blocks are written one at a time and released as they go, so the peak memory
    is one recording rather than the whole cohort -- writing into a memmap and
    then concatenating in RAM would defeat the entire point.
    """
    if not blocks:
        raise ValueError("nothing to materialise")
    total = int(sum(block.shape[0] for block in blocks))
    shape = (total, *blocks[0].shape[1:])

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"epochs-{key}.npy"

    if path.exists():
        existing = np.load(path, mmap_mode="r")
        if existing.shape == shape and existing.dtype == np.float32:
            if progress:
                print(f"  reusing materialised epochs at {path}", flush=True)
            return existing
        # A file of the right name and the wrong shape is a key collision or a
        # half-written file from an interrupted run. Either way it is not what
        # was asked for, and rewriting is cheaper than reasoning about it.
        path.unlink()

    if progress:
        print(
            f"  materialising {total} epochs ({total * int(np.prod(shape[1:])) * 4 / 1e9:.1f} GB) "
            f"to {path}",
            flush=True,
        )
    written = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=shape
    )
    offset = 0
    for index, block in enumerate(blocks):
        written[offset : offset + block.shape[0]] = block
        offset += block.shape[0]
        blocks[index] = np.empty((0, *block.shape[1:]), dtype=np.float32)
    written.flush()
    del written
    return np.load(path, mmap_mode="r")


def block_views(array: np.ndarray, lengths: list[int]) -> list[np.ndarray]:
    """Split a materialised array back into per-recording views.

    Views, not copies: a slice of a memmap is a memmap, so the context-window
    datasets can index per recording without any of it becoming resident.
    """
    views = []
    offset = 0
    for length in lengths:
        views.append(array[offset : offset + length])
        offset += length
    return views
