"""The benchmark sweep, shaped for a Kaggle notebook session.

Paste this into one cell. It is written to be run repeatedly: each session
continues where the last one stopped, so the 12-hour cap costs nothing but a
restart.

Why it is laid out this way
---------------------------

**Three directories, three different lifetimes.** Kaggle gives you a read-only
input mount, a 20 GB ``/kaggle/working`` that is saved as the notebook's output,
and scratch space that is not. The materialised epoch stores are about 10 GB and
are rebuilt from the cache in minutes, so they go to scratch. The predictions,
checkpoints and resume state are small and must survive the session, so they go
to working.

**Resumption is the whole design.** The benchmark runner reads its predictions
file and skips cells that already have rows; training writes optimiser, epoch
and RNG state after every pass. So a session that is cut off mid-cell loses at
most one pass, and the next session is started by adding the previous session's
output as an input dataset.

**Budgets run in priority order**, not numeric order, for the same reason: if
the session ends early, the cells that matter are the ones that finished.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Paths. Edit the two dataset slugs to match what you uploaded.
# --------------------------------------------------------------------------

CACHE_INPUT = Path("/kaggle/input/sleep-edf-epoch-cache")
"""The epoch cache, uploaded once as a private Kaggle Dataset. Not the
recordings: training reads the cache, and the 7 GB of EDF is only needed by
`prepare`, which has already been run."""

ENCODER = CACHE_INPUT / "cohort_pretrain_encoder.pt"
"""The self-supervised encoder D3 cells start from. Upload it alongside the
cache: it is 1.9 MB and lives in `runs/`, which the repository does not track,
so cloning the code is not enough to get it."""

PREVIOUS = Path("/kaggle/input/sleepstatelab-benchmark-previous")
"""The previous session's output, added as an input dataset. Absent on the
first run, which is fine."""

WORKING = Path("/kaggle/working")
SCRATCH = Path("/kaggle/temp")
"""Not saved as output. The right home for 10 GB of rebuildable stores."""

REPO = WORKING / "sleep-state-lab"
STORE = SCRATCH / "store"
OUTPUT = WORKING / "predictions_benchmark.csv"
RUNS = WORKING / "runs"


def run(command: list[str], **kwargs: object) -> None:
    print(f"$ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, **kwargs)  # type: ignore[arg-type]


def setup() -> None:
    """Clone the code, install it, and carry forward anything already done."""
    if not REPO.exists():
        run(["git", "clone", "--depth", "1",
             "https://github.com/Elnazkarami/sleep-state-lab.git", str(REPO)])
    os.chdir(REPO)

    # Kaggle images ship torch with CUDA; everything else is small. scipy is
    # left to pip rather than pinned to the version the macOS rebuild needed --
    # that was a dyld problem on macOS 27, not a correctness one, and Linux
    # wheels are unaffected.
    run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[plots]"])

    STORE.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    # Resume: bring forward the previous session's predictions and checkpoints.
    if PREVIOUS.exists():
        earlier = PREVIOUS / "predictions_benchmark.csv"
        if earlier.exists() and not OUTPUT.exists():
            shutil.copy2(earlier, OUTPUT)
            print(f"carried forward {earlier}")
        earlier_runs = PREVIOUS / "runs"
        if earlier_runs.exists():
            shutil.copytree(earlier_runs, RUNS, dirs_exist_ok=True)
            print(f"carried forward checkpoints from {earlier_runs}")
    else:
        print("no previous session found; starting fresh")


def completed_cells() -> set[str]:
    if not OUTPUT.exists():
        return set()
    sys.path.insert(0, str(REPO / "src"))
    from sleepstatelab.evaluation.predictions import read_predictions

    return {row["model"] for row in read_predictions(OUTPUT)}


def sweep(
    *,
    split: str,
    encoder: str,
    budgets: tuple[float, ...] = (0.25, 0.1, 1.0),
    seeds: tuple[int, ...] = (0, 1, 2),
    reduce_validation: bool = False,
) -> None:
    """One pass over the plan, budget by budget, resuming as it goes."""
    for budget in budgets:
        print(f"\n{'=' * 60}\nBUDGET {budget:.0%}\n{'=' * 60}", flush=True)
        command = [
            sys.executable, "-u", "-m", "sleepstatelab.cli", "benchmark",
            "--config", "configs/default.yaml",
            "--data-root", str(CACHE_INPUT),
            "--cache-dir", str(CACHE_INPUT),
            "--store-dir", str(STORE),
            "--split", split,
            "--pretrained-encoder", encoder,
            "--device", "cuda",
            "--budgets", str(budget),
            "--seeds", *[str(s) for s in seeds],
            "--models", "D2", "D3",
            "--runs-dir", str(RUNS),
            "--output", str(OUTPUT),
        ]
        if reduce_validation:
            command.append("--reduce-validation")
        try:
            run(command)
        except subprocess.CalledProcessError as error:
            # A failed budget should not sink the ones after it; the log says
            # what happened and the next session resumes whatever is missing.
            print(f"budget {budget} stopped: {error}", flush=True)


def main() -> None:
    setup()
    done = completed_cells()
    print(f"\n{len(done)} cell(s) already have predictions: {sorted(done)}\n")

    environment = {"PYTHONPATH": str(REPO / "src"), **os.environ}
    os.environ.update(environment)

    # The split is committed (docs/cohort_split.json); the encoder is not, so
    # it rides along with the uploaded cache.
    if not ENCODER.exists():
        raise SystemExit(
            f"{ENCODER} is missing. Upload runs/cohort_pretrain/encoder.pt into "
            "the cache dataset as cohort_pretrain_encoder.pt -- D3 cells cannot "
            "start without it."
        )
    sweep(split="docs/cohort_split.json", encoder=str(ENCODER))

    print("\nsession finished. Save Version to persist /kaggle/working, then")
    print("add this notebook's output as an input dataset to continue.")


if __name__ == "__main__":
    main()
