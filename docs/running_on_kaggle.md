# Running the benchmark on Kaggle Notebooks

The benchmark needs more memory than a 16 GB laptop has. Building one cell
materialises the training, validation and test epochs at once — about 10 GB —
and on the machine this project was developed on that exhausted the memory
compressor and panicked the kernel twice. Kaggle gives roughly 29 GB of RAM and
a GPU for free, which is the smallest change that makes the job possible rather
than the fastest one.

**The session cap is not a problem any more.** Kaggle stops a notebook at 12
hours. The benchmark runner skips cells that already have predictions, and
training writes its optimiser, epoch and RNG state after every pass, so an
interrupted session loses at most one pass and the next one continues.

## What you need to upload

Not the recordings. Training reads the epoch cache; the 7 GB of EDF is only
needed by `prepare`, which has already been run.

Create one **private** Kaggle Dataset containing:

```
sleep-edf-epoch-cache/
├── 9d2205f7906f4883/            ← the preprocessing-identity directory,
│   ├── SC400-n1.npz                with all 153 cached recordings (4.4 GB)
│   ├── SC400-n2.npz
│   └── ...
└── cohort_pretrain_encoder.pt   ← runs/cohort_pretrain/encoder.pt (1.9 MB)
```

The directory name `9d2205f7906f4883` matters: it is the hash of the
preprocessing settings, and `--cache-dir` expects to find it one level down. Do
not flatten it.

The encoder has to ride along because `runs/` is not tracked by git, so cloning
the repository does not bring it. Without it, every D3 cell fails.

Everything else — the code, the cohort split (`docs/cohort_split.json`) — comes
from the public repository, which the notebook clones.

### Uploading 4.4 GB

The web uploader is unreliable at this size. Use the CLI:

```bash
pip install kaggle                       # token from kaggle.com → Settings → API
mkdir -p upload/9d2205f7906f4883
cp /Volumes/maxone/sleep-edf-data/cache/9d2205f7906f4883/*.npz upload/9d2205f7906f4883/
cp /Volumes/maxone/sleep-state-lab/runs/cohort_pretrain/encoder.pt \
   upload/cohort_pretrain_encoder.pt

cd upload
kaggle datasets init -p .
# edit dataset-metadata.json: set "title" and "id" to <your-username>/sleep-edf-epoch-cache
kaggle datasets create -p . --dir-mode zip
```

Uploading takes a while and only has to happen once.

## The notebook

One cell:

```python
!cp /kaggle/input/sleep-edf-epoch-cache/../sleep-state-lab/scripts/kaggle_run.py . 2>/dev/null
!git clone --depth 1 https://github.com/Elnazkarami/sleep-state-lab.git /tmp/ssl
%run /tmp/ssl/scripts/kaggle_run.py
```

Settings for the notebook:

| setting | value | why |
| --- | --- | --- |
| Accelerator | GPU T4 ×2 or P100 | either is comparable to the M2 Pro this was developed on |
| Internet | **on** | the notebook clones the repository and pip-installs |
| Persistence | Variables and files | so `/kaggle/working` survives |

## Where things go, and why

| path | holds | survives the session? |
| --- | --- | --- |
| `/kaggle/input/...` | the uploaded cache and encoder | read-only, permanent |
| `/kaggle/temp/store` | materialised epochs, ~10 GB | no — and should not, it rebuilds in minutes |
| `/kaggle/working` | predictions, checkpoints, resume state | yes, if you **Save Version** |

`/kaggle/working` has a 20 GB limit and is saved as the notebook's output. The
materialised stores would eat it for no reason, so they go to scratch.

## Continuing in a second session

1. **Save Version** on the finished session, so its `/kaggle/working` becomes a
   dataset.
2. In the next notebook, add that output as an input dataset and name it
   `sleepstatelab-benchmark-previous`.
3. Run the same cell. It copies the earlier predictions and checkpoints into
   place and the runner skips what is done.

## Quota arithmetic

Kaggle gives about 30 GPU-hours a week. The full 18-cell sweep is roughly 13
hours on hardware of this class, so a complete sweep plus a second participant
split fits inside one week's allowance.

Spend it in this order, which is not the order the cells are numbered in:

1. **A second participant split at 10% and 25%** — 12 cells, ~4 hours. Every
   result currently rests on one split; this is the largest single weakness and
   the cheapest to address.
2. **`--reduce-validation` at 10% and 25%** — 12 cells, ~4 hours. The honest
   limited-label setting, in which validation shrinks with the budget. At
   present every cell selects its checkpoint on all 16 validation participants,
   which is more supervision than a 10% budget implies, and that caveat
   currently qualifies every number reported.
3. **The four missing 100%-budget cells** — ~2 hours. Finishes the table.
   Deliberately last: the 100% row's answer is already established twice, and
   putting error bars on a null result is the least valuable thing here.

To run the second split, generate it with a different seed first:

```bash
sleepstatelab split --config configs/default.yaml --cache-dir <cache> \
    --name cohort78-b --output docs/cohort_split_b.json
```

— editing `split.seed` in the configuration, then point `sweep(split=...)` at
the new file.

## Bringing the results home

The predictions CSV is a few hundred megabytes at most; the report regenerates
from it locally:

```bash
sleepstatelab report predictions_benchmark.csv --part test \
    --output docs/results_benchmark.md --json docs/benchmark_metrics.json
sleepstatelab transitions predictions_benchmark.csv --part test \
    --output docs/results_benchmark_transitions.md
python scripts/make_results_figure.py
```

Nothing is retyped: every table and the figure come from the saved per-epoch
predictions.

## What will differ from the laptop runs

Kaggle's GPUs are CUDA, not Apple MPS, and its Python stack will not match the
one here exactly. Predictions were identical between CPU and MPS to 7.75e-07 on
this project, so results should be comparable — but cells run on Kaggle are
internally consistent with each other and not bit-identical to the ones in the
README. Say so wherever the two are placed side by side, as the README already
does for the numpy 1.26 → 2.5 rebuild.
