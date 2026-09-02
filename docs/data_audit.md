# Data: what is read, what is checked, what is thrown away

The dataset is [Sleep-EDF Database Expanded 1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/),
**Sleep Cassette only** (`SC4*`). Sleep Telemetry (`ST7*`) files carry a
different montage and a different protocol; discovery does not match them, and a
telemetry file in the data root is listed as ignored rather than quietly mixed
in.

The dataset is not redistributed here. It is read from a configurable data root
that contains nothing this repository wrote.

## Discovery and pairing

`sleepstatelab.data.discovery` walks the data root and pairs files by
`(participant, night)` parsed from `SC4ssN…`:

* `ss` is the **participant**, `N` is the **night**. Most participants have two
  nights. They are one person, and every split keeps them together.
* the hypnogram's trailing letters differ from the PSG's — `SC4001E0-PSG.edf`
  pairs with `SC4001EC-Hypnogram.edf` — so pairing is by participant and night,
  never by file stem.
* a PSG with no hypnogram, or the reverse, is reported by name and not offered.

**What is on disk is not the cohort.** Sleep Cassette holds 153 recordings from
78 participants. Every manifest and every command's output says how many were
found and states this explicitly. Nothing in this package downloads a dataset.

## What the manifest records

`sleepstatelab audit` writes one entry per recording: participant, recording and
night; both file paths; the SHA-256 of each file; a verdict against PhysioNet's
published `SHA256SUMS.txt` when one is supplied (`match` / `mismatch` /
`unchecked`); both start timestamps; the PSG duration; epoch counts (in the
signal, in the hypnogram, and past the end of the signal); annotation coverage;
per-channel unit, physical range, sampling rate, prefiltering and transducer as
the header declares them; the stage counts; the exclusions; misaligned or
overlapping annotations; and a list of problems that is empty only when nothing
was wrong.

## What is validated, and why each check exists

| check | failure it catches |
| --- | --- |
| PSG and hypnogram start at the same second | every label shifted in time, invisible downstream |
| declared physical dimension is a voltage this package converts | amplitudes wrong by a factor of 1,000 |
| both EEG channels present, in the requested order | channel 0 silently becoming the other derivation |
| sampling rate is exactly 100 Hz | a resampled spectrum compared against an unresampled one |
| annotation onsets and durations are whole epochs | a partial epoch labelled as though it were whole |
| no epoch scored twice | a contradiction in the scoring, hidden by last-write-wins |
| epoch counts reconcile: signal, annotated, beyond-signal, retained | epochs appearing or vanishing between stages |

Resampling is never done silently: a recording at the wrong rate raises.

## Epoching

Epoch *i* is samples `[i·3000, (i+1)·3000)` of the PSG at 100 Hz — 30 seconds,
3,000 samples per channel — which is the same grid the hypnogram's onsets sit
on.

* **Original indices survive.** Every retained epoch carries its index in the
  recording. An excluded epoch leaves a hole, and the hole is visible as a jump
  in that index. No code in this package treats two rows as neighbours because
  they are adjacent in an array; D2's context windows are specified to check the
  index difference, which is the only reason they can be trusted at a gap.
* **Nothing is normalised at this stage.** The cache holds microvolts as the
  header declares them. Normalising here would bake a training set into a cache.

## Label mapping

| annotation | label |
| --- | --- |
| `Sleep stage W` | Wake |
| `Sleep stage 1` | N1 |
| `Sleep stage 2` | N2 |
| `Sleep stage 3`, `Sleep stage 4` | N3 |
| `Sleep stage R` | REM |
| `Movement time` | **excluded** (`movement_time`) |
| `Sleep stage ?`, `Sleep stage e` | **excluded** (`unscored`) |
| anything else | **excluded** (`unknown_annotation`) |
| no annotation covering the epoch | **excluded** (`no_annotation`) |

Stages 3 and 4 are merged as current AASM practice does. **The original
annotation text is stored per epoch**, so the merge can be undone. Movement and
unknown annotations never become a stage and never become wake.

## Quality control

Bit flags per epoch, computed over both channels
(`sleepstatelab.data.epochs.QCPolicy`, version `sleepstatelab-eeg-qc-1.0`):

| flag | rule | rejects by default |
| --- | --- | --- |
| `qc_flatline` | channel standard deviation below 0.1 µV | yes |
| `qc_clipped` | more than 2% of samples at the amplifier rail declared in the header | yes |
| `qc_high_amplitude` | any sample beyond 500 µV | yes |
| `qc_muscle` | more than 50% of 0.5–45 Hz power above 30 Hz | **no** |

Muscle contamination is measured and recorded but does not reject: wake epochs
legitimately carry muscle activity, and rejecting on it would delete part of the
class it describes.

`qc_high_amplitude` **cannot fire on Sleep Cassette**, whose declared physical
range is about ±192 µV. It is kept for montages with a wider range, and this is
stated rather than left as an unexplained column of zeros.

## Preprocessing

Applied between the cache and a model, never inside the cache:

1. zero-phase Butterworth band-pass, 0.3–35 Hz, order 4, applied per epoch so no
   filter state crosses a gap. Zero-phase because a causal filter shifts slow
   waves in time, and a stage boundary in the wrong epoch is a labelling error;
2. normalisation — see below;
3. clipping at ±20 normalised units, so one movement artifact cannot dominate a
   batch's gradients.

### Normalisation

Default `train_robust_channel`: one median and one inter-quartile range **per
channel**, estimated over the **training participants only**, then frozen and
applied unchanged to validation and test. The statistics, the participants they
were fitted on, and a hash of both are saved and recorded in every checkpoint.

**Amplitude is deliberately preserved.** Per-epoch z-scoring — dividing each 30
seconds by its own standard deviation — is a common default in EEG code and it
deletes the absolute amplitude that separates N3 from N1, which is one of the
things a human scorer reads directly. It is available as `per_epoch_zscore` so
its cost can be measured, and it is not the default.

## Primary preparation versus the sensitivity analysis

**Primary:** every valid scored epoch, with no sleep-stage-based cropping. A
Sleep Cassette recording runs about twenty hours around one night, so wake is
roughly 70% of the epochs. That is the real class balance of the recordings as
scored, and accuracy on it is close to meaningless — which is why the primary
metric is participant-averaged macro-F1 and why a class-prior baseline is
reported in every table.

**Sensitivity analysis:** `configs/sleep_window_sensitivity.yaml` crops to the
annotated sleep interval plus a 30-minute margin, which moves wake to roughly
30% of the epochs. It is a separately labelled analysis, never mixed with
primary results, and results produced under it say so.

## Splits

Participant-disjoint train / validation / test, generated deterministically from
a seed (`sleepstatelab.data.splits.grouped_split`). All nights of a participant
land in one split; the `Split` class refuses to be constructed otherwise, and a
split file records a hash of its three participant lists so an edited file is
detected on load.

**No test participant contributes to** normalisation statistics, hyperparameter
choice, early stopping, calibration, or self-supervised pretraining. The first
is enforced in `build_datasets`, the second and third by selecting only on
validation, and the last is a stated requirement of the D3 contract.

## A worked audit of one real recording

`docs/audit/SC400-n1_audit.md` is the audit of a real recording — file
checksums, header claims, waveforms and spectra for one epoch of each stage, the
hypnogram against time, retained counts and every exclusion. It was generated by
`sleepstatelab audit-report` and nothing on that page was typed in by hand.
