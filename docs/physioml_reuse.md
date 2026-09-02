# What was taken from PhysioML, and how

PhysioML is the existing classical-machine-learning and traceability project by
the same author. SleepStateLab is a separate repository that owns neural-network
training and representation learning. The two are not merged, and PhysioML was
not modified by this work.

## The relationship, stated plainly

| | PhysioML | SleepStateLab |
| --- | --- | --- |
| owns | classical models, provenance/traceability, CDFS integration, WESAD + Sleep-EDF pipelines | neural encoders, self-supervised pretraining, sleep-state decoding |
| datasets | WESAD and Sleep-EDF Expanded | Sleep-EDF Expanded, Sleep Cassette only |
| runtime | needs its own extras; CDFS optional | **runs with no CDFS deployment of any kind** |
| this repository depends on it | — | no: PhysioML is not imported, vendored, or installed |

SleepStateLab has no import of `physioml` anywhere, and no code path that
requires a CDFS deployment. This was checked with `grep -r physioml src/`, which
matches only documentation strings.

## Licensing

PhysioML carries a proprietary licence: all rights reserved to Elnaz Alikarami,
and use requires her prior written approval. SleepStateLab is by the same author
under the same terms, so re-use within it is permitted by the rights holder. It
is nonetheless attributed here rather than silently copied, because a component
whose origin is undocumented is a component nobody can audit later.

## What was taken

**1. The spectral feature definitions** — `physioml.neural.features`
(feature set `sleep-eeg`, version 1.0, revision `4f18d97`) → this repository's
`sleepstatelab.features.spectral` (feature set `sleepstatelab-eeg-spectral`,
version 1.0).

*Kept:* the band edges (delta 0.5–4, theta 4–8, alpha 8–12, sigma 12–16, beta
16–30 Hz), the 0.5–30 Hz total band that relative power is taken against, the
Welch estimate on four-second segments, the trapezoidal band integration, the
Hjorth parameters, normalised spectral entropy, the 95% spectral edge, the
95th-percentile amplitude and the zero-crossing rate.

*Changed:* the electro-oculogram and chin-electromyogram features are dropped,
because SleepStateLab reads the two EEG derivations only. The extractor takes a
whole `[channels, samples]` epoch instead of one channel at a time. Column names
carry the channel index as well as the derivation, so a one-channel ablation
still produces a traceable matrix.

*Why re-implemented rather than imported:* importing would make a proprietary
sibling repository a hard dependency of every run, and it would couple this
project's array layout to PhysioML's `FeatureTable`. The definitions are
attributed and versioned instead, which is the arrangement the licence permits
and the one that survives either repository moving on.

**2. The EEG quality-control thresholds** — `physioml.neural.qc` policy
`sleep-eeg-1.0` → `sleepstatelab.data.epochs.QCPolicy`, version
`sleepstatelab-eeg-qc-1.0`.

*Kept:* the flat-line standard-deviation floor (0.1 µV), the clipping fraction
(2% of samples), the muscle-share ceiling (50% of 0.5–45 Hz power above 30 Hz),
and — importantly — the decision that muscle contamination *warns* rather than
rejects, because wake epochs legitimately carry muscle activity and rejecting on
it deletes part of the class it describes.

*Changed:* saturation is detected against the amplifier rail declared in the EDF
header (about ±192 µV on Sleep Cassette) rather than against the epoch's own
maximum. PhysioML's fixed 500 µV ceiling is retained as a second, absolute
check, and this repository records that it *cannot fire* on Sleep Cassette
rather than leaving a column of zeros for someone to discover.

**3. Two design decisions, adopted with attribution and no code.**
Both nights of a participant are one participant. Stage 4 folds into N3 under
current practice while the original annotation text is retained. Both are
PhysioML's decisions and both are correct; they are re-implemented here because
they are one line each, not because they are new.

## What was deliberately *not* taken

* **PhysioML's EDF reader.** An established reader was preferred, as the brief
  asks: MNE reads the signals and the annotations. What MNE does not do is say
  what the file *claims* — it has already applied the physical calibration by
  the time anything is visible — so `sleepstatelab.data.edf_header` parses the
  header directly for the unit, rate and timestamp checks.
* **PhysioML's `SleepEDF` adapter.** It trims every night to the sleep period
  plus a margin by default. That is a reasonable choice for the study it was
  written for and the wrong default here: SleepStateLab's primary preparation
  keeps every valid scored epoch, and the crop is an opt-in sensitivity
  analysis (`configs/sleep_window_sensitivity.yaml`).
* **The provenance core, the CDFS client, the calibration and personalisation
  machinery, WESAD, the peripheral pipeline.** Out of scope, and copying them
  would make this a fork rather than a project.

## PhysioML's published sleep numbers are context, not baselines

PhysioML reports, for 20 participants of Sleep Cassette:
balanced accuracy 0.725 and Cohen's κ 0.710 for a random forest, on
20,626 epochs, under subject-wise cross-validation.

That number **cannot be compared** with anything in this repository, for four
reasons, each sufficient on its own:

1. it was computed on recordings cropped to the sleep period plus 30 minutes,
   which changes wake from about 70% of the epochs to about 30%;
2. it used four channels including the EOG and chin EMG, which are much of what
   separates REM;
3. it used a different participant set and a different cross-validation scheme;
4. it averages over folds pooled by epoch, not equally over participants.

The classical baselines are therefore **re-run inside this repository** on the
matched participants, channels, eligible epochs, label mapping and split, and
only those re-run numbers are compared with D1.

## The 20-participant subset has been used before

The development subset — the first 20 Sleep Cassette participants — was used in
PhysioML for the results quoted above. It is **not** an untouched confirmation
cohort, and it is not described as one anywhere here. It is a development set
whose prior use is recorded. When a confirmation cohort is needed, it will have
to come from the participants neither project has looked at.
