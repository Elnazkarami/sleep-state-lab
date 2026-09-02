# Evaluation protocol

## Everything is computed from saved predictions

A model writes one row per **participant, recording, epoch, model and run**, and
every metric and every table in this repository is computed from those rows. The
row carries the true label, the predicted label, and **all five probabilities**
— without them the N1/REM confusion cannot be examined afterwards and no
recalibration study is possible without re-running the model.

The file is CSV, with these columns:

```
run_id, model, split_id, split_part, seed, participant_id, recording_id,
epoch_index, true_label, pred_label,
p_Wake, p_N1, p_N2, p_N3, p_REM, qc_flags
```

`epoch_index` is the epoch's original index in its recording, so a prediction
can always be put back against the 30 seconds it was made for.

No score is ever typed into documentation. `sleepstatelab report` reads a
prediction file and emits the markdown tables, and those tables are pasted whole.

## The primary metric

**Macro-F1 computed within each participant, then averaged equally across
people.** One participant, one vote. A pooled macro-F1 over all epochs at once is
dominated by whoever contributed the most epochs — on Sleep Cassette, whoever
slept longest with the recorder running — so it is reported *beside* the primary
metric and never instead of it. The two are always labelled.

## The five-class order

`("Wake", "N1", "N2", "N3", "REM")`, fixed, defined once in
`sleepstatelab.labels`, and positional in every probability vector, class-weight
array, confusion matrix and checkpoint.

## How absent classes are handled

Within one participant a stage can be absent from the truth, absent from the
predictions, or both. The rule:

* no true epochs **and** no predicted epochs → the class is **omitted** from
  that participant's macro average. There was no question to get right, and
  scoring it zero would punish a model for a stage that never occurred;
* no true epochs but some predicted → **F1 = 0, and it counts**. Predicting a
  stage that never happened is an error;
* true epochs but never predicted → **F1 = 0, and it counts**.

Every per-stage table reports **how many participants** the figure was averaged
over, so a number resting on three people cannot be read as one resting on
twenty.

## Also reported, always

* stage support (epoch counts per stage);
* per-stage precision, recall and F1 — precision and recall pooled, F1 both
  pooled and averaged over participants, each labelled;
* balanced accuracy (mean recall over stages present in the truth);
* Cohen's κ, computed from the confusion matrix under the documented class
  order — the metric automatic sleep staging is compared on;
* the 5×5 confusion matrix, rows truth, columns prediction;
* quality-control coverage: what share of scored epochs carried no QC flag;
* per-participant scores, so a cohort mean cannot hide one person's failure.

## What a comparison requires

Two numbers in this repository are only comparable when they share: the
participants, the channels and their order, the eligible-epoch mask (label
mapping and QC rejection), the split identity, the preprocessing identity, and
the preparation mode (primary or sleep-window). The prediction rows and the
checkpoint together record all six, and a checkpoint whose contract does not
match the configuration it is being loaded into is refused rather than run.

## Cohort size and what an estimate is worth

The first release was executed on a **six-participant pilot**: four training,
one validation, one test. A test estimate from one held-out person is one
person's night. It demonstrates that the pipeline runs on real recordings and
produces coherent numbers; it establishes nothing about how a model generalises.
The benchmark this repository is built for needs the full cohort, several seeds,
and the controls in `docs/model_contracts.md`.
