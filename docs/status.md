# What was built, what was run, and what was not

The short version is the table at the top of the README. This is the long one,
component by component, with the distinction the project is organised around:
**implemented** means the code exists and is tested; **executed on synthetic
data** means it has been run on generated recordings; **executed on real data**
means it has been run on Sleep-EDF Expanded.

## The pipeline

| component | implemented | synthetic | real data |
| --- | :-: | :-: | :-: |
| discovery, pairing, EDF header validation, manifest | yes | yes | 153 recordings, 78 participants |
| epoching, label mapping, quality control, gap handling | yes | yes | 414,961 epochs stored, 414,481 eligible |
| participant-disjoint splits, nested label budgets | yes | yes | 47 / 16 / 15 participants |
| preprocessing, train-only normalisation, disk-backed store | yes | yes | yes |
| classical baselines: class prior, logistic, random forest | yes | yes | pilot only |
| D1 — epoch CNN, 489,477 parameters | yes | yes | cohort, 15 held out |
| D2 — temporal transformer over 11 epochs, 756,229 parameters | yes | yes | cohort, 15 held out |
| D3 — masked-reconstruction pretraining, then D2 | yes | yes | cohort, 15 held out |
| saved per-epoch predictions and generated report tables | yes | yes | yes |
| transition analysis | yes | — | cohort |
| sleep-window sensitivity analysis | yes | yes | 12 recordings, quantified |
| label-budget benchmark harness | yes | yes | 14 of 18 cells |

## The six required controls

Every one is implemented. Five have been run on the cohort.

| control | what it rules out | run | result |
| --- | --- | :-: | --- |
| shuffled neighbours | that a temporal model uses the *order* of its context | yes | costs D2 0.020 — order matters, modestly |
| masked context | that it uses the context *at all* | yes | costs D2 0.076 — the context is real |
| compute-matched D2 | that D2's deficit was undertraining | yes | pattern unchanged at equal passes |
| training-fitted temporal smoothing | that a transition table already does the job | yes | 0.655 against D1's 0.673 — it does not, on the cohort |
| pretrained encoder, no temporal context | that the gain is the representation, not the transformer | yes | 0.673, identical to random init |
| frozen random vs frozen pretrained probe | that fine-tuning is doing the work | yes | 0.537 against 0.291 — the representation is real |
| one-channel training vs channel loss at inference | that robustness and design are the same question | implemented | not run |

## The benchmark

18 cells: 2 models × 3 budgets × 3 seeds. **14 complete.**

| budget | D2 | D3 | cells |
| --- | :-: | :-: | --- |
| 10% (5 participants) | 3 seeds | 3 seeds | complete |
| 25% (12 participants) | 3 seeds | 3 seeds | complete |
| 100% (47 participants) | seed 0 | seed 0 | **seeds 1–2 missing** |

The missing four are the most expensive cells — about two hours each — and the
least informative, since the 100% budget's answer is established twice over
(benchmark seed 0: +0.000; the separate cohort run: −0.027). They stopped
because the machine did; see below.

## What has not been done at all

* **The stricter limited-label setting.** Every cell selected its checkpoint on
  all 16 validation participants, which is more supervision than a 10% label
  budget implies. `--reduce-validation` implements the honest version and has
  not been run. This is disclosed everywhere a budget result appears.
* **More than one participant split.** Every result rests on one split with one
  seed. The split generator is deterministic and takes a seed; nothing has been
  repeated across splits.
* **Within-run resumption.** Checkpoints save the best model, not the optimiser,
  epoch counter and RNG state, so an interrupted cell restarts from zero. A
  kernel panic cost a two-hour cell this way.
* **Any clinical or mechanistic claim.** None is made, and the transition
  analysis says so on its own first line.
