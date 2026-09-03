# Model contracts: D1 and D2 implemented, D3 specified

**Status, stated once and not softened anywhere else in this repository:**

| | status |
| --- | --- |
| D1 — epoch CNN | **implemented and trained** (synthetic and a six-participant real pilot) |
| D2 — temporal transformer over 11 epochs | **implemented and trained** (synthetic and the same pilot) |
| D3 — D2 with a self-supervised pretrained encoder | **specified below; not implemented** |
| the label-budget benchmark | **not run** |
| the required controls | **two of six implemented** (shuffled neighbours, masked context); the rest are listed below and not run |

No checkpoint, score, or conclusion for D3 exists, and the D1-versus-D2 pilot
numbers are not a comparison — the two were not given matched compute, which the
results section states wherever they appear.

---

## D1 — the epoch encoder (implemented)

One 30-second epoch of two-channel EEG in; five logits, an epoch embedding, and
temporal tokens out.

* input `[batch, 2, 3000]` — channel 0 is `EEG Fpz-Cz`, channel 1 is
  `EEG Pz-Oz`, and a checkpoint records that order;
* stem: `Conv1d(2 → 32, kernel 49, stride 6)`, GroupNorm, GELU, MaxPool 2;
* three blocks of two `kernel 9` convolutions each, widths 64 → 96 → 128, each
  followed by MaxPool 2;
* tokens: `[batch, 128, 31]`, about half a second of signal each;
* embedding: mean- and max-pool over time, concatenated, projected to 128;
* head: dropout 0.3, `Linear(128 → 5)`, unnormalised logits;
* 489,477 parameters — 488,832 in the encoder, 645 in the head;
* GroupNorm rather than BatchNorm, so inference on one participant does not
  depend on who else is in the batch and no training-set statistics ride along
  inside the model.

The encoder and the head are separate modules, and
`D1Classifier.from_encoder(encoder)` attaches a fresh head to an existing
backbone. That is the seam D2 and D3 are built on: they must reuse *this*
encoder, not a re-implementation of it, or the comparison measures architecture
rather than initialisation.

---

## D2 — temporal context (implemented)

Implemented in `sleepstatelab.models.d2` with its windows built by
`sleepstatelab.training.windows`. What follows was the contract; it is now also
the description of what the code does, and `tests/test_d2.py` asserts each of
the rules below rather than trusting this page.

**Parameter count: 756,229** — 488,832 in the encoder (bit-for-bit D1's),
266,752 in the temporal stack, 645 in the head.

**Architecture.** The D1 epoch encoder applied to each of 11 consecutive epochs
— five before, the central epoch, five after — producing 11 epoch embeddings,
then a small transformer encoder over that sequence with learned positional
embeddings, then a head on the *central* position only. One prediction per
window, for the central epoch's stage.

**This is an offline model and must be described as one.** It uses 2.5 minutes
of future signal. It is not a real-time scorer and any comparison against one
would be unfair in D2's favour.

**Boundary and gap rules, which are the part that is easy to get wrong.**

* A context window never crosses a recording boundary, and never crosses a
  participant. Recordings are separate sequences.
* A context position whose epoch index is not exactly `centre + offset` — that
  is, one on the other side of an excluded epoch — is **masked**, not filled
  with its neighbour and not silently skipped. The epoch indices kept by
  `sleepstatelab.data.epochs` are what makes this decidable.
* At the start and end of a recording the missing positions are masked the same
  way. Padding with zeros would teach the model that a boundary looks like a
  flat signal.
* The central epoch must itself be eligible; a window is never built around an
  epoch that has no label.

**Training.** Identical supervised settings to D1 — same loss, same class
weighting from training participants only, same optimiser, same stopping rule on
validation participant macro-F1 — so that D2 minus D1 is temporal context and
nothing else. This is enforced rather than intended: both models are trained by
the same `train_supervised` routine, and both take their class weights from
`class_weights_from_counts`, with a test asserting the two datasets produce
identical weights.

**What is implemented and what that leaves.** The model, the windowing, the
masking, training, checkpointing, prediction and the shuffled-neighbour control
all run, on synthetic data and on the six-participant pilot. What has *not* been
done is a comparison: on the pilot, D1 was given 20 passes over the training set
and D2 six, because a D2 pass costs about eleven times a D1 pass on a CPU. Those
two numbers are not compute-matched and are not a measurement of what temporal
context is worth.

**Shared encodings.** Overlapping windows share ten of their eleven epochs, so
training and inference encode a contiguous stretch of a recording once and
gather windows out of it: 32 centres cost 42 encodings instead of 352. This is
the same model, not an approximation — `test_segment_forward_equals_window_forward`
asserts the two paths give the same logits, and
`test_segment_and_window_predictions_agree` asserts the same probabilities end to
end. `--no-segments` runs the window-by-window path, kept so the fast one can be
checked against it.

What it does change is what a batch is: several stretches rather than several
independent windows, so examples within a step are correlated in time. The
encoder carries no batch statistics — GroupNorm throughout — so nothing in the
model depends on batch composition; what remains is a slightly less diverse
gradient at each step.

---

## D3 — pretrained temporal model (specified, not implemented)

**Inference architecture is identical to D2.** Same encoder, same transformer,
same context window, same masking. The *only* difference is that D3's epoch
encoder starts from self-supervised masked reconstruction instead of random
initialisation. If anything else differs, the comparison is void.

**Pretraining task.** Masked reconstruction of raw EEG patches:

* the input epoch is divided into fixed-length patches of raw samples;
* a random subset of patches is masked **before** the encoder sees anything;
* the masked patches are replaced by a learned mask token — **the hidden target
  values must not enter the encoder in any form**, including through
  normalisation statistics computed over the unmasked signal;
* a lightweight decoder reconstructs the masked patches from the encoder's
  tokens;
* the loss is computed **only on masked, valid samples** — never on visible
  patches, never on samples belonging to an excluded epoch.

**Pretraining uses training participants only.** No validation or test
participant contributes a single sample to pretraining, and the pretraining
manifest records exactly whose data it saw. A pretraining run that has seen the
test participants is not a limited-label experiment; it is leakage with extra
steps.

---

## The planned comparison

**Primary planned comparison: D3 minus D2 in mean participant macro-F1 at the
25% labelled-training-participant budget.** One number, declared before the
benchmark is run.

* Budgets: nested 10%, 25%, 100% of the *training* participants
  (`sleepstatelab.data.splits.label_budget_subsets`, nested by construction so
  "more labels" is not confounded with "different people").
* **Validation-label access must be disclosed with every result.** As
  implemented, reducing the training budget does *not* reduce the validation
  set: a 10% run still selects its checkpoint using the full validation labels,
  which is more supervision than a genuine 10%-label setting would have. Any
  reported budget result states this, or reduces validation with the budget and
  states that instead.
* Multiple seeds per cell, with the spread reported, not a single run.
* Matched supervised training settings across D2 and D3: same schedule, same
  batch size, same epochs, same early-stopping rule.

## Required controls, before any claim about pretraining

Each of these exists to close a specific alternative explanation.

0. **Shuffled neighbours** — *implemented*: `sleepstatelab predict
   --shuffle-context` permutes the non-central positions of every window,
   keeping the centre and the amount of real context fixed. It asks whether the
   **order** of the context is used. It cannot answer whether the context is
   used at all: a model that averaged its neighbours would be invariant to
   shuffling while depending on them completely, which is close to what the
   pilot actually found.
0b. **Masked context** — *implemented*: `sleepstatelab predict --mask-context`
   marks every non-central position absent, reducing a temporal model to its
   encoder on the central epoch. This is the control that answers whether the
   context contributes anything, and the two must be read together.
1. **Pretrained epoch CNN without temporal context** — is the gain from the
   representation, or only from having a transformer?
2. **Frozen random versus frozen pretrained encoder, same probe** — does the
   pretrained representation carry more, or is fine-tuning doing the work?
3. **A longer-trained D2** — is the gain pretraining, or just extra compute?
   D3 has seen more gradient steps in total; D2 must be given the same budget.
4. **D1 plus a training-fitted temporal smoothing baseline** — how much of the
   context benefit is available from a transition-probability smoother fitted on
   training participants? This is often most of it.
5. **One-channel models** — trained *with* one channel, kept strictly distinct
   from a two-channel model that loses a channel at inference. These are
   different questions and reporting them together would confuse a robustness
   claim with a design claim.

## Transitions, and what may not be claimed

Transition analysis uses annotations **for evaluation only** — epochs are
labelled near or far from a scored stage change, and performance is reported
separately for each. Transition proximity never enters training, sampling, or
model selection.

What classification performance and embedding pictures **cannot** establish, and
what will not be claimed from them:

* that sleep stages are attractors, or that transitions are bifurcations;
* any mechanism, cortical or thalamic;
* that an embedding cluster is a biological state rather than an artefact of
  amplitude, electrode impedance, or which participant a point came from;
* that anything here is a clinically useful biomarker.

A separation visible in a two-dimensional projection of a 128-dimensional
embedding is a property of the projection until it is shown to be more.
