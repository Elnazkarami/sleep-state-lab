"""Temporal smoothing: how much of D2's context benefit a transition table buys.

The control this answers is the cheapest and most deflating one available. A
temporal model is supposed to be worth having because sleep is not a sequence of
independent 30-second draws -- stages persist, and the transitions between them
are far from uniform. But that fact is available without a transformer: fit a
5x5 transition matrix on the training participants, and decode a single-epoch
model's per-epoch probabilities through it.

If D2 does not beat D1 plus this, then whatever D2 learned about time is
something a table of transition counts already knew.

**It is fitted on training participants only**, like everything else, and it
reads a model's saved probabilities rather than its internals -- so it applies to
any model in this repository without touching it.

**It respects gaps.** Viterbi runs over runs of *consecutive* epochs. Two epochs
either side of an excluded one are not neighbours, and decoding across the hole
would assert a transition that was never observed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sleepstatelab.labels import STAGES


@dataclass(frozen=True, slots=True)
class TransitionModel:
    """Stage priors and transition probabilities, estimated from labels."""

    prior: tuple[float, ...]
    transition: tuple[tuple[float, ...], ...]
    fitted_on: tuple[str, ...]
    n_transitions: int
    smoothing: float

    @property
    def log_prior(self) -> np.ndarray:
        return np.log(np.asarray(self.prior, dtype=np.float64))

    @property
    def log_transition(self) -> np.ndarray:
        return np.log(np.asarray(self.transition, dtype=np.float64))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        stay = [self.transition[i][i] for i in range(len(STAGES))]
        parts = "  ".join(f"{name} {value:.3f}" for name, value in zip(STAGES, stay, strict=True))
        return (
            f"{self.n_transitions} transitions from {len(self.fitted_on)} "
            f"participant(s); P(stay) {parts}"
        )


def fit_transitions(
    labels_by_recording: list[tuple[np.ndarray, np.ndarray]],
    participants: list[str],
    *,
    smoothing: float = 1.0,
) -> TransitionModel:
    """Count transitions between epochs that are genuinely adjacent.

    Each item is ``(epoch_index, label)`` for one recording. A pair counts only
    when the epoch indices differ by exactly one: an excluded epoch between them
    means the transition was never observed, and counting it would invent
    evidence for a stage change that may not have happened.

    Laplace smoothing keeps every transition possible. A zero would let one
    unobserved pairing veto a path through the whole night.
    """
    n = len(STAGES)
    counts = np.full((n, n), smoothing, dtype=np.float64)
    prior = np.full(n, smoothing, dtype=np.float64)
    observed = 0

    for index, labels in labels_by_recording:
        index = np.asarray(index)
        labels = np.asarray(labels)
        for value in labels:
            prior[int(value)] += 1.0
        adjacent = np.flatnonzero(np.diff(index) == 1)
        for position in adjacent:
            counts[int(labels[position]), int(labels[position + 1])] += 1.0
            observed += 1

    # A stage never seen as the source of a transition leaves an all-zero row,
    # which divides to NaN and then poisons every path through the decoder.
    # It cannot happen with the default smoothing, and it is handled anyway:
    # nothing was observed about where that stage goes, and uniform is the
    # honest statement of that.
    totals = counts.sum(axis=1, keepdims=True)
    uniform = np.full_like(counts, 1.0 / n)
    transition = np.where(totals > 0, counts / np.where(totals > 0, totals, 1.0), uniform)

    return TransitionModel(
        prior=tuple(prior / prior.sum()) if prior.sum() > 0 else tuple([1.0 / n] * n),
        transition=tuple(tuple(row) for row in transition),
        fitted_on=tuple(sorted(set(participants))),
        n_transitions=observed,
        smoothing=smoothing,
    )


def viterbi(
    log_emission: np.ndarray, model: TransitionModel, *, prior_correction: bool = True
) -> np.ndarray:
    """Most likely stage sequence for one run of consecutive epochs.

    ``log_emission`` is ``[n_epochs, 5]``. A classifier's softmax is a posterior,
    ``P(stage | epoch)``, and what a hidden Markov decoder wants is a likelihood,
    ``P(epoch | stage)``. Dividing by the training prior converts one to the
    other up to a constant, which is what ``prior_correction`` does. Without it
    the class prior is applied twice -- once by the classifier and again by the
    transition model -- and the decoder becomes far too fond of N2.
    """
    n_epochs, n_states = log_emission.shape
    if n_states != len(STAGES):
        raise ValueError(f"expected {len(STAGES)} classes, got {n_states}")
    if n_epochs == 0:
        return np.empty(0, dtype=np.int64)

    emission = log_emission.copy()
    if prior_correction:
        emission = emission - model.log_prior[None, :]

    log_transition = model.log_transition
    scores = model.log_prior + emission[0]
    backpointers = np.zeros((n_epochs, n_states), dtype=np.int64)
    for step in range(1, n_epochs):
        candidates = scores[:, None] + log_transition
        backpointers[step] = candidates.argmax(axis=0)
        scores = candidates.max(axis=0) + emission[step]

    path = np.zeros(n_epochs, dtype=np.int64)
    path[-1] = int(scores.argmax())
    for step in range(n_epochs - 1, 0, -1):
        path[step - 1] = backpointers[step, path[step]]
    return path


def consecutive_runs(epoch_index: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ranges of positions whose epoch indices step by exactly one."""
    index = np.asarray(epoch_index)
    if index.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(index) != 1) + 1
    bounds = [0, *breaks.tolist(), int(index.size)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def smooth_probabilities(
    probabilities: np.ndarray,
    epoch_index: np.ndarray,
    model: TransitionModel,
    *,
    floor: float = 1e-12,
) -> np.ndarray:
    """Decode one recording's probabilities, one consecutive run at a time.

    Returns a one-hot ``[n, 5]``: Viterbi produces a sequence, not a calibrated
    distribution, and returning something that looks like a probability would
    invite it to be read as one. The saved rows keep the original probabilities
    under the unsmoothed model, so nothing is lost.
    """
    log_emission = np.log(np.clip(probabilities, floor, None))
    out = np.zeros_like(probabilities)
    for start, stop in consecutive_runs(epoch_index):
        path = viterbi(log_emission[start:stop], model)
        out[np.arange(start, stop), path] = 1.0
    return out


def write_transition_model(path: Path | str, model: TransitionModel) -> None:
    import json

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(model.to_dict(), indent=2))
