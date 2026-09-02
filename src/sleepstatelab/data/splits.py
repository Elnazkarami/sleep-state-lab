"""Participant-disjoint splits, generated deterministically.

One rule, and everything here exists to make it impossible to break: **all
nights of one participant go to one split.** Sleep-EDF records most people
twice, and two nights of the same person share their skull, their electrode
placement and their spectrum. A model that has seen one night of a person can
identify them in the other, and it will score well while having learned less
than the number says.

The generator is deterministic given a seed and a participant list. The split's
*identity* is a hash of its three sorted participant lists, so two runs can be
compared for having used the same split rather than for having used the same
file name.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from sleepstatelab.provenance import digest


class SplitError(RuntimeError):
    """Raised when a split cannot be made, or when one is not participant-disjoint."""


@dataclass(frozen=True, slots=True)
class Split:
    """One train/validation/test division, named by the people on each side."""

    name: str
    seed: int
    strategy: str
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        self.check()

    @property
    def identity(self) -> str:
        return digest(
            {
                "strategy": self.strategy,
                "train": sorted(self.train),
                "val": sorted(self.val),
                "test": sorted(self.test),
            }
        )

    @property
    def participants(self) -> tuple[str, ...]:
        return tuple(sorted({*self.train, *self.val, *self.test}))

    def part_of(self, participant_id: str) -> str:
        for name in ("train", "val", "test"):
            if participant_id in getattr(self, name):
                return name
        raise KeyError(f"{participant_id!r} is in no part of split {self.name!r}")

    def check(self) -> None:
        """Refuse a split that shares a participant. Called on construction."""
        parts = {"train": set(self.train), "val": set(self.val), "test": set(self.test)}
        for left in ("train", "val", "test"):
            for right in ("train", "val", "test"):
                if left >= right:
                    continue
                shared = parts[left] & parts[right]
                if shared:
                    raise SplitError(
                        f"split {self.name!r} puts {sorted(shared)} in both "
                        f"{left} and {right}"
                    )
        if not self.train or not self.test:
            raise SplitError(f"split {self.name!r} has an empty train or test side")

    def summary(self) -> str:
        return (
            f"{self.name} [{self.identity}] {self.strategy} seed={self.seed}: "
            f"train {len(self.train)} / val {len(self.val)} / test {len(self.test)} "
            f"participants; test = {', '.join(self.test)}"
        )

    def write(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self) | {"identity": self.identity}
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def read(cls, path: Path | str) -> Split:
        payload = json.loads(Path(path).read_text())
        split = cls(
            name=payload["name"],
            seed=payload["seed"],
            strategy=payload["strategy"],
            train=tuple(payload["train"]),
            val=tuple(payload["val"]),
            test=tuple(payload["test"]),
        )
        stored = payload.get("identity")
        if stored and stored != split.identity:
            raise SplitError(
                f"{path} records identity {stored} but its participant lists hash to "
                f"{split.identity}; the file has been edited"
            )
        return split


def grouped_split(
    participants: list[str] | tuple[str, ...],
    *,
    seed: int = 0,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    name: str = "grouped",
) -> Split:
    """A deterministic participant-disjoint three-way split.

    Sizes are floors with a guarantee: validation and test each get at least one
    participant, and training keeps the remainder. With six participants and the
    default fractions that is 4/1/1, which is small enough that the estimate is
    a pilot rather than a benchmark -- and this function will not pretend
    otherwise, it simply returns the split it was asked for.
    """
    ordered = sorted(set(participants))
    if len(ordered) < 3:
        raise SplitError(
            f"a train/validation/test split needs at least three participants, got "
            f"{len(ordered)}: {ordered}"
        )
    if not 0 < train_fraction < 1 or not 0 <= val_fraction < 1:
        raise SplitError("fractions must lie in (0, 1)")
    if train_fraction + val_fraction >= 1:
        raise SplitError(
            "train and validation fractions leave nothing for test: "
            f"{train_fraction} + {val_fraction} >= 1"
        )

    shuffled = list(ordered)
    np.random.default_rng(seed).shuffle(shuffled)
    total = len(shuffled)
    n_val = max(1, round(total * val_fraction))
    n_test = max(1, total - round(total * train_fraction) - n_val)
    n_train = total - n_val - n_test
    if n_train < 1:
        raise SplitError(
            f"{total} participants cannot be divided {train_fraction}/{val_fraction} "
            "and leave anyone to train on"
        )
    test = tuple(sorted(shuffled[:n_test]))
    val = tuple(sorted(shuffled[n_test : n_test + n_val]))
    train = tuple(sorted(shuffled[n_test + n_val :]))
    return Split(
        name=name, seed=seed, strategy="grouped_random", train=train, val=val, test=test
    )


def label_budget_subsets(
    split: Split, budgets: tuple[float, ...] = (0.1, 0.25, 1.0), *, seed: int = 0
) -> dict[float, tuple[str, ...]]:
    """Nested subsets of the *training* participants, for the label-budget study.

    Nested by construction: the participants at 10% are a prefix of those at 25%,
    which are a prefix of the whole. Nesting matters because an un-nested draw
    confounds "more labels" with "different people", and with twenty-odd
    participants the difference between two random subsets is not small.

    Validation and test participants are untouched. The disclosure that goes with
    any result from this: the validation set is *not* reduced with the budget
    unless a run says so, so a 10% run still selects its checkpoint using the
    full validation labels.
    """
    ordered = list(split.train)
    np.random.default_rng(seed).shuffle(ordered)
    subsets: dict[float, tuple[str, ...]] = {}
    for budget in sorted(budgets):
        if not 0 < budget <= 1:
            raise SplitError(f"label budget {budget} is not in (0, 1]")
        take = max(1, round(len(ordered) * budget))
        subsets[budget] = tuple(sorted(ordered[:take]))
    return subsets
