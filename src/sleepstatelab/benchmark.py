"""The label-budget benchmark: D3 minus D2 at nested labelled-participant budgets.

This is the comparison the repository was built to make, and the reason so much
of the rest of it is about refusing to compare things that are not comparable.

**The primary planned comparison, declared before it is run:** D3 minus D2 in
mean participant macro-F1 at the **25%** labelled-training-participant budget.
One number. Everything else here is context for it.

**What a cell is.** One budget, one seed, one model. D2 starts from random
weights; D3 starts from the self-supervised encoder, which saw the *training*
participants' signal and no labels at all. Both then train identically -- same
loop, same loss, same class weighting, same optimiser, same stopping rule -- so
the difference between them is the initialisation.

**Budgets are nested.** The participants at 10% are a subset of those at 25%,
which are a subset of the whole. An un-nested draw would confound "more labels"
with "different people", and with 47 training participants the difference
between two random subsets is not small.

**Validation-label access is disclosed, not hidden.** As implemented, reducing
the training budget does *not* reduce the validation set: a 10% run still
selects its checkpoint using all 16 validation participants' labels, which is
more supervision than a real 10%-label setting would have. Every result this
module writes carries that disclosure, and `--reduce-validation` runs the
stricter version instead.

**It is resumable.** A cell whose checkpoint and predictions already exist is
skipped. Eighteen cohort-scale cells is days of compute, and a benchmark that
cannot survive an interruption is a benchmark that will not be run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_BUDGETS: tuple[float, ...] = (0.1, 0.25, 1.0)
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)
PRIMARY_BUDGET = 0.25
"""The budget the planned comparison is made at. Declared here so that reading
it off a table afterwards is not a choice anyone gets to make."""


@dataclass(frozen=True, slots=True)
class Cell:
    """One run of the benchmark: a model, a budget, a seed."""

    model: str
    budget: float
    seed: int
    participants: tuple[str, ...]

    @property
    def name(self) -> str:
        """What this cell's predictions are saved under."""
        return f"{self.model}@{self.budget:.0%}-s{self.seed}"

    @property
    def run_id(self) -> str:
        return f"benchmark-{self.model.lower()}-b{int(self.budget * 100)}-s{self.seed}"

    def checkpoint_path(self, root: Path | str) -> Path:
        return Path(root) / self.run_id / "checkpoint.pt"


@dataclass(frozen=True, slots=True)
class Plan:
    """Every cell the benchmark will run, and what it costs."""

    cells: tuple[Cell, ...]
    budgets: tuple[float, ...]
    seeds: tuple[int, ...]
    models: tuple[str, ...]
    reduce_validation: bool
    validation_disclosure: str
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def summary(self) -> str:
        lines = [
            f"{len(self.cells)} cell(s): {len(self.models)} model(s) x "
            f"{len(self.budgets)} budget(s) x {len(self.seeds)} seed(s)",
            f"primary comparison: D3 minus D2 at the {PRIMARY_BUDGET:.0%} budget, "
            "in mean participant macro-F1",
            self.validation_disclosure,
            "",
        ]
        for budget in self.budgets:
            people = next(c.participants for c in self.cells if c.budget == budget)
            lines.append(
                f"  {budget:>5.0%}: {len(people)} training participant(s) -- "
                f"{', '.join(people[:6])}{' ...' if len(people) > 6 else ''}"
            )
        return "\n".join(lines)


def build_plan(
    split: Any,
    *,
    budgets: tuple[float, ...] = DEFAULT_BUDGETS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    models: tuple[str, ...] = ("D2", "D3"),
    split_seed: int = 0,
    reduce_validation: bool = False,
) -> Plan:
    """Every cell, with each budget's nested participant subset resolved."""
    from sleepstatelab.data.splits import label_budget_subsets

    subsets = label_budget_subsets(split, tuple(budgets), seed=split_seed)
    for smaller, larger in zip(sorted(budgets), sorted(budgets)[1:], strict=False):
        if not set(subsets[smaller]) <= set(subsets[larger]):
            raise ValueError(
                f"the {smaller:.0%} subset is not contained in the {larger:.0%} one; "
                "budgets must be nested or the comparison confounds more labels "
                "with different people"
            )

    cells = tuple(
        Cell(model=model, budget=budget, seed=seed, participants=subsets[budget])
        for budget in budgets
        for seed in seeds
        for model in models
    )
    disclosure = (
        "validation labels are reduced with the budget"
        if reduce_validation
        else (
            "DISCLOSURE: validation labels are NOT reduced with the budget -- every "
            "cell selects its checkpoint on all validation participants, which is "
            "more supervision than the budget implies"
        )
    )
    return Plan(
        cells=cells,
        budgets=tuple(budgets),
        seeds=tuple(seeds),
        models=tuple(models),
        reduce_validation=reduce_validation,
        validation_disclosure=disclosure,
        notes={
            "primary_budget": PRIMARY_BUDGET,
            "primary_comparison": "D3 - D2 in mean participant macro-F1",
            "split_id": split.identity,
            "nested": True,
        },
    )


def difference_at_budget(
    results: list[Any], budget: float = PRIMARY_BUDGET, *, minuend: str = "D3", subtrahend: str = "D2"
) -> dict[str, Any]:
    """The planned comparison, computed from evaluated results.

    Reports the difference in mean participant macro-F1 at one budget, averaged
    over seeds, with the spread across seeds beside it. The spread is not
    decoration: with three seeds a difference smaller than their range is not a
    difference anyone should describe as one.
    """
    import numpy as np

    def scores(model: str) -> list[float]:
        prefix = f"{model}@{budget:.0%}-s"
        return [
            r.participant_macro_f1_mean
            for r in results
            if r.model.startswith(prefix)
        ]

    first, second = scores(minuend), scores(subtrahend)
    if not first or not second:
        return {
            "budget": budget,
            "available": False,
            "reason": (
                f"no results for {minuend} and {subtrahend} at {budget:.0%}; "
                "the benchmark has not been run at this budget"
            ),
        }
    return {
        "budget": budget,
        "available": True,
        f"{minuend}_mean": float(np.mean(first)),
        f"{minuend}_seeds": len(first),
        f"{subtrahend}_mean": float(np.mean(second)),
        f"{subtrahend}_seeds": len(second),
        "difference": float(np.mean(first) - np.mean(second)),
        f"{minuend}_spread": float(np.max(first) - np.min(first)) if len(first) > 1 else None,
        f"{subtrahend}_spread": float(np.max(second) - np.min(second)) if len(second) > 1 else None,
    }
