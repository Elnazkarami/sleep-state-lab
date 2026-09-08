"""The label-budget benchmark plan: nesting, naming, and the declared comparison.

The benchmark is eighteen cohort-scale training runs. Everything that can be
checked before spending days of compute is checked here.
"""

from __future__ import annotations

import pytest

from sleepstatelab.benchmark import (
    DEFAULT_BUDGETS,
    PRIMARY_BUDGET,
    build_plan,
    difference_at_budget,
)
from sleepstatelab.data.splits import grouped_split

pytestmark = pytest.mark.synthetic

PEOPLE = [f"SC4{i:02d}" for i in range(40)]


@pytest.fixture
def split():
    return grouped_split(PEOPLE, seed=3, name="bench")


def test_the_plan_covers_every_model_budget_and_seed(split):
    plan = build_plan(split, budgets=(0.1, 0.25, 1.0), seeds=(0, 1), models=("D2", "D3"))
    assert len(plan.cells) == 2 * 3 * 2
    assert {c.model for c in plan.cells} == {"D2", "D3"}
    assert {c.seed for c in plan.cells} == {0, 1}


def test_budgets_are_nested(split):
    """Un-nested budgets confound more labels with different people."""
    plan = build_plan(split, budgets=(0.1, 0.25, 1.0), seeds=(0,))
    by_budget = {c.budget: set(c.participants) for c in plan.cells}
    assert by_budget[0.1] <= by_budget[0.25] <= by_budget[1.0]
    assert by_budget[1.0] == set(split.train)


def test_a_budget_never_reaches_validation_or_test(split):
    plan = build_plan(split, budgets=DEFAULT_BUDGETS, seeds=(0,))
    for cell in plan.cells:
        assert not set(cell.participants) & set(split.val)
        assert not set(cell.participants) & set(split.test)


def test_cell_names_carry_the_budget_and_seed(split):
    plan = build_plan(split, budgets=(0.25,), seeds=(2,), models=("D3",))
    cell = plan.cells[0]
    assert cell.name == "D3@25%-s2"
    assert cell.run_id == "benchmark-d3-b25-s2"
    assert cell.checkpoint_path("runs").parts[-2:] == ("benchmark-d3-b25-s2", "checkpoint.pt")


def test_the_validation_disclosure_is_always_present(split):
    lenient = build_plan(split, seeds=(0,))
    assert "NOT reduced" in lenient.validation_disclosure
    assert "NOT reduced" in lenient.summary()

    strict = build_plan(split, seeds=(0,), reduce_validation=True)
    assert "reduced with the budget" in strict.validation_disclosure


def test_the_primary_comparison_is_declared_not_chosen():
    """Picking the budget after seeing the table is how a null result becomes a
    finding. It is fixed in the module."""
    assert PRIMARY_BUDGET == 0.25


def test_the_difference_reports_the_spread_across_seeds():
    class Result:
        def __init__(self, model, score):
            self.model = model
            self.participant_macro_f1_mean = score

    results = [
        Result("D3@25%-s0", 0.70),
        Result("D3@25%-s1", 0.72),
        Result("D2@25%-s0", 0.66),
        Result("D2@25%-s1", 0.68),
    ]
    found = difference_at_budget(results, budget=0.25)
    assert found["available"]
    assert found["difference"] == pytest.approx(0.04)
    assert found["D3_spread"] == pytest.approx(0.02)
    assert found["D2_seeds"] == 2


def test_an_unrun_budget_says_so_rather_than_returning_a_number():
    found = difference_at_budget([], budget=0.25)
    assert not found["available"]
    assert "has not been run" in found["reason"]


def test_the_plan_round_trips(split, tmp_path):
    import json

    plan = build_plan(split, seeds=(0,))
    plan.write(tmp_path / "plan.json")
    payload = json.loads((tmp_path / "plan.json").read_text())
    assert len(payload["cells"]) == len(plan.cells)
    assert payload["notes"]["split_id"] == split.identity
