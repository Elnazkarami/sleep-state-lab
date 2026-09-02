"""The metrics compute what the documentation says they compute."""

from __future__ import annotations

import numpy as np
import pytest

from sleepstatelab.evaluation.metrics import (
    cohens_kappa,
    evaluate_rows,
    participant_macro_f1,
)
from sleepstatelab.labels import STAGES

pytestmark = pytest.mark.synthetic


def _rows(participant: str, truth: list[str], predicted: list[str]) -> list[dict]:
    rows = []
    for index, (true_label, pred_label) in enumerate(zip(truth, predicted, strict=True)):
        probabilities = {f"p_{name}": 0.0 for name in STAGES}
        probabilities[f"p_{pred_label}"] = 1.0
        rows.append(
            {
                "run_id": "t",
                "model": "m",
                "split_id": "s",
                "split_part": "test",
                "seed": 0,
                "participant_id": participant,
                "recording_id": f"{participant}-n1",
                "epoch_index": index,
                "true_label": true_label,
                "pred_label": pred_label,
                "qc_flags": 0,
                **probabilities,
            }
        )
    return rows


def test_perfect_prediction_scores_one():
    truth = ["Wake", "N1", "N2", "N3", "REM"]
    result = evaluate_rows(_rows("A", truth, truth))
    assert result.participant_macro_f1_mean == pytest.approx(1.0)
    assert result.cohens_kappa == pytest.approx(1.0)


def test_absent_class_with_no_predictions_does_not_count():
    """A participant who never reached N3, and was never predicted N3, is scored
    over four classes -- not penalised for a stage that never came up."""
    truth = np.array([0, 0, 1, 1, 2, 4])
    score, _, counted = participant_macro_f1(truth, truth)
    assert counted.tolist() == [True, True, True, False, True]
    assert score == pytest.approx(1.0)


def test_predicting_a_stage_that_never_happened_is_penalised():
    truth = np.array([0, 0, 0, 0])
    predicted = np.array([0, 0, 0, 3])
    score, f1, counted = participant_macro_f1(truth, predicted)
    assert counted[3], "a predicted-but-absent class must count"
    assert f1[3] == 0.0
    assert score < 1.0


def test_participant_average_differs_from_pooled_when_sizes_differ():
    """The reason both are reported: one large participant can carry the pooled
    number while the per-person mean says something else."""
    big = _rows("BIG", ["Wake"] * 40 + ["N2"] * 40, ["Wake"] * 40 + ["N2"] * 40)
    small = _rows("SMALL", ["Wake", "N2"], ["N2", "Wake"])
    result = evaluate_rows(big + small)
    assert result.participant_macro_f1_mean == pytest.approx(0.5)
    assert result.pooled_macro_f1 > result.participant_macro_f1_mean


def test_kappa_of_a_constant_predictor_is_zero():
    truth = ["Wake"] * 7 + ["N2"] * 3
    result = evaluate_rows(_rows("A", truth, ["Wake"] * 10))
    assert result.cohens_kappa == pytest.approx(0.0, abs=1e-9)
    assert result.pooled_accuracy == pytest.approx(0.7)


def test_kappa_matches_a_worked_example():
    confusion = np.array([[20, 5], [10, 15]])
    # observed 35/50 = 0.7; expected = (30*25 + 20*25)/2500 = 0.5; kappa = 0.4
    assert cohens_kappa(confusion) == pytest.approx(0.4)


def test_confusion_rows_are_truth():
    result = evaluate_rows(_rows("A", ["Wake", "Wake"], ["Wake", "N2"]))
    assert result.confusion[0][0] == 1
    assert result.confusion[0][2] == 1
