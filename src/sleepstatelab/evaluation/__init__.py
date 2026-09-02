"""Saved predictions, and every metric computed from them.

Nothing in this package reports a number that was not read back off disk. A
model writes one row per epoch it predicted; the metrics read those rows; the
report tables are generated from the same rows. There is no path by which a
score can be typed into documentation.
"""

from sleepstatelab.evaluation.metrics import (
    EvaluationResult,
    evaluate_predictions,
    participant_macro_f1,
)
from sleepstatelab.evaluation.predictions import (
    PREDICTION_COLUMNS,
    PredictionWriter,
    read_predictions,
)

__all__ = [
    "PREDICTION_COLUMNS",
    "EvaluationResult",
    "PredictionWriter",
    "evaluate_predictions",
    "participant_macro_f1",
    "read_predictions",
]
