"""Classical baselines, on exactly the epochs the neural models see.

The comparison in this repository is only meaningful if the baselines and the
CNN are given the same problem, so the eligible-epoch mask, the participants,
the channels, the label mapping and the split all come from the same
configuration and the same cache. Nothing here re-derives a dataset.

Three models, each answering a different question:

``class_prior``
    Predicts the training class distribution for every epoch, so its argmax is
    the commonest training stage. This is the number a result has to beat, and
    on a Sleep Cassette night without cropping it is a high one: wake is about
    70% of the epochs, so accuracy alone says almost nothing. Its participant
    macro-F1 is the honest floor.

``logistic``
    Spectral features and a linear model. If this reaches the CNN's score, the
    CNN is answering a question the features had already answered.

``random_forest``
    The same features, non-linear. Where the two disagree, the disagreement is
    about interactions between bands rather than about the bands.

PhysioML's published sleep numbers were computed on cropped recordings with a
different feature set and different participants. They are context, not a
baseline, and these three are re-run here on matched data for that reason.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from sleepstatelab.labels import STAGES


@dataclass
class ClassPrior:
    """Predicts the training class distribution, unchanged, for every epoch.

    Written out rather than taken from scikit-learn's ``DummyClassifier`` so
    that it emits a full probability vector in this package's class order, which
    is what the saved predictions require.
    """

    prior_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> ClassPrior:
        counts = np.array([np.count_nonzero(y == index) for index in range(len(STAGES))])
        total = counts.sum()
        # A Laplace smoothing of one epoch per class: a stage absent from the
        # training participants gets a small non-zero probability rather than a
        # zero that would make a log-loss infinite.
        self.prior_ = (counts + 1.0) / (total + len(STAGES))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.prior_ is None:
            raise RuntimeError("ClassPrior has not been fitted")
        return np.tile(self.prior_, (len(x), 1))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)


def logistic() -> Any:
    """Standardised features, multinomial logistic regression, balanced classes.

    The scaler lives inside the pipeline so it is fitted on the training fold
    only. Fitting it on everything would let the test participants' distribution
    shape the transform applied to training data -- quieter than sharing
    participants, and it inflates the score the same way.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=0
                ),
            ),
        ]
    )


def random_forest() -> Any:
    """300 trees, leaves of at least five epochs, classes balanced per bootstrap."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=0,
                ),
            ),
        ]
    )


BASELINES: dict[str, Callable[[], Any]] = {
    "class_prior": ClassPrior,
    "logistic": logistic,
    "random_forest": random_forest,
}


def build_baseline(name: str) -> Any:
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; expected one of {sorted(BASELINES)}")
    return BASELINES[name]()


def probabilities_in_stage_order(model: Any, x: np.ndarray) -> np.ndarray:
    """``[n, 5]`` probabilities in ``labels.STAGES`` order, whatever the model saw.

    A scikit-learn classifier's ``predict_proba`` has one column per class it was
    *trained* on. If a training split contains no N3, the matrix has four columns
    and every downstream index is silently wrong by one. This maps the columns
    back by ``classes_`` and leaves the absent stage at zero.
    """
    raw = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = getattr(model.named_steps.get("model"), "classes_", None)
    if classes is None:
        if raw.shape[1] != len(STAGES):
            raise ValueError(
                f"model produced {raw.shape[1]} probability columns and exposes no "
                "classes_ to map them by"
            )
        return raw
    full = np.zeros((raw.shape[0], len(STAGES)), dtype=np.float64)
    for column, class_index in enumerate(np.asarray(classes, dtype=int)):
        full[:, int(class_index)] = raw[:, column]
    return full


def run_baselines(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    models: tuple[str, ...] = tuple(BASELINES),
) -> dict[str, np.ndarray]:
    """Fit each baseline on the training features and predict on the evaluation set.

    Returns probabilities in stage order per model. Fitting and prediction are
    kept together so no caller can accidentally fit on the evaluation features.
    """
    out: dict[str, np.ndarray] = {}
    for name in models:
        model = build_baseline(name)
        model.fit(train_x, train_y)
        out[name] = probabilities_in_stage_order(model, eval_x)
    return out
