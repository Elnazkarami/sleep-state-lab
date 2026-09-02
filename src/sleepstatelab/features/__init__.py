"""Engineered descriptions of an epoch, for the classical baselines only.

Nothing the neural models see passes through this package. The features exist so
that the comparison in the README is between a learned representation and a
*fair* hand-built one -- computed on the same epochs, under the same quality
control, with the same participants held out.
"""

from sleepstatelab.features.spectral import (
    BANDS,
    FEATURE_SET,
    FEATURE_SET_VERSION,
    epoch_features,
    feature_matrix,
    feature_names,
)

__all__ = [
    "BANDS",
    "FEATURE_SET",
    "FEATURE_SET_VERSION",
    "epoch_features",
    "feature_matrix",
    "feature_names",
]
