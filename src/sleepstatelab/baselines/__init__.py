"""The models a learned representation has to beat before it is interesting."""

from sleepstatelab.baselines.classical import (
    BASELINES,
    ClassPrior,
    build_baseline,
    run_baselines,
)

__all__ = ["BASELINES", "ClassPrior", "build_baseline", "run_baselines"]
