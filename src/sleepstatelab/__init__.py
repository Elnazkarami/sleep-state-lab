"""SleepStateLab: self-supervised EEG representation learning and sleep-state decoding.

The package is organised so that every stage of the experiment is an importable
module and the command line is a thin wrapper over it. Nothing here reads a
notebook, and no result in the documentation is produced by one.

Stages, in the order they run:

``sleepstatelab.data``
    Discover Sleep-EDF Expanded recordings, pair each PSG with its hypnogram,
    validate what the header claims, and cut the signals into scored 30-second
    epochs with their gaps preserved.
``sleepstatelab.features``
    Spectral descriptions of an epoch, for the classical baselines only.
``sleepstatelab.baselines``
    The three models a learned representation has to beat before it is
    interesting: the training class prior, logistic regression, random forest.
``sleepstatelab.models``
    D1, the trainable epoch encoder, kept separate from its classification head
    so D2 and D3 can reuse exactly the same backbone.
``sleepstatelab.training``
    The supervised loop, its checkpoints, and the provenance recorded with them.
``sleepstatelab.evaluation``
    One saved row per participant/recording/epoch/model/run, and every reported
    table generated from those rows.
"""

from sleepstatelab.labels import STAGE_INDEX, STAGES

__version__ = "0.1.0"
__all__ = ["STAGES", "STAGE_INDEX", "__version__"]
