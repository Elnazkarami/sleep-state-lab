"""The models. D1 is implemented; D2 and D3 are contracts, in docs/model_contracts.md.

The encoder is separated from the head on purpose. D2 puts a temporal
transformer over the *same* encoder's outputs, and D3 initialises that same
encoder from self-supervised pretraining; if the backbone differed between them
by so much as a pooling choice, the D3-minus-D2 comparison would be measuring
the difference between two architectures rather than the effect of pretraining.
"""

from sleepstatelab.models.d1 import D1Classifier, StageHead
from sleepstatelab.models.encoder import EncoderOutput, EpochEncoder

__all__ = ["D1Classifier", "EncoderOutput", "EpochEncoder", "StageHead"]
