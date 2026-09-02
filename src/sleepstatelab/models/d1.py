"""D1: the epoch encoder with a linear stage head on top.

The model the first release trains, and the floor the temporal models have to
clear. It sees one 30-second epoch and nothing else -- no neighbours, no
hypnogram context -- which is a harder problem than the one a human scorer
solves, and deliberately so: it isolates what a single epoch carries.

The head is a separate module with its own name so a checkpoint can be loaded
into an encoder alone. ``D1Classifier.from_encoder`` is how D3's pretrained
backbone will be attached to a fresh head.
"""

from __future__ import annotations

import torch
from torch import nn

from sleepstatelab.labels import STAGES
from sleepstatelab.models.encoder import EncoderOutput, EpochEncoder


class StageHead(nn.Module):
    """Dropout and one linear layer to five unnormalised logits.

    Linear on purpose. A deeper head would make it ambiguous whether a gain came
    from the representation or from the classifier sitting on it, and the whole
    experiment is about the representation.
    """

    def __init__(self, embedding_dim: int, n_classes: int = len(STAGES), dropout: float = 0.3) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(embedding_dim, n_classes)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(embedding))


class D1Classifier(nn.Module):
    """Encoder plus head. Returns logits, and keeps the representation reachable."""

    def __init__(
        self,
        encoder: EpochEncoder | None = None,
        n_classes: int = len(STAGES),
        dropout: float = 0.3,
        **encoder_kwargs: object,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else EpochEncoder(**encoder_kwargs)  # type: ignore[arg-type]
        self.head = StageHead(self.encoder.embedding_dim, n_classes, dropout)
        self.n_classes = n_classes

    @classmethod
    def from_encoder(
        cls, encoder: EpochEncoder, *, n_classes: int = len(STAGES), dropout: float = 0.3
    ) -> D1Classifier:
        """Attach a fresh head to an existing backbone.

        The route a pretrained encoder takes into a supervised model, and the
        reason the head is not built inside the encoder.
        """
        return cls(encoder=encoder, n_classes=n_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x).embedding)

    def represent(self, x: torch.Tensor) -> EncoderOutput:
        """The encoder's outputs, without the head. For probes and for D2/D3."""
        return self.encoder(x)

    def n_parameters(self) -> dict[str, int]:
        return {
            "encoder": self.encoder.n_parameters(),
            "head": sum(p.numel() for p in self.head.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
