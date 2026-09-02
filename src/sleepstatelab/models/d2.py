"""D2: the same epoch encoder, with a small transformer over eleven epochs.

A human scorer does not look at one epoch. They look at where it sits: a
30-second stretch that would be ambiguous alone is obvious after two minutes of
slow waves. D2 is the smallest model that can use that, and it is deliberately
the *only* thing that differs from D1 -- same encoder, same head, same loss,
same optimiser, same stopping rule -- so that the difference between them is
temporal context and not architecture.

**It is an offline model and this repository will not describe it otherwise.**
Predicting the central epoch of eleven means using 2.5 minutes of future signal.
That is legitimate for overnight scoring and it is not a real-time scorer;
comparing it against one would be unfair in D2's favour.

**Masking is the part that has to be right.** A context position is either a
genuine neighbour -- an epoch that exists, in the same recording, at exactly the
expected index -- or it is absent. Absent positions get a learned token *and* are
excluded from attention. They are never filled with the nearest available epoch,
which would fabricate a neighbour, and never zero-filled, which would teach the
model that a recording boundary looks like a flat signal.

The central position is never masked: a window is only ever built around an
epoch that is eligible and labelled.
"""

from __future__ import annotations

import torch
from torch import nn

from sleepstatelab.labels import STAGES
from sleepstatelab.models.d1 import StageHead
from sleepstatelab.models.encoder import EpochEncoder

DEFAULT_CONTEXT = 11
"""Five epochs before, the central epoch, five after. 5.5 minutes in total."""


class D2Classifier(nn.Module):
    """Epoch encoder over a window, transformer across it, head on the centre.

    Input is ``[batch, context, channels, samples]`` with a boolean
    ``[batch, context]`` mask, ``True`` where a position is a genuine neighbour.
    """

    def __init__(
        self,
        encoder: EpochEncoder | None = None,
        *,
        context: int = DEFAULT_CONTEXT,
        n_layers: int = 2,
        n_heads: int = 4,
        feedforward_multiplier: int = 2,
        n_classes: int = len(STAGES),
        dropout: float = 0.3,
        transformer_dropout: float = 0.1,
        **encoder_kwargs: object,
    ) -> None:
        super().__init__()
        if context < 1 or context % 2 == 0:
            raise ValueError(
                f"context must be odd so that a centre exists, got {context}"
            )
        self.encoder = encoder if encoder is not None else EpochEncoder(**encoder_kwargs)  # type: ignore[arg-type]
        self.context = context
        self.centre = context // 2
        self.n_classes = n_classes
        width = self.encoder.embedding_dim

        self.position = nn.Parameter(torch.zeros(1, context, width))
        nn.init.trunc_normal_(self.position, std=0.02)

        self.absent = nn.Parameter(torch.zeros(1, 1, width))
        """The learned stand-in for a position that has no neighbour. It exists
        so that an absent position is a *stated* absence rather than a number
        that happens to be zero, and it is what the model sees at a recording
        boundary."""
        nn.init.trunc_normal_(self.absent, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=n_heads,
            dim_feedforward=feedforward_multiplier * width,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Nested tensors are disabled explicitly: torch cannot use them with
        # norm_first and warns once per construction otherwise, which is noise
        # in every training log.
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(width)
        self.head = StageHead(width, n_classes, dropout)

    @classmethod
    def from_encoder(
        cls, encoder: EpochEncoder, **kwargs: object
    ) -> D2Classifier:
        """Attach the temporal stack to an existing backbone.

        The route a D1-trained or self-supervised encoder takes into D2, and the
        reason the encoder is a constructor argument rather than something built
        inside.
        """
        return cls(encoder=encoder, **kwargs)  # type: ignore[arg-type]

    def embed_window(self, x: torch.Tensor) -> torch.Tensor:
        """Encode every epoch in every window: ``[batch, context, embedding]``.

        The windows are folded into the batch dimension so the encoder sees a
        plain stack of epochs -- it has no idea it is being used inside a
        temporal model, which is what keeps it identical to D1's.
        """
        if x.dim() != 4:
            raise ValueError(
                f"expected [batch, context, channels, samples], got {tuple(x.shape)}"
            )
        batch, context, channels, samples = x.shape
        if context != self.context:
            raise ValueError(
                f"this model was built for a context of {self.context} epochs, "
                f"got {context}"
            )
        flat = x.reshape(batch * context, channels, samples)
        embedded = self.encoder(flat).embedding
        return embedded.reshape(batch, context, -1)

    def forward(
        self,
        x: torch.Tensor,
        second: torch.Tensor,
        third: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Logits, from either layout the datasets produce.

        ``forward(windows, mask)`` -- ``[batch, context, channels, samples]``
        with a ``[batch, context]`` mask -- gives ``[batch, classes]``, one
        prediction per window.

        ``forward(signals, gather, mask)`` -- a contiguous stretch of a
        recording, the indices that cut windows out of it, and their mask --
        gives ``[batch, centres, classes]``. Same model, shared encodings.

        One entry point rather than two because the training loop and the
        prediction loop both call ``model(*inputs)``, and a model that had to be
        called differently depending on its dataset is a model that will one day
        be called the wrong way.
        """
        if third is None:
            return self.classify(self.embed_window(x), second)
        return self.forward_segment(x, second, third)

    def classify(self, embedded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """The temporal half, given epoch embeddings already computed.

        Separated from ``forward`` so that inference can encode each epoch of a
        recording once and reuse it across the windows that overlap it, which is
        eleven times less work and provably the same arithmetic -- there is a
        test that asserts the two paths agree.
        """
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if mask.shape != embedded.shape[:2]:
            raise ValueError(
                f"mask {tuple(mask.shape)} does not match the window "
                f"{tuple(embedded.shape[:2])}"
            )
        if not bool(mask[:, self.centre].all()):
            raise ValueError(
                "the central position of every window must be present; a window "
                "is only ever built around an eligible, labelled epoch"
            )

        present = mask.unsqueeze(-1)
        tokens = torch.where(present, embedded, self.absent.to(embedded.dtype))
        tokens = tokens + self.position.to(embedded.dtype)
        # `src_key_padding_mask` marks positions to ignore, so it is the
        # negation of "this position is a real neighbour".
        encoded = self.transformer(tokens, src_key_padding_mask=~mask)
        centre = self.norm(encoded[:, self.centre])
        return self.head(centre)

    def forward_segment(
        self, signals: torch.Tensor, gather: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Predict many overlapping windows from one contiguous stretch of epochs.

        The arithmetic is identical to calling ``forward`` on each window
        separately; the difference is that an epoch shared by several windows is
        encoded **once** rather than once per window. Eleven-epoch windows
        stepped by one epoch share ten of their eleven, so a stretch of ``L``
        centres costs ``L + 10`` encodings instead of ``11 L`` -- about nine
        times less work for a long stretch, which is the difference between a
        D2 pass costing eleven D1 passes and costing one and a fifth.

        ``signals`` is ``[batch, rows, channels, samples]``: a contiguous run of
        stored epochs. ``gather`` is ``[batch, centres, context]``, indexing into
        those rows, and ``mask`` marks which of those positions are genuine
        neighbours. A position that is absent may hold any index -- it is
        replaced by the learned absent token before attention -- so the caller is
        free to point it anywhere in range.

        Returns ``[batch, centres, n_classes]``.

        A test asserts this agrees with the per-window path to float tolerance.
        If it ever stops agreeing, the fast path is wrong and the slow one is
        the truth.
        """
        if signals.dim() != 4:
            raise ValueError(
                f"expected [batch, rows, channels, samples], got {tuple(signals.shape)}"
            )
        if gather.shape[-1] != self.context:
            raise ValueError(
                f"gather indexes {gather.shape[-1]} positions, this model was "
                f"built for a context of {self.context}"
            )
        batch, rows, channels, samples = signals.shape
        embedded = self.encoder(
            signals.reshape(batch * rows, channels, samples)
        ).embedding.reshape(batch, rows, -1)

        centres = gather.shape[1]
        width = embedded.shape[-1]
        flat = gather.reshape(batch, centres * self.context, 1).expand(-1, -1, width)
        windows = torch.gather(embedded, 1, flat).reshape(
            batch, centres, self.context, width
        )
        logits = self.classify(
            windows.reshape(batch * centres, self.context, width),
            mask.reshape(batch * centres, self.context),
        )
        return logits.reshape(batch, centres, self.n_classes)

    def n_parameters(self) -> dict[str, int]:
        temporal = sum(p.numel() for p in self.transformer.parameters())
        temporal += int(self.position.numel() + self.absent.numel())
        temporal += sum(p.numel() for p in self.norm.parameters())
        return {
            "encoder": self.encoder.n_parameters(),
            "temporal": temporal,
            "head": sum(p.numel() for p in self.head.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
