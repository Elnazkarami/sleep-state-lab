"""The epoch encoder: 30 seconds of two-channel EEG in, a representation out.

Deliberately compact. The question this repository asks is whether
self-supervised pretraining helps under limited labels, and that question is
answered by holding the architecture fixed while the initialisation changes. A
larger encoder would make each run slower without making the comparison sharper,
and with twenty-odd participants it would mostly make overfitting easier.

**Three outputs, and why each exists.**

``tokens`` -- ``[batch, embedding, time]``, one vector per receptive-field span
of the epoch. D2's transformer attends over these, and D3's masked
reconstruction predicts through them. D1 does not use them, but it produces them,
so that the backbone D2 and D3 inherit is bit-for-bit the one D1 trained.

``embedding`` -- a single vector per epoch, mean and max pooled over time and
projected. Max pooling is included because a sleep spindle is a brief event, and
a mean over 30 seconds averages it away.

``logits`` -- produced by the head, not by the encoder. The encoder never sees
the number of classes.

**Normalisation is GroupNorm, not BatchNorm.** Batch statistics over a batch of
sleep epochs are dominated by whichever stages happen to be in the batch, and a
class-weighted sampler changes them again; worse, the running statistics a
BatchNorm carries into evaluation are a piece of the training distribution
leaking into inference on a new participant. GroupNorm has no batch dependence
and no running state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class EncoderOutput:
    """What the encoder returns. Named rather than a tuple, so a caller cannot
    silently swap the embedding and the tokens."""

    embedding: torch.Tensor
    """``[batch, embedding_dim]``."""

    tokens: torch.Tensor
    """``[batch, embedding_dim, n_tokens]``, for the temporal models."""


def _groups(channels: int) -> int:
    """A group count that divides the channel count, at most eight."""
    for candidate in (8, 4, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1


class ConvBlock(nn.Module):
    """Two same-width convolutions and a halving pool."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int) -> None:
        super().__init__()
        padding = kernel // 2
        self.body = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, padding=padding, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel, padding=padding, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
            nn.MaxPool1d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class EpochEncoder(nn.Module):
    """A 1D convolutional encoder over one 30-second epoch.

    Input ``[batch, channels, samples]``; with the package's defaults that is
    ``[batch, 2, 3000]``. The stem's stride-6 convolution over a 49-sample
    kernel is a half-second window stepped every 60 ms: long enough to see a
    slow wave's shape, short enough not to smear a spindle. Each block then
    halves the time axis, so the tokens the temporal models attend over are
    about half a second of signal each.
    """

    def __init__(
        self,
        in_channels: int = 2,
        stem_channels: int = 32,
        stem_kernel: int = 49,
        stem_stride: int = 6,
        block_channels: tuple[int, ...] = (64, 96, 128),
        block_kernel: int = 9,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                stem_channels,
                stem_kernel,
                stride=stem_stride,
                padding=stem_kernel // 2,
                bias=False,
            ),
            nn.GroupNorm(_groups(stem_channels), stem_channels),
            nn.GELU(),
            nn.MaxPool1d(2),
        )
        blocks = []
        width = stem_channels
        for out_channels in block_channels:
            blocks.append(ConvBlock(width, out_channels, block_kernel))
            width = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.token_dim = width
        self.project = nn.Sequential(
            nn.Linear(2 * width, embedding_dim),
            nn.GELU(),
        )
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        """Kaiming for the convolutions, Xavier for the linear layers.

        Written out rather than left to torch's defaults so that a run's
        initialisation is a property of this file and a documented one: with
        GELU activations, fan-out Kaiming keeps the variance of the activations
        roughly constant through eight convolutional layers.
        """
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GroupNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        if x.dim() != 3:
            raise ValueError(f"expected [batch, channels, samples], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"encoder was built for {self.in_channels} channel(s), got {x.shape[1]}. "
                "Channel order and count are part of the checkpoint's contract."
            )
        tokens = self.blocks(self.stem(x))
        pooled = torch.cat([tokens.mean(dim=2), tokens.amax(dim=2)], dim=1)
        return EncoderOutput(embedding=self.project(pooled), tokens=tokens)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def token_count(self, samples: int) -> int:
        """How many tokens an epoch of ``samples`` produces, by construction."""
        with torch.no_grad():
            probe = torch.zeros(1, self.in_channels, samples)
            return int(self.forward(probe).tokens.shape[2])
