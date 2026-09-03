"""Masked reconstruction: the self-supervised stage D3's encoder starts from.

The task is deliberately plain. Cut an epoch into one-second patches, hide half
of them, and ask the model to reconstruct what was hidden from what was left.
Nothing about sleep stages enters it, and no label is read.

**The property this file exists to guarantee: a hidden value never reaches the
encoder.** Masking happens in signal space, before the encoder sees anything --
the hidden samples are replaced by a learned per-channel constant, so what the
encoder receives carries no trace of what it is being asked to predict. It is
easy to write a masked autoencoder that leaks: attenuate instead of replace,
normalise using statistics computed over the whole epoch, or hand the decoder
the original alongside its own output, and the loss falls beautifully while the
representation learns nothing. ``test_pretrain.py`` asserts the absence of that
leak directly: changing the signal *inside* the masked patches must not change
the encoder's input by a single element, nor its embedding.

**The encoder is not modified.** D3's whole claim rests on its backbone being
the same object D2 trains, so pretraining adds a decoder and a mask token beside
the encoder and touches nothing inside it. The decoder is thrown away afterwards.

**The loss is computed on masked samples only.** Scoring the visible ones would
reward copying, which is a task the model can solve without learning anything
about EEG.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from sleepstatelab.models.encoder import EpochEncoder


@dataclass(frozen=True)
class MaskedBatch:
    """What the encoder is shown, what it must predict, and where."""

    visible: torch.Tensor
    """``[batch, channels, samples]``: the signal with hidden patches replaced."""

    target: torch.Tensor
    """The original signal. Never given to the encoder."""

    masked: torch.Tensor
    """``[batch, channels, samples]`` bool, ``True`` where a sample is hidden and
    must be reconstructed."""


def patch_mask(
    n_samples: int,
    n_patches_masked: int,
    patch_samples: int,
    *,
    batch: int,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """A random patch mask, ``[batch, samples]``, ``True`` where hidden.

    Patches rather than individual samples because hiding scattered samples is a
    task solved by interpolation: the neighbours of a hidden sample are still
    there, and nothing about the signal's structure has to be learned. A hidden
    second is long enough that filling it in requires knowing what EEG does.

    A tail shorter than one patch is never masked, so every hidden region is a
    whole patch and the count of hidden samples is exact.
    """
    n_patches = n_samples // patch_samples
    if n_patches < 1:
        raise ValueError(
            f"an epoch of {n_samples} samples holds no whole patch of {patch_samples}"
        )
    if not 0 < n_patches_masked <= n_patches:
        raise ValueError(
            f"cannot mask {n_patches_masked} of {n_patches} patches"
        )
    scores = torch.rand(batch, n_patches, generator=generator, device=device)
    chosen = scores.argsort(dim=1)[:, :n_patches_masked]
    patches = torch.zeros(batch, n_patches, dtype=torch.bool, device=device)
    patches.scatter_(1, chosen, True)

    mask = torch.zeros(batch, n_samples, dtype=torch.bool, device=device)
    mask[:, : n_patches * patch_samples] = patches.repeat_interleave(
        patch_samples, dim=1
    )
    return mask


class MaskedReconstruction(nn.Module):
    """An epoch encoder, a mask token, and a light decoder over its tokens.

    The decoder is small on purpose. A powerful decoder can reconstruct from a
    weak representation, which moves the work out of the encoder -- and the
    encoder is the only part that is kept.
    """

    def __init__(
        self,
        encoder: EpochEncoder | None = None,
        *,
        n_channels: int = 2,
        n_samples: int = 3000,
        patch_samples: int = 0,
        mask_ratio: float = 0.5,
        decoder_width: int = 64,
        decoder_kernel: int = 9,
        **encoder_kwargs: object,
    ) -> None:
        super().__init__()
        self.encoder = (
            encoder
            if encoder is not None
            else EpochEncoder(in_channels=n_channels, **encoder_kwargs)  # type: ignore[arg-type]
        )
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.mask_ratio = mask_ratio

        # The patch is aligned to the encoder's own downsampling, and by default
        # derived from it: one token per patch. Alignment is not cosmetic. The
        # decoder reconstructs a patch from the token that covers it, so a patch
        # spanning two tokens would be asking one output block to answer for
        # inputs it does not see -- and, worse, a decoder that upsampled tokens
        # smoothly instead could not represent anything faster than about half a
        # hertz. Sleep spindles live at 12-16 Hz. A reconstruction target the
        # decoder physically cannot draw teaches the encoder to ignore it.
        self.n_tokens = self.encoder.token_count(n_samples)
        self.patch_samples = patch_samples or (n_samples // self.n_tokens)
        if self.patch_samples < 1:
            raise ValueError(
                f"{n_samples} samples over {self.n_tokens} tokens leaves no patch"
            )
        self.covered_samples = self.n_tokens * self.patch_samples
        """How much of the epoch the decoder can draw. Any tail beyond it is
        never masked and therefore never appears in the loss."""

        n_patches = n_samples // self.patch_samples
        self.n_patches = min(n_patches, self.n_tokens)
        self.n_patches_masked = max(1, round(self.n_patches * mask_ratio))

        self.mask_value = nn.Parameter(torch.zeros(1, n_channels, 1))
        """What a hidden sample is replaced by, one learned constant per channel.

        A constant rather than noise, and certainly rather than the original
        attenuated: the encoder must be able to see *that* a region is hidden,
        and must not be able to see anything about what was there."""

        # Each token predicts its own patch outright, for both channels: the
        # standard masked-autoencoder head. Two properties come out of it. The
        # reconstruction has full sample resolution, so a spindle is a thing the
        # decoder can draw rather than something it must smooth away. And it is
        # small -- about 20,000 parameters against the encoder's 489,000 -- so
        # the work has to happen in the part that is kept.
        self.decoder = nn.Sequential(
            nn.Conv1d(self.encoder.embedding_dim, decoder_width, 1),
            nn.GELU(),
            nn.Conv1d(decoder_width, self.patch_samples * n_channels, 1),
        )
        self.smooth = nn.Conv1d(
            n_channels, n_channels, decoder_kernel, padding=decoder_kernel // 2
        )
        """Joins the patches back up. Without it the reconstruction can step at
        every patch boundary, which is an artefact of the decoder rather than
        anything about the signal."""

    def apply_mask(
        self, x: torch.Tensor, generator: torch.Generator | None = None
    ) -> MaskedBatch:
        """Hide patches, and return what the encoder may see beside what it may not."""
        if x.dim() != 3:
            raise ValueError(f"expected [batch, channels, samples], got {tuple(x.shape)}")
        batch, channels, samples = x.shape
        if samples != self.n_samples:
            raise ValueError(
                f"this model was built for {self.n_samples} samples, got {samples}"
            )
        per_sample = patch_mask(
            samples,
            self.n_patches_masked,
            self.patch_samples,
            batch=batch,
            generator=generator,
            device=x.device,
        )
        masked = per_sample.unsqueeze(1).expand(batch, channels, samples)
        # torch.where, not multiplication: a hidden sample is *replaced*, so no
        # arithmetic trace of its value survives into the encoder's input.
        visible = torch.where(masked, self.mask_value.to(x.dtype).expand_as(x), x)
        return MaskedBatch(visible=visible, target=x, masked=masked)

    def reconstruct(self, visible: torch.Tensor) -> torch.Tensor:
        """The decoder's guess at the epoch, from the encoder's tokens.

        Returned at the epoch's full length. The tail the tokens do not cover --
        at most one patch, 24 samples of 3,000 with this package's defaults -- is
        filled with zeros, and is never masked, so it never reaches the loss.
        """
        tokens = self.encoder(visible).tokens
        blocks = self.decoder(tokens)
        batch, _, n_tokens = blocks.shape
        # [batch, patch * channels, tokens] -> [batch, channels, tokens * patch]
        drawn = (
            blocks.reshape(batch, self.n_channels, self.patch_samples, n_tokens)
            .permute(0, 1, 3, 2)
            .reshape(batch, self.n_channels, n_tokens * self.patch_samples)
        )
        drawn = self.smooth(drawn)
        if drawn.shape[-1] == self.n_samples:
            return drawn
        if drawn.shape[-1] > self.n_samples:
            return drawn[..., : self.n_samples]
        pad = torch.zeros(
            *drawn.shape[:-1],
            self.n_samples - drawn.shape[-1],
            dtype=drawn.dtype,
            device=drawn.device,
        )
        return torch.cat([drawn, pad], dim=-1)

    def forward(
        self, x: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, MaskedBatch]:
        batch = self.apply_mask(x, generator)
        return self.reconstruct(batch.visible), batch

    def loss(
        self,
        prediction: torch.Tensor,
        batch: MaskedBatch,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mean squared error over hidden samples only.

        ``valid`` optionally marks samples that are real measurements, so a
        padded or excluded region can never contribute a target. Nothing in the
        current pipeline produces one -- epochs reaching pretraining are whole
        and eligible -- and it is here because a reconstruction loss that
        silently trains on padding is a bug that produces a plausible curve.
        """
        counted = batch.masked if valid is None else (batch.masked & valid)
        total = counted.sum()
        if total == 0:
            return prediction.sum() * 0.0
        error = (prediction - batch.target) ** 2
        return (error * counted).sum() / total

    def n_parameters(self) -> dict[str, int]:
        return {
            "encoder": self.encoder.n_parameters(),
            "decoder": sum(p.numel() for p in self.decoder.parameters())
            + sum(p.numel() for p in self.smooth.parameters())
            + int(self.mask_value.numel()),
            "total": sum(p.numel() for p in self.parameters()),
        }
