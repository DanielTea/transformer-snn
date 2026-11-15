"""Causal local attention with spike coincidences and discrete votes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SpikeCoincidenceState:
    keys: torch.Tensor
    values: torch.Tensor
    length: int = 0


class _CoincidenceVotes(torch.autograd.Function):
    """Use integer votes in the forward pass and a smooth training derivative."""

    @staticmethod
    def forward(ctx, gaps, table):
        ctx.save_for_backward(gaps)
        ctx.top_exponent = table.shape[0] - 2
        bins = gaps.round().clamp_(0, table.shape[0] - 1).long()
        return table[bins]

    @staticmethod
    def backward(ctx, grad):
        (gaps,) = ctx.saved_tensors
        slope = -math.log(2.0) * torch.exp2(ctx.top_exponent - gaps)
        return grad * slope, None


class SpikeCoincidence(nn.Module):
    """Pool value spikes by local query–key coincidence.

    The dot product sums signed coincidences between emitted spike counts.
    Scale that count, add per-head age bias, and round the gap from the
    best visible score. Gaps 0 through 4 receive 16, 8, 4, 2 or 1 votes.
    Larger gaps and masked keys receive zero. The readout is a normalized
    weighted sum. PyTorch simulates the counts with dense arithmetic
    inside each window, and uses the surrogate only during training.
    """

    def __init__(self, heads: int, head_dim: int, window: int) -> None:
        super().__init__()
        if min(heads, head_dim, window) < 1:
            raise ValueError("heads, head dimension and window must be positive")
        self.heads = heads
        self.head_dim = head_dim
        self.window = window
        self.scale = head_dim ** -0.5
        bias = -torch.linspace(0.0, 2.0, window)
        self.age_bias = nn.Parameter(bias.expand(heads, -1).clone())
        self.register_buffer("_votes", torch.tensor([16.0, 8.0, 4.0, 2.0, 1.0, 0.0]), persistent=False)
        offsets = torch.arange(window)
        self.register_buffer("_positions", offsets, persistent=False)
        self.register_buffer("_ages", offsets[:, None] - torch.arange(-window, window),
                             persistent=False)

    def _read(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              ages: torch.Tensor) -> torch.Tensor:
        visible = (ages >= 0) & (ages < self.window)
        bias = self.age_bias[:, ages.clamp(0, self.window - 1)].to(q.dtype)
        scores = (q @ k.transpose(-1, -2)) * self.scale + bias
        scores = scores.masked_fill(~visible, -torch.inf)
        best = scores.amax(dim=-1, keepdim=True).detach()
        votes = _CoincidenceVotes.apply(best - scores, self._votes.to(scores.dtype))
        return (votes @ v) / votes.sum(dim=-1, keepdim=True).clamp_min(1)

    def _validate(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dims: int) -> None:
        if q.ndim != dims or q.shape != k.shape or q.shape != v.shape:
            raise ValueError("q, k and v must have the same shape")
        if q.shape[1] != self.heads or q.shape[-1] != self.head_dim:
            raise ValueError("q, k and v must match the configured heads and head dimension")

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Score an entire sequence with at most two windows of keys per block."""
        self._validate(q, k, v, 4)
        batch, heads, length, width = q.shape
        if length == 0:
            return v

        window = self.window
        first_end = min(window, length)
        first = self._read(q[:, :, :first_end], k[:, :, :first_end], v[:, :, :first_end],
                           self._ages[:first_end, window : window + first_end])
        if length <= window:
            return first

        pad = (-length) % window
        blocks = (length + pad) // window - 1
        queries = F.pad(q[:, :, window:], (0, 0, 0, pad))
        queries = queries.unflatten(2, (blocks, window)).transpose(1, 2)
        queries = queries.reshape(batch * blocks, heads, window, width)

        def key_blocks(x):
            padded = F.pad(x, (0, 0, 0, pad))
            windows = padded.unfold(2, 2 * window, window).permute(0, 2, 1, 4, 3)
            return windows.reshape(batch * blocks, heads, 2 * window, width)

        rest = self._read(queries, key_blocks(k), key_blocks(v), self._ages)
        rest = rest.reshape(batch, blocks, heads, window, width).transpose(1, 2).flatten(2, 3)
        return torch.cat((first, rest), dim=2)[:, :, :length]

    @torch.no_grad()
    def step(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
             state: SpikeCoincidenceState | None = None) -> tuple[torch.Tensor, SpikeCoincidenceState]:
        """Consume one set of spikes using a fixed-size ring of keys and values."""
        self._validate(q, k, v, 3)
        shape = (q.shape[0], self.heads, self.window, self.head_dim)
        if state is None:
            state = SpikeCoincidenceState(k.new_empty(shape), v.new_empty(shape))
        elif (state.keys.shape != shape or state.values.shape != shape
              or state.keys.device != k.device or state.keys.dtype != k.dtype
              or state.values.device != v.device or state.values.dtype != v.dtype):
            raise ValueError("streaming state must keep the same batch size, window, device and dtype")

        position = state.length
        slot = position % self.window
        state.keys[:, :, slot].copy_(k)
        state.values[:, :, slot].copy_(v)
        state.length += 1
        visible = min(state.length, self.window)
        ages = (slot - self._positions[:visible]).remainder(self.window)[None, :]
        result = self._read(q.unsqueeze(2), state.keys[:, :, :visible], state.values[:, :, :visible], ages)
        return result.squeeze(2), state
