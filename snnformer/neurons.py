"""Spiking neurons.

Every neuron here takes a tensor shaped ``[T, ...]`` (T = number of time steps)
and returns a tensor of the same shape whose values are *spike counts
normalised to [0, 1]*:

* ``levels == 1``  -> binary spikes {0, 1} (classic LIF, sigmoid surrogate).
* ``levels == D``  -> integer spikes {0, 1, ..., D} x (v_th / D) (I-LIF style
  "integer-valued training, spike-driven inference"): at inference an output
  of k/D is emitted as k binary spikes over D virtual sub-steps, so the next
  (linear) layer only ever performs accumulates.

Integer neurons are parameterised by their *range* ``v_th`` (the membrane
value at which they saturate, emitting D spikes); each spike is worth
``v_th / D`` and the output is returned in these physical units, so changing
the range does not rescale downstream attention scores.  At inference the
spike value is folded into the next layer's weights (still accumulate-only).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _HeavisideSigmoid(torch.autograd.Function):
    """H(x) forward, derivative of sigmoid(alpha * x) backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (x,) = ctx.saved_tensors
        sg = torch.sigmoid(ctx.alpha * x)
        return grad * ctx.alpha * sg * (1.0 - sg), None


def heaviside(x: torch.Tensor, alpha: float = 4.0) -> torch.Tensor:
    """Binary spike with a sigmoid surrogate gradient."""
    return _HeavisideSigmoid.apply(x, alpha)


class _IntegerSpike(torch.autograd.Function):
    """clamp(round(x * D), 0, D) / D forward, straight-through backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, levels: int) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.levels = levels
        return torch.clamp(torch.round(x * levels), 0, levels) / levels

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (x,) = ctx.saved_tensors
        half = 0.5 / ctx.levels
        inside = (x > -half) & (x < 1.0 + half)
        return grad * inside.to(grad.dtype), None


def integer_spike(x: torch.Tensor, levels: int) -> torch.Tensor:
    return _IntegerSpike.apply(x, levels)


class SpikingNeuron(nn.Module):
    """Leaky integrate-and-fire neuron with binary or integer spikes.

    Membrane update (per time step): ``h_t = beta * u_{t-1} + x_t`` with
    ``beta = 1 - 1/tau``.  Binary neurons fire when ``h_t >= v_th`` and use a
    hard reset (as in Spikformer); integer neurons emit
    ``clamp(round(h_t * D / v_th), 0, D) * v_th / D`` and use a soft reset.
    ``unit`` is the value of one spike (used for spike counting).
    """

    def __init__(
        self,
        levels: int = 4,
        tau: float = 2.0,
        v_th: float = 1.0,
        alpha: float = 4.0,
        detach_reset: bool = True,
    ) -> None:
        super().__init__()
        self.levels = int(levels)
        self.beta = 1.0 - 1.0 / tau
        self.v_th = v_th
        self.alpha = alpha
        self.detach_reset = detach_reset
        self.unit = 1.0 if self.levels == 1 else v_th / self.levels

    def _fire(self, h: torch.Tensor) -> torch.Tensor:
        if self.levels == 1:
            return heaviside(h - self.v_th, self.alpha)
        return integer_spike(h / self.v_th, self.levels) * self.v_th

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        if T == 1:  # single step: no membrane history
            return self._fire(x)
        u = torch.zeros_like(x[0])
        out = []
        for t in range(T):
            h = self.beta * u + x[t]
            s = self._fire(h)
            r = s.detach() if self.detach_reset else s
            if self.levels == 1:
                u = h * (1.0 - r)  # hard reset to 0
            else:
                u = h - r  # soft reset (subtract emitted charge)
            out.append(s)
        return torch.stack(out)

    def extra_repr(self) -> str:
        return f"levels={self.levels}, beta={self.beta:.3f}, v_th={self.v_th}"


class AnalogNeuron(nn.Module):
    """Non-spiking stand-in (GELU) used to build the ANN reference model."""

    levels = 0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)
