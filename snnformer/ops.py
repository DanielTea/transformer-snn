"""Synaptic-operation counting and energy estimation.

Follows the convention of the spiking-transformer literature (Spikformer,
Spike-driven Transformer, QKFormer, ...): a layer that receives real values
costs MACs, a layer that receives spikes costs one accumulate (AC) per spike
per fan-out synapse, with 45 nm CMOS figures E_MAC = 4.6 pJ, E_AC = 0.9 pJ
(Horowitz, ISSCC 2014). An integer spike output of k units is counted as k binary
spikes (emitted as k spikes over D virtual sub-steps at inference).

These are *estimates* of the kind reported in papers, not measurements.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

E_MAC_PJ = 4.6
E_AC_PJ = 0.9


class OpCounter:
    active: "OpCounter | None" = None

    def __init__(self, levels: int, unit: float = 1.0) -> None:
        self.levels = levels  # 0 => analog network
        self.unit = unit  # value of one spike
        self.mac = 0.0
        self.ac = 0.0
        self.nonlinear = 0.0  # exp/div ops (softmax) that are not spike-driven
        self.spikes = 0.0
        self.neurons = 0.0
        self.samples = 0

    # -- helpers used by attention modules --------------------------------
    def spike_count(self, x: torch.Tensor, unit: float | None = None) -> torch.Tensor:
        return x.sum() / (unit or self.unit)

    def spike_matmul(self, x: torch.Tensor, y: torch.Tensor, unit: float | None = None) -> None:
        """x @ y where x carries spikes; each spike adds the non-zero entries of a row of y."""
        if self.levels == 0:
            self.mac += x.shape[-2] * x.shape[-1] * y.shape[-1] * x[..., 0, 0].numel()
            return
        # sum_i sum_c x_ic * D * nnz(y_c,:)  ==  sum_c (sum_i x_ic) * D * nnz(y_c,:)
        nz_per_row = (y != 0).to(x.dtype).sum(-1)  # [..., d]
        self.ac += float((x.sum(-2) * nz_per_row).sum() / (unit or self.unit))

    def elementwise(self, n: float, real: bool = False) -> None:
        if real or self.levels == 0:
            self.mac += n
        else:
            self.ac += n

    def energy_mj(self) -> float:
        return (self.mac * E_MAC_PJ + self.ac * E_AC_PJ) * 1e-9 / max(self.samples, 1)

    def summary(self) -> dict:
        s = max(self.samples, 1)
        return {
            "mac_per_sample": self.mac / s,
            "ac_per_sample": self.ac / s,
            "nonlinear_per_sample": self.nonlinear / s,
            "energy_mJ_per_sample": self.energy_mj(),
            "firing_rate": self.spikes / max(self.neurons, 1),
        }


def _linear_hook(mod, inp, out):
    c = OpCounter.active
    if c is None:
        return
    x = inp[0]
    real = getattr(mod, "real_input", False) or c.levels == 0
    if isinstance(mod, nn.Linear):
        fan_out = mod.out_features
        if real:
            c.mac += x.numel() * fan_out
        else:
            c.ac += float(c.spike_count(x)) * fan_out
    elif isinstance(mod, nn.Conv2d):
        k = mod.kernel_size[0] * mod.kernel_size[1]
        if real:
            c.mac += out.numel() * (mod.in_channels // mod.groups) * k
        else:
            # each input spike reaches (out_channels/groups) * k^2 synapses (/stride^2 on average)
            stride = mod.stride[0] * mod.stride[1]
            c.ac += float(c.spike_count(x)) * (mod.out_channels // mod.groups) * k / stride


def _neuron_hook(mod, inp, out):
    c = OpCounter.active
    if c is None or c.levels == 0:
        return
    # firing rate = emitted binary spikes / (neurons x virtual sub-steps)
    c.spikes += float(out.sum() / mod.unit)
    c.neurons += out.numel() * mod.levels


@contextmanager
def count_ops(model: nn.Module, levels: int, unit: float = 1.0):
    from .neurons import SpikingNeuron

    counter = OpCounter(levels, unit)
    handles = []
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            handles.append(m.register_forward_hook(_linear_hook))
        elif isinstance(m, SpikingNeuron):
            handles.append(m.register_forward_hook(_neuron_hook))
    prev, OpCounter.active = OpCounter.active, counter
    try:
        yield counter
    finally:
        OpCounter.active = prev
        for h in handles:
            h.remove()
