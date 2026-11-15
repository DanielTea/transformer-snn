from __future__ import annotations

import torch
import torch.nn as nn

from .config import InputNeuron, SeqBN, SNNConfig, make_neuron
from .ops import OpCounter


class SpikingMLP(nn.Module):
    """Standard spiking MLP (Meta-SpikeFormer ordering): SN -> fc1 -> BN -> SN -> fc2 -> BN."""

    def __init__(self, dim: int, cfg: SNNConfig) -> None:
        super().__init__()
        hidden = int(dim * cfg.mlp_ratio)
        self.sn1 = InputNeuron(dim, cfg)
        self.fc1 = nn.Linear(dim, hidden)
        self.bn1 = SeqBN(hidden)
        self.sn2 = make_neuron(cfg)
        self.fc2 = nn.Linear(hidden, dim)
        self.bn2 = SeqBN(dim)

    def forward(self, x):
        return self.bn2(self.fc2(self.sn2(self.bn1(self.fc1(self.sn1(x))))))


class CoincidenceGLU(nn.Module):
    r"""Coincidence-gated spiking MLP (CD-GLU) -- this work.

    h = g * u,  g = H(BN(W_g s)) binary gate spikes,  u = SN(BN(W_u s)) (multi-level)

    A spiking analogue of SwiGLU: a hidden unit transmits only when its value
    neuron and its gate neuron fire together (dendritic coincidence detection).
    Because g is binary, ``g * u`` is pure masking -- no multiplications -- and
    the hidden firing rate (hence fc_o's synaptic ops) is the *product* of the
    two rates.  Hidden width is 2/3 of the plain MLP's so parameters match.
    """

    def __init__(self, dim: int, cfg: SNNConfig) -> None:
        super().__init__()
        hidden = int(round(dim * cfg.mlp_ratio * 2 / 3))
        self.hidden = hidden
        self.sn_in = InputNeuron(dim, cfg)
        self.fc_gu = nn.Linear(dim, 2 * hidden)
        self.bn_g, self.bn_u = SeqBN(hidden), SeqBN(hidden)
        # gate: binary spikes (threshold 0 -> ~50% open at init); ANN uses a sigmoid gate
        self.sn_g = make_neuron(cfg, levels=1, v_th=0.0) if cfg.neuron in ("ilif", "lif") else nn.Sigmoid()
        self.sn_u = make_neuron(cfg)
        self.fc_o = nn.Linear(hidden, dim)
        self.bn_o = SeqBN(dim)

    def forward(self, x):
        g, u = self.fc_gu(self.sn_in(x)).chunk(2, dim=-1)
        g = self.sn_g(self.bn_g(g))
        u = self.sn_u(self.bn_u(u))
        if (c := OpCounter.active) is not None:
            c.elementwise(float(g.sum()))  # masking: one gated transfer per open gate
        return self.bn_o(self.fc_o(g * u))


MLPS = {"mlp": SpikingMLP, "cdglu": CoincidenceGLU}


def make_mlp(dim: int, cfg: SNNConfig) -> nn.Module:
    return MLPS[cfg.mlp](dim, cfg)
