from __future__ import annotations

from dataclasses import asdict, dataclass

import torch.nn as nn

from .neurons import AnalogNeuron, SpikingNeuron


@dataclass
class SNNConfig:
    """Everything that distinguishes one spiking-transformer variant from another.

    neuron:  "ilif" integer LIF (multi-level spikes, I-LIF style),
             "lif"  binary LIF (Spikformer style), "ann" non-spiking GELU reference.
    attn:    "csa" (ours), "ssa" (Spikformer), "ssa_norm" (SSA / #visible keys),
             "sdsa" (Spike-driven Transformer V1), "wta" (top-K winner-take-all,
             Zhou et al. 2026), "softmax" (softmax over spike Q/K;
             not spike-driven -- used as reference, and by the ANN model).
    mlp:     "cdglu" (ours, coincidence-gated) or "mlp" (standard 2-layer).
    """

    neuron: str = "ilif"
    levels: int = 4
    T: int = 1
    tau: float = 2.0
    v_th: float = 1.0  # range of integer neurons: they saturate (emit D spikes) at h = v_th
    attn: str = "csa"
    mlp: str = "cdglu"
    mlp_ratio: float = 4.0
    qk_levels: int | None = None  # spike levels of Q/K neurons (None -> same as levels; 1 -> binary)
    softmax_temp_init: float = 1.0  # initial inverse temperature of softmax attention
    wta_k: int = 1  # winners per query for the WTA baseline
    sel_alpha: float = 4.0  # surrogate sharpness of the attention-selection spikes
    csa_compete: bool = True  # ablation: False -> soft (score-weighted) selection
    csa_normalize: bool = True  # ablation: False -> no divisive normalisation
    csa_levels: int = 1  # spikes per key on the attention map (1 = binary selection; K>1 = graded)
    pre_bn: bool = True  # BN on the residual stream before each block's input neuron (see InputNeuron)

    @property
    def spike_levels(self) -> int:
        return {"ilif": self.levels, "lif": 1, "ann": 0, "relu": 0, "relu1": 0, "qrelu": 0}[self.neuron]

    @property
    def spike_unit(self) -> float:
        """Value of one spike of the (integer) neurons that feed weight layers."""
        return self.v_th / self.levels if self.neuron == "ilif" else 1.0

    def to_dict(self) -> dict:
        return asdict(self)


def make_neuron(cfg: SNNConfig, levels: int | None = None, v_th: float = 1.0) -> nn.Module:
    """For integer neurons ``v_th`` is relative to the configured range ``cfg.v_th``;
    binary neurons use ``v_th`` as an absolute firing threshold."""
    if cfg.neuron == "ann":
        return AnalogNeuron()
    if cfg.neuron == "relu":
        return nn.ReLU()
    if cfg.neuron == "relu1":  # diagnostic: clamp to [0, 1], no quantisation
        return nn.Hardtanh(0.0, 1.0)
    if cfg.neuron == "qrelu":  # diagnostic: quantise (step 1/levels), no upper clamp
        from .neurons import integer_spike
        lv = cfg.levels
        class _Q(nn.Module):
            def forward(self, x):
                return integer_spike(x / 64.0, lv * 64) * 64.0
        return _Q()
    lv = cfg.spike_levels if levels is None else levels
    return SpikingNeuron(levels=lv, tau=cfg.tau, v_th=v_th * cfg.v_th if lv > 1 else v_th)


class SeqBN(nn.Module):
    """BatchNorm over the channel (last) dim of a [T, B, N, C] tensor.

    BN is used instead of LayerNorm because it folds into the preceding
    linear layer at inference (keeps the network multiply-free)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        shp = x.shape
        return self.bn(x.reshape(-1, shp[-1])).reshape(shp)


class InputNeuron(nn.Module):
    """Spiking neuron that reads the real-valued residual (membrane-shortcut) stream.

    The residual stream is an unbounded running sum, while a spiking neuron
    saturates (an integer neuron emits at most D spikes).  Without
    normalisation, residual channels drift into saturation, which erases the
    information they carry and zeroes their (straight-through) gradient -- in
    our experiments this alone prevents every spiking variant from learning
    associative recall.  With ``pre_bn`` the stream is batch-normalised first.
    At inference this per-channel affine map folds into the neuron's firing
    threshold and bias, so it adds no multiplications."""

    def __init__(self, dim: int, cfg: SNNConfig) -> None:
        super().__init__()
        self.bn = SeqBN(dim) if cfg.pre_bn else nn.Identity()
        self.sn = make_neuron(cfg)

    def forward(self, x):
        return self.sn(self.bn(x))
