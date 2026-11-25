from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .neurons import AnalogNeuron, SpikingNeuron


@dataclass
class SNNConfig:
    """Everything that distinguishes one spiking-transformer variant from another.

    neuron:  "ilif" integer LIF (multi-level spikes, I-LIF style),
             "lif"  binary LIF (Spikformer style), "ann" non-spiking GELU reference.
    attn:    "csa" (ours), "ssa" (Spikformer), "ssa_norm" (SSA / #visible keys),
             "sdsa" (Spike-driven Transformer V1), "wta" (top-K winner-take-all,
             Zhou et al. 2026), "dendritic" (fixed-delay spike-addressed memory),
             "dendritic_event" (spike-addressed memory with learned capture and commit),
             "temporal_spike" (token-time integrate-and-fire and local spike coincidence),
             "softmax" (softmax over spike Q/K;
             not spike-driven -- used as reference, and by the ANN model).
    mlp:     "cdglu" (ours, coincidence-gated) or "mlp" (standard 2-layer).
    """

    neuron: str = "ilif"
    levels: int = 4
    T: int = 1
    tau: float = 2.0
    v_th: float = 1.0  # range of integer neurons: they saturate (emit D spikes) at h = v_th
    attn: str = "csa"
    position_embeddings: bool = True
    address_bits: int = 4  # complementary binary detectors per dendritic memory head
    memory_block: int = 32  # block size for the parallel prefix implementation
    memory_delay: int = 1  # odd heads bind current values to addresses this many tokens back
    memory_relay: bool = False  # feed the output neuron one short-range value tap per head
    local_window: int = 32  # token window in the local and hybrid Transformer references
    temporal_carry: bool = True  # retain integrate-and-fire charge across tokens
    pulse_memory_cells: int = 0  # optional paired-pulse bank in the last temporal block
    pulse_preserve_reads: bool = True  # False clears each probed cell after reading it
    pulse_write_holds: tuple[int, ...] = ()  # minimum accepted-write separation per cell
    pulse_read_bins: int = 2  # the legacy first-pulse bit, or four timing channels
    pulse_fixed_cuts: bool = False  # equal-width timing bins for the timing control
    pulse_global_cuts: bool = False  # learned, query-independent timing cuts for each cell
    charge_memory: str = "none"  # "conserved" transfers or "independent" writes
    event_width: int | None = None  # optional bottleneck for the hybrid's event branch
    successor_orders: tuple[int, ...] = (2, 3, 4, 5, 7)
    successor_slots: int = 2048  # tagged successor entries per order
    successor_spikes: bool = True  # quantize the cache's predictive mass
    successor_scaled_votes: bool = False  # smaller spike weights for shorter suffix matches
    mlp: str = "cdglu"
    mlp_ratio: float = 4.0
    # The reference Transformer's units: gated "swiglu", and as controls at the same number of weights "reglu" (gated,
    # open only above zero), "gelu" (plain and smooth) and "relu" (plain with a threshold). Further controls of the
    # gate: "rampglu", "shuntglu" and "sigglu" are bounded gates, "bilinear" is the product of the two drives.
    feed_forward: str = "swiglu"
    age_offsets: float = 0.0  # above 0 the reference addresses by an offset per head and distance at this rate, without RoPE
    untied_output: bool = False  # the reference's output layer has weights of its own, not the embedding's
    # The reference's normalization: "layer" (LayerNorm), or as a control "pool": the stream divided by its mean
    # absolute activity and a learned rest, with a gain per channel and no centring, as a stack's inhibitory pool
    # raises every threshold with the stream's activity.
    norm: str = "layer"
    qk_levels: int | None = None  # spike levels of Q/K neurons (None -> same as levels; 1 -> binary)
    softmax_temp_init: float = 1.0  # initial inverse temperature of softmax attention
    wta_k: int = 1  # winners per query for the WTA baseline
    sel_alpha: float = 4.0  # surrogate sharpness of the attention-selection spikes
    csa_compete: bool = True  # ablation: False -> soft (score-weighted) selection
    csa_normalize: bool = True  # ablation: False -> no divisive normalisation
    csa_levels: int = 1  # spikes per key on the attention map (1 = binary selection; K>1 = graded)
    pre_bn: bool = True  # BN on the residual stream before each block's input neuron (see InputNeuron)
    causal_norm: bool = False  # in causal models, normalize with running stats from prior batches

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
    linear layer at inference (keeps the network multiply-free).

    In causal training, using current batch statistics would let a token
    depend on future tokens. Normalize with statistics from completed
    batches, then update them for the next training forward.
    """

    def __init__(self, dim: int, *, causal: bool = False) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.causal = causal

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        if not (self.causal and self.bn.training):
            return self.bn(flat).reshape_as(x)

        bn = self.bn
        # Backpropagation needs the original statistics even after the running
        # buffers have been updated.
        normalized = F.batch_norm(
            flat, bn.running_mean.clone(), bn.running_var.clone(),
            bn.weight, bn.bias, training=False, eps=bn.eps,
        )
        if flat.shape[0] > 1:
            with torch.no_grad():
                var, mean = torch.var_mean(flat.detach().to(bn.running_mean.dtype), dim=0, correction=1)
                bn.num_batches_tracked.add_(1)
                momentum = bn.momentum if bn.momentum is not None else 1.0 / bn.num_batches_tracked.item()
                bn.running_mean.lerp_(mean, momentum)
                bn.running_var.lerp_(var, momentum)
        return normalized.reshape_as(x)


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

    def __init__(self, dim: int, cfg: SNNConfig, *, causal: bool = False) -> None:
        super().__init__()
        self.bn = SeqBN(dim, causal=causal and cfg.causal_norm) if cfg.pre_bn else nn.Identity()
        self.sn = make_neuron(cfg)

    def forward(self, x):
        return self.sn(self.bn(x))
