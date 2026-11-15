"""A causal language model whose hidden synapses communicate in spike counts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from .config import SNNConfig
from .spike_attention import SpikeCoincidence, SpikeCoincidenceState
from .token_if import TokenIF

if TYPE_CHECKING:
    from .charge_memory import ChargeTransferState
    from .pulse_memory import PulseMemoryState


class SpikeSynapse(nn.Linear):
    """Dyadic synaptic weights keep sums of FP32 spike contributions exact."""

    def __init__(self, inputs: int, outputs: int, cap: int) -> None:
        super().__init__(inputs, outputs, bias=False)
        # Weights are bounded by 1. Leave mantissa space for the residual sum.
        max_scale = (2 ** 22) / (inputs * cap)
        self.scale = 2. ** min(12, math.floor(math.log2(max_scale)))
        if self.scale < 1:
            raise ValueError("spike count cap and synapse width exceed the FP32 accumulator range")

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        bounded = self.weight.clamp(-1, 1)
        quantized = (bounded * self.scale).round().detach() / self.scale
        return F.linear(spikes, quantized + (bounded - bounded.detach()))


class SpikingPopulation(nn.Module):
    """Integrate-and-fire populations with an optional negative rail."""

    def __init__(self, dim: int, cfg: SNNConfig, gain: float, *, signed: bool = True) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.neurons = TokenIF(cfg.spike_levels, cfg.temporal_carry)
        self.gain = gain
        self.signed = signed

    def _currents(self, current: torch.Tensor) -> torch.Tensor:
        drive = self.norm(current) * self.gain
        return torch.cat((drive.relu(), (-drive).relu()), dim=-1) if self.signed else drive.relu()

    def _decode(self, counts: torch.Tensor) -> torch.Tensor:
        if not self.signed:
            return counts
        positive, negative = counts.chunk(2, dim=-1)
        return positive - negative

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        return self._decode(self.neurons(self._currents(current)))

    def step(self, current: torch.Tensor, phase: torch.Tensor | None):
        counts, phase = self.neurons.step(self._currents(current), phase)
        return self._decode(counts), phase


@dataclass
class _TemporalBlockState:
    qkv_phase: torch.Tensor | None = None
    read_phase: torch.Tensor | None = None
    attention_residual_phase: torch.Tensor | None = None
    hidden_phase: torch.Tensor | None = None
    output_phase: torch.Tensor | None = None
    attention: SpikeCoincidenceState | None = None
    delay_history: torch.Tensor | None = None
    memory_phase: torch.Tensor | PulseMemoryState | ChargeTransferState | None = None


class _TemporalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int, cfg: SNNConfig, *, with_memory: bool = False) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        # Three groups read older spikes. The remaining channels stay at zero delay.
        lag_width = dim // 6
        self.delay_widths = (dim - 3 * lag_width, lag_width, lag_width, lag_width)
        cap = cfg.spike_levels
        pulse_scale = max(1., cap / 8)
        gain = min(2. * pulse_scale, cap / 2)
        self.qkv = SpikeSynapse(dim, 3 * dim, cap)
        self.qkv_emit = SpikingPopulation(3 * dim, cfg, gain=min(1.25 * pulse_scale, cap / 2))
        self.attention = SpikeCoincidence(heads, self.head_dim, cfg.local_window)
        self.attention.scale /= pulse_scale ** 2
        self.read_emit = SpikingPopulation(dim, cfg, gain)
        self.attention_out = SpikeSynapse(dim, dim, cap)
        self.attention_residual = SpikingPopulation(dim, cfg, gain)
        self.expand = SpikeSynapse(dim, hidden, cap)
        self.hidden_emit = SpikingPopulation(hidden, cfg, gain, signed=False)
        self.contract = SpikeSynapse(hidden, dim, cap)
        self.output_emit = SpikingPopulation(dim, cfg, gain)
        self.memory = None
        if with_memory:
            cells = cfg.pulse_memory_cells
            if cfg.charge_memory != "none":
                from .charge_memory import ChargeTransferMemory

                self.memory = ChargeTransferMemory(dim, cells, conserve=cfg.charge_memory == "conserved",
                                                   spike_cap=cap)
            else:
                from .pulse_memory import PulseMemory

                self.memory = PulseMemory(dim, cells, carry_reads=cfg.pulse_preserve_reads, spike_cap=cap,
                                          write_holds=cfg.pulse_write_holds or None, read_bins=cfg.pulse_read_bins,
                                          fixed_cuts=cfg.pulse_fixed_cuts, global_cuts=cfg.pulse_global_cuts)
            self.memory_out = SpikeSynapse(self.memory.read_width, dim, cap=1)
            nn.init.normal_(self.memory_out.weight, std=0.025)

    def _split_heads(self, counts: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(group.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)
                     for group in counts.chunk(3, dim=-1))

    def forward(self, spikes: torch.Tensor, *, memory_options=None) -> torch.Tensor:
        groups = spikes.split(self.delay_widths, dim=-1)
        delayed = torch.cat([groups[0]] + [F.pad(group, (0, 0, lag, 0))[:, :spikes.shape[1]]
                                         for lag, group in enumerate(groups[1:], start=1)], dim=-1)
        q, k, v = self._split_heads(self.qkv_emit(self.qkv(delayed)))
        current = self.attention(q, k, v).transpose(1, 2).flatten(-2)
        read = self.read_emit(current)
        spikes = self.attention_residual(spikes + self.attention_out(read))
        hidden = self.hidden_emit(self.expand(spikes))
        current = spikes + self.contract(hidden)
        if self.memory is not None:
            memory_read = self.memory(spikes, **memory_options)
            trace = memory_options.get("trace")
            if trace is not None:
                trace.update(spike_inputs=spikes.detach(), pre_memory_current=current.detach())
            current = current + self.memory_out(memory_read)
        return self.output_emit(current)

    def step(self, spikes: torch.Tensor, state: _TemporalBlockState, position: int) -> torch.Tensor:
        count = len(self.delay_widths) - 1
        if state.delay_history is None:
            state.delay_history = spikes.new_zeros(spikes.shape[0], count, spikes.shape[1])
        history = state.delay_history
        groups = [spikes[:, :self.delay_widths[0]]]
        start = self.delay_widths[0]
        for lag, width in enumerate(self.delay_widths[1:], start=1):
            groups.append(history[:, (position - lag) % count, start:start + width])
            start += width
        delayed = torch.cat(groups, dim=-1)
        history[:, position % count].copy_(spikes)
        qkv, state.qkv_phase = self.qkv_emit.step(self.qkv(delayed), state.qkv_phase)
        q, k, v = self._split_heads(qkv.unsqueeze(1))
        current, state.attention = self.attention.step(q.squeeze(2), k.squeeze(2), v.squeeze(2), state.attention)
        read, state.read_phase = self.read_emit.step(current.flatten(1), state.read_phase)
        spikes, state.attention_residual_phase = self.attention_residual.step(
            spikes + self.attention_out(read), state.attention_residual_phase)
        hidden, state.hidden_phase = self.hidden_emit.step(self.expand(spikes), state.hidden_phase)
        current = spikes + self.contract(hidden)
        if self.memory is not None:
            memory_read, state.memory_phase = self.memory.step(spikes, state.memory_phase)
            current = current + self.memory_out(memory_read)
        spikes, state.output_phase = self.output_emit.step(current, state.output_phase)
        return spikes


@dataclass
class TemporalLMState:
    layers: list[_TemporalBlockState]
    input_phase: torch.Tensor | None = None
    length: int = 0


class TemporalSpikingLM(nn.Module):
    """Decoder with byte-time charge, delay synapses and local spike coincidence.

    Residual and Q/K/V connections carry signed counts from two populations.
    The inside of the feed-forward path uses rectifying spike neurons.
    Synapses form continuous currents within the receiving cells. The final
    readout produces continuous vocabulary logits from the last spike counts.
    """

    def __init__(self, cfg: SNNConfig, vocab=256, dim=128, depth=4, heads=4, ctx=128) -> None:
        super().__init__()
        if min(vocab, dim, depth, heads, ctx) < 1 or dim % heads or cfg.mlp_ratio <= 0:
            raise ValueError("model sizes must be positive and dim must be divisible by heads")
        if cfg.neuron not in ("lif", "ilif") or cfg.spike_levels < 1 or cfg.T != 1:
            raise ValueError("temporal_spike uses integrate-and-fire with one charge update per token")
        if cfg.position_embeddings or not cfg.causal_norm or not cfg.pre_bn:
            raise ValueError("temporal_spike uses relative age and causal synaptic normalization")
        if cfg.pulse_memory_cells < 0:
            raise ValueError("pulse_memory_cells cannot be negative")
        if cfg.charge_memory not in ("none", "conserved", "independent"):
            raise ValueError("charge_memory must be none, conserved or independent")
        if cfg.charge_memory != "none" and (cfg.pulse_memory_cells < 2 or cfg.pulse_memory_cells % 2
                                            or cfg.pulse_read_bins != 4 or cfg.pulse_write_holds
                                            or cfg.pulse_fixed_cuts or cfg.pulse_global_cuts
                                            or not cfg.pulse_preserve_reads):
            raise ValueError("charge memory needs an even cell count and query-dependent four-pulse reads")
        if cfg.pulse_write_holds and len(cfg.pulse_write_holds) != cfg.pulse_memory_cells:
            raise ValueError("provide one write hold duration per pulse memory cell")
        if cfg.pulse_read_bins not in (2, 4):
            raise ValueError("pulse memory supports two or four read pulses")
        if (cfg.pulse_fixed_cuts or cfg.pulse_global_cuts) and cfg.pulse_read_bins != 4:
            raise ValueError("fixed or global cuts require four read pulses")
        if cfg.pulse_fixed_cuts and cfg.pulse_global_cuts:
            raise ValueError("fixed and global cuts are mutually exclusive")
        cap = cfg.spike_levels
        self.cfg = cfg
        self.ctx = ctx
        hidden = max(1, round(cfg.mlp_ratio * dim))
        self.embedding = nn.Embedding(vocab, dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.input_emit = SpikingPopulation(dim, cfg, min(2. * max(1., cap / 8), cap / 2))
        self.blocks = nn.ModuleList(
            _TemporalBlock(dim, heads, hidden, cfg, with_memory=cfg.pulse_memory_cells > 0 and i == depth - 1)
            for i in range(depth)
        )
        self.head = SpikeSynapse(dim, vocab, cap)
        self.head.weight = self.embedding.weight
        self.logit_scale = 1. / max(1., cap / 8)

    def forward(self, tokens: torch.Tensor, *, memory_clear_at=None, memory_stats=None,
                memory_training_stats=None, memory_ablate_min_age: int | None = None,
                memory_trace=None) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("forward expects a nonempty sequence of shape [batch, tokens]")
        memory_options = {name: value for name, value in (
            ("clear", memory_clear_at), ("stats", memory_stats), ("training_stats", memory_training_stats),
            ("ablate_min_age", memory_ablate_min_age), ("trace", memory_trace),
        ) if value is not None}
        if not self.cfg.pulse_memory_cells and memory_options:
            raise ValueError("paired-pulse diagnostics require a memory bank")
        if self.cfg.charge_memory != "none" and (memory_training_stats is not None or memory_trace is not None):
            raise ValueError("write-credit diagnostics require an overwrite memory bank")
        spikes = self.input_emit(self.embedding(tokens))
        for block in self.blocks:
            spikes = block(spikes, memory_options=memory_options)
        return self.head(spikes) * self.logit_scale

    def replay_memory_read(self, pre_memory_current: torch.Tensor, read: torch.Tensor) -> torch.Tensor:
        """Recompute logits for a captured current and alternative last-block read."""
        block = self.blocks[-1]
        if block.memory is None or self.cfg.temporal_carry or self.training:
            raise ValueError("memory replay requires an eval-mode decoder with token-local output neurons")
        current = pre_memory_current + block.memory_out(read)
        return self.head(block.output_emit(current)) * self.logit_scale

    @torch.no_grad()
    def step(self, tokens: torch.Tensor, state: TemporalLMState | None = None):
        """Decode one token per stream in eval mode and update bounded state in place."""
        if self.training or tokens.ndim != 1:
            raise ValueError("step requires eval mode and tokens of shape [batch]")
        current = self.embedding(tokens)
        if state is None:
            state = TemporalLMState([_TemporalBlockState() for _ in self.blocks])
        elif (len(state.layers) != len(self.blocks) or state.input_phase is None
              or state.input_phase.shape != (current.shape[0], 2 * current.shape[1])
              or state.input_phase.device != current.device):
            raise ValueError("streaming state must match the model, batch size and device")
        spikes, state.input_phase = self.input_emit.step(current, state.input_phase)
        for block, layer in zip(self.blocks, state.layers):
            spikes = block.step(spikes, layer, state.length)
        state.length += 1
        return self.head(spikes) * self.logit_scale, state
