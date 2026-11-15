"""Spike-addressed memory with delayed, gated writes and shunting reads."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import InputNeuron, SeqBN, make_neuron
from .neurons import heaviside
from .ops import OpCounter


def address_events(bits: torch.Tensor) -> torch.Tensor:
    """Conjoin complementary bit events into exactly one active address.

    Bit 0 is the low bit. The product form lets the spike surrogate pass
    gradients through the address used on reads and writes.
    """
    events = torch.ones_like(bits[..., :1])
    for bit in bits.split(1, dim=-1):
        events = torch.cat((events * (1 - bit), events * bit), dim=-1)
    return events


def prefix_read(q: torch.Tensor, k: torch.Tensor, values: torch.Tensor, block: int) -> torch.Tensor:
    """Inclusive causal read of addressed sums, without sequence-square storage.

    q/k: [..., N, buckets]; values: [..., N, channels]. For every block,
    materialize just one bucket table and accumulate those tables across
    blocks. Separately match the keys inside the current block.
    """
    n = q.shape[-2]
    width = min(n, block)
    pad = (-n) % width

    def partition(x):
        x = F.pad(x, (0, 0, 0, pad)) if pad else x
        return x.unflatten(-2, (-1, width))

    q, k, values = map(partition, (q, k, values))
    match = q @ k.transpose(-1, -2)
    visible = torch.ones((width, width), dtype=torch.bool, device=q.device).tril()
    read = match.masked_fill(~visible, 0.0) @ values
    if q.shape[-3] > 1:
        totals = k.transpose(-1, -2) @ values
        previous = F.pad(totals[..., :-1, :, :].cumsum(-3), (0, 0, 0, 0, 1, 0))
        read = read + q @ previous
    return read.flatten(-3, -2)[..., :n, :]


@dataclass
class DendriticState:
    """One layer's running sums/counts and ring buffer of recent addresses."""

    table: torch.Tensor
    history: torch.Tensor
    relay: torch.Tensor | None = None
    length: int = 0
    pending_address: torch.Tensor | None = None


class DendriticDelayAttention(nn.Module):
    """Route by spike word, then read a normalized sum at that address.

    Even heads bind values to the present address. Odd heads can bind values
    to an earlier address. A gate on the value token decides whether a write
    occurs. Read/write use the same binary code, with one table per head.
    """

    write_bias = 0.75

    def __init__(self, dim, heads, cfg, causal=False):
        super().__init__()
        if dim % heads or not 1 <= cfg.address_bits <= 8 or cfg.memory_block < 1 or cfg.memory_delay < 0:
            raise ValueError("Dendritic attention requires divisible heads, 1–8 address bits, "
                             "positive block width and nonnegative delay")
        if cfg.neuron not in ("ilif", "lif"):
            raise ValueError("Dendritic attention requires spiking value neurons")
        self.cfg, self.causal = cfg, causal
        self.heads, self.dh = heads, dim // heads
        self.buckets = 1 << cfg.address_bits
        self.max_delay = cfg.memory_delay if heads > 1 else 0
        self.register_buffer("delays", torch.arange(heads) % 2 * self.max_delay, persistent=False)
        self.register_buffer("relay_delays", 1 + torch.arange(heads) % 2, persistent=False)
        self.relay_width = min(heads, 2) if cfg.memory_relay else 0
        self.register_buffer("bit_weights", 2 ** torch.arange(cfg.address_bits), persistent=False)
        norm_opts = dict(causal=causal and cfg.causal_norm)
        self.in_sn = InputNeuron(dim, cfg, causal=causal)
        # One shared projection; split into value channels, address bits and a write gate.
        size = dim + heads * (cfg.address_bits + 1)
        self.encode = nn.Linear(dim, size, bias=False)
        self.encode_bn = SeqBN(size, **norm_opts)
        self.value_sn = make_neuron(cfg)
        self.gain = nn.Parameter(torch.tensor(2.0))
        self.out_sn = make_neuron(cfg)
        self.proj = nn.Linear(dim, dim)
        self.proj_bn = SeqBN(dim, **norm_opts)

    def _encode(self, x):
        return self._encode_spikes(self.in_sn(x))

    def _encode_spikes(self, spikes):
        v, address, gate = self.encode_bn(self.encode(spikes)).split(
            (self.heads * self.dh, self.heads * self.cfg.address_bits, self.heads), dim=-1
        )
        v = self.value_sn(v).unflatten(-1, (self.heads, self.dh)).transpose(-3, -2)
        address = address.unflatten(-1, (self.heads, self.cfg.address_bits)).transpose(-3, -2)
        bits = heaviside(address, self.cfg.sel_alpha)
        write = heaviside(gate.transpose(-1, -2).unsqueeze(-1) + self.write_bias, self.cfg.sel_alpha)
        return bits, write, v

    def _output(self, read, x, local=None):
        v = read[..., :-1] / read[..., -1:].clamp_min(1.0)
        if local is not None:
            v = v + 0.5 * local
        v = v.to(x.dtype).transpose(-3, -2).reshape(x.shape)
        return self.proj_bn(self.proj(self.out_sn(self.gain * v)))

    def _shift(self, events, delays=None):
        """Retrieve each head's events at a fixed lag, zero before the start."""
        if delays is None:
            if not self.max_delay:
                return events
            delays = self.delays
        source = torch.arange(events.shape[-2], device=events.device) - delays[:, None]
        idx = source.clamp_min(0)[..., None].expand_as(events)
        return events.gather(-2, idx) * (source >= 0).unsqueeze(-1)

    def _write_events(self, addresses, write, spikes):
        return self._shift(addresses) * write

    def forward(self, x):
        spikes = self.in_sn(x)
        bits, write, v = self._encode_spikes(spikes)
        # Keep prefix accumulators in single precision under mixed precision.
        upcast = v.dtype in (torch.bfloat16, torch.float16)
        if upcast:
            bits, write, v = bits.float(), write.float(), v.float()
        addresses = address_events(bits)
        keys = self._write_events(addresses, write, spikes)
        payload = torch.cat((v, torch.ones_like(write)), dim=-1)
        with torch.autocast(device_type=v.device.type, enabled=False) if upcast else nullcontext():
            read = (
                prefix_read(addresses, keys, payload, self.cfg.memory_block)
                if self.causal else addresses @ (keys.transpose(-1, -2) @ payload)
            )
        local = self._shift(v, self.relay_delays) if self.relay_width else None
        if (c := OpCounter.active) is not None:
            # Charge for detector decisions, writes, read accumulators and
            # divisions; address decoding and memory traffic are not priced.
            c.elementwise(bits.numel() + write.numel())
            accepted = keys.detach().sum(-1)
            c.ac += float((accepted * (1 + v.detach().sum(-1) / c.unit)).sum())
            c.elementwise(v.numel())
            if local is not None:
                c.elementwise(v.numel())
            c.nonlinear += v.numel()
        return self._output(read, x, local)

    def _route_step(self, address, write, spikes, state):
        target = address
        if self.max_delay:
            slots = (state.length - self.delays).remainder(self.max_delay)
            past = state.history.gather(-1, slots[:, None].expand(*address.shape, 1)).squeeze(-1)
            target = torch.where(self.delays > 0, past, address)
            write = write * (state.length >= self.delays).unsqueeze(-1)
            state.history[..., state.length % self.max_delay] = address
        return target, write

    @torch.no_grad()
    def step(self, x, state: DendriticState | None = None):
        """Consume one token [T,B,1,C] and return its output and mutated state.

        Every head touches one write bucket (if gated on) and one read bucket.
        The table and delay ring do not grow with the number of tokens.
        """
        if self.training or not self.causal or x.ndim != 4 or x.shape[-2] != 1:
            raise ValueError("step requires eval-mode causal attention and one token")
        spikes = self.in_sn(x)
        bits, write, v = self._encode_spikes(spikes)
        address = (bits.squeeze(-2).long() * self.bit_weights).sum(-1)
        v, write = v.squeeze(-2), write.squeeze(-2)
        if v.dtype in (torch.bfloat16, torch.float16):
            v, write = v.float(), write.float()
        if state is None:
            table = v.new_zeros((*address.shape, self.buckets, self.dh + 1))
            history = address.new_zeros((*address.shape, self.max_delay))
            relay = v.new_zeros((*address.shape, self.relay_width, self.dh)) if self.relay_width else None
            state = DendriticState(table, history, relay)
        target, write = self._route_step(address, write, spikes, state)
        payload = torch.cat((v * write, write), dim=-1).unsqueeze(-2)
        state.table.scatter_add_(-2, target[..., None, None].expand_as(payload), payload)
        read = state.table.gather(-2, address[..., None, None].expand_as(payload))
        local = None
        if state.relay is not None:
            slots = (state.length - self.relay_delays).remainder(self.relay_width)
            local = state.relay.gather(-2, slots[:, None, None].expand_as(payload[..., :-1]))
            state.relay[..., state.length % self.relay_width, :] = v
        state.length += 1
        return self._output(read, x, local), state
