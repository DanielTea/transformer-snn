"""Learned capture and commit events for spike-addressed sequence memory."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SeqBN
from .dendritic import DendriticDelayAttention
from .neurons import heaviside
from .ops import OpCounter


class _EventScan(torch.autograd.Function):
    """Hard recurrence h[t] = (1 - reset[t]) h[t-1] + update[t].

    In the forward pass, nonzero updates occur only at reset events. Gather
    the most recent update without unrolling a sequential graph. In backward,
    a state's adjoint is the sum of its downstream gradients up to the next
    reset. This is the derivative of the affine recurrence at its hard gates.
    """

    @staticmethod
    def forward(ctx, reset, update):
        n = reset.shape[-2]
        positions = torch.arange(1, n + 1, device=reset.device)[:, None]
        latest = torch.where(reset.bool(), positions, 0).cummax(-2).values
        hidden = F.pad(update, (0, 0, 1, 0)).gather(-2, latest.expand_as(update))
        ctx.save_for_backward(reset, hidden)
        return hidden

    @staticmethod
    def backward(ctx, grad):
        reset, hidden = ctx.saved_tensors
        n = reset.shape[-2]
        positions = torch.arange(n, device=reset.device)[:, None]
        first = torch.where(reset.bool(), positions, n).flip(-2).cummin(-2).values.flip(-2)
        next_reset = F.pad(first[..., 1:, :], (0, 0, 0, 1), value=n)
        suffix = F.pad(grad.flip(-2).cumsum(-2).flip(-2), (0, 0, 0, 1))
        adjoint = suffix[..., :-1, :] - suffix.gather(-2, next_reset.expand_as(grad))
        previous = F.pad(hidden[..., :-1, :], (0, 0, 1, 0))
        return -(adjoint * previous).sum(-1, keepdim=True), adjoint


class DendriticEventAttention(DendriticDelayAttention):
    """Each head can capture an address, then commit a value at a later event.

    Capture takes precedence over commit. Tokens without an event preserve
    the pending address exactly. A commit reads and clears the pending
    address. There is one pending address per head, in addition to the
    parent class's table of completed writes.
    """

    write_bias = -0.5

    def __init__(self, dim, heads, cfg, causal=False):
        if not causal:
            raise ValueError("event binding requires causal attention")
        super().__init__(dim, heads, replace(cfg, memory_delay=0), causal=True)
        self.capture = nn.Linear(dim, heads, bias=False)
        self.capture_bn = SeqBN(heads, causal=cfg.causal_norm)

    def _capture_events(self, spikes):
        logit = self.capture_bn(self.capture(spikes))
        return heaviside(logit.transpose(-1, -2).unsqueeze(-1) - 0.75, self.cfg.sel_alpha)

    def _write_events(self, addresses, write, spikes):
        capture = self._capture_events(spikes).to(addresses.dtype)
        commit = write * (1.0 - capture)
        held = _EventScan.apply(capture + commit, addresses * capture)
        prior = F.pad(held[..., :-1, :], (0, 0, 1, 0))
        if (counter := OpCounter.active) is not None:
            counter.elementwise(capture.numel())
        return prior * commit

    def _route_step(self, address, write, spikes, state):
        capture = self._capture_events(spikes).squeeze(-2).squeeze(-1).bool()
        commit = write.squeeze(-1).bool() & ~capture
        if state.pending_address is None:
            state.pending_address = torch.full_like(address, -1)
        pending = state.pending_address
        write = (commit & (pending >= 0)).to(write.dtype).unsqueeze(-1)
        target = pending.clamp_min(0)
        pending.masked_fill_(commit, -1)
        pending.copy_(torch.where(capture, address, pending))
        return target, write
