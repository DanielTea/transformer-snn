"""Nondestructive phase probes that transmit the time of a threshold crossing."""

from __future__ import annotations

import torch
from torch import nn

from .neurons import heaviside


PHASE_THRESHOLD = 256


class PhaseRead(nn.Module):
    """Distribute one threshold of drive over two or four ordered subpulses.

    The phase is in [0, 255]. A probe emits exactly one spike and preserves
    the phase. The legacy two-pulse read exposes its first event only. The
    four-pulse read exposes all four temporal channels with one active event.
    """

    def __init__(self, read_bins: int, fixed_cuts: bool = False) -> None:
        super().__init__()
        if read_bins not in (2, 4):
            raise ValueError("read_bins must be 2 or 4")
        self.read_bins = read_bins
        self.fixed_cuts = fixed_cuts
        self.channels = 1 if read_bins == 2 else read_bins

    def cuts(self, cut_logits: torch.Tensor) -> torch.Tensor:
        """Map logits to ordered integer cumulative drive amounts over [0, 256]."""
        if cut_logits.shape[-1] != self.read_bins - 1:
            raise ValueError("one logit is required for each cut between pulses")
        if self.fixed_cuts:
            stride = PHASE_THRESHOLD // self.read_bins
            return cut_logits.new_tensor(range(stride, PHASE_THRESHOLD, stride)).expand_as(cut_logits)

        drive = cut_logits.sigmoid() * PHASE_THRESHOLD
        if self.read_bins == 4:
            drive = drive.sort(dim=-1).values
        return drive.detach().round() + (drive - drive.detach())

    def pulses(self, cuts: torch.Tensor) -> torch.Tensor:
        """Return the successive nonnegative drive increments for cumulative cuts."""
        start = torch.zeros_like(cuts[..., :1])
        finish = torch.full_like(start, PHASE_THRESHOLD)
        return torch.cat((start, cuts, finish), dim=-1).diff(dim=-1)

    def forward(self, phase: torch.Tensor, cuts: torch.Tensor, probe_gate: torch.Tensor) -> torch.Tensor:
        """Compute the equivalent timing code using threshold surrogates."""
        margin = (phase.unsqueeze(-1) + cuts - PHASE_THRESHOLD) / PHASE_THRESHOLD
        crossed = heaviside(margin, alpha=8.0)
        if self.read_bins == 2:
            events = crossed
        else:
            start = torch.zeros_like(crossed[..., :1])
            finish = torch.ones_like(start)
            events = torch.cat((start, crossed, finish), dim=-1).diff(dim=-1)
        return events * probe_gate.unsqueeze(-1)

    @torch.no_grad()
    def step(self, phase: torch.Tensor, cuts: torch.Tensor, probe_gate: torch.Tensor):
        """Execute the pulse sequence with subtractive reset and return timing plus phase."""
        timed = []
        for pulse in self.pulses(cuts).unbind(-1):
            charged = phase + pulse * probe_gate
            spike = (charged >= PHASE_THRESHOLD).to(phase.dtype)
            phase = charged - PHASE_THRESHOLD * spike
            timed.append(spike)
        return torch.stack(timed[:self.channels], dim=-1), phase
