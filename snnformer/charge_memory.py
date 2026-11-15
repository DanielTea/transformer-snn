"""Hierarchical spiking memory with bounded charge transfer within leaf pairs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .dendritic_event import _EventScan
from .neurons import heaviside
from .pulse_memory import _latest_before, _write_eligibility
from .pulse_timing import PHASE_THRESHOLD, PhaseRead
from .temporal_snn import SpikeSynapse


_MAX_CHARGE = PHASE_THRESHOLD - 1
_NEUTRAL_CHARGE = PHASE_THRESHOLD // 2
_FAST_DRIVE = 32
_HOLD_LENGTHS = (64, 128, 256, 512)


@dataclass
class ChargeTransferState:
    left: torch.Tensor
    right: torch.Tensor
    cooldown: torch.Tensor


def _round_charge(value: torch.Tensor) -> torch.Tensor:
    return value.detach().round() + (value - value.detach())


def _clip(value: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return torch.clamp(value, min=lower, max=upper)


def _scan_clipped_maps(shift: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Evaluate every prefix of f(x) = clip(x + shift, lower, upper)."""
    width = shift.shape[1]
    stride = 1
    while stride < width:
        next_shift = shift[:, stride:]
        next_lower, next_upper = lower[:, stride:], upper[:, stride:]
        lower = torch.cat((lower[:, :stride], _clip(lower[:, :-stride] + next_shift, next_lower, next_upper)), dim=1)
        upper = torch.cat((upper[:, :stride], _clip(upper[:, :-stride] + next_shift, next_lower, next_upper)), dim=1)
        shift = torch.cat((shift[:, :stride], shift[:, :-stride] + next_shift), dim=1)
        stride *= 2
    return _clip(_NEUTRAL_CHARGE + shift, lower, upper)


class ChargeTransferMemory(nn.Module):
    """Read a pair's total and imbalance through separate four-pulse probes.

    The leaves start at 128. A coarse commit sets both to the same learned
    charge and starts a hold on further commits. Fine updates remain active
    during that hold. The conservative update moves charge between leaves
    in integer ticks, constrained by both leaf capacities. The control uses
    independently bounded leaf updates. Reads precede updates, and a coarse
    commit supersedes that token's fine update. A clear resets charge and
    releases the hold before the current token is read or updated.
    """

    def __init__(self, dim: int, cells: int, *, conserve: bool = True, spike_cap: int = 32) -> None:
        super().__init__()
        if dim < 1 or cells < 2 or cells % 2 or spike_cap < 1:
            raise ValueError("dim and spike_cap must be positive, and cells must be positive and even")
        self.cells = cells
        self.pairs = cells // 2
        self.conserve = conserve
        self.read_width = 4 * cells
        self.project = SpikeSynapse(dim, 12 * self.pairs, spike_cap)
        self.timing = PhaseRead(4)
        biases = torch.zeros(12)
        biases[[0, 4, 8]] = -0.5
        biases[5:8] = biases[9:12] = torch.tensor([0.25, 0.5, 0.75]).logit()
        self.bias = nn.Parameter(biases[:, None].expand(-1, self.pairs).clone())
        self.write_holds = tuple(_HOLD_LENGTHS[pair % len(_HOLD_LENGTHS)] for pair in range(self.pairs))
        self.register_buffer("_write_holds", torch.tensor(self.write_holds), persistent=False)

    def _events(self, spikes: torch.Tensor):
        drives = self.project(spikes).unflatten(-1, (12, self.pairs)) + self.bias
        commit, charge, left, right = drives[..., :4, :].unbind(-2)
        coarse, fine = drives[..., 4:8, :], drives[..., 8:12, :]
        gates = heaviside(torch.stack((coarse[..., 0, :], fine[..., 0, :]), dim=-1))
        cuts = self.timing.cuts(torch.stack((coarse[..., 1:, :], fine[..., 1:, :]), dim=-1).movedim(-3, -1))
        # The terminal axes of both tensors are [pair, view, optional cut].
        charge = _round_charge(64 + 127 * charge.sigmoid())
        left = _round_charge(_FAST_DRIVE * left.tanh())
        right = _round_charge(_FAST_DRIVE * right.tanh())
        if self.conserve:
            left = _round_charge((left - right) / 2)
            right = -left
        return heaviside(commit), charge, left, right, gates, cuts

    @staticmethod
    def _limits(total: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (total - _MAX_CHARGE).clamp_min(0), total.clamp_max(_MAX_CHARGE)

    @staticmethod
    def _prior(post: torch.Tensor, initial: int, clear: torch.Tensor | None) -> torch.Tensor:
        prior = F.pad(post[:, :-1], (0, 0, 1, 0), value=initial)
        return prior if clear is None else prior.masked_fill(clear, initial)

    @staticmethod
    def _leaf_scan(shift, lower, upper, write_gate, charge, clear):
        if clear is not None:
            # A clear precedes a same-token fine update, if a commit does not win.
            from_neutral = _clip(_NEUTRAL_CHARGE + shift, lower, upper)
            lower = torch.where(clear, from_neutral, lower)
            upper = torch.where(clear, from_neutral, upper)
        keep = 1 - write_gate
        return _scan_clipped_maps(shift * keep, lower * keep + charge * write_gate,
                                  upper * keep + charge * write_gate)

    @staticmethod
    def _phases(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        coarse = (left + right) / 2
        fine = (left - right + _MAX_CHARGE) / 2
        return _round_charge(torch.stack((coarse, fine), dim=-1))

    def forward(
        self,
        spikes: torch.Tensor,
        *,
        clear: torch.Tensor | None = None,
        stats: dict[str, torch.Tensor] | None = None,
        ablate_min_age: int | None = None,
    ) -> torch.Tensor:
        """Read only prior states. Age measures tokens since the last coarse commit."""
        if spikes.ndim != 3 or spikes.shape[1] < 1:
            raise ValueError("expected nonempty [batch, tokens, channels] spikes")
        if ablate_min_age is not None and ablate_min_age < 1:
            raise ValueError("ablate_min_age must be positive")
        if clear is not None:
            if clear.shape != spikes.shape[:-1] or clear.dtype != torch.bool or clear.device != spikes.device:
                raise ValueError("clear must be a boolean mask with one entry per token on its device")
            clear = clear.unsqueeze(-1)

        proposals, charge, left_drive, right_drive, gates, cuts = self._events(spikes)
        write_gate = proposals * _write_eligibility(proposals, self._write_holds, _HOLD_LENGTHS[0], clear)
        if self.conserve:
            keep = 1 - write_gate
            if clear is not None:
                keep = keep * ~clear
            reset = (1 - keep).transpose(1, 2).unsqueeze(-1)
            update = (2 * (charge - _NEUTRAL_CHARGE) * write_gate).transpose(1, 2).unsqueeze(-1)
            coarse_post = 2 * _NEUTRAL_CHARGE + _EventScan.apply(reset, update).squeeze(-1).transpose(1, 2)
            coarse_prior = self._prior(coarse_post, 2 * _NEUTRAL_CHARGE, clear)
            lower, upper = self._limits(coarse_prior)
            left = self._leaf_scan(left_drive, lower, upper, write_gate, charge, clear)
            right = coarse_post - left
        else:
            lower, upper = left_drive.new_zeros(()), left_drive.new_full((), _MAX_CHARGE)
            left = self._leaf_scan(left_drive, lower, upper, write_gate, charge, clear)
            right = self._leaf_scan(right_drive, lower, upper, write_gate, charge, clear)

        left_prior = self._prior(left, _NEUTRAL_CHARGE, clear)
        right_prior = self._prior(right, _NEUTRAL_CHARGE, clear)
        phases = self._phases(left_prior, right_prior)
        if ablate_min_age is not None:
            last_write = _latest_before(write_gate.bool())
            positions = torch.arange(1, spikes.shape[1] + 1, device=spikes.device)[None, :, None]
            old = (last_write > 0) & (positions - last_write >= ablate_min_age)
            if clear is not None:
                latest_clear = torch.where(clear, positions, 0).cummax(1).values
                old = old & (last_write >= latest_clear)
            coarse_phase, fine_phase = phases.unbind(-1)
            hidden = coarse_phase.masked_fill(old, _NEUTRAL_CHARGE)
            phases = torch.stack((hidden, fine_phase), dim=-1)
            if stats is not None:
                coarse_cuts = cuts[..., 0, :].detach()
                changed = ((coarse_phase.detach().unsqueeze(-1) + coarse_cuts >= PHASE_THRESHOLD) !=
                           (hidden.detach().unsqueeze(-1) + coarse_cuts >= PHASE_THRESHOLD)).any(dim=-1)
                old_probe = old & gates[..., 0].bool()
                stats.update(old_probe_events=old_probe.sum(), altered_probe_events=(old_probe & changed).sum())

        reads = self.timing(phases, cuts, gates)
        if stats is not None:
            self._record_stats(stats, proposals, write_gate, left_drive, right_drive, left, right,
                               left_prior, right_prior, gates, reads)
        return reads.flatten(-3)

    @staticmethod
    @torch.no_grad()
    def _record_stats(stats, proposals, write_gate, left_drive, right_drive, left, right,
                      left_prior, right_prior, gates, reads):
        fine_slot = ~write_gate.bool()
        left_change, right_change = left - left_prior, right - right_prior
        proposed_drift = left_drive + right_drive
        actual_drift = left_change + right_change
        proposed, accepted = proposals.sum(), write_gate.sum()
        stats.update(
            probe_events=gates.sum(), read_spikes=reads.sum(),
            read_bin_counts=reads.sum(dim=(0, 1, 2, 3)),
            coarse_probe_events=gates[..., 0].sum(), fine_probe_events=gates[..., 1].sum(),
            write_events=accepted, proposed_write_events=proposed,
            accepted_write_events=accepted, rejected_write_events=proposed - accepted,
            fine_proposed_events=(fine_slot & ((left_drive != 0) | (right_drive != 0))).sum(),
            fine_applied_events=(fine_slot & ((left_change != 0) | (right_change != 0))).sum(),
            fine_proposed_move_ticks=((left_drive.abs() + right_drive.abs()) * fine_slot).sum(),
            fine_applied_move_ticks=((left_change.abs() + right_change.abs()) * fine_slot).sum(),
            fine_proposed_drift_events=(fine_slot & (proposed_drift != 0)).sum(),
            fine_proposed_drift_ticks=(proposed_drift.abs() * fine_slot).sum(),
            fine_coarse_drift_events=(fine_slot & (actual_drift != 0)).sum(),
            fine_coarse_drift_ticks=(actual_drift.abs() * fine_slot).sum(),
        )

    @torch.no_grad()
    def step(self, spikes: torch.Tensor, state: ChargeTransferState | None = None):
        """Fire the four pulse substeps for each view, then update both leaves."""
        if spikes.ndim != 2:
            raise ValueError("step expects [batch, channels] spikes")
        proposals, charge, left_drive, right_drive, gates, cuts = self._events(spikes)
        if state is None:
            left = torch.full_like(charge, _NEUTRAL_CHARGE)
            right = torch.full_like(charge, _NEUTRAL_CHARGE)
            cooldown = torch.zeros_like(charge, dtype=torch.long)
        else:
            left, right, cooldown = state.left, state.right, state.cooldown
            if any(leaf.shape != charge.shape or leaf.device != charge.device or leaf.dtype != charge.dtype
                   for leaf in (left, right)):
                raise ValueError("streaming leaf state must match the batch, device and dtype")
            if cooldown.shape != charge.shape or cooldown.device != charge.device or cooldown.dtype != torch.long:
                raise ValueError("cooldown must match the batch and device with integer dtype")
        reads, _ = self.timing.step(self._phases(left, right), cuts, gates)
        write = proposals.bool() & (cooldown == 0)
        cooldown = torch.where(write, self._write_holds - 1, (cooldown - 1).clamp_min(0))
        if self.conserve:
            total = left + right
            lower, upper = self._limits(total)
            left = _clip(left + left_drive, lower, upper)
            right = total - left
        else:
            left = (left + left_drive).clamp(0, _MAX_CHARGE)
            right = (right + right_drive).clamp(0, _MAX_CHARGE)
        left, right = torch.where(write, charge, left), torch.where(write, charge, right)
        return reads.flatten(-3), ChargeTransferState(left, right, cooldown)
