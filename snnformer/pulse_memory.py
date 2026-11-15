"""Paired-pulse reads from overwritable fixed-point membrane state."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .dendritic_event import _EventScan
from .neurons import heaviside
from .pulse_timing import PHASE_THRESHOLD as _THRESHOLD
from .pulse_timing import PhaseRead
from .temporal_snn import SpikeSynapse


@dataclass
class PulseMemoryState:
    phase: torch.Tensor
    cooldown: torch.Tensor


def _latest_before(events: torch.Tensor) -> torch.Tensor:
    """Most recent event strictly before each position, using 0 for no event."""
    positions = torch.arange(1, events.shape[1] + 1, device=events.device)[None, :, None]
    latest = torch.where(events, positions, 0).cummax(1).values
    return F.pad(latest[:, :-1], (0, 0, 1, 0))


@torch.no_grad()
def _write_eligibility(
    proposals: torch.Tensor,
    holds: torch.Tensor,
    shortest_hold: int,
    clear: torch.Tensor | None,
) -> torch.Tensor:
    """Greedy per-cell holds. A clear releases every cell before the next write."""
    width = proposals.shape[1]
    positions = torch.arange(width, device=proposals.device)[None, :, None]
    offers = proposals.bool()
    next_offer = torch.where(offers, positions, width).flip(1).cummin(1).values.flip(1)
    next_offer = F.pad(next_offer, (0, 0, 0, 1), value=width)
    deadline = (positions + holds.clamp_max(width)).clamp_max(width)
    if clear is not None:
        next_clear = torch.where(clear, positions, width).flip(1).cummin(1).values.flip(1)
        deadline = deadline.minimum(F.pad(next_clear, (0, 0, 0, 1), value=width)[:, 1:])

    # Doubling successor distance enumerates the accepted-write chain without
    # unrolling every token. The one-past-end index is an absorbing sentinel.
    successor = F.pad(next_offer.gather(1, deadline.expand_as(offers)), (0, 0, 0, 1), value=width)
    accepted = next_offer[:, :1]
    max_writes = width if clear is not None else (width + shortest_hold - 1) // shortest_hold
    while accepted.shape[1] < max_writes:
        accepted = torch.cat((accepted, successor.gather(1, accepted)), dim=1)
        successor = successor.gather(1, successor)
    schedule = torch.zeros_like(next_offer, dtype=torch.bool).scatter_(1, accepted, True)[:, :-1]

    previous = _latest_before(schedule)
    eligible = (previous == 0) | (positions + 1 - previous >= holds)
    if clear is not None:
        last_clear = torch.where(clear, positions + 1, 0).cummax(1).values
        eligible = eligible | (last_clear > previous)
    return eligible


class PulseMemory(nn.Module):
    """Learn which cells to probe and write from the incoming spike counts.

    Each cell stores an integer charge in [0, 255]. A probe adds q ticks and
    emits a bit when that crosses threshold. Adding the complementary 256 - q
    ticks restores the charge after subtractive reset. The destructive control
    clears cells selected for probing after that same read. External clears
    happen before reads. Writes overwrite selected cells after all reads.
    An optional hold starts on each accepted write. Proposals during the hold
    have no effect. External clears release the hold before the current write.
    """

    def __init__(
        self,
        dim: int,
        cells: int,
        carry_reads: bool = True,
        spike_cap: int = 32,
        *,
        write_holds: tuple[int, ...] | None = None,
        read_bins: int = 2,
        fixed_cuts: bool = False,
        global_cuts: bool = False,
    ) -> None:
        super().__init__()
        if min(dim, cells, spike_cap) < 1:
            raise ValueError("dim, cells and spike_cap must be positive")
        if read_bins not in (2, 4) or (read_bins == 2 and fixed_cuts):
            raise ValueError("expected two learned read bins or four learned/fixed read bins")
        if global_cuts and (read_bins != 4 or fixed_cuts):
            raise ValueError("global cuts require four trainable read bins")
        if write_holds is not None and (len(write_holds) != cells or any(
            not isinstance(hold, int) or not 0 <= hold <= torch.iinfo(torch.long).max for hold in write_holds
        )):
            raise ValueError("write_holds must contain one nonnegative duration per cell")
        self.cells = cells
        self.carry_reads = carry_reads
        self.read_bins = read_bins
        self.global_cuts = global_cuts
        self.timing = PhaseRead(read_bins, fixed_cuts) if read_bins == 4 else None
        self.read_width = cells * (1 if self.timing is None else self.timing.channels)
        self.project = SpikeSynapse(dim, (read_bins + 2) * cells, spike_cap)
        initial_bias = torch.tensor([-0.5, 0.0, -0.5, 0.0] + [0.0] * (read_bins - 2))[:, None]
        if global_cuts:
            initial_bias[[1, 4, 5], 0] = torch.tensor([0.25, 0.5, 0.75]).logit()
        self.bias = nn.Parameter(initial_bias.expand(read_bins + 2, cells).clone())
        self.write_holds = None if write_holds is None else tuple(write_holds)
        protected = [i for i, hold in enumerate(self.write_holds or ()) if hold >= 2]
        self.register_buffer("_write_holds", torch.tensor(write_holds, dtype=torch.long)
                             if write_holds is not None else None, persistent=False)
        self.register_buffer("_protected_cells", torch.tensor(protected, dtype=torch.long), persistent=False)
        protected_holds = [self.write_holds[i] for i in protected]
        self.register_buffer("_protected_holds", torch.tensor(protected_holds, dtype=torch.long), persistent=False)
        self._shortest_hold = min(protected_holds, default=1)

    @staticmethod
    def _mask(spikes: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.shape != spikes.shape[:-1] or mask.dtype != torch.bool or mask.device != spikes.device:
            raise ValueError("event masks must be boolean, with one entry per token on its device")
        return mask.unsqueeze(-1)

    def _events(self, spikes: torch.Tensor, probe: torch.Tensor | None, write: torch.Tensor | None,
                training_stats: dict[str, torch.Tensor] | None = None):
        drives = self.project(spikes).unflatten(-1, (self.read_bins + 2, self.cells)) + self.bias
        probe_drive, pulse_drive, write_drive, charge_drive, *extra_cuts = drives.unbind(-2)
        if training_stats is not None:
            training_stats["write_drive"] = write_drive.detach()
        probe_gate = heaviside(probe_drive)
        write_gate = heaviside(write_drive)
        if probe is not None:
            probe_gate = probe_gate * probe
        if write is not None:
            write_gate = write_gate * write

        if self.timing is None:
            pulse = pulse_drive.sigmoid() * _THRESHOLD
            pulse = pulse.detach().round() + (pulse - pulse.detach())
        else:
            cut_drives = torch.stack((pulse_drive, *extra_cuts), dim=-1)
            if self.global_cuts:
                cut_drives = torch.stack((self.bias[1], *self.bias[4:]), dim=-1).expand_as(cut_drives)
            pulse = self.timing.cuts(cut_drives)
        charge = charge_drive.sigmoid() * (_THRESHOLD - 1)
        charge = charge.detach().round() + (charge - charge.detach())
        return probe_gate, pulse, write_gate, charge

    @torch.no_grad()
    def _eligible_cells(self, proposals: torch.Tensor, clear: torch.Tensor | None) -> torch.Tensor:
        offers = proposals.index_select(2, self._protected_cells)
        eligible = _write_eligibility(offers, self._protected_holds, self._shortest_hold, clear)
        return torch.ones_like(proposals, dtype=torch.bool).index_copy_(2, self._protected_cells, eligible)

    def _accepted_writes(self, proposals: torch.Tensor, clear: torch.Tensor | None = None) -> torch.Tensor:
        if self._protected_cells.numel() == 0:
            return proposals
        # The hard schedule decides eligibility. At eligible cells, the local
        # proposal surrogate still receives a gradient, including for a zero
        # proposal. Scheduling through earlier proposals stays discrete.
        return proposals * self._eligible_cells(proposals, clear)

    def _read_phase(self, phase: torch.Tensor, cuts: torch.Tensor, probe_gate: torch.Tensor) -> torch.Tensor:
        if self.timing is None:
            return probe_gate * heaviside((phase + cuts - _THRESHOLD) / _THRESHOLD, alpha=8.0)
        return self.timing(phase, cuts, probe_gate).flatten(-2)

    def forward(
        self,
        spikes: torch.Tensor,
        *,
        probe: torch.Tensor | None = None,
        write: torch.Tensor | None = None,
        clear: torch.Tensor | None = None,
        stats: dict[str, torch.Tensor] | None = None,
        training_stats: dict[str, torch.Tensor] | None = None,
        ablate_min_age: int | None = None,
        trace: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return binary reads and optionally capture the factual bank computation."""
        if spikes.ndim != 3 or spikes.shape[1] == 0:
            raise ValueError("forward expects a nonempty sequence of shape [batch, tokens, channels]")
        if ablate_min_age is not None and ablate_min_age < 1:
            raise ValueError("ablate_min_age must be positive")
        probe_gate, pulse, proposals, charge = self._events(
            spikes, self._mask(spikes, probe), self._mask(spikes, write), training_stats)
        clear = self._mask(spikes, clear)
        write_gate = self._accepted_writes(proposals, clear)
        if training_stats is not None:
            training_stats.update(proposed_write_counts=proposals.sum(dim=(0, 1)),
                                  accepted_write_counts=write_gate.sum(dim=(0, 1)))

        # A nondestructive pulse pair leaves the carried state unchanged.
        # Each remaining transition is a hard keep or an overwrite, which can
        # be scanned as an affine recurrence independently for each cell.
        keep = 1.0 - write_gate
        if not self.carry_reads:
            keep = keep * (1.0 - probe_gate)
        if clear is not None:
            keep = keep * ~clear
        reset = (1.0 - keep).transpose(1, 2).unsqueeze(-1)
        update = (write_gate * charge).transpose(1, 2).unsqueeze(-1)
        stored = _EventScan.apply(reset, update).squeeze(-1).transpose(1, 2)
        prior = F.pad(stored[:, :-1], (0, 0, 1, 0))
        if clear is not None:
            prior = prior * ~clear
        if trace is not None or ablate_min_age is not None:
            last_write = _latest_before(write_gate.bool())
            live_phase = last_write != 0
            if clear is not None or not self.carry_reads:
                live_phase = live_phase & (last_write == _latest_before((1.0 - keep).bool()))
            if clear is not None:
                live_phase = live_phase & ~clear
            if trace is not None:
                pre_write = prior.detach() if self.carry_reads else prior.detach() * (1.0 - probe_gate.detach())
                trace.update(accepted_writes=write_gate.bool(), write_charge=charge.detach(),
                             pre_write_phase=pre_write, probe_gate=probe_gate.detach(), cuts=pulse.detach(),
                             phase_before_read=prior.detach(), latest_accepted_write=last_write - 1,
                             live_phase=live_phase)
        if ablate_min_age is not None:
            if trace is not None:
                trace["reads"] = self._read_phase(trace["phase_before_read"], trace["cuts"], trace["probe_gate"])
            positions = torch.arange(1, spikes.shape[1] + 1, device=spikes.device)[None, :, None]
            old = live_phase & (positions - last_write >= ablate_min_age)
            if stats is not None:
                old_probes = old & probe_gate.bool()
                if self.timing is None:
                    changed = (prior + pulse >= _THRESHOLD) != (pulse >= _THRESHOLD)
                else:
                    restored_crossings = prior.unsqueeze(-1) + pulse >= _THRESHOLD
                    changed = (restored_crossings != (pulse >= _THRESHOLD)).any(dim=-1)
                stats.update(old_probe_events=old_probes.sum(), altered_probe_events=(old_probes & changed).sum())
            prior = prior.masked_fill(old, 0)
        read = self._read_phase(prior, pulse, probe_gate)
        if trace is not None and ablate_min_age is None:
            trace["reads"] = read.detach()
        if stats is not None:
            stats.update(probe_events=probe_gate.detach().sum(), write_events=write_gate.detach().sum(),
                         read_spikes=read.detach().sum())
            if self.timing is not None:
                stats["read_bin_counts"] = read.detach().unflatten(-1, (self.cells, self.read_bins)).sum(dim=(0, 1, 2))
            if self.write_holds is not None:
                proposed = proposals.detach().sum()
                accepted = stats["write_events"]
                stats.update(proposed_write_events=proposed, rejected_write_events=proposed - accepted,
                             accepted_write_events=accepted)
        return read

    @torch.no_grad()
    def counterfactual_read(
        self,
        trace: dict[str, torch.Tensor],
        write_positions: torch.Tensor,
        write_cells: torch.Tensor,
        *,
        min_age: int = 128,
        max_age: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one replaced payload per row under the factual schedule.

        Indices are zero-based [batch] tensors on the trace device. The pair
        (-1, -1) skips that row. The returned [batch, tokens] mask marks probes
        of the selected live phase within the inclusive age band.
        """
        phase = trace["phase_before_read"]
        batch, width, cells = phase.shape
        if not 1 <= min_age <= max_age:
            raise ValueError("expected 1 <= min_age <= max_age")
        for indices in (write_positions, write_cells):
            if indices.shape != (batch,) or indices.dtype != torch.long or indices.device != phase.device:
                raise ValueError("write indices must be long tensors with one entry per row on the trace device")
        skipped = (write_positions == -1) & (write_cells == -1)
        in_bounds = (write_positions >= 0) & (write_positions < width) & (write_cells >= 0) & (write_cells < cells)
        if not (in_bounds | skipped).all():
            raise ValueError("write index is outside the trace")
        rows = torch.arange(batch, device=phase.device)
        positions = write_positions.clamp_min(0)
        columns = write_cells.clamp_min(0)
        if not (skipped | trace["accepted_writes"][rows, positions, columns]).all():
            raise ValueError("interventions require an accepted write")

        location = columns[:, None, None].expand(-1, width, 1)
        last_write = trace["latest_accepted_write"].gather(2, location).squeeze(-1)
        live = trace["live_phase"].gather(2, location).squeeze(-1)
        probed = trace["probe_gate"].gather(2, location).squeeze(-1).bool()
        age = torch.arange(width, device=phase.device)[None, :] - write_positions[:, None]
        eligible = live & probed & (last_write == write_positions[:, None]) & (age >= min_age) & (age <= max_age)
        eligible = eligible & ~skipped[:, None]
        previous = trace["pre_write_phase"][rows, positions, columns]
        selected_phase = phase.gather(2, location)
        replacement = torch.where(eligible.unsqueeze(-1), previous[:, None, None], selected_phase)
        phase = phase.scatter(2, location, replacement)
        return self._read_phase(phase, trace["cuts"], trace["probe_gate"]), eligible

    @torch.no_grad()
    def step(
        self,
        spikes: torch.Tensor,
        phase: torch.Tensor | PulseMemoryState | None = None,
        *,
        probe: torch.Tensor | None = None,
        write: torch.Tensor | None = None,
        clear: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | PulseMemoryState]:
        """Read then update single-token state, with a cooldown when configured."""
        if spikes.ndim != 2:
            raise ValueError("step expects spikes of shape [batch, channels]")
        probe_gate, pulse, write_gate, charge = self._events(
            spikes, self._mask(spikes, probe), self._mask(spikes, write))
        clear = self._mask(spikes, clear)
        cooldown = None
        if self.write_holds is not None:
            if phase is None:
                cooldown = torch.zeros_like(charge, dtype=torch.long)
            elif isinstance(phase, PulseMemoryState):
                cooldown, phase = phase.cooldown, phase.phase
            else:
                raise ValueError("protected memory requires a phase and cooldown state")
            if cooldown.shape != charge.shape or cooldown.device != charge.device or cooldown.dtype != torch.long:
                raise ValueError("cooldown must match batch and device with integer dtype")
            if clear is not None:
                cooldown = cooldown.masked_fill(clear, 0)
            write_gate = write_gate * (cooldown == 0)
            cooldown = torch.where(write_gate.bool(), self._write_holds - 1, cooldown - 1).clamp_min_(0)
        elif isinstance(phase, PulseMemoryState):
            raise ValueError("unprotected memory requires a phase tensor")
        if phase is None:
            phase = torch.zeros_like(charge)
        elif phase.shape != charge.shape or phase.device != charge.device or phase.dtype != charge.dtype:
            raise ValueError("streaming state must match the batch, dtype and device")
        if clear is not None:
            phase = phase.masked_fill(clear, 0)

        if self.timing is None:
            first = phase + pulse * probe_gate
            read = (first >= _THRESHOLD).to(phase.dtype)
            after_read = first - read * _THRESHOLD
            complementary_pulse = (_THRESHOLD - pulse) * probe_gate
            restored = (after_read + complementary_pulse).remainder(_THRESHOLD)
        else:
            read, restored = self.timing.step(phase, pulse, probe_gate)
            read = read.flatten(-2)
        if not self.carry_reads:
            restored = restored * (1.0 - probe_gate)
        updated = torch.where(write_gate.bool(), charge, restored)
        return read, updated if cooldown is None else PulseMemoryState(updated, cooldown)
