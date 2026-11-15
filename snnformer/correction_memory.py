"""Spike addressed memory with exact integer correction events."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CorrectionState:
    charge: torch.Tensor
    last_change: torch.Tensor | None = None
    position: int = 0
    previous_address: torch.Tensor | None = None
    previous_prediction: torch.Tensor | None = None


def _bounds(address, target, tolerance, limit):
    active = address.unsqueeze(-1)
    lower = (target - tolerance).clamp_min(-limit).unsqueeze(-2)
    upper = (target + tolerance).clamp_max(limit).unsqueeze(-2)
    return -limit + active * (lower + limit), limit + active * (upper - limit)


def _interval_scan(lower, upper, initial):
    # A clip followed by another clip is a clip with its endpoints
    # projected into the second interval. Compose adjacent prefixes.
    stride = 1
    while stride < lower.shape[1]:
        later_lower, later_upper = lower[:, stride:], upper[:, stride:]
        lower = torch.cat((lower[:, :stride], lower[:, :-stride].clamp(later_lower, later_upper)), dim=1)
        upper = torch.cat((upper[:, :stride], upper[:, :-stride].clamp(later_lower, later_upper)), dim=1)
        stride *= 2
    return initial[:, None].clamp(lower, upper)


def interval_spikes(charge, lower, upper):
    """Signed spikes needed to move charge just inside the allowed interval."""
    return (lower - charge).clamp_min(0) - (charge - upper).clamp_min(0)


def correction_reads(address, target, tolerance, limit, state=None, *, bind_previous=False,
                     hide_age=None, stats=None, key_channels=0, shuffle_keys=False, pulse_counts=None):
    """Read before writing, moving each addressed cell by the least signed error."""
    prior_address = None
    if bind_previous:
        prior_address = None if state is None else state.previous_address
        first = torch.zeros_like(address[:, :1]) if prior_address is None else prior_address[:, None]
        write = torch.cat((first, address[:, :-1]), dim=1)
    else:
        write = address
    lower, upper = _bounds(write, target, tolerance, limit)
    if state is None:
        state = CorrectionState(torch.zeros_like(lower[:, 0]))
    after = _interval_scan(lower, upper, state.charge)
    before = torch.cat((state.charge[:, None], after[:, :-1]), dim=1)
    if pulse_counts is not None:
        pulse_counts.append((after - before).abs().sum())
    position = state.position + address.shape[1]
    last_change = state.last_change
    selected = address.unsqueeze(-1)
    ages = hide_age if isinstance(hide_age, tuple) else None

    if hide_age is not None or stats is not None or last_change is not None:
        positions = torch.arange(state.position + 1, position + 1, device=address.device)[None, :, None, None, None]
        if last_change is None:
            if state.position:
                raise ValueError("read age needs change tracking from the beginning of the stream")
            last_change = torch.zeros_like(state.charge, dtype=positions.dtype)
        previous = torch.cat((last_change[:, None], torch.where(before != after, positions, 0)), dim=1)
        latest = previous.cummax(1).values
        last_change = latest[:, -1].clone()
        elapsed = positions - latest[:, :-1]
        if stats is not None:
            read_cells = selected.bool() & (before[..., key_channels:] != 0)
            write_spikes = (after - before).detach().abs()
            stats["address_events"] = address.detach().sum(dtype=torch.int64)
            stats["changed_cells"] = (before != after).sum(dtype=torch.int64)
            stats["correction_spikes"] = write_spikes.sum(dtype=torch.int64)
            stats["nonzero_read_cells"] = read_cells.sum(dtype=torch.int64)
            stats["age_sum"] = (elapsed[..., key_channels:] * read_cells).sum(dtype=torch.int64)
            if key_channels:
                stats["key_write_spikes"] = write_spikes[..., :key_channels].sum(dtype=torch.int64)

    if shuffle_keys:
        # Keep the values and the write trajectory intact while permuting
        # the attached forecast keys within each head.
        keys = before[..., :key_channels].roll(1, dims=-2)
        before = torch.cat((keys, before[..., key_channels:]), dim=-1)

    def read_at(age):
        charge = before
        if age is not None:
            hidden = elapsed >= age
            if key_channels:
                hidden[..., :key_channels] = False
            charge = before.masked_fill(hidden, 0)
        return (selected * charge).sum(-2, dtype=before.dtype)

    read = read_at(hide_age) if ages is None else torch.stack([read_at(age) for age in ages])
    prior_address = address[:, -1].clone() if bind_previous else None
    return read, CorrectionState(after[:, -1].clone(), last_change, position, prior_address)


def correction_step(address, target, tolerance, limit, state, *, bind_previous=False):
    write = address
    if bind_previous:
        write = torch.zeros_like(address) if state is None else state.previous_address
    lower, upper = _bounds(write, target, tolerance, limit)
    if state is None:
        state = CorrectionState(torch.zeros_like(lower))
    read = (address.unsqueeze(-1) * state.charge).sum(-2, dtype=state.charge.dtype)
    correction = interval_spikes(state.charge, lower, upper)
    if state.last_change is not None:
        state.last_change.masked_fill_(correction != 0, state.position + 1)
    state.charge.add_(correction)
    state.previous_address = address if bind_previous else None
    state.position += 1
    return read, state
