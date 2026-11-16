"""Byte decoder with fixed-point synapses and coincidence-routed spike packets."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .correction_memory import (
    CorrectionState, _interval_scan, _write_credit_scan, correction_reads, correction_step, interval_spikes,
)


_VOCAB = 256
_CHARGE_BITS = 6
_TICKS = 1 << _CHARGE_BITS
_PACKET_BITS = 2
_PACKET_UNITS = 1 << _PACKET_BITS
_PACKET_TICKS = _TICKS // _PACKET_UNITS
_WEIGHT_LIMIT = 127
_HIDDEN_CAP = 32
_OUTPUT_CAP = 128
_FP32_EXACT = 1 << 24
_DELAYS = ((1, 2, 4, 8), (1, 3, 9, 27), (1, 4, 16, 64))
_STEERED_DELAYS = ((1, 1, 2, 3), (1, 2, 4, 8), (1, 3, 9, 27))
_LONG_DELAY = (16, 64, 128)
_FEEDBACK_WINDOW = 8
_DELAY_ROUTES = ("steered_static", "steered_feedback", "steered_margin")
_PHASE_ROUTES = ("steered_timed", "steered_phase_swap", "steered_phase_delay")
_CORRECTION_ROUTES = ("steered_correction", "steered_correction_residual", "steered_correction_binding")
_PREDICTION_ROUTES = ("steered_correction_observed", "steered_correction_error", "steered_correction_code")
_STEERED_ROUTES = (
    "steered", "steered_local", "convergent", "steered_balanced",
    *_CORRECTION_ROUTES, *_PREDICTION_ROUTES, *_PHASE_ROUTES, *_DELAY_ROUTES,
)


@dataclass(frozen=True)
class EventLMConfig:
    dim: int = 96
    hidden: int = 288
    depth: int = 3
    route: str = "coincidence"
    carry: bool = True
    continuous: bool = False
    memory_heads: int = 4
    memory_slots: int = 16
    resolution: int = 1
    memory_width: int | None = None
    forecast_key_channels: int = 0
    match_forecast_key: bool = False
    learn_context_keys: bool = False
    learn_packet_thresholds: bool = False
    output_reference: bool = False
    compact_delays: bool = False
    departure_coincidence: bool = False
    delay_window: int = 1
    delay_residual: bool = False
    router_charge_bits: int = _CHARGE_BITS
    activity_inhibition: bool = False
    local_trace: bool = False
    trace_write_credit: bool = False
    trace_read_veto: bool = False
    delay_filter: bool = False
    opponent_steering: bool = False
    spread_coincidence: bool = False
    residual_membrane: bool = False
    compact_axon_state: bool = False


def _weight_ticks(weight, quantized=True, *, charge_bits=_CHARGE_BITS):
    if not weight.is_floating_point():
        return weight
    limit = _WEIGHT_LIMIT << (charge_bits - _CHARGE_BITS)
    ticks = (weight * (1 << charge_bits)).clamp(-limit, limit)
    return ticks.detach().round() + (ticks - ticks.detach()) if quantized else ticks


def _unit_gain_ticks(weight):
    ticks = _weight_ticks(weight)
    if not ticks.is_floating_point():
        return ticks
    # Keep a gradient at the silent endpoint.
    return ticks.detach().clamp(0, _TICKS) + (ticks - ticks.detach())


class _Synapse(nn.Module):
    def __init__(self, inputs: int, outputs: int, gain: float, *, byte_input: bool = False,
                 continuous: bool = False, charge_bits: int = _CHARGE_BITS):
        super().__init__()
        self.byte_input = byte_input
        self.continuous = continuous
        self.charge_bits = charge_bits
        shape = (inputs, outputs) if byte_input else (outputs, inputs)
        self.weight = nn.Parameter(torch.empty(shape))
        nn.init.normal_(self.weight, std=gain if byte_input else gain / math.sqrt(inputs))

    def freeze_integer(self):
        if self.weight.is_floating_point():
            ticks = _weight_ticks(self.weight, charge_bits=self.charge_bits)
            self.weight = nn.Parameter(ticks.to(torch.int32), requires_grad=False)

    def weight_ticks(self):
        return _weight_ticks(self.weight, quantized=not self.continuous, charge_bits=self.charge_bits)

    def forward(self, spikes):
        weight = self.weight_ticks()
        if self.byte_input:
            # A byte addresses the single active afferent of a 256-way input population.
            return F.embedding(spikes, weight)
        return F.linear(spikes, weight)


class _IntegrateFire(nn.Module):
    """Separate excitation and inhibition rails with subtractive reset."""

    def __init__(self, cap: int, charge_bits: int, carry: bool, *, continuous: bool = False):
        super().__init__()
        self.cap = cap
        self.bits = charge_bits
        self.threshold = 1 << charge_bits
        self.carry = carry
        self.continuous = continuous

    def _charge(self, bounded):
        rails = torch.stack((bounded.clamp_min(0), (-bounded).clamp_min(0)), dim=-1)
        return rails.to(torch.int32)

    @staticmethod
    def _signed(counts):
        return counts[..., 0] - counts[..., 1]

    def forward(self, current, quanta=None, *, reference=None):
        bounded = current.clamp(-self.threshold * self.cap, self.threshold * self.cap)
        if reference is not None:
            charge = bounded.to(torch.int32)
            # Keep the original signed rounding at half-threshold ties.
            phase = self.threshold // 2 - (charge < 0).to(torch.int32)
            emitted = (charge + phase - reference.to(torch.int32) * self.threshold) >> self.bits
            if current.is_floating_point():
                surrogate = bounded / self.threshold - reference
                return emitted.to(current.dtype) + (surrogate - surrogate.detach())
            return emitted
        if quanta is not None:
            units = quanta.to(torch.int32)
            quantum = self.threshold // _PACKET_UNITS
            threshold = units * quantum
            charge = bounded.to(torch.int32)
            magnitude = (charge.abs() + threshold // 2) // threshold
            limit = (self.cap * _PACKET_UNITS) // units
            counts = magnitude.minimum(limit) * charge.sign()
            if current.is_floating_point():
                surrogate = bounded / (quanta * quantum)
                return counts.to(current.dtype) + (surrogate - surrogate.detach())
            return counts
        if self.continuous:
            return bounded / self.threshold
        charge = self._charge(bounded)
        initial = self.threshold // 2
        if self.carry:
            if current.shape[1] * self.threshold * self.cap + initial > torch.iinfo(torch.int32).max:
                raise ValueError("byte sequence exceeds the int32 charge scan")
            emissions = (charge.cumsum(1, dtype=torch.int32) + initial) >> self.bits
            emissions = emissions.diff(dim=1, prepend=torch.zeros_like(emissions[:, :1]))
        else:
            emissions = (charge + initial) >> self.bits
        counts = self._signed(emissions)
        if current.is_floating_point():
            surrogate = bounded / self.threshold
            return counts.to(current.dtype) + (surrogate - surrogate.detach())
        return counts

    def step(self, current, phase, quanta=None, *, reference=None):
        if reference is not None:
            return self(current, reference=reference), None
        if quanta is not None:
            return self(current, quanta), None
        bounded = current.clamp(-self.threshold * self.cap, self.threshold * self.cap)
        total = self._charge(bounded) + (phase if self.carry and phase is not None else self.threshold // 2)
        counts = total >> self.bits
        phase = (total & (self.threshold - 1)) if self.carry else None
        return self._signed(counts), phase


class _ResidualMembraneFire(_IntegrateFire):
    """Carry the signed rounding residue of a residual neuron across bytes."""

    def __init__(self, cap, bits):
        super().__init__(cap, bits, True)
        if (cap + 1) * self.threshold - 1 > torch.iinfo(torch.int32).max:
            raise ValueError("residual charge exceeds the int32 membrane range")
        self.phase_dtype = next(dtype for dtype in (torch.uint8, torch.int16, torch.int32)
                                if self.threshold - 1 <= torch.iinfo(dtype).max)

    def forward(self, current, *, state=None):
        if (current.shape[1] * self.cap + 1) * self.threshold - 1 > torch.iinfo(torch.int32).max:
            raise ValueError("byte sequence exceeds the int32 residual charge scan")
        limit = self.cap * self.threshold
        bounded = current.clamp(-limit, limit)
        phase = self.threshold // 2
        if state is not None and state.output_phase is not None:
            phase = state.output_phase[:, None].to(torch.int32)
        total = bounded.to(torch.int32).cumsum(1, dtype=torch.int32) + phase
        emitted = total >> self.bits
        counts = emitted.diff(dim=1, prepend=torch.zeros_like(emitted[:, :1]))
        if state is not None:
            state.output_phase = (total[:, -1] & (self.threshold - 1)).to(self.phase_dtype)
        if current.is_floating_point():
            surrogate = bounded / self.threshold
            return counts.to(current.dtype) + (surrogate - surrogate.detach())
        return counts

    def step(self, current, phase):
        limit = self.cap * self.threshold
        initial = self.threshold // 2 if phase is None else phase.to(torch.int32)
        total = current.clamp(-limit, limit).to(torch.int32) + initial
        return total >> self.bits, (total & (self.threshold - 1)).to(self.phase_dtype)


class _PacketThresholds(nn.Module):
    """A shared firing threshold for each content and steering group."""

    def __init__(self, widths):
        super().__init__()
        self.relative = nn.Parameter(torch.ones(len(widths)))
        groups = torch.arange(len(widths)).repeat_interleave(torch.tensor(widths))
        self.register_buffer("groups", groups, persistent=False)

    def quanta(self):
        if not self.relative.is_floating_point():
            return self.relative
        units = self.relative.clamp(0.5, 2.0) * _PACKET_UNITS
        return units.detach().round() + (units - units.detach())

    def forward(self):
        return self.quanta().index_select(0, self.groups)

    def freeze_integer(self):
        if self.relative.is_floating_point():
            self.relative = nn.Parameter(self.quanta().to(torch.int32), requires_grad=False)


class _DelayCorrection(nn.Module):
    """Emit the difference between the anchor and matching packet on one path."""

    def __init__(self, widths, cap):
        super().__init__()
        self.gain = nn.Parameter(torch.zeros(len(widths)))
        groups = torch.arange(len(widths)).repeat_interleave(torch.tensor(widths))
        self.register_buffer("groups", groups, persistent=False)
        self.emit = _IntegrateFire(2 * cap, _CHARGE_BITS, False)

    def gains(self):
        return _unit_gain_ticks(self.gain)

    def freeze_integer(self):
        if self.gain.is_floating_point():
            self.gain = nn.Parameter(self.gains().to(torch.int32), requires_grad=False)

    def forward(self, anchor, chosen):
        return self.emit_difference(chosen - anchor)

    def emit_difference(self, difference):
        return self.emit(difference * self.gains().index_select(0, self.groups))


def _even_prefix(count, slots, bits):
    """Count evenly scheduled pulses in the first slots of a fixed clock."""
    rounded = (count.to(torch.int32) * slots.to(torch.int32) + (1 << (bits - 1))) >> bits
    if not count.is_floating_point():
        return rounded
    surrogate = count * slots / (1 << bits)
    return rounded.to(count.dtype) + (surrogate - surrogate.detach())


class _SpreadCoincidence(nn.Module):
    """Redistribute part of a pulse packet across the steering clock."""

    def __init__(self, widths, cap, resolution):
        super().__init__()
        self.clock_bits = (2 * resolution - 1).bit_length()
        self.clock_slots = 1 << self.clock_bits
        units = max(_TICKS, self.clock_slots)
        if cap * units + units // 2 > torch.iinfo(torch.int32).max:
            raise ValueError("spread coincidence would exceed the integer phase counter range")
        self.mix = nn.Parameter(torch.zeros(len(widths)))
        groups = torch.arange(len(widths)).repeat_interleave(torch.tensor(widths))
        self.register_buffer("groups", groups, persistent=False)

    def mix_ticks(self):
        return _unit_gain_ticks(self.mix)

    def freeze_integer(self):
        if self.mix.is_floating_point():
            self.mix = nn.Parameter(self.mix_ticks().to(torch.int32), requires_grad=False)

    def forward(self, packet, steering, paired):
        count = packet.abs()
        spread = _even_prefix(count, self.mix_ticks().index_select(0, self.groups), _CHARGE_BITS)
        aperture = steering.clamp_max(self.clock_slots)
        timed = _even_prefix(spread, aperture, self.clock_bits)
        packed = (count - spread).minimum(steering)
        # Retain the original sign-rail gradients when the new route is silent.
        selected = paired + (packed + timed - count.minimum(steering)) * packet.sign()
        return selected, packet - selected


class _DelayFilter(nn.Module):
    """Use changes across fixed taps to emit a correction to the anchor packet."""

    def __init__(self, width, delay, cap):
        super().__init__()
        midpoint = (delay + 1) // 2
        self.taps = (midpoint, 1) if midpoint > 1 else (1,)
        self.weight = nn.Parameter(torch.zeros(width, len(self.taps)))
        self.emit = _IntegrateFire(cap, _CHARGE_BITS, False)

    def freeze_integer(self):
        if self.weight.is_floating_point():
            self.weight = nn.Parameter(self.weight_ticks().to(torch.int32), requires_grad=False)

    def weight_ticks(self):
        return _weight_ticks(self.weight)

    def forward(self, anchor, packets):
        differences = torch.stack(packets, dim=-1) - anchor.unsqueeze(-1)
        current = (differences * self.weight_ticks()).sum(-1, dtype=anchor.dtype)
        correction = self.emit(current)
        return (anchor + correction).clamp(-self.emit.cap, self.emit.cap)


def _route_packets(present, delayed, route):
    if route == "coincidence":
        # Packets occupy the first count slots on their sign rail. Intersecting
        # two trains therefore takes min on each rail. Every unmatched pulse
        # goes to the second synapse, including the opposite-sign case.
        paired = torch.minimum(present.clamp_min(0), delayed.clamp_min(0))
        paired = paired - torch.minimum((-present).clamp_min(0), (-delayed).clamp_min(0))
    else:
        # Same two destinations and parameters. Every other slot is routed to
        # the first synapse, with no dependence on the current spike packet.
        half = delayed.sign() * ((delayed.abs().to(torch.int32) + 1) >> 1)
        paired = half
        if delayed.is_floating_point():
            surrogate = delayed * 0.5
            paired = half.to(delayed.dtype) + (surrogate - surrogate.detach())
    return paired, delayed - paired


def _steer_packet(packet, steering, start=None):
    # The same steering spikes open both sign rails. Every pulse that does
    # not coincide with a steering pulse remains available on the other path.
    positive = packet.clamp_min(0)
    negative = (-packet).clamp_min(0)
    if start is None:
        selected = torch.minimum(positive, steering) - torch.minimum(negative, steering)
    else:
        # Intersect the content interval [0, |packet|) with [start, start + steering).
        end = start + steering
        left = start.clamp_min(0)
        selected = (torch.minimum(positive, end) - left).clamp_min(0)
        selected = selected - (torch.minimum(negative, end) - left).clamp_min(0)
    return selected, packet - selected


def _phase_swap(packet, steering, start):
    """Use one timing bit to swap the destinations of intact pulse groups."""
    matched, rest = _steer_packet(packet, steering)
    selected = matched + (start + 1) * (rest - matched)
    return selected, packet - selected


def _balanced_steer(packet, steering):
    """Start at an even split and let either steering rail transfer pulses."""
    magnitude = packet.abs()
    half = (magnitude.to(torch.int32) + 1) >> 1
    if packet.is_floating_point():
        surrogate = magnitude * 0.5
        half = half.to(packet.dtype) + (surrogate - surrogate.detach())
    selected = torch.minimum((half + steering).clamp_min(0), magnitude) * packet.sign()
    return selected, packet - selected


def _steering_start(drive, phase_bits=2, cap=_HIDDEN_CAP):
    """Start tick from the input charge modulo a firing threshold."""
    drive = drive.clamp(0, _TICKS * cap)
    phase = (drive.to(torch.int32) + _TICKS // 2) & (_TICKS - 1)
    start = (phase >> (_CHARGE_BITS - phase_bits)) - (1 << (phase_bits - 1))
    if drive.is_floating_point():
        surrogate = drive / (_TICKS >> phase_bits)
        start = start.to(drive.dtype) + (surrogate - surrogate.detach())
    return start


def _lag(spikes, delay):
    padding = min(delay, spikes.shape[1])
    return F.pad(spikes[:, :spikes.shape[1] - padding], (0, 0, padding, 0))


def _agreement(present, delayed):
    """Matched pulses minus extra pulses, relative to the error of silence."""
    present = present.to(torch.int32).unsqueeze(-1)
    return present.abs() - (present - delayed.to(torch.int32)).abs()


class _DelaySelector(nn.Module):
    """Select a per-channel delay using past packet errors and a learned prior."""

    def __init__(self, taps, widths, extra_delay, feedback, margin):
        super().__init__()
        self.feedback = feedback
        self.margin = margin
        delays = sorted(set((*taps, extra_delay)))
        self.delays = tuple(delays)
        self.history_size = delays[-1] + _FEEDBACK_WINDOW
        self.register_buffer("ages", torch.tensor(delays), persistent=False)
        prior = torch.zeros(sum(widths), len(delays))
        first = 0
        for width, tap in zip(widths, taps, strict=True):
            prior[first:first + width, delays.index(tap)] = 4 / _TICKS
            first += width
        self.prior = nn.Parameter(prior)

    def freeze_integer(self):
        if self.prior.is_floating_point():
            self.prior = nn.Parameter(_weight_ticks(self.prior).to(torch.int32), requires_grad=False)

    def _select(self, packets, credit, available):
        prior = _weight_ticks(self.prior).masked_fill(~available, -4096)
        if self.margin:
            anchor = prior.argmax(-1, keepdim=True).expand_as(credit[..., :1])
            # Require an average gain of one spike per feedback byte before an override.
            credit = (credit - credit.gather(-1, anchor) - self.margin).clamp_min(0)
        scores = (prior + credit).masked_fill(~available, -4096)
        choice = scores.argmax(-1, keepdim=True).expand_as(packets[..., :1])
        packet = packets.gather(-1, choice).squeeze(-1)
        if self.training and self.prior.is_floating_point():
            # Only the hard winner moves spikes. Use soft weights for prior gradients.
            surrogate = ((scores / 16).softmax(-1) * packets.detach()).sum(-1)
            packet = packet + (surrogate - surrogate.detach())
        return packet

    def forward(self, spikes, delay_limit):
        packets = torch.stack([_lag(spikes, delay) for delay in self.delays], dim=-1)
        credit = 0
        if self.feedback:
            evidence = _agreement(spikes, packets)
            past = F.pad(evidence[:, :-1].cumsum(1, dtype=torch.int32), (0, 0, 0, 0, 1, 0))
            trim = min(_FEEDBACK_WINDOW, spikes.shape[1])
            credit = past - F.pad(past[:, :-trim], (0, 0, 0, 0, trim, 0))
        age = torch.arange(spikes.shape[1], device=spikes.device).clamp_min(1)
        if delay_limit is not None:
            age = age.clamp_max(delay_limit)
        return self._select(packets, credit, age[:, None, None] >= self.ages)

    def step(self, history, position):
        read_at = (position - self.ages) % self.history_size
        packets = history.index_select(1, read_at).transpose(1, 2)
        credit = 0
        if self.feedback:
            times = torch.arange(position - _FEEDBACK_WINDOW, position, device=history.device)
            recent = history.index_select(1, times % self.history_size)
            read_at = (times[:, None] - self.ages) % self.history_size
            past = history.index_select(1, read_at.flatten()).unflatten(1, read_at.shape).transpose(-1, -2)
            credit = _agreement(recent, past).sum(1, dtype=torch.int32)
        return self._select(packets, credit, max(1, position) >= self.ages)


@dataclass
class _CorrectionAccess:
    state: CorrectionState | None
    prefix: int
    hide_age: int | tuple[int | None, ...] | None
    stats: dict | None
    shuffle_keys: bool = False
    local_layers: list[_DelayState] | None = None


class _SpikeCorrection(nn.Module):
    """An address spike probes a bank and enables a bounded correction."""

    def __init__(self, dim, width, heads, slots, resolution, bind_previous, *, observed=False, forecast=False,
                 key_channels=0, match_key=False):
        super().__init__()
        if heads < 1 or slots < 2 or width < heads or width % heads:
            raise ValueError("memory needs at least two slots and a positive read width divisible by its heads")
        if forecast and width > dim:
            raise ValueError("the forecast code must fit in the learned input spike population")
        if not 0 <= key_channels < width // heads:
            raise ValueError("each cell needs at least one payload channel")
        self.heads = heads
        self.slots = slots
        self.width = width
        self.resolution = resolution
        self.bind_previous = bind_previous
        self.key_channels = key_channels
        self.match_key = match_key
        self.encode = _Synapse(dim, heads * (slots + 1) + (0 if observed else width), gain=1.0)
        self.code = _Synapse(dim, width, gain=0.25 if forecast else 1.0) if observed else None
        # A single byte afferent contributes at most two counts per unit
        # of resolution. Forecast the same code within that range.
        self.forecast = _IntegrateFire(2 * resolution, _CHARGE_BITS, False) if forecast else None
        self.address = _IntegrateFire(_HIDDEN_CAP * resolution, _CHARGE_BITS - 2, False)
        self.value = _IntegrateFire(_HIDDEN_CAP * resolution, _CHARGE_BITS, False)
        self.register_buffer("slot_ids", torch.arange(slots), persistent=False)

    def _value_code(self, spikes):
        emit = self.value if self.forecast is None else self.forecast
        return emit(self.code(spikes)).unflatten(-1, (self.heads, -1))

    def observation_code(self, byte_spikes):
        if self.forecast is not None:
            return byte_spikes[..., :self.width].unflatten(-1, (self.heads, -1))
        return self._value_code(byte_spikes)

    def _events(self, spikes, observed):
        encoded = self.encode(spikes)
        if self.code is None:
            drive, target, tolerance = encoded.split((self.heads * self.slots, self.width, self.heads), dim=-1)
            target = self.value(target).unflatten(-1, (self.heads, -1))
        else:
            drive, tolerance = encoded.split((self.heads * self.slots, self.heads), dim=-1)
            target = self.observation_code(observed)
        counts = self.address(drive.clamp_min(0)).unflatten(-1, (self.heads, self.slots))
        winner = counts.argmax(-1, keepdim=True)
        active = counts.amax(-1, keepdim=True).clamp(0, 1)
        address = (winner == self.slot_ids).to(counts.dtype) * active
        if self.training and counts.is_floating_point():
            # One address fires in the forward pass. Gradients can also
            # reach runner-up neurons so the route can change during training.
            probabilities = (counts / (2 * self.resolution)).softmax(-1)
            address = address + active.detach() * (probabilities - probabilities.detach())
        tolerance = self.value(tolerance.clamp_min(0)).unsqueeze(-1)
        return address, target, tolerance

    def _write_target(self, observed, prediction, tolerance):
        error = (observed - prediction).clamp(-self.value.cap, self.value.cap)
        if not self.key_channels:
            return error, tolerance
        # The key records the forecast that produced the error. A
        # zero-width interval preserves that key exactly.
        key = prediction[..., :self.key_channels]
        payload = error[..., self.key_channels:]
        return torch.cat((key, payload), dim=-1), torch.cat((torch.zeros_like(key), tolerance.expand_as(payload)), dim=-1)

    def _read_payload(self, read, prediction, tolerance, address, stats=None, pulse_counts=None):
        if not self.key_channels:
            return read
        key, payload = read[..., :self.key_channels], read[..., self.key_channels:]
        if not self.match_key:
            return payload
        query = prediction[..., :self.key_channels]
        excess = interval_spikes(key, query - tolerance, query + tolerance).abs()
        # The inhibitory packets share a rail. Overlapping pulses make
        # one veto, and coincidence removes the same slots from the read.
        veto = excess.amax(-1, keepdim=True) * address.amax(-1, keepdim=True)
        muted, passed = _steer_packet(payload, veto)
        if pulse_counts is not None:
            pulse_counts.append(veto.sum())
        if stats is not None:
            if read.ndim > prediction.ndim:
                veto, muted, passed_for_stats = veto[0], muted[0], passed[0]
            else:
                passed_for_stats = passed
            stats["veto_spikes"] = veto.detach().sum(dtype=torch.int64)
            stats["muted_payload_spikes"] = muted.detach().abs().sum(dtype=torch.int64)
            stats["passed_payload_spikes"] = passed_for_stats.detach().abs().sum(dtype=torch.int64)
        return passed

    def forward(self, spikes, access, *, observed=None, prediction_code=None, pulse_counts=None):
        prefix = access.prefix if access is not None else 0
        state = None if access is None else access.state
        observation = None if observed is None else observed[:, prefix:]
        address, target, tolerance = self._events(spikes[:, prefix:], observation)
        read_tolerance = tolerance
        prediction = None
        if prediction_code is not None:
            prediction = prediction_code[:, prefix:]
            previous = None if state is None else state.previous_prediction
            first = torch.zeros_like(prediction[:, :1]) if previous is None else previous[:, None]
            previous = torch.cat((first, prediction[:, :-1]), dim=1)
            target, tolerance = self._write_target(target, previous, tolerance)
        read, state = correction_reads(
            address, target, tolerance, self.value.cap, state, bind_previous=self.bind_previous,
            hide_age=None if access is None else access.hide_age, stats=None if access is None else access.stats,
            key_channels=self.key_channels, shuffle_keys=access is not None and access.shuffle_keys,
            pulse_counts=pulse_counts,
        )
        if access is not None:
            if prediction_code is not None:
                state.previous_prediction = prediction[:, -1].clone()
            access.state = state
        read = self._read_payload(read, prediction, read_tolerance, address,
                                  None if access is None else access.stats, pulse_counts)
        if pulse_counts is not None:
            pulse_counts.append(read.abs().sum())
        return F.pad(read.flatten(-2), (0, 0, prefix, 0)) if prefix else read.flatten(-2)

    def step(self, spikes, state, *, observed=None, prediction_code=None):
        address, target, tolerance = self._events(spikes, observed)
        read_tolerance = tolerance
        if prediction_code is not None and state is not None:
            target, tolerance = self._write_target(target, state.previous_prediction, tolerance)
        read, state = correction_step(
            address, target, tolerance, self.value.cap, state, bind_previous=self.bind_previous,
        )
        if prediction_code is not None:
            state.previous_prediction = prediction_code
        read = self._read_payload(read, prediction_code, read_tolerance, address)
        return read.flatten(-2), state


@dataclass
class _DelayState:
    history: torch.Tensor | None = None
    relay_phase: torch.Tensor | None = None
    steering_phase: torch.Tensor | None = None
    output_phase: torch.Tensor | None = None
    correction: CorrectionState | None = None
    trace_charge: torch.Tensor | None = None


class _SpikeTrace(nn.Module):
    """A causal charge trace that moves only when an interval excludes it."""

    def __init__(self, cap, *, write_credit=False, veto_widths=None):
        super().__init__()
        self.cap = cap
        self.write_credit = write_credit
        self.read_veto = _DelayCorrection(veto_widths, cap) if veto_widths is not None else None

    def _bounds(self, present, radius):
        return (present - radius).clamp_min(-self.cap), (present + radius).clamp_max(self.cap)

    def _read(self, before, change, steering):
        read, _ = _steer_packet(before, steering)
        if self.read_veto is not None:
            veto = self.read_veto.emit_difference(change.abs())
            _, read = _steer_packet(read, veto)
        return read

    def forward(self, present, steering, initial, pulse_counts, *, write_radius=None):
        if initial is None:
            initial = torch.zeros_like(present[:, 0])
        lower, upper = self._bounds(present, steering if write_radius is None else write_radius)
        needs_credit = self.write_credit and torch.is_grad_enabled() and (
            lower.requires_grad or upper.requires_grad or initial.requires_grad
        )
        scan = _write_credit_scan if needs_credit else _interval_scan
        after = scan(lower, upper, initial)
        before = torch.cat((initial[:, None], after[:, :-1]), dim=1)
        change = after - before
        read = self._read(before, change, steering)
        if pulse_counts is not None:
            pulse_counts.extend((change.abs().sum(), read.abs().sum()))
        return read, after[:, -1].clone()

    def step(self, present, steering, before, *, write_radius=None):
        if before is None:
            before = torch.zeros_like(present)
        lower, upper = self._bounds(present, steering if write_radius is None else write_radius)
        change = interval_spikes(before, lower, upper)
        return self._read(before, change, steering), before + change


class _OpponentSteering(nn.Module):
    """Use the opposite synaptic current to emit write-protection spikes."""

    def __init__(self, width, cap, charge_bits):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))
        self.current_limit = cap << charge_bits
        self.emit = _IntegrateFire(cap, charge_bits + _CHARGE_BITS, False)

    def gain_ticks(self):
        if not self.weight.is_floating_point():
            return self.weight
        ticks = _weight_ticks(self.weight)
        return ticks.detach().clamp_min(0) + (ticks - ticks.detach())

    def freeze_integer(self):
        if self.weight.is_floating_point():
            self.weight = nn.Parameter(self.gain_ticks().to(torch.int32), requires_grad=False)

    def forward(self, current, quanta):
        opposite = (-current).clamp(0, self.current_limit)
        return self.emit(opposite * self.gain_ticks(), quanta)


class _DelayBlock(nn.Module):
    def __init__(self, cfg: EventLMConfig, depth: int):
        super().__init__()
        self.steered = cfg.route in _STEERED_ROUTES
        schedule = _STEERED_DELAYS if self.steered else _DELAYS
        self.delays = schedule[depth % len(schedule)]
        width, extra = divmod(cfg.hidden, len(self.delays))
        self.widths = tuple(width + (group < extra) for group in range(len(self.delays)))
        self.route = cfg.route
        self.correction = None
        self.memory_residual = None
        if cfg.route in _CORRECTION_ROUTES and depth == cfg.depth - 1:
            self.correction = _SpikeCorrection(
                cfg.dim, self.widths[-1], cfg.memory_heads, cfg.memory_slots, cfg.resolution,
                bind_previous=cfg.route == "steered_correction_binding",
            )
            if cfg.route != "steered_correction":
                self.memory_residual = _Synapse(self.widths[-1], cfg.dim, gain=0.15)
        replace_delay = self.correction is not None and self.memory_residual is None
        self.local_delays = self.delays[:-1] if replace_delay else self.delays
        self.local_widths = self.widths[:len(self.local_delays)]
        self.balanced = cfg.route == "steered_balanced"
        self.phase_bits = 2 if cfg.route == "steered_timed" else int(cfg.route in _PHASE_ROUTES)
        self.alternate_delays = ()
        if cfg.route == "steered_phase_delay":
            self.alternate_delays = tuple(delay + 1 if delay < self.delays[-1] else delay - 1 for delay in self.delays)
        self.selector = None
        if cfg.route in _DELAY_ROUTES:
            self.selector = _DelaySelector(self.delays, self.widths, _LONG_DELAY[depth % len(_LONG_DELAY)],
                                           feedback=cfg.route != "steered_static",
                                           margin=_FEEDBACK_WINDOW if cfg.route == "steered_margin" else 0)
        self.history_size = self.local_delays[-1] if self.selector is None else self.selector.history_size
        self.compact_delays = cfg.compact_delays
        self.departure_coincidence = cfg.departure_coincidence
        self.delay_window = cfg.delay_window
        self.resolution = cfg.resolution
        if self.delay_window > 1:
            offsets = torch.arange(min(self.delay_window, self.history_size))
            self.register_buffer("delay_offsets", offsets, persistent=False)
        counts_per_channel = 2 if self.departure_coincidence else 1
        self.delay_widths = tuple(counts_per_channel * width for width in self.widths)
        self.delay_sizes = tuple(delay * width for delay, width in zip(self.local_delays, self.delay_widths))
        cap = _HIDDEN_CAP * cfg.resolution
        self.trace = None
        if cfg.local_trace:
            veto_widths = self.widths if cfg.trace_read_veto else None
            self.trace = _SpikeTrace(cap, write_credit=cfg.trace_write_credit, veto_widths=veto_widths)
        self.opponent_steering = (
            _OpponentSteering(cfg.hidden, cap, cfg.router_charge_bits) if cfg.opponent_steering else None
        )
        self.spread_coincidence = (
            _SpreadCoincidence(self.widths, cap, cfg.resolution) if cfg.spread_coincidence else None
        )
        self.delay_correction = _DelayCorrection(self.widths, cap) if cfg.delay_residual else None
        self.delay_filters = None
        if cfg.delay_filter:
            # For an anchor at d, compare with the halfway point and one byte back.
            self.delay_filters = nn.ModuleList(
                _DelayFilter(width, delay, cap) if delay > 1 else None
                for width, delay in zip(self.widths, self.local_delays, strict=True)
            )
        self.packet_thresholds = _PacketThresholds(self.widths) if cfg.learn_packet_thresholds else None
        self.activity_inhibition = None
        if cfg.activity_inhibition:
            # One pool receives one unit of charge per spike on either input rail.
            pool_bits = ((cfg.dim * cfg.resolution - 1) // _PACKET_UNITS).bit_length()
            self.activity_inhibition = _IntegrateFire(_HIDDEN_CAP * _PACKET_UNITS, pool_bits, False)
        bits = cfg.router_charge_bits
        self.expand = _Synapse(cfg.dim, cfg.hidden * (2 if self.steered else 1), gain=1.0,
                               continuous=cfg.continuous, charge_bits=bits)
        self.relay = _IntegrateFire(cap, bits, cfg.carry, continuous=cfg.continuous)
        if self.steered:
            self.steer = _IntegrateFire(cap, bits, cfg.carry, continuous=cfg.continuous)
        self.project = _Synapse((4 if self.steered else 3) * cfg.hidden, cfg.dim, gain=0.65,
                                continuous=cfg.continuous, charge_bits=bits)
        output_bits = bits + (_PACKET_BITS if self.packet_thresholds is not None else 0)
        self.residual_membrane = cfg.residual_membrane
        if self.residual_membrane:
            self.emit = _ResidualMembraneFire(cap, output_bits)
        else:
            self.emit = _IntegrateFire(cap, output_bits, cfg.carry, continuous=cfg.continuous)
        self.axon_dtype = None
        if cfg.compact_axon_state:
            self.axon_dtype = next(dtype for dtype in (torch.int8, torch.int16, torch.int32)
                                   if cap <= torch.iinfo(dtype).max)

    def _firing_quanta(self, spikes, quanta):
        if self.activity_inhibition is None:
            return quanta
        activity = spikes.abs().sum(-1, keepdim=True, dtype=spikes.dtype)
        return _PACKET_UNITS + self.activity_inhibition(activity)

    def _trace_radius(self, steering, current, quanta):
        if self.opponent_steering is None:
            return steering
        return steering + self.opponent_steering(current, quanta)

    def _departure_packets(self, present, steering):
        if not self.departure_coincidence:
            return present, None
        # A marked pulse occupies the same slot on the same sign rail when it arrives.
        paired, rest = _steer_packet(present, steering)
        packets = torch.stack((present, paired), dim=-1).flatten(-2)
        return packets, (paired, rest)

    def _match_delay(self, candidates, query, steering, available):
        # A steering train exposes the same slots of the present packet and of each delayed packet.
        aperture = steering.unsqueeze(-1)
        keys = candidates.clamp(-aperture, aperture)
        mismatches = (keys - query.unsqueeze(-1)).abs().sum(-2, dtype=keys.dtype)
        limit = 2 * self.relay.cap * query.shape[-1] + 1
        mismatches = mismatches.masked_fill(~available, limit)
        index = mismatches.argmin(-1, keepdim=True)
        winner = candidates.gather(-1, index.unsqueeze(-2).expand_as(candidates[..., :1])).squeeze(-1)
        if self.training and candidates.is_floating_point():
            scale = self.resolution * math.isqrt(query.shape[-1])
            weights = (-mismatches / scale).softmax(-1)
            surrogate = (candidates.detach() * weights.unsqueeze(-2)).sum(-1)
            winner = winner + (surrogate - surrogate.detach())
        return winner

    def _delayed_groups(self, groups, steering, pulse_counts):
        if self.delay_window == 1:
            return [_lag(group, delay) for group, delay in zip(groups, self.local_delays)], None
        positions = torch.arange(groups[0].shape[1], device=groups[0].device)
        result, anchors = [], []
        for group, aperture, delay in zip(groups, steering.split(self.widths, dim=-1), self.local_delays, strict=True):
            choices = min(self.delay_window, delay)
            if choices == 1:
                winner = anchor = _lag(group, delay)
            else:
                # Each window starts at the anchor lag and looks toward the present.
                candidates = F.pad(group, (0, 0, delay, 0)).unfold(1, choices, 1)[:, :group.shape[1]]
                available = positions[:, None] >= delay - self.delay_offsets[:choices]
                query = group.clamp(-aperture, aperture)
                winner = self._match_delay(candidates, query, aperture, available)
                anchor = candidates[..., 0]
                if pulse_counts is not None:
                    pulse_counts.append(available.any(-1).sum(dtype=torch.int32) * group.shape[0])
            result.append(winner)
            if self.delay_correction is not None:
                anchors.append(anchor)
        return result, torch.cat(anchors, dim=-1) if anchors else None

    def _chunk_delays(self, groups, state):
        # Chunk histories are chronological and cover the full span of each tap.
        batch, steps = groups[0].shape[:2]
        history_dtype = groups[0].dtype
        if self.axon_dtype is not None and not (groups[0].is_floating_point() and torch.is_grad_enabled()):
            history_dtype = self.axon_dtype
        if state.history is None:
            state.history = groups[0].new_zeros(batch, sum(self.delay_sizes), dtype=history_dtype)
        delayed, tails = [], []
        history = state.history.split(self.delay_sizes, dim=-1)
        for storage, group, delay in zip(history, groups, self.local_delays, strict=True):
            joined = torch.cat((storage.reshape(batch, delay, group.shape[-1]).to(group.dtype), group), dim=1)
            delayed.append(joined[:, :steps])
            tails.append(joined[:, -delay:].flatten(1))
        state.history = torch.cat(tails, dim=-1).to(history_dtype)
        return delayed

    def _current(self, residual, present, delayed, steering, start, alternate=None, recall=None, quanta=None, *,
                 departure=None, anchor=None, coincidence_read=None):
        if alternate is not None:
            # The last half of the phase interval opens the adjacent delay
            # only when a steering packet has actually fired.
            switch = (start + 1) * (steering > 0)
            delayed = delayed + switch * (alternate - delayed)
            start = None
        if not self.steered:
            routes = (present, *_route_packets(present, delayed, self.route))
        elif departure is not None:
            delayed, marked = delayed.unflatten(-1, (-1, 2)).unbind(-1)
            paired, _ = _steer_packet(marked, steering)
            routes = departure + (paired, delayed - paired)
        elif anchor is not None:
            selected, bypass = _steer_packet(anchor, steering)
            replacement, _ = _steer_packet(delayed, steering)
            correction = self.delay_correction(selected, replacement)
            routes = _steer_packet(present, steering) + (selected + correction, bypass)
        elif coincidence_read is not None:
            _, bypass = _steer_packet(delayed, steering)
            routes = _steer_packet(present, steering) + (coincidence_read, bypass)
        elif self.balanced:
            routes = _balanced_steer(present, steering) + _balanced_steer(delayed, steering)
        else:
            split = _phase_swap if self.route == "steered_phase_swap" else _steer_packet
            delay_routes = (_route_packets(present, delayed, "tonic") if self.route == "steered_local"
                            else split(delayed, steering, start))
            routes = split(present, steering, start) + delay_routes
        if self.spread_coincidence is not None:
            routes = self.spread_coincidence(present, steering, routes[0]) + routes[2:]
        if quanta is not None:
            # Route physical pulse counts first. The same synaptic charge
            # then requires fewer pulses at a higher firing threshold.
            routes = tuple(packet * quanta for packet in routes)
        current = residual * self.emit.threshold + self.project(torch.cat(routes, dim=-1))
        if self.memory_residual is not None:
            current = current + self.memory_residual(recall)
        return current

    def forward(self, spikes, *, delay_limit=None, shuffle_phase=False, correction_access=None, pulse_counts=None,
                local_state=None):
        drive = self.expand(spikes)
        quanta = self.packet_thresholds() if self.packet_thresholds is not None else None
        firing_quanta = self._firing_quanta(spikes, quanta)
        steering = start = None
        if self.steered:
            drive, steer_drive = drive.chunk(2, dim=-1)
            steering = self.steer(steer_drive if self.balanced else steer_drive.clamp_min(0), firing_quanta)
            if self.phase_bits:
                start = _steering_start(steer_drive, self.phase_bits, self.steer.cap)
                if shuffle_phase:
                    start = start.roll(1, dims=-1)
        present = self.relay(drive, firing_quanta)
        coincidence_read = None
        if self.trace is not None:
            initial = None if local_state is None else local_state.trace_charge
            radius = self._trace_radius(steering, steer_drive, firing_quanta)
            coincidence_read, trace_charge = self.trace(present, steering, initial, pulse_counts, write_radius=radius)
            if local_state is not None:
                local_state.trace_charge = trace_charge
        packets, departure = self._departure_packets(present, steering)
        alternate = recall = anchor = None
        if self.selector is None:
            groups = packets.split(self.delay_widths, dim=-1)
            if local_state is None:
                delayed_groups, anchor = self._delayed_groups(groups, steering, pulse_counts)
            else:
                delayed_groups = self._chunk_delays(groups, local_state)
            if self.delay_filters is not None:
                filtered = [base if kernel is None else kernel(base, [_lag(group, tap) for tap in kernel.taps])
                            for group, base, kernel in zip(groups, delayed_groups, self.delay_filters, strict=True)]
                coincidence_read, _ = _steer_packet(torch.cat(filtered, dim=-1), steering)
            if self.correction is not None:
                recall = self.correction(spikes, correction_access)
                if self.memory_residual is None:
                    delayed_groups.append(recall)
            delayed = torch.cat(delayed_groups, dim=-1)
            if self.alternate_delays:
                alternate = torch.cat([_lag(group, delay) for group, delay in
                                       zip(groups, self.alternate_delays, strict=True)], dim=-1)
        else:
            delayed = self.selector(present, delay_limit)
        current = self._current(spikes, present, delayed, steering, start, alternate, recall, quanta,
                                departure=departure, anchor=anchor, coincidence_read=coincidence_read)
        if self.residual_membrane:
            return self.emit(current, state=local_state)
        return self.emit(current)

    def _step_compact(self, packets, steering, state, position):
        # Recycle a packet's slot as soon as its fixed delay has elapsed.
        batch = packets.shape[0]
        if state.history is None:
            state.history = packets.new_zeros(batch, sum(self.delay_sizes), dtype=self.axon_dtype)
        groups = packets.split(self.delay_widths, dim=-1)
        history = state.history.split(self.delay_sizes, dim=-1)
        apertures = steering.split(self.widths, dim=-1)
        previous, slots, base_packets, filtered = [], [], [], []
        for index, (storage, packet, aperture, delay) in enumerate(
            zip(history, groups, apertures, self.local_delays, strict=True)
        ):
            ring = storage.reshape(batch, delay, packet.shape[-1])
            slot = ring[:, position % delay]
            slots.append(slot)
            old = slot.to(packets.dtype)
            base_packets.append(old)
            if self.delay_filters is not None:
                kernel = self.delay_filters[index]
                if kernel is None:
                    filtered.append(old)
                else:
                    taps = [ring[:, (position - tap) % delay].to(packets.dtype) for tap in kernel.taps]
                    filtered.append(kernel(old, taps))
            choices = min(self.delay_window, delay)
            if choices == 1:
                previous.append(old)
            else:
                ages = delay - self.delay_offsets[:choices]
                candidates = ring.index_select(1, (position - ages) % delay).transpose(1, 2).to(packets.dtype)
                query = packet.clamp(-aperture, aperture)
                previous.append(self._match_delay(candidates, query, aperture, position >= ages))
        delayed = torch.cat(previous, dim=-1)
        anchor = torch.cat(base_packets, dim=-1) if self.delay_correction is not None else None
        filtered_packets = torch.cat(filtered, dim=-1) if filtered else None
        for slot, packet in zip(slots, groups, strict=True):
            slot.copy_(packet)
        return delayed, anchor, filtered_packets

    def step(self, spikes, state, position):
        drive = self.expand(spikes)
        quanta = self.packet_thresholds() if self.packet_thresholds is not None else None
        firing_quanta = self._firing_quanta(spikes, quanta)
        steering = start = None
        if self.steered:
            drive, steer_drive = drive.chunk(2, dim=-1)
            steer_charge = steer_drive if self.balanced else steer_drive.clamp_min(0)
            steering, state.steering_phase = self.steer.step(steer_charge, state.steering_phase, firing_quanta)
            if self.phase_bits:
                start = _steering_start(steer_drive, self.phase_bits, self.steer.cap)
        present, state.relay_phase = self.relay.step(drive, state.relay_phase, firing_quanta)
        coincidence_read = None
        if self.trace is not None:
            radius = self._trace_radius(steering, steer_drive, firing_quanta)
            coincidence_read, state.trace_charge = self.trace.step(present, steering, state.trace_charge,
                                                                  write_radius=radius)
        packets, departure = self._departure_packets(present, steering)
        alternate = recall = anchor = None
        if self.compact_delays:
            delayed, anchor, filtered = self._step_compact(packets, steering, state, position)
            if filtered is not None:
                coincidence_read, _ = _steer_packet(filtered, steering)
        else:
            ring_size = self.history_size
            local_width = sum(self.local_widths)
            if state.history is None:
                state.history = present.new_zeros(present.shape[0], ring_size, local_width)
            if self.selector is None:
                previous = []
                neighbors = []
                first = 0
                for group, (width, delay) in enumerate(zip(self.local_widths, self.local_delays, strict=True)):
                    previous.append(state.history[:, (position - delay) % ring_size, first:first + width])
                    if self.alternate_delays:
                        neighbor_at = (position - self.alternate_delays[group]) % ring_size
                        neighbors.append(state.history[:, neighbor_at, first:first + width])
                    first += width
                if self.correction is not None:
                    recall, state.correction = self.correction.step(spikes, state.correction)
                    if self.memory_residual is None:
                        previous.append(recall)
                delayed = torch.cat(previous, dim=-1)
                if neighbors:
                    alternate = torch.cat(neighbors, dim=-1)
            else:
                delayed = self.selector.step(state.history, position)
            state.history[:, position % ring_size].copy_(present[:, :local_width])
        current = self._current(spikes, present, delayed, steering, start, alternate, recall, quanta,
                                departure=departure, anchor=anchor, coincidence_read=coincidence_read)
        spikes, state.output_phase = self.emit.step(current, state.output_phase)
        return spikes


@dataclass
class EventLMState:
    layers: list[_DelayState]
    input_phase: torch.Tensor | None = None
    output_phase: torch.Tensor | None = None
    previous_bytes: torch.Tensor | None = None
    position: int = 0
    prediction_memory: CorrectionState | None = None


@dataclass
class EventChunkState:
    local_prefix: torch.Tensor | None
    correction: CorrectionState | None
    local_layers: list[_DelayState] | None = None


class EventLM(nn.Module):
    """One-hot byte events in, signed output spike counts over 256 bytes out.

    Floating-point master weights use a straight-through training surrogate.
    The CPU export uses only integer tensors to execute the same circuit.
    """

    def __init__(self, cfg: EventLMConfig = EventLMConfig()):
        super().__init__()
        if min(cfg.dim, cfg.depth) < 1 or cfg.hidden < len(_DELAYS[0]):
            raise ValueError("invalid decoder dimensions")
        if cfg.route not in ("coincidence", "tonic", *_STEERED_ROUTES):
            raise ValueError("invalid spike packet route")
        if cfg.continuous and (cfg.route != "steered" or cfg.carry):
            raise ValueError("the continuous control uses the reset-per-byte steered topology")
        if cfg.route in _PREDICTION_ROUTES and cfg.carry:
            raise ValueError("prediction memory requires reset-per-byte local neurons (--no-carry)")
        if (cfg.forecast_key_channels or cfg.match_forecast_key) and cfg.route != "steered_correction_code":
            raise ValueError("forecast keys need the direct forecast bank")
        if not isinstance(cfg.forecast_key_channels, int) or (cfg.match_forecast_key and cfg.forecast_key_channels < 1):
            raise ValueError("matching needs a positive integer number of forecast key channels")
        if cfg.learn_context_keys and (cfg.route != "steered_correction_code" or cfg.forecast_key_channels < 1):
            raise ValueError("learned context keys need reserved channels in the direct forecast bank")
        if cfg.learn_packet_thresholds and (cfg.carry or cfg.continuous or
                                            cfg.route not in ("steered", "steered_correction_code")):
            raise ValueError("learned packet thresholds need a reset-per-byte spike router")
        if cfg.compact_delays and cfg.route not in ("steered", "steered_correction_code"):
            raise ValueError("compact delays need a fixed steered delay schedule")
        if cfg.compact_axon_state and (not cfg.compact_delays or cfg.continuous):
            raise ValueError("compact axon state needs compact delays and integer pulse counts")
        if cfg.departure_coincidence and (not cfg.compact_delays or cfg.carry or cfg.continuous):
            raise ValueError("departure coincidence needs compact delays and reset-per-byte spike neurons")
        if not isinstance(cfg.delay_window, int) or cfg.delay_window < 1:
            raise ValueError("delay_window must be a positive integer")
        if cfg.delay_window > 1 and (not cfg.compact_delays or cfg.carry or cfg.continuous or cfg.departure_coincidence):
            raise ValueError("delay matching needs compact delays and reset-per-byte spike neurons")
        if cfg.delay_residual and cfg.delay_window == 1:
            raise ValueError("delay corrections need at least two candidate taps")
        if cfg.local_trace and (not cfg.compact_delays or cfg.carry or cfg.continuous
                                or cfg.delay_window != 1 or cfg.departure_coincidence):
            raise ValueError("local traces need compact fixed delays and reset-per-byte spike routing")
        if cfg.opponent_steering and not cfg.local_trace:
            raise ValueError("opponent steering needs a local charge trace")
        if cfg.trace_write_credit and not cfg.local_trace:
            raise ValueError("trace write credit needs a local charge trace")
        if cfg.trace_read_veto and not cfg.local_trace:
            raise ValueError("trace read veto needs a local charge trace")
        if cfg.residual_membrane and not cfg.local_trace:
            raise ValueError("residual membranes need chunkable local trace state")
        if cfg.spread_coincidence and (cfg.carry or cfg.continuous
                                       or cfg.route not in ("steered", "steered_correction_code")):
            raise ValueError("spread coincidence needs a reset-per-byte steered spike router")
        if cfg.delay_filter and (not cfg.compact_delays or cfg.carry or cfg.continuous or cfg.delay_window != 1
                                 or cfg.departure_coincidence or cfg.local_trace):
            raise ValueError("delay filters need the fixed-delay coincidence path and reset-per-byte spike routing")
        if not isinstance(cfg.router_charge_bits, int) or not _CHARGE_BITS <= cfg.router_charge_bits < 31:
            raise ValueError("router_charge_bits must be an integer from 6 to 30")
        extra_charge_bits = cfg.router_charge_bits - _CHARGE_BITS
        if extra_charge_bits and (cfg.carry or cfg.continuous or cfg.learn_packet_thresholds
                                  or cfg.route not in ("steered", "steered_correction_code")):
            raise ValueError("finer router charge needs reset-per-byte neurons with fixed firing thresholds")
        if not isinstance(cfg.resolution, int) or cfg.resolution < 1:
            raise ValueError("spike resolution must be a positive integer")
        if cfg.activity_inhibition:
            if (cfg.carry or cfg.continuous or cfg.learn_packet_thresholds
                    or cfg.route not in ("steered", "steered_correction_code")):
                raise ValueError("activity inhibition needs a reset-per-byte router with a fixed threshold floor")
            if cfg.dim * _HIDDEN_CAP * cfg.resolution >= _FP32_EXACT:
                raise ValueError("the inhibition pool would exceed the exact FP32 integer range")
        output_bits = 3
        output_cap = _OUTPUT_CAP * cfg.resolution
        if cfg.output_reference:
            if cfg.carry or cfg.continuous or cfg.route not in ("steered", "steered_correction_code"):
                raise ValueError("output reference needs a reset-per-byte spike router")
            if _VOCAB * output_cap * (1 << output_bits) > _FP32_EXACT:
                raise ValueError("pooled output charge would exceed the exact FP32 integer range")
        projection_width = (4 if cfg.route in _STEERED_ROUTES else 3) * cfg.hidden
        cap = _HIDDEN_CAP * cfg.resolution
        if cfg.opponent_steering and ((cap * _WEIGHT_LIMIT) << cfg.router_charge_bits) > _FP32_EXACT:
            raise ValueError("opponent steering charge would exceed the exact FP32 integer range")
        # The additional read projection is bounded by one delay group's width.
        if cfg.route in _CORRECTION_ROUTES and cfg.route != "steered_correction":
            projection_width += cfg.hidden // len(_STEERED_DELAYS[0])
        if cfg.memory_width is not None:
            if cfg.route not in _PREDICTION_ROUTES:
                raise ValueError("memory_width requires a prediction bank")
            projection_width = max(projection_width, cfg.memory_width)
        unit_drive = (max(projection_width, cfg.dim) * _WEIGHT_LIMIT + _TICKS) * _HIDDEN_CAP
        if unit_drive >= _FP32_EXACT:
            raise ValueError("synaptic accumulator would exceed the exact FP32 integer range")
        packet_units = _PACKET_UNITS if cfg.learn_packet_thresholds else 1
        max_drive = (unit_drive * cfg.resolution * packet_units) << extra_charge_bits
        if max_drive > torch.iinfo(torch.int32).max:
            raise ValueError("synaptic accumulator would exceed the int32 integer range")
        self._check_weights = not cfg.continuous and max_drive >= _FP32_EXACT
        self.cfg = cfg
        self.input = _Synapse(_VOCAB, cfg.dim, gain=0.9, byte_input=True, continuous=cfg.continuous)
        taps = 2 if cfg.route == "convergent" else 0
        self.delayed_input = nn.ModuleList(_Synapse(_VOCAB, cfg.dim, gain=0.5, byte_input=True) for _ in range(taps))
        self.input_emit = _IntegrateFire(cap, _CHARGE_BITS, cfg.carry, continuous=cfg.continuous)
        self.blocks = nn.ModuleList(_DelayBlock(cfg, i) for i in range(cfg.depth))
        self.local_history = len(self.delayed_input) + sum(block.history_size for block in self.blocks)
        self.output = _Synapse(cfg.dim, _VOCAB, gain=0.8, continuous=cfg.continuous)
        self.output_emit = _IntegrateFire(output_cap, output_bits, cfg.carry, continuous=cfg.continuous)
        self.output_reference = None
        if cfg.output_reference:
            self.output_reference = _IntegrateFire(output_cap, output_bits + _VOCAB.bit_length() - 1, False)
        self.prediction_memory = None
        if cfg.route in _PREDICTION_ROUTES:
            width = self.blocks[-1].widths[-1] if cfg.memory_width is None else cfg.memory_width
            self.prediction_memory = _SpikeCorrection(
                cfg.dim, width, cfg.memory_heads, cfg.memory_slots, cfg.resolution, bind_previous=True, observed=True,
                forecast=cfg.route == "steered_correction_code", key_channels=cfg.forecast_key_channels,
                match_key=cfg.match_forecast_key,
            )
            self.recall_synapse = _Synapse(width - cfg.memory_heads * cfg.forecast_key_channels, cfg.dim, gain=0.15)
            self.recall_emit = _IntegrateFire(cap, _CHARGE_BITS, False)

    @property
    def has_correction(self):
        return self.prediction_memory is not None or self.blocks[-1].correction is not None

    def _prediction_code(self, local_spikes):
        if self.cfg.route == "steered_correction_code":
            return self.prediction_memory._value_code(local_spikes)
        if self.cfg.route != "steered_correction_error":
            return None
        # The local winner is encoded by the same input synapses as an
        # observed byte. A later recall never contributes to the write.
        guess = self.byte(self.output_emit(self.output(local_spikes)))
        return self.prediction_memory._value_code(self.input_emit(self.input(guess) * self.cfg.resolution))

    def _apply_recall(self, local_spikes, recall):
        current = self.recall_synapse(recall)
        if self.prediction_memory.forecast is not None:
            # The original spikes and the correction share the output
            # synapses. Only the correction needs to fire here.
            cap = self.recall_emit.cap
            return (local_spikes + self.recall_emit(current)).clamp(-cap, cap)
        return self.recall_emit(local_spikes * _TICKS + current)

    def _reference_counts(self, current):
        if self.output_reference is None:
            return None
        limit = self.output_emit.threshold * self.output_emit.cap
        pooled = current.clamp(-limit, limit).sum(-1, keepdim=True, dtype=current.dtype)
        return self.output_reference(pooled)

    @torch.no_grad()
    def _check_accumulator(self):
        # High-resolution training uses the actual quantized weights to bound
        # the magnitude of every partial sum. Every input count is at most cap.
        def row_sums(synapse):
            return synapse.weight_ticks().abs().sum(-1)

        bounds = [row_sums(self.output).amax()]
        for block in self.blocks:
            bounds.append(row_sums(block.expand).amax())
            if block.delay_filters is not None:
                bounds.extend(2 * row_sums(kernel).amax() for kernel in block.delay_filters if kernel is not None)
            if block.trace is not None and block.trace.read_veto is not None:
                bounds.append(2 * block.trace.read_veto.gains().amax())
            packet_units = _PACKET_UNITS if block.packet_thresholds is not None else 1
            projected = row_sums(block.project) * packet_units + block.emit.threshold
            if block.correction is not None:
                bounds.append(row_sums(block.correction.encode).amax())
            if block.memory_residual is not None:
                projected = projected + row_sums(block.memory_residual)
            bounds.append(projected.amax())
        if self.prediction_memory is not None:
            bounds.append(row_sums(self.prediction_memory.encode).amax())
            bounds.append(row_sums(self.prediction_memory.code).amax())
            bounds.append(row_sums(self.recall_synapse).amax() + _TICKS)
        limit = _FP32_EXACT // self.input_emit.cap
        if not torch.stack(bounds).amax().item() <= limit:
            raise ValueError("quantized weights exceed the exact FP32 accumulator range at this spike resolution")

    def forward(self, tokens, *, delay_limit=None, shuffle_phase=False, correction_access=None,
                return_prediction=False, memory_pulses=None):
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("expected a nonempty [batch, bytes] tensor")
        if delay_limit is not None and (delay_limit < 1 or self.cfg.route not in _DELAY_ROUTES):
            raise ValueError("delay_limit needs a positive value and a selectable delay line")
        if shuffle_phase and self.cfg.route not in _PHASE_ROUTES:
            raise ValueError("shuffle_phase needs timed steering")
        if return_prediction and self.cfg.route != "steered_correction_code":
            raise ValueError("return_prediction needs a spike forecast")
        if self._check_weights and self.input.weight.is_floating_point():
            self._check_accumulator()
        drive = self.input(tokens)
        for delay, synapse in enumerate(self.delayed_input, 1):
            padding = min(delay, tokens.shape[1])
            delayed = synapse(tokens[:, :tokens.shape[1] - padding])
            drive = drive + F.pad(delayed, (0, 0, padding, 0))
        observed = self.input_emit(drive * self.cfg.resolution)
        spikes = observed
        local_layers = None if correction_access is None else correction_access.local_layers
        for index, block in enumerate(self.blocks):
            spikes = block(spikes, delay_limit=delay_limit, shuffle_phase=shuffle_phase,
                           correction_access=correction_access, pulse_counts=memory_pulses,
                           local_state=None if local_layers is None else local_layers[index])
        if self.prediction_memory is not None:
            prediction = self._prediction_code(spikes)
            recall = self.prediction_memory(
                spikes, correction_access, observed=observed, prediction_code=prediction, pulse_counts=memory_pulses,
            )
            spikes = self._apply_recall(spikes, recall)
        current = self.output(spikes)
        counts = self.output_emit(current, reference=self._reference_counts(current))
        if return_prediction:
            return counts, prediction, self.prediction_memory.observation_code(observed)
        return counts

    def forward_chunk(self, tokens, state=None, *, hide_age=None, stats=None, shuffle_keys=False):
        """Carry the bank and either replay local context or carry the trace and delays.

        In the post-router bank variants, a tuple of ages adds a leading
        axis of read conditions. All conditions share one write trajectory.
        """
        if self.cfg.carry or not (self.has_correction or self.cfg.local_trace):
            raise ValueError("chunked memory needs reset-per-byte local neurons")
        if hide_age is not None and not self.has_correction:
            raise ValueError("hiding by age needs a correction bank")
        if shuffle_keys and not self.cfg.match_forecast_key:
            raise ValueError("shuffling memory keys requires a bank that matches stored keys")
        ages = hide_age if isinstance(hide_age, tuple) else (hide_age,)
        if isinstance(hide_age, tuple) and (self.prediction_memory is None or not ages):
            raise ValueError("a group of memory ages needs a post-router bank")
        if tokens.ndim != 2 or tokens.shape[1] == 0 or any(age is not None and age < 1 for age in ages):
            raise ValueError("expected nonempty [batch, bytes] and a positive age when hiding old state")
        correction = None if state is None else state.correction
        if self.cfg.local_trace:
            if state is None:
                layers = [_DelayState() for _ in self.blocks]
            else:
                previous = state.local_layers
                if (previous is None or len(previous) != len(self.blocks) or previous[0].history is None
                        or previous[0].history.shape[0] != tokens.shape[0]):
                    raise ValueError("chunk state must match the trace decoder and batch size")
                layers = [copy.copy(layer) for layer in previous]
            access = _CorrectionAccess(correction, 0, hide_age, stats, shuffle_keys, local_layers=layers)
            counts = self(tokens, correction_access=access)
            return counts, EventChunkState(None, access.state, layers)
        if state is not None and state.local_prefix is None:
            raise ValueError("chunk state must contain the local byte prefix")
        prefix = 0 if state is None else state.local_prefix.shape[1]
        if state is not None:
            tokens = torch.cat((state.local_prefix, tokens), dim=1)
        access = _CorrectionAccess(correction, prefix, hide_age, stats, shuffle_keys)
        counts = self(tokens, correction_access=access)
        return counts[..., prefix:, :], EventChunkState(tokens[:, -self.local_history:].clone(), access.state)

    @torch.no_grad()
    def to_integer(self):
        """Export an eval-mode CPU decoder with int32 synapses."""
        if self.cfg.continuous:
            raise ValueError("the continuous control has no integer export")
        model = copy.deepcopy(self).cpu().eval()
        for submodule in model.modules():
            if isinstance(submodule, (_Synapse, _DelaySelector, _PacketThresholds, _DelayCorrection, _DelayFilter,
                                      _OpponentSteering, _SpreadCoincidence)):
                submodule.freeze_integer()
        return model

    @torch.no_grad()
    def step(self, tokens, state: EventLMState | None = None):
        """Apply one byte per stream using bounded integer state."""
        if self.input.weight.is_floating_point() or self.training:
            raise ValueError("step requires an eval-mode integer export from to_integer()")
        if tokens.ndim != 1 or tokens.device.type != "cpu":
            raise ValueError("step expects a CPU tensor of shape [batch]")
        if state is None:
            state = EventLMState([_DelayState() for _ in self.blocks])
        else:
            history = state.layers[0].history if len(state.layers) == len(self.blocks) else None
            if history is None or history.shape[0] != tokens.shape[0]:
                raise ValueError("streaming state must belong to the same decoder and batch size")
        drive = self.input(tokens)
        if self.delayed_input:
            taps = len(self.delayed_input)
            if state.previous_bytes is None:
                state.previous_bytes = tokens.new_zeros((tokens.shape[0], taps))
            for delay, synapse in enumerate(self.delayed_input, 1):
                if state.position >= delay:
                    drive = drive + synapse(state.previous_bytes[:, (state.position - delay) % taps])
            state.previous_bytes[:, state.position % taps].copy_(tokens)
        observed, state.input_phase = self.input_emit.step(drive * self.cfg.resolution, state.input_phase)
        spikes = observed
        for block, layer in zip(self.blocks, state.layers):
            spikes = block.step(spikes, layer, state.position)
        if self.prediction_memory is not None:
            recall, state.prediction_memory = self.prediction_memory.step(
                spikes, state.prediction_memory, observed=observed, prediction_code=self._prediction_code(spikes),
            )
            spikes = self._apply_recall(spikes, recall)
        current = self.output(spikes)
        counts, state.output_phase = self.output_emit.step(
            current, state.output_phase, reference=self._reference_counts(current),
        )
        state.position += 1
        return counts, state

    @staticmethod
    def byte(counts):
        """Decode the winning byte directly from its spike evidence."""
        return counts.argmax(dim=-1)
