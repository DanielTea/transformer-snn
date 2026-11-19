"""Byte decoder with fixed-point synapses and coincidence-routed spike packets."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

from .coasting_heads import CoastingHeads, HeadState
from .fast_synapses import FastSynapses, SynapseState
from .neurons import integer_spike
from .correction_memory import (
    CorrectionState, _scan_charge, _scan_translated_charge, correction_reads, correction_step, interval_spikes,
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
_TRACE_TAP_COUNT = 4
_DELAY_ROUTES = ("steered_static", "steered_feedback", "steered_margin")
_PHASE_ROUTES = ("steered_timed", "steered_phase_swap", "steered_phase_delay")
_CORRECTION_ROUTES = ("steered_correction", "steered_correction_residual", "steered_correction_binding")
_PREDICTION_ROUTES = ("steered_correction_observed", "steered_correction_error", "steered_correction_code")
_STEERED_ROUTES = (
    "steered", "steered_local", "convergent", "steered_balanced",
    *_CORRECTION_ROUTES, *_PREDICTION_ROUTES, *_PHASE_ROUTES, *_DELAY_ROUTES,
)
_TRACE_READ_DRIVE_SOURCES = ("write", "present", "write-rails", "present-rails")
_TRACE_ARRIVAL_MODES = ("selected", "unmatched")
_CLOCK_SOURCES = ("surprise", "periodic")
_READ_MODES = ("soft", "race", "age")
# The most words a state read counts a cell's age in.
_READ_BEATS = 16
_READ_CLOCKS = ("word", "end", "learned")
# Signed probe and tag pulses give a cell this many times ``resolution`` root channels of charge spread at the start.
_RACE_UNIT = 1.5
# Multiples of the golden ratio spread the starting phases of a race's cells evenly.
_GOLDEN = (math.sqrt(5) - 1) / 2
# The kinds of slow synapses a granule cluster can have. The slowest then passes a 256th of its charge on per byte.
_GRANULE_TRACES = 4
# Bytes over which a slow synapse's held charge is summed in one piece.
_TRACE_STRETCH = 512
# The most spikes of resting charge a granule cell can hold below its threshold.
_GRANULE_REST_LIMIT = 16
# The most thresholds of charge that the gate dendrite of a gated cell answers with spikes, before inhibition.
_GATE_CAP = 16
# What opens a new stretch on a granule cluster's beat fibres, and what such a fibre brings of a stretch.
_GRANULE_CLOCKS = ("word", "end", "every", "learned")
_GRANULE_HOLDS = ("mean", "sum", "last")
# The most closed stretches a cluster's beat fibres can hold.
_GRANULE_BEATS = 8
# A clock cell's threshold is this many bits above a stream neuron's. A clock hears the whole stream, and against
# a lower threshold one step of all its synapses together would carry it from never beating to always beating.
_CLOCK_BITS = 4
# The clock cell of a read's head hears the read's tags, and its threshold is this many bits above a stream neuron's.
_READ_CLOCK_BITS = 4
# The most pairs of pacemaker cells a state read can hear, the largest count of a pacemaker in spikes of either
# sign, and the shortest period of a bank in bytes. The longest period is twice the read window.
_READ_PHASES = 64
_PHASE_SPIKES = 64
_PHASE_SHORTEST = 3.0
# The synapses from the pacemakers start with this gain. At a quarter the pacemakers charge a probe or tag neuron
# about as much as the stream does, and the heads start on a few cells picked by chance. At an eighth the reads
# start nearly as broad as without pacemakers, and the synapses still hold several ticks each.
_PHASE_GAIN = 0.125
# The pacemakers that turn the probes and tags of a read: the fastest turns by one radian a byte, and with ever
# more pairs the slowest would turn this many times slower.
_TURN_SHORTEST = 2 * math.pi
_TURN_SPAN = 10_000.0


def _delay_group_widths(channels, groups):
    width, extra = divmod(channels, groups)
    return tuple(width + (group < extra) for group in range(groups))


def _spike_storage_dtype(cap):
    return next(dtype for dtype in (torch.int8, torch.int16, torch.int32) if cap <= torch.iinfo(dtype).max)


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
    inhibition_by_expansion: bool = False
    local_trace: bool = False
    trace_write_credit: bool = False
    bank_write_credit: bool = False
    trace_read_veto: bool = False
    delay_filter: bool = False
    opponent_steering: bool = False
    spread_coincidence: bool = False
    residual_membrane: bool = False
    compact_axon_state: bool = False
    tag_phase_address: bool = False
    trace_coincidence_start: int | None = None
    trace_write_confirm: bool = False
    steering_gradient_at_zero: bool = False
    redistribute_inhibition: bool = False
    tag_read_competition: bool = False
    late_steering: bool = False
    residual_correction_start: int | None = None
    trace_integration: bool = False
    learn_inhibition_thresholds: bool = False
    inhibition_threshold_bits: int = _PACKET_BITS
    trace_tap_read: str | None = None
    compact_trace_state: bool = False
    steering_transfer: str | None = None
    delay_group_widths: tuple[int, ...] | None = None
    trace_read_drive: str | None = None
    trace_read_drive_bits: int = _CHARGE_BITS
    trace_read_drive_carry: bool = False
    trace_read_share_steering: bool = False
    trace_arrival: str | None = None
    trace_arrival_groups: int = 2
    clock_start: int | None = None
    clock_threshold: int = 0
    clock_leak: int = 0
    clock_source: str = "surprise"
    synapse_orders: tuple[int, ...] | None = None
    synapse_groups: int = 4
    synapse_cells: int = 4096
    synapse_levels: int = 4
    synapse_gate: int = 1
    head_orders: tuple[int, ...] | None = None
    head_cells: int = 4096
    head_gate: int = 1
    head_tape: int = 8192
    head_lookup: int = 0
    head_miss: int = 0
    head_relock: int = 0
    read_after: tuple[int, ...] | None = None
    read_width: int = 256
    read_heads: int = 4
    read_window: int = 256
    read_mode: str = "soft"
    read_levels: int = 4
    read_resolution: int | None = None
    read_signed: bool = True
    read_budget: int | None = None
    read_address_rate: float = 1.0
    read_synapse_gain: float = 1.0
    read_beats: int = 0
    read_beat_start: float = 0.0
    read_clock: str = "word"
    read_clock_rate: float = 0.25
    read_marks: bool = False
    granule_after: tuple[int, ...] | None = None
    granule_clusters: int = 32
    granule_cells: int = 256
    granule_fan: int = 128
    granule_reach: int | None = None
    granule_gain: float = 0.25
    granule_rest: float = 0.5
    granule_price: float = 0.0
    granule_bits: int | None = None
    granule_budget: int | None = None
    granule_races: int = 1
    granule_soften: float = 1.0
    granule_soft: bool = False
    granule_slope: bool = False
    granule_delays: int = 1
    granule_relays: int = 0
    granule_traces: int = 0
    granule_beats: int = 0
    granule_clock: str = "word"
    granule_hold: str = "sum"
    granule_clock_rate: float = 0.25
    stack: bool = False
    granule_cap: int | None = None
    granule_gate: int = 0
    read_cap: int | None = None
    read_phases: int = 0
    read_turn: bool = False
    # An output neuron fires up to this many eighths of a bit of evidence for or against its byte.
    output_cap: int = _OUTPUT_CAP

    @property
    def group_widths(self) -> tuple[int, ...]:
        if self.delay_group_widths is not None:
            return tuple(self.delay_group_widths)
        return _delay_group_widths(self.hidden, len(_STEERED_DELAYS[0]))

    @property
    def trace_arrival_width(self) -> int:
        return sum(self.group_widths[:self.trace_arrival_groups]) if self.trace_arrival is not None else 0

    @property
    def inhibition_bits(self) -> int:
        population = 2 * self.hidden if self.inhibition_by_expansion else self.dim
        return ((population * self.resolution - 1) // _PACKET_UNITS).bit_length()


def _weight_ticks(weight, quantized=True, *, charge_bits=_CHARGE_BITS):
    if not weight.is_floating_point():
        return weight
    limit = _WEIGHT_LIMIT << (charge_bits - _CHARGE_BITS)
    ticks = (weight * (1 << charge_bits)).clamp(-limit, limit)
    return ticks.detach().round() + (ticks - ticks.detach()) if quantized else ticks


def _unrounded_sums(weight, charge_bits):
    """The summed sizes of each neuron's synapses in weight ticks, before the ticks are rounded."""
    limit = _WEIGHT_LIMIT << (charge_bits - _CHARGE_BITS)
    return (weight * (1 << charge_bits)).clamp(-limit, limit).abs().sum(-1)


def _shrink_synapses(weight, factor, charge_bits):
    """Shrink in place all synapses of the neurons whose ``factor`` is below one, and count those neurons.

    ``weight`` holds one neuron's synapses along its last axis. A synapse beyond the largest weight counts as the
    largest weight. The synapses of every other neuron stay as they are, bit for bit. The count is a tensor, so
    that a pass over a whole decoder waits for no result on a graphics processor.
    """
    over = (factor < 1).unsqueeze(-1)
    limit = (_WEIGHT_LIMIT << (charge_bits - _CHARGE_BITS)) / (1 << charge_bits)
    weight.copy_(torch.where(over, weight.clamp(-limit, limit) * factor.unsqueeze(-1), weight))
    return over.sum()


def emitted_pulses(population, counts):
    """A firing population's pulses for a pulse budget, whose price pushes on them with the population's share."""
    total = counts.abs().sum()
    share = population.price_share
    if share == 1:
        return total
    held = total.detach()
    return held if share == 0 else held + share * (total - held)


def _unit_gain_ticks(weight):
    ticks = _weight_ticks(weight)
    if not ticks.is_floating_point():
        return ticks
    # Keep a gradient at the silent endpoint.
    return ticks.detach().clamp(0, _TICKS) + (ticks - ticks.detach())


def _weights_on_charge_grid(weight, bits):
    return _weight_ticks(weight, charge_bits=bits) / (1 << bits)


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

    # The share of a pulse budget's price that pushes on this population. Its pulses count in full either way.
    price_share = 1.0

    def __init__(self, cap: int, charge_bits: int, carry: bool, *, continuous: bool = False,
                 quanta_per_threshold: int = _PACKET_UNITS):
        super().__init__()
        self.cap = cap
        self.bits = charge_bits
        self.threshold = 1 << charge_bits
        self.carry = carry
        self.continuous = continuous
        self.quanta_per_threshold = quanta_per_threshold

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
            quantum = self.threshold // self.quanta_per_threshold
            threshold = units * quantum
            charge = bounded.to(torch.int32)
            magnitude = (charge.abs() + threshold // 2) // threshold
            limit = (self.cap * self.quanta_per_threshold) // units
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

    def __init__(self, cap, bits, *, phase_field="output_phase"):
        super().__init__(cap, bits, True)
        if (cap + 1) * self.threshold - 1 > torch.iinfo(torch.int32).max:
            raise ValueError("residual charge exceeds the int32 membrane range")
        self.phase_dtype = next(dtype for dtype in (torch.uint8, torch.int16, torch.int32)
                                if self.threshold - 1 <= torch.iinfo(dtype).max)
        self.phase_field = phase_field

    def forward(self, current, *, state=None):
        if (current.shape[1] * self.cap + 1) * self.threshold - 1 > torch.iinfo(torch.int32).max:
            raise ValueError("byte sequence exceeds the int32 residual charge scan")
        limit = self.cap * self.threshold
        bounded = current.clamp(-limit, limit)
        phase = self.threshold // 2
        previous = None if state is None else getattr(state, self.phase_field)
        if previous is not None:
            phase = previous[:, None].to(torch.int32)
        total = bounded.to(torch.int32).cumsum(1, dtype=torch.int32) + phase
        emitted = total >> self.bits
        counts = emitted.diff(dim=1, prepend=torch.zeros_like(emitted[:, :1]))
        if state is not None:
            setattr(state, self.phase_field, (total[:, -1] & (self.threshold - 1)).to(self.phase_dtype))
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

    def __init__(self, widths, quanta_per_threshold=_PACKET_UNITS):
        super().__init__()
        self.relative = nn.Parameter(torch.ones(len(widths)))
        self.quanta_per_threshold = quanta_per_threshold
        groups = torch.arange(len(widths)).repeat_interleave(torch.tensor(widths))
        self.register_buffer("groups", groups, persistent=False)

    def quanta(self):
        if not self.relative.is_floating_point():
            return self.relative
        units = self.relative.clamp(0.5, 2.0) * self.quanta_per_threshold
        return units.detach().round() + (units - units.detach())

    def forward(self):
        return self.quanta().index_select(0, self.groups)

    def freeze_integer(self):
        if self.relative.is_floating_point():
            self.relative = nn.Parameter(self.quanta().to(torch.int32), requires_grad=False)


class _InhibitionRedistribution(nn.Module):
    """Transfer inhibitory pulses between pairs of delay groups."""

    def __init__(self, dim, widths, pool):
        super().__init__()
        self.synapse = _Synapse(dim, len(widths) // 2, gain=0.0)
        self.emit = _IntegrateFire(pool.cap, pool.bits, False)
        groups = torch.arange(len(widths)).repeat_interleave(torch.tensor(widths))
        self.register_buffer("groups", groups, persistent=False)

    def forward(self, spikes, pooled):
        limit = pooled * self.emit.threshold
        transfer = self.emit(self.synapse(spikes).clamp(-limit, limit))
        # Pair the first and last groups, then the two middle groups.
        inhibition = pooled + torch.cat((transfer, -transfer.flip(-1)), dim=-1)
        return inhibition.index_select(-1, self.groups)


class _DelayCorrection(nn.Module):
    """Emit scaled corrections with one conductance per channel group."""

    def __init__(self, widths, cap, *, input_bits=0, quanta_per_threshold=_PACKET_UNITS):
        super().__init__()
        self.gain = nn.Parameter(torch.zeros(len(widths)))
        groups = torch.arange(len(widths)).repeat_interleave(torch.tensor(widths))
        self.register_buffer("groups", groups, persistent=False)
        self.emit = _IntegrateFire(2 * cap, _CHARGE_BITS + input_bits, False,
                                   quanta_per_threshold=quanta_per_threshold)

    def gains(self):
        return _unit_gain_ticks(self.gain)

    def freeze_integer(self):
        if self.gain.is_floating_point():
            self.gain = nn.Parameter(self.gains().to(torch.int32), requires_grad=False)

    def forward(self, anchor, chosen):
        return self.emit_difference(chosen - anchor)

    def emit_difference(self, difference, quanta=None):
        return self.emit(difference * self.gains().index_select(0, self.groups), quanta)


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


def _first_arrival(candidates, latencies, surrogate_scale=None):
    index = latencies.argmin(-1, keepdim=True)
    packet = candidates.gather(-1, index.unsqueeze(-2).expand_as(candidates[..., :1])).squeeze(-1)
    if surrogate_scale is not None:
        weights = (-latencies / surrogate_scale).softmax(-1)
        surrogate = (candidates.detach() * weights.unsqueeze(-2)).sum(-1)
        packet = packet + (surrogate - surrogate.detach())
    return packet, index.squeeze(-1)


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
    local_position: int = 0
    clock: ClockState | None = None
    synapses: SynapseState | None = None
    heads: HeadState | None = None


class _SpikeCorrection(nn.Module):
    """An address spike probes a bank and enables a bounded correction."""

    def __init__(self, dim, width, heads, slots, resolution, bind_previous, *, observed=False, forecast=False,
                 key_channels=0, match_key=False, tag_phase_address=False, write_credit=False,
                 tag_read_competition=False):
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
        self.write_credit = write_credit
        self.encode = _Synapse(dim, heads * (slots + 1) + (0 if observed else width), gain=1.0)
        self.code = _Synapse(dim, width, gain=0.25 if forecast else 1.0) if observed else None
        # A single byte afferent contributes at most two counts per unit
        # of resolution. Forecast the same code within that range.
        self.forecast = _IntegrateFire(2 * resolution, _CHARGE_BITS, False) if forecast else None
        self.address = _IntegrateFire(_HIDDEN_CAP * resolution, _CHARGE_BITS - 2, False)
        self.value = _IntegrateFire(_HIDDEN_CAP * resolution, _CHARGE_BITS, False)
        self.read_inhibition = _DelayCorrection((1,) * heads, self.forecast.cap) if tag_read_competition else None
        self.register_buffer("slot_ids", torch.arange(slots), persistent=False)
        self.register_buffer("read_slot_ids", torch.arange(slots << key_channels)
                             if tag_phase_address and tag_read_competition else None, persistent=False)
        phase_negative = None
        if tag_phase_address:
            phases = torch.arange(1 << key_channels)
            phase_negative = ((phases[:, None] >> torch.arange(key_channels)) & 1).bool()
        self.register_buffer("phase_negative", phase_negative, persistent=False)

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
        tolerance = self.value(tolerance.clamp_min(0)).unsqueeze(-1)
        return counts, target, tolerance

    def _compete(self, counts, inhibition=None):
        preference = eligible = counts
        slot_ids = self.slot_ids
        if inhibition is not None:
            preference = counts - inhibition
            lower_bound = -self.read_inhibition.emit.cap
            eligible = preference.masked_fill(counts <= 0, lower_bound - 1)
            if self.read_slot_ids is not None:
                slot_ids = self.read_slot_ids
        winner = eligible.argmax(-1, keepdim=True)
        active = counts.amax(-1, keepdim=True).clamp(0, 1)
        address = (winner == slot_ids).to(counts.dtype) * active
        if self.training and counts.is_floating_point():
            # One address fires in the forward pass. Gradients can also
            # reach runner-up neurons so the route can change during training.
            probabilities = (preference / (2 * self.resolution)).softmax(-1)
            address = address + active.detach() * (probabilities - probabilities.detach())
        return address

    @staticmethod
    def _tag_excess(key, query, tolerance):
        outside = interval_spikes(key, query - tolerance, query + tolerance)
        return outside.abs().amax(-1, keepdim=True)

    def _read_ranking_loss(self, cells, counts, inhibition, error):
        values = cells[:, :-1, ..., self.key_channels:].detach() / self.resolution
        desired = (error.unsqueeze(-2) / self.resolution).expand_as(values)
        eligible = counts[:, :-1] > 0
        floor = torch.finfo(values.dtype).min
        with torch.no_grad():
            utility = -F.smooth_l1_loss(values, desired, reduction="none").mean(-1)
            targets = utility.masked_fill(~eligible, floor).softmax(-1)
        preference = (counts[:, :-1].detach() - inhibition[:, :-1]) / (2 * self.resolution)
        log_probs = preference.masked_fill(~eligible, floor).log_softmax(-1)
        present = eligible.any(-1)
        loss = -(targets * log_probs).sum(-1)
        return (loss * present).sum() / present.sum().clamp_min(1)

    def _addresses(self, counts, prediction, tolerance, stats=None, ranking=None):
        address = self._phase_address(self._compete(counts), prediction)
        if self.read_inhibition is None:
            return address, None

        counts = self._phase_address(counts, prediction)
        query = prediction[..., :self.key_channels].unsqueeze(-2)

        def select_read(cells):
            keys = cells[..., :self.key_channels]
            excess = self._tag_excess(keys, query, tolerance.unsqueeze(-2)).squeeze(-1)
            excess = excess * (counts > 0)
            inhibition = self.read_inhibition.emit_difference(excess.transpose(-1, -2)).transpose(-1, -2)
            selected = self._compete(counts, inhibition)
            if ranking is not None:
                error, losses = ranking
                losses.append(self._read_ranking_loss(cells, counts, inhibition, error))
            if stats is not None:
                stats["rerouted_read_heads"] = (selected != address).any(-1).sum(dtype=torch.int64)
                stats["bank_read_inhibition_spikes"] = inhibition.detach().sum(dtype=torch.int64)
            return selected

        return address, select_read

    def _write_target(self, observed, prediction, tolerance):
        error = (observed - prediction).clamp(-self.value.cap, self.value.cap)
        if not self.key_channels:
            return error, tolerance
        # The key records the forecast that produced the error. A
        # zero-width interval preserves that key exactly.
        key = prediction[..., :self.key_channels]
        payload = error[..., self.key_channels:]
        return torch.cat((key, payload), dim=-1), torch.cat((torch.zeros_like(key), tolerance.expand_as(payload)), dim=-1)

    def _phase_address(self, address, prediction):
        if self.phase_negative is None:
            return address
        # Each negative tag rail delays the same address spike by its slot interval.
        tag = prediction[..., :self.key_channels]
        negative = (tag < 0).to(address.dtype)
        if self.training and negative.is_floating_point():
            surrogate = (0.5 - tag / (2 * self.forecast.cap)).clamp(0, 1)
            negative = negative + (surrogate - surrogate.detach())
        negative = negative.unsqueeze(-2)
        matching = torch.where(self.phase_negative, negative, 1 - negative)
        phase = matching.prod(-1, dtype=address.dtype)
        return (address.unsqueeze(-1) * phase.unsqueeze(-2)).flatten(-2)

    def _read_payload(self, read, prediction, tolerance, address, stats=None, pulse_counts=None):
        if not self.key_channels:
            return read
        key, payload = read[..., :self.key_channels], read[..., self.key_channels:]
        if not self.match_key:
            return payload
        query = prediction[..., :self.key_channels]
        # The inhibitory packets share a rail. Overlapping pulses make
        # one veto, and coincidence removes the same slots from the read.
        veto = self._tag_excess(key, query, tolerance) * address.amax(-1, keepdim=True)
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

    def forward(self, spikes, access, *, observed=None, prediction_code=None, pulse_counts=None, read_losses=None):
        prefix = access.prefix if access is not None else 0
        state = None if access is None else access.state
        observation = None if observed is None else observed[:, prefix:]
        counts, target, tolerance = self._events(spikes[:, prefix:], observation)
        read_tolerance = tolerance
        prediction = None if prediction_code is None else prediction_code[:, prefix:]
        stats = None if access is None else access.stats
        ranking = None
        if read_losses is not None:
            error = target[:, 1:, ..., self.key_channels:] - prediction[:, :-1, ..., self.key_channels:]
            ranking = (error.detach(), read_losses)
        address, select_read = self._addresses(counts, prediction, tolerance, stats, ranking)
        if prediction_code is not None:
            previous = None if state is None else state.previous_prediction
            first = torch.zeros_like(prediction[:, :1]) if previous is None else previous[:, None]
            previous = torch.cat((first, prediction[:, :-1]), dim=1)
            target, tolerance = self._write_target(target, previous, tolerance)
        read, state = correction_reads(
            address, target, tolerance, self.value.cap, state, bind_previous=self.bind_previous,
            hide_age=None if access is None else access.hide_age, stats=stats,
            key_channels=self.key_channels, shuffle_keys=access is not None and access.shuffle_keys,
            pulse_counts=pulse_counts, write_credit=self.write_credit, select_read=select_read,
        )
        if access is not None and access.stats is not None and self.phase_negative is not None:
            by_phase = address.detach().unflatten(-1, (self.slots, -1))
            for index, count in enumerate(by_phase.sum((0, 1, 2, 3), dtype=torch.int64)):
                access.stats[f"phase_{index}_address_events"] = count
        if access is not None:
            if prediction_code is not None:
                state.previous_prediction = prediction[:, -1].clone()
            access.state = state
        read = self._read_payload(read, prediction, read_tolerance, address, stats, pulse_counts)
        if pulse_counts is not None:
            pulse_counts.append(read.abs().sum())
        return F.pad(read.flatten(-2), (0, 0, prefix, 0)) if prefix else read.flatten(-2)

    def step(self, spikes, state, *, observed=None, prediction_code=None):
        counts, target, tolerance = self._events(spikes, observed)
        address, select_read = self._addresses(counts, prediction_code, tolerance)
        read_tolerance = tolerance
        if prediction_code is not None and state is not None:
            target, tolerance = self._write_target(target, state.previous_prediction, tolerance)
        read, state = correction_step(
            address, target, tolerance, self.value.cap, state, bind_previous=self.bind_previous, select_read=select_read,
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
    trace_read_phase: torch.Tensor | None = None


class _TraceInput(nn.Module):
    """Emit signed input current into each held-charge cell."""

    def __init__(self, width, cap):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, width))
        self.emit = _IntegrateFire(cap, _CHARGE_BITS, False)

    def weight_ticks(self):
        return _weight_ticks(self.weight)

    def freeze_integer(self):
        if self.weight.is_floating_point():
            self.weight = nn.Parameter(self.weight_ticks().to(torch.int32), requires_grad=False)

    def forward(self, present, delayed):
        immediate, arriving = self.weight_ticks().unbind()
        return self.emit(immediate * present + arriving * delayed)


class _ReadSteeringShare(nn.Module):
    """Match overflow requests to spare steering within each delay group."""

    def __init__(self, widths, cap, bits):
        super().__init__()
        self.widths = widths
        self.request = _IntegrateFire(cap, bits, False)
        starts, ends = [], []
        first = 0
        for width in widths:
            starts.extend([first] * width)
            first += width
            ends.extend([first] * width)
        self.register_buffer("group_start", torch.tensor(starts, dtype=torch.long), persistent=False)
        self.register_buffer("group_end", torch.tensor(ends, dtype=torch.long), persistent=False)

    def forward(self, current, aperture):
        threshold = self.request.threshold
        charge = aperture * threshold
        magnitude = current.abs()
        overflow = (magnitude - charge).clamp_min(0)
        overflow = overflow.minimum(self.request.cap * threshold - charge)
        # Request a whole aperture pulse even for a fractional overflow.
        requests = self.request(overflow + overflow.sign() * (threshold // 2 - 1))
        slack = (charge - magnitude).clamp_min(0)
        spare = slack.to(torch.int32) >> self.request.bits
        if slack.is_floating_point():
            surrogate = slack / threshold
            spare = spare.to(slack.dtype) + (surrogate - surrogate.detach())

        # The two pulse trains meet in their group's common pulse order.
        runs = torch.stack((requests, spare), dim=-1)
        prefix = F.pad(runs, (0, 0, 1, 0)).cumsum(dim=-2, dtype=runs.dtype)
        start = prefix.index_select(-2, self.group_start)
        earlier_wants, earlier_spare = (prefix[..., :-1, :] - start).unbind(-1)
        total_wants, total_spare = (prefix.index_select(-2, self.group_end) - start).unbind(-1)
        take = (total_spare - earlier_wants).clamp_min(0).minimum(requests)
        give = (total_wants - earlier_spare).clamp_min(0).minimum(spare)
        return aperture + (take - give)


class _TraceReadDrive(nn.Module):
    """Send one more spike input to each existing held-charge read neuron."""

    def __init__(self, width, cap, source, *, bits=_CHARGE_BITS, carry=False, steering_widths=None):
        super().__init__()
        self.from_write = source.startswith("write")
        self.separate_rails = source.endswith("-rails")
        shape = (2, width) if self.separate_rails else (width,)
        self.weight = nn.Parameter(torch.zeros(shape))
        self.emit = (_ResidualMembraneFire(cap, bits, phase_field="trace_read_phase") if carry
                     else _IntegrateFire(cap, bits, False))
        self.steering_share = (_ReadSteeringShare(steering_widths, cap, bits)
                               if steering_widths is not None else None)

    def weight_ticks(self):
        return _weight_ticks(self.weight, charge_bits=self.emit.bits)

    def freeze_integer(self):
        if self.weight.is_floating_point():
            self.weight = nn.Parameter(self.weight_ticks().to(torch.int32), requires_grad=False)

    def drive_ticks(self, present, correction):
        source = correction if self.from_write else present
        weights = self.weight_ticks()
        if self.separate_rails:
            positive, negative = weights.unbind(0)
            return positive * source.clamp_min(0) + negative * (-source).clamp_min(0)
        return weights * source

    def forward(self, read, steering, present, correction, *, state=None):
        current = read * self.emit.threshold + self.drive_ticks(present, correction)
        aperture = steering.clamp(max=self.emit.cap)
        if self.steering_share is not None:
            aperture = self.steering_share(current, aperture)
        limit = aperture * self.emit.threshold
        current = current.clamp(-limit, limit)
        if not self.emit.carry:
            return self.emit(current)
        if current.ndim == 2:
            return self.emit(current[:, None], state=state)[:, 0]
        return self.emit(current, state=state)


class _SpikeTrace(nn.Module):
    """A causal charge trace with interval correction and optional input spikes."""

    def __init__(self, cap, *, write_credit=False, veto_widths=None, confirm_widths=None, integration_width=None,
                 compact_state=False, read_drive=None):
        super().__init__()
        self.cap = cap
        self.write_credit = write_credit
        self.storage_dtype = _spike_storage_dtype(cap) if compact_state else None
        self.read_veto = _DelayCorrection(veto_widths, cap) if veto_widths is not None else None
        self.write_confirm = _DelayCorrection(confirm_widths, cap) if confirm_widths is not None else None
        self.input = _TraceInput(integration_width, cap) if integration_width is not None else None
        self.read_drive = read_drive

    def _bounds(self, present, radius, delayed_content=None):
        lower = present - radius
        upper = present + radius
        if self.write_confirm is not None:
            difference = delayed_content - present
            extension = self.write_confirm.emit_difference(difference)
            # Extend the interval toward the delayed packet when the two disagree.
            # Select the rail before firing so a silent conductance can learn.
            lower = lower + torch.where(difference < 0, extension, 0)
            upper = upper + torch.where(difference > 0, extension, 0)
        return lower.clamp(-self.cap, self.cap), upper.clamp(-self.cap, self.cap)

    def _read(self, before, change, steering, present, state):
        read, _ = _steer_packet(before, steering)
        if self.read_veto is not None:
            veto = self.read_veto.emit_difference(change.abs())
            _, read = _steer_packet(read, veto)
        if self.read_drive is not None:
            read = self.read_drive(read, steering, present, change, state=state)
        return read

    def _scan(self, lower, upper, initial, present, delayed_content):
        if self.input is None:
            return _scan_charge(lower, upper, initial, write_credit=self.write_credit), None
        pulses = self.input(present, delayed_content)
        after = _scan_translated_charge(lower, upper, initial, pulses, self.cap, write_credit=self.write_credit)
        return after, pulses

    def _store(self, charge):
        if self.storage_dtype is None or (charge.is_floating_point() and torch.is_grad_enabled()):
            return charge
        return charge.to(self.storage_dtype)

    def forward(self, present, steering, initial, pulse_counts, *, write_radius=None, delayed_content=None, state=None):
        if initial is None:
            initial = torch.zeros_like(present[:, 0])
        elif self.storage_dtype is not None:
            initial = initial.to(present.dtype)
        radius = steering if write_radius is None else write_radius
        lower, upper = self._bounds(present, radius, delayed_content)
        after, pulses = self._scan(lower, upper, initial, present, delayed_content)
        before = torch.cat((initial[:, None], after[:, :-1]), dim=1)
        moved = before if pulses is None else before + pulses
        change = after - moved
        read = self._read(before, change, steering, present, state)
        if pulse_counts is not None:
            pulse_counts.append(change.abs().sum())
            if self.read_drive is None:
                pulse_counts.append(read.abs().sum())
        return read, self._store(after[:, -1].clone())

    def step(self, present, steering, before, *, write_radius=None, delayed_content=None, state=None):
        if before is None:
            before = torch.zeros_like(present)
        elif self.storage_dtype is not None:
            before = before.to(present.dtype)
        radius = steering if write_radius is None else write_radius
        lower, upper = self._bounds(present, radius, delayed_content)
        moved = before if self.input is None else before + self.input(present, delayed_content)
        change = interval_spikes(moved, lower, upper)
        return self._read(before, change, steering, present, state), self._store(moved + change)


class _OpponentSteering(nn.Module):
    """Use the opposite synaptic current to emit write-protection spikes."""

    def __init__(self, width, cap, charge_bits, quanta_per_threshold=_PACKET_UNITS):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))
        self.current_limit = cap << charge_bits
        self.emit = _IntegrateFire(cap, charge_bits + _CHARGE_BITS, False,
                                   quanta_per_threshold=quanta_per_threshold)

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


class _TraceTapRead(nn.Module):
    """Race arriving axon packets using inhibited firing times from the held read."""

    def __init__(self, widths, delays, cap, resolution, mode, *, packet_thresholds):
        super().__init__()
        self.widths = widths
        self.delays = delays
        self.choices = tuple(min(_TRACE_TAP_COUNT, delay) for delay in delays)
        self.active_widths = tuple(width for width, count in zip(widths, self.choices, strict=True) if count > 1)
        self.cap = cap
        self.resolution = resolution
        self.race = mode == "race"
        self.weight = nn.Parameter(torch.zeros(sum(self.active_widths)))
        self.emit = _IntegrateFire((4 if packet_thresholds else 2) * cap, _CHARGE_BITS, False)

    def gain_ticks(self):
        if not self.weight.is_floating_point():
            return self.weight
        # One master weight learns both the mismatch conductance and the packet correction.
        return _weight_ticks(self.weight * 4).clamp(1 - _TICKS, _TICKS)

    def freeze_integer(self):
        if self.weight.is_floating_point():
            self.weight = nn.Parameter(self.gain_ticks().to(torch.int32), requires_grad=False)

    def forward(self, candidates, anchor, read, steering, position, quanta=None, pulse_counts=None):
        if read.ndim == 3:
            position = torch.arange(read.shape[1], device=read.device) + position
        gains = iter(self.gain_ticks().split(self.active_widths))
        currents, indices = [], []
        for taps, fixed, query, aperture, delay, choices in zip(
            candidates, anchor.split(self.widths, dim=-1), read.split(self.widths, dim=-1),
            steering.split(self.widths, dim=-1), self.delays, self.choices, strict=True,
        ):
            gain = next(gains) if choices > 1 else None
            if taps is None:
                currents.append(torch.zeros_like(fixed))
                indices.append(torch.full_like(query[..., 0], -1, dtype=torch.long))
                continue
            active = query.ne(0).any(-1) & (position >= delay)
            if self.race:
                window = aperture.unsqueeze(-1)
                exposed = taps.clamp(-window, window)
                mismatch = (exposed - query.unsqueeze(-1)).abs()
                latency = (mismatch * (_TICKS + gain[:, None])).sum(-2, dtype=taps.dtype)
                scale = (_TICKS * self.resolution * math.isqrt(gain.numel())
                         if self.training and taps.is_floating_point() else None)
                selected, index = _first_arrival(taps, latency, scale)
            else:
                selected = taps[..., -1]
                index = torch.full_like(active, taps.shape[-1] - 1, dtype=torch.long)
            currents.append(torch.where(active.unsqueeze(-1), selected - fixed, 0) * gain)
            indices.append(index.masked_fill(~active, -1))

        winners = torch.stack(indices, dim=-1)
        if pulse_counts is not None:
            pulse_counts.append(winners.ge(0).sum(dtype=torch.int32))
        correction = self.emit(torch.cat(currents, dim=-1))
        limit = self.cap if quanta is None else (self.cap * _PACKET_UNITS) // quanta
        delayed = (anchor + correction).clamp(-limit, limit)
        before, _ = _steer_packet(anchor, steering)
        after, _ = _steer_packet(delayed, steering)
        return (read + (after - before)).clamp(-limit, limit), delayed, winners


class _SteeringTransfer(nn.Module):
    """Use content spikes to transfer steering pulses within each neuron pair."""

    def __init__(self, widths, cap, quanta_per_threshold, source):
        super().__init__()
        self.widths = widths
        self.cap = cap
        self.quanta_per_threshold = quanta_per_threshold
        self.use_delay = source == "delayed"
        self.weights = nn.ParameterList(nn.Parameter(torch.zeros(width // 2, width)) for width in widths)
        self.emit = _IntegrateFire(2 * cap, _CHARGE_BITS, False)
        channels = []
        start = 0
        for width in widths:
            channels.extend(range(start, start + width - width % 2))
            start += width
        self.register_buffer("paired_channels", torch.tensor(channels, dtype=torch.long), persistent=False)

    def freeze_integer(self):
        for group, weight in enumerate(self.weights):
            if weight.is_floating_point():
                self.weights[group] = nn.Parameter(_weight_ticks(weight).to(torch.int32), requires_grad=False)

    def forward(self, steering, present, delayed, quanta):
        source = delayed if self.use_delay else present
        drive = torch.cat([
            F.linear(packet, _weight_ticks(weight))
            for packet, weight in zip(source.split(self.widths, dim=-1), self.weights, strict=True)
            if weight.shape[0]
        ], dim=-1)
        ceiling = (self.cap if quanta is None
                   else (self.cap * self.quanta_per_threshold) // quanta.to(torch.int32))
        stock = steering.index_select(-1, self.paired_channels).unflatten(-1, (-1, 2))
        room = (ceiling - steering).index_select(-1, self.paired_channels).unflatten(-1, (-1, 2))
        first, second = stock.unbind(-1)
        first_room, second_room = room.unbind(-1)
        threshold = self.emit.threshold
        lower = -first.minimum(second_room) * threshold
        upper = second.minimum(first_room) * threshold
        transfer = self.emit(drive.clamp(lower, upper))
        adjustment = torch.stack((transfer, -transfer), dim=-1).flatten(-2)
        return steering.index_add(-1, self.paired_channels, adjustment)


_LOCAL_STATE_FIELDS = ("history", "relay_phase", "steering_phase", "output_phase", "trace_charge",
                       "trace_read_phase")


def _state_rows(state, rows):
    """The local state of a subset of streams."""
    return _DelayState(**{name: None if getattr(state, name) is None else getattr(state, name)[rows]
                          for name in _LOCAL_STATE_FIELDS})


def _join_state_rows(state, parts):
    """Stack per-stream states. A stream without events keeps resting state."""
    for name in _LOCAL_STATE_FIELDS:
        values = [getattr(part, name) for part in parts]
        present = next((value for value in values if value is not None), None)
        if present is not None:
            setattr(state, name, torch.cat([torch.zeros_like(present) if value is None else value
                                            for value in values]))


def _write_state_rows(state, rows, part, batch):
    for name in _LOCAL_STATE_FIELDS:
        value = getattr(part, name)
        if value is None:
            continue
        whole = getattr(state, name)
        if whole is None:
            whole = value.new_zeros((batch, *value.shape[1:]))
            setattr(state, name, whole)
        whole[rows] = value


@dataclass
class ClockState:
    """What the surprise clock carries from one byte to the next."""

    membrane: torch.Tensor
    forecast: torch.Tensor | None
    held: torch.Tensor
    events: torch.Tensor


def _unmatched_winner_spikes(counts, tokens, previous=None):
    """Surprise at each arriving byte, in forecast spikes.

    The forecast neurons fired before the byte arrived. The winner's spikes
    that the arriving byte's own neuron did not match are the error spikes.
    """
    first = counts[:, :1] if previous is None else previous[:, None].to(counts.device, counts.dtype)
    forecast = torch.cat((first, counts[:, :-1]), dim=1)
    surprise = forecast.amax(-1) - forecast.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    if previous is None:
        surprise[:, 0] = 0
    return surprise


def _clock_ticks(surprise, threshold, leak, membrane=None):
    """One integrate-and-fire neuron per stream with subtractive reset.

    The input saturates at one threshold and the leak holds the membrane at
    or above rest, so the membrane ends every byte inside [0, threshold) and
    the neuron fires at most once per byte.
    """
    rows = surprise.clamp(max=threshold).tolist()
    start = [0] * len(rows) if membrane is None else membrane.tolist()
    ticks, final = [], []
    for row, charge in zip(rows, start, strict=True):
        fired = []
        for value in row:
            charge = max(charge + value - leak, 0)
            spike = charge >= threshold
            if spike:
                charge -= threshold
            fired.append(spike)
        ticks.append(fired)
        final.append(charge)
    return torch.tensor(ticks, dtype=torch.bool), torch.tensor(final, dtype=torch.int32)


class _DelayBlock(nn.Module):
    def __init__(self, cfg: EventLMConfig, depth: int):
        super().__init__()
        self.steered = cfg.route in _STEERED_ROUTES
        schedule = _STEERED_DELAYS if self.steered else _DELAYS
        self.delays = schedule[depth % len(schedule)]
        self.widths = cfg.group_widths
        self.route = cfg.route
        self.correction = None
        self.memory_residual = None
        if cfg.route in _CORRECTION_ROUTES and depth == cfg.depth - 1:
            self.correction = _SpikeCorrection(
                cfg.dim, self.widths[-1], cfg.memory_heads, cfg.memory_slots, cfg.resolution,
                bind_previous=cfg.route == "steered_correction_binding",
                write_credit=cfg.bank_write_credit,
            )
            if cfg.route != "steered_correction":
                self.memory_residual = _Synapse(self.widths[-1], cfg.dim, gain=0.15)
        replace_delay = self.correction is not None and self.memory_residual is None
        self.local_delays = self.delays[:-1] if replace_delay else self.delays
        self.local_widths = self.widths[:len(self.local_delays)]
        self.balanced = cfg.route == "steered_balanced"
        correction_start = cfg.residual_correction_start
        self.residual_correction = correction_start is not None and depth >= correction_start
        # Appended silent steering neurons can learn as their content pathway starts firing.
        self.steering_gradient_at_zero = cfg.steering_gradient_at_zero or (self.residual_correction and correction_start != 0)
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
        candidate_window = _TRACE_TAP_COUNT if cfg.trace_tap_read is not None else self.delay_window
        if candidate_window > 1:
            offsets = torch.arange(min(candidate_window, self.history_size))
            self.register_buffer("delay_offsets", offsets, persistent=False)
        counts_per_channel = 2 if self.departure_coincidence else 1
        self.delay_widths = tuple(counts_per_channel * width for width in self.widths)
        self.delay_sizes = tuple(delay * width for delay, width in zip(self.local_delays, self.delay_widths))
        cap = _HIDDEN_CAP * cfg.resolution
        self.trace = None
        self.trace_coincidence_start = cfg.trace_coincidence_start
        if cfg.local_trace:
            veto_widths = self.widths if cfg.trace_read_veto else None
            confirm_widths = self.widths if cfg.trace_write_confirm else None
            integration_width = cfg.hidden if cfg.trace_integration else None
            read_drive = (_TraceReadDrive(cfg.hidden, cap, cfg.trace_read_drive, bits=cfg.trace_read_drive_bits,
                                          carry=cfg.trace_read_drive_carry,
                                          steering_widths=self.widths if cfg.trace_read_share_steering else None)
                          if cfg.trace_read_drive is not None else None)
            self.trace = _SpikeTrace(cap, write_credit=cfg.trace_write_credit, veto_widths=veto_widths,
                                     confirm_widths=confirm_widths, integration_width=integration_width,
                                     compact_state=cfg.compact_trace_state, read_drive=read_drive)
        self.trace_taps = (
            _TraceTapRead(self.widths, self.local_delays, cap, cfg.resolution, cfg.trace_tap_read,
                          packet_thresholds=cfg.learn_packet_thresholds)
            if cfg.trace_tap_read is not None else None
        )
        quanta_per_threshold = 1 << cfg.inhibition_threshold_bits
        self.steering_transfer = (_SteeringTransfer(self.widths, cap, quanta_per_threshold, cfg.steering_transfer)
                                  if cfg.steering_transfer is not None else None)
        self.opponent_steering = (_OpponentSteering(cfg.hidden, cap, cfg.router_charge_bits, quanta_per_threshold)
                                  if cfg.opponent_steering else None)
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
            self.activity_inhibition = _IntegrateFire(_HIDDEN_CAP * _PACKET_UNITS, cfg.inhibition_bits, False)
        self.inhibition_thresholds = (_PacketThresholds(self.widths, quanta_per_threshold)
                                      if cfg.learn_inhibition_thresholds else None)
        self.inhibition_redistribution = (
            _InhibitionRedistribution(cfg.dim, self.widths, self.activity_inhibition)
            if cfg.redistribute_inhibition else None
        )
        bits = cfg.router_charge_bits
        self.expand = _Synapse(cfg.dim, cfg.hidden * (2 if self.steered else 1), gain=1.0,
                               continuous=cfg.continuous, charge_bits=bits)
        self.relay = _IntegrateFire(cap, bits, cfg.carry, continuous=cfg.continuous,
                                    quanta_per_threshold=quanta_per_threshold)
        if self.steered:
            self.steer = _IntegrateFire(cap, bits, cfg.carry, continuous=cfg.continuous,
                                        quanta_per_threshold=quanta_per_threshold)
        self.project = _Synapse((4 if self.steered else 3) * cfg.hidden, cfg.dim, gain=0.65,
                                continuous=cfg.continuous, charge_bits=bits)
        self.trace_arrival = cfg.trace_arrival
        self.arrival_width = cfg.trace_arrival_width
        self.arrival_project = (_Synapse(self.arrival_width, cfg.dim, gain=0.0, charge_bits=bits)
                                if self.trace_arrival is not None else None)
        output_bits = bits + (_PACKET_BITS if self.packet_thresholds is not None else 0)
        self.residual_membrane = cfg.residual_membrane
        if self.residual_membrane:
            self.emit = _ResidualMembraneFire(cap, output_bits)
        else:
            self.emit = _IntegrateFire(cap, output_bits, cfg.carry, continuous=cfg.continuous)
        self.late_steering = (_DelayCorrection(self.widths, cap, input_bits=bits,
                                              quanta_per_threshold=quanta_per_threshold)
                              if cfg.late_steering else None)
        self.late_emit = _IntegrateFire(cap, output_bits, False) if cfg.late_steering else None
        self.axon_dtype = None
        if cfg.compact_axon_state:
            axon_cap = 2 * cap if cfg.learn_packet_thresholds else cap
            self.axon_dtype = _spike_storage_dtype(axon_cap)

    def _firing_quanta(self, spikes, quanta):
        if self.activity_inhibition is None:
            return quanta
        activity = spikes.abs().sum(-1, keepdim=True, dtype=spikes.dtype)
        inhibition = self.activity_inhibition(activity)
        if self.inhibition_redistribution is not None:
            inhibition = self.inhibition_redistribution(spikes, inhibition)
        if self.inhibition_thresholds is not None:
            units = self.inhibition_thresholds.quanta_per_threshold
            # Each shunt pulse still adds one quarter of the original threshold.
            return (self.inhibition_thresholds() + inhibition * (units // _PACKET_UNITS)).clamp_min(units)
        return _PACKET_UNITS + inhibition

    def _steering_current(self, drive):
        if self.balanced:
            return drive
        if self.steering_gradient_at_zero and drive.requires_grad:
            # Use the right derivative at the silent boundary during training.
            return torch.where(drive >= 0, drive, 0)
        return drive.clamp_min(0)

    def _apply_late_steering(self, current, first, present, steering, firing_quanta, quanta):
        width = sum(self.widths)
        drive = F.linear(first, self.expand.weight_ticks()[width:])
        back = self.late_steering.emit_difference(drive.clamp(0, self.steer.cap * self.steer.threshold), firing_quanta)
        _, bypass = _steer_packet(present, steering)
        redirected, _ = _steer_packet(bypass, back)
        if quanta is not None:
            redirected = redirected * quanta
        match, miss = self.project.weight_ticks()[:, :2 * width].split(width, dim=-1)
        # Transfer the same packets from bypass to coincidence with opposite synaptic signs.
        revised = current + F.linear(redirected, match - miss)
        return first + self.late_emit(revised, reference=first)

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
        scale = self.resolution * math.isqrt(query.shape[-1]) if self.training and candidates.is_floating_point() else None
        winner, _ = _first_arrival(candidates, mismatches, scale)
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
        history = ((None,) * len(groups) if state is None or state.history is None
                   else state.history.split(self.delay_sizes, dim=-1))
        delayed, tails, tap_candidates = [], [], []
        for index, (storage, group, delay) in enumerate(zip(history, groups, self.local_delays, strict=True)):
            joined = (F.pad(group, (0, 0, delay, 0)) if storage is None
                      else torch.cat((storage.reshape(batch, delay, group.shape[-1]).to(group.dtype), group), dim=1))
            delayed.append(joined[:, :steps])
            if self.trace_taps is not None:
                choices = self.trace_taps.choices[index]
                tap_candidates.append(joined.unfold(1, choices, 1)[:, :steps] if choices > 1 else None)
            if state is not None:
                tails.append(joined[:, -delay:].flatten(1))
        if state is not None:
            state.history = torch.cat(tails, dim=-1).to(history_dtype)
        return delayed, tap_candidates

    def _current(self, residual, present, delayed, steering, start, alternate=None, recall=None, quanta=None, *,
                 departure=None, anchor=None, coincidence_read=None):
        if alternate is not None:
            # The last half of the phase interval opens the adjacent delay
            # only when a steering packet has actually fired.
            switch = (start + 1) * (steering > 0)
            delayed = delayed + switch * (alternate - delayed)
            start = None
        arrival = None
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
            selected, bypass = _steer_packet(delayed, steering)
            routes = _steer_packet(present, steering) + (coincidence_read, bypass)
            if self.arrival_project is not None:
                arrival = selected[..., :self.arrival_width]
                if self.trace_arrival == "unmatched":
                    held = coincidence_read[..., :self.arrival_width]
                    _, arrival = _route_packets(held, arrival, "coincidence")
        elif self.balanced:
            routes = _balanced_steer(present, steering) + _balanced_steer(delayed, steering)
        else:
            split = _phase_swap if self.route == "steered_phase_swap" else _steer_packet
            delay_routes = (_route_packets(present, delayed, "tonic") if self.route == "steered_local"
                            else split(delayed, steering, start))
            routes = split(present, steering, start) + delay_routes
        if self.spread_coincidence is not None:
            routes = self.spread_coincidence(present, steering, routes[0]) + routes[2:]
        trace_routes = None
        if self.trace_coincidence_start is not None:
            # The held read already coincides with steering. Match it to
            # the current content on its own sign rail.
            trace_routes = _route_packets(coincidence_read, present, "coincidence") + routes[2:]
        if quanta is not None:
            # Route physical pulse counts first. The same synaptic charge
            # then requires fewer pulses at a higher firing threshold.
            routes = tuple(packet * quanta for packet in routes)
            if trace_routes is not None:
                trace_routes = tuple(packet * quanta for packet in trace_routes)
        combined = torch.cat(routes, dim=-1)
        if trace_routes is None:
            projected = self.project(combined)
        else:
            weights = self.project.weight_ticks()
            boundary = self.trace_coincidence_start
            standard = F.linear(combined, weights[:boundary])
            conditioned = F.linear(torch.cat(trace_routes, dim=-1), weights[boundary:])
            projected = torch.cat((standard, conditioned), dim=-1)
        current = residual * self.emit.threshold + projected
        if arrival is not None:
            if quanta is not None:
                arrival = arrival * quanta[:self.arrival_width]
            current = current + self.arrival_project(arrival)
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
            steering = self.steer(self._steering_current(steer_drive), firing_quanta)
            if self.phase_bits:
                start = _steering_start(steer_drive, self.phase_bits, self.steer.cap)
                if shuffle_phase:
                    start = start.roll(1, dims=-1)
        present = self.relay(drive, firing_quanta)
        coincidence_read = None
        packets, departure = self._departure_packets(present, steering)
        alternate = recall = anchor = None
        tap_candidates = ()
        if self.selector is None:
            groups = packets.split(self.delay_widths, dim=-1)
            if local_state is None and self.trace_taps is None:
                delayed_groups, anchor = self._delayed_groups(groups, steering, pulse_counts)
            else:
                delayed_groups, tap_candidates = self._chunk_delays(groups, local_state)
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
        if self.steering_transfer is not None:
            steering = self.steering_transfer(steering, present, delayed, firing_quanta)
        if self.trace is not None:
            initial = None if local_state is None else local_state.trace_charge
            radius = self._trace_radius(steering, steer_drive, firing_quanta)
            coincidence_read, trace_charge = self.trace(
                present, steering, initial, pulse_counts, write_radius=radius, delayed_content=delayed,
                state=local_state,
            )
            if local_state is not None:
                local_state.trace_charge = trace_charge
        if self.trace_taps is not None:
            position = 0 if correction_access is None else correction_access.local_position
            coincidence_read, delayed, _ = self.trace_taps(
                tap_candidates, delayed, coincidence_read, steering, position, quanta, pulse_counts,
            )
        current = self._current(spikes, present, delayed, steering, start, alternate, recall, quanta,
                                departure=departure, anchor=anchor, coincidence_read=coincidence_read)
        if self.residual_membrane:
            return self.emit(current, state=local_state)
        reference = spikes if self.residual_correction else None
        first = self.emit(current, reference=reference)
        if reference is not None:
            first = first + reference
        if self.late_steering is not None:
            return self._apply_late_steering(current, first, present, steering, firing_quanta, quanta)
        return first

    def _step_compact(self, packets, steering, state, position):
        # Recycle a packet's slot as soon as its fixed delay has elapsed.
        batch = packets.shape[0]
        if state.history is None:
            state.history = packets.new_zeros(batch, sum(self.delay_sizes), dtype=self.axon_dtype)
        groups = packets.split(self.delay_widths, dim=-1)
        history = state.history.split(self.delay_sizes, dim=-1)
        apertures = steering.split(self.widths, dim=-1)
        previous, slots, base_packets, filtered, tap_candidates = [], [], [], [], []
        for index, (storage, packet, aperture, delay) in enumerate(
            zip(history, groups, apertures, self.local_delays, strict=True)
        ):
            ring = storage.reshape(batch, delay, packet.shape[-1])
            slot = ring[:, position % delay]
            slots.append(slot)
            old = slot.to(packets.dtype)
            base_packets.append(old)
            if self.trace_taps is not None:
                tap_count = self.trace_taps.choices[index]
                candidates = None
                if tap_count > 1 and position >= delay:
                    addresses = (position - delay + self.delay_offsets[:tap_count]) % delay
                    candidates = ring.index_select(1, addresses).transpose(1, 2).to(packets.dtype)
                tap_candidates.append(candidates)
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
        return delayed, anchor, filtered_packets, tap_candidates

    def step(self, spikes, state, position):
        drive = self.expand(spikes)
        quanta = self.packet_thresholds() if self.packet_thresholds is not None else None
        firing_quanta = self._firing_quanta(spikes, quanta)
        steering = start = None
        if self.steered:
            drive, steer_drive = drive.chunk(2, dim=-1)
            steer_charge = self._steering_current(steer_drive)
            steering, state.steering_phase = self.steer.step(steer_charge, state.steering_phase, firing_quanta)
            if self.phase_bits:
                start = _steering_start(steer_drive, self.phase_bits, self.steer.cap)
        present, state.relay_phase = self.relay.step(drive, state.relay_phase, firing_quanta)
        coincidence_read = None
        packets, departure = self._departure_packets(present, steering)
        alternate = recall = anchor = None
        tap_candidates = ()
        if self.compact_delays:
            delayed, anchor, filtered, tap_candidates = self._step_compact(packets, steering, state, position)
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
        if self.steering_transfer is not None:
            steering = self.steering_transfer(steering, present, delayed, firing_quanta)
        if self.trace is not None:
            radius = self._trace_radius(steering, steer_drive, firing_quanta)
            coincidence_read, state.trace_charge = self.trace.step(
                present, steering, state.trace_charge, write_radius=radius, delayed_content=delayed, state=state,
            )
        if self.trace_taps is not None:
            coincidence_read, delayed, _ = self.trace_taps(
                tap_candidates, delayed, coincidence_read, steering, position, quanta,
            )
        current = self._current(spikes, present, delayed, steering, start, alternate, recall, quanta,
                                departure=departure, anchor=anchor, coincidence_read=coincidence_read)
        if self.residual_correction:
            correction, state.output_phase = self.emit.step(current, state.output_phase, reference=spikes)
            spikes = spikes + correction
        else:
            spikes, state.output_phase = self.emit.step(current, state.output_phase)
        if self.late_steering is not None:
            spikes = self._apply_late_steering(current, spikes, present, steering, firing_quanta, quanta)
        return spikes


class _Level(nn.Module):
    """A level of a stack. It has no relay neurons of its own: the stream passes as it is to the state read and to
    the cells that follow the level."""

    history_size = 0
    correction = None

    def forward(self, spikes, **_):
        return spikes


@dataclass
class EventLMState:
    layers: list[_DelayState]
    input_phase: torch.Tensor | None = None
    output_phase: torch.Tensor | None = None
    previous_bytes: torch.Tensor | None = None
    position: int = 0
    prediction_memory: CorrectionState | None = None
    clock: ClockState | None = None
    synapses: SynapseState | None = None
    heads: HeadState | None = None


@dataclass
class EventChunkState:
    local_prefix: torch.Tensor | None
    correction: CorrectionState | None
    local_layers: list[_DelayState] | None = None
    position: int = 0
    clock: ClockState | None = None
    synapses: SynapseState | None = None
    heads: HeadState | None = None


def _word_bytes():
    """The bytes that make up words: letters, digits and the bytes of longer characters."""
    inside = torch.zeros(_VOCAB, dtype=torch.bool)
    inside[list(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")] = True
    inside[128:] = True
    return inside


def _word_starts(inside):
    """The bytes at which a word starts: a word byte after a byte that is none, as [batch, bytes]."""
    return inside & ~F.pad(inside, (1, 0))[:, :inside.shape[1]]


def _word_ends(inside):
    """The first byte after every word: a byte that is no word byte after one that is, as [batch, bytes]."""
    return ~inside & F.pad(inside, (1, 0))[:, :inside.shape[1]]


class _StateRead(nn.Module):
    """A read of the stream's own earlier cells, addressed by what they hold.

    Every byte position is a cell that leaves a tag and a payload. The
    present cell sends a probe, the cells whose tags match it pass their
    payloads on, and the sum joins the stream through a firing population
    like a block's correction. ``soft`` weighs the cells with a softmax, as
    attention does, and is the reference for what such an address is worth.
    ``race`` is its spike form. Probe, tag and payload are spike packets. A
    probe pulse that meets a tag pulse of its own sign on its channel charges
    the cell, and one that meets a pulse of the other sign discharges it. The
    cells within reach of the most charged one fire, and every cell spike
    passes the cell's payload pulses on. With a ``read_budget`` the race is
    one of rates instead: every unit of charge multiplies a cell's rate by e,
    and the pool's inhibition ends the race after that many spikes, so the
    spike counts are a softmax in whole spikes. The threshold of the read
    neurons grows with the number of cell spikes, so they count the mean
    payload in pulses of their own resolution. Without ``read_signed`` the
    probe and the tag keep only their positive pulses. ``age`` is the control
    without an address: every cell's share follows from its age alone.

    With ``read_beats`` a cell also has an age in beats of a clock: the number
    of beats after it, counted up to that many. Each head has a learned offset
    for every such age beside its offsets for the age in bytes. The ``word``
    clock beats where a word starts, which the bytes themselves tell, so a
    head can prefer the cells of the word before last however long the words
    between are. The ``end`` clock beats at the first byte after a word, so
    that a word has one age from every later byte. With a ``learned`` clock
    every head has a clock cell of its own, and its spike is the beat. The
    cell hears the tags of its read, the pulses by which the cells are
    addressed, through synapses that start silent. A head then counts a
    cell's age in whatever its clock cell has learned to mark. The whole
    beats decide the ages, and the clock cells learn from the mean offsets
    over clocks that beat with their chances. With ``read_beat_start`` every
    head starts with that much charge for one age of its own, the first head
    for the present stretch, the next for the one before, and so on, so the
    heads do not all turn to the same one.

    With ``read_marks`` a beat also marks the cell it falls on, and each head
    has a second offset for the marked cell of every age. Under the clock at
    word ends the marked cell of age nought is the first byte after the last
    complete word, the one of age one the first byte after the word before
    it, and so on. One offset then addresses the single cell that holds the
    stream's state at the end of a chosen word.

    With ``read_phases`` the probe and tag neurons also hear a bank of that
    many pairs of pacemaker cells. A pacemaker fires in a fixed rhythm: the
    counts of a pair follow the cosine and the sine of a phase that turns
    once in the pair's period, and the periods run from three bytes to twice
    the window. The phases at two bytes tell how far apart the bytes are, so
    a probe that carries its own phases turned back by some bytes meets the
    tag of the cell that many bytes back. The match of probe and tag then
    addresses a cell by what it holds, by how old it is, and by both at once,
    and all of it is learned by the synapses from the stream and from the
    pacemakers. The age offsets remain beside it.

    With ``read_turn`` the pacemakers gate where they were only heard. The
    neurons of a probe or tag come in pairs that share two dendrites, and
    every pair has a pair of pacemaker cells with a period of its own. Each
    spike of a pacemaker lets the charge of a dendrite into a body once, on
    the rail of the pacemaker's sign, so the two bodies fire for the two
    charges turned by the pacemaker's phase. A probe sent at one byte and a
    tag left at an earlier one then match by what both hold and by the phase
    that passed between them, at each pair's own period. One head can so look
    for one thing a few bytes back and for another thing far back, where an
    age offset gives a head one preference for every probe. The fastest pair
    turns by a radian a byte and the periods grow in equal ratios. Most pairs
    turn little within a window and match by what the cells hold alone. This
    is the spike form of a Transformer's rotary phases.
    """

    def __init__(self, cfg: EventLMConfig):
        super().__init__()
        self.heads, self.window, self.mode, self.levels = cfg.read_heads, cfg.read_window, cfg.read_mode, cfg.read_levels
        self.resolution = cfg.resolution
        # The age offsets and the gains are charges, which move further than a synapse does. This factor on
        # their stored values lets them cover that range at the synapses' learning rate.
        self.rate = cfg.read_address_rate
        bits = cfg.router_charge_bits
        cap = _HIDDEN_CAP * cfg.resolution
        # A payload pulse of a single addressed cell arrives as this many read pulses.
        self.read_resolution = cfg.resolution if cfg.read_resolution is None else cfg.read_resolution
        if self.mode != "age":
            self.probe = _Synapse(cfg.dim, cfg.read_width, gain=cfg.read_synapse_gain, charge_bits=bits)
            self.tag = _Synapse(cfg.dim, cfg.read_width, gain=cfg.read_synapse_gain, charge_bits=bits)
            # Each head has its own gain on the match of probe and tag.
            self.gain = nn.Parameter(torch.zeros(self.heads, 1, 1))
        self.phases = cfg.read_phases
        if self.phases:
            # The probe and tag neurons also hear the pacemakers, through synapses of their own.
            gain = _PHASE_GAIN * cfg.read_synapse_gain
            self.probe_phase = _Synapse(2 * self.phases, cfg.read_width, gain=gain, charge_bits=bits)
            self.tag_phase = _Synapse(2 * self.phases, cfg.read_width, gain=gain, charge_bits=bits)
            self.register_buffer("pacemakers", _pacemaker_counts(self.window, self.phases, 2 * self.window),
                                 persistent=False)
        self.payload = _Synapse(cfg.dim, cfg.read_width, gain=cfg.read_synapse_gain, charge_bits=bits)
        self.turn = cfg.read_turn
        if self.turn:
            # Every pair of a head's probe or tag neurons has a pacemaker pair, slower from pair to pair, and the
            # heads share the pacemakers.
            pairs = cfg.read_width // self.heads // 2
            self.turn_longest = _TURN_SHORTEST * _TURN_SPAN ** ((pairs - 1) / pairs)
            self.register_buffer("turners", _pacemaker_counts(self.window, pairs, self.turn_longest, _TURN_SHORTEST),
                                 persistent=False)
            # A pulse of a turned neuron takes a threshold of charge at every spike of a pacemaker's largest count.
            self.turned = _IntegrateFire(cap if cfg.read_cap is None else cfg.read_cap,
                                         bits + _PHASE_SPIKES.bit_length() - 1, False)
        # A cell's charge starts from a learned offset for its age.
        self.age = nn.Parameter(torch.zeros(self.heads, self.window))
        self.beats, self.clock, self.beat_mark = cfg.read_beats, None, None
        if self.beats:
            # And from a learned offset for its age in beats, where the last one stands for all older cells.
            self.beat_age = nn.Parameter(torch.zeros(self.heads, self.beats + 1))
            if cfg.read_marks:
                # A cell on which a beat fell has a further offset for its age.
                self.beat_mark = nn.Parameter(torch.zeros(self.heads, self.beats + 1))
            if cfg.read_beat_start:
                with torch.no_grad():
                    own = torch.arange(self.heads) % (self.beats + 1)
                    self.beat_age[torch.arange(self.heads), own] = cfg.read_beat_start / self.rate
            if cfg.read_clock != "learned":
                # A fixed clock reads the bytes: it beats where a word starts, or at the first byte after one.
                self.beat_at = _word_starts if cfg.read_clock == "word" else _word_ends
                self.register_buffer("word_bytes", _word_bytes(), persistent=False)
            else:
                # A head's clock cell hears the tags of the read. It starts deaf, so that it first beats at its
                # own pace.
                self.clock = _Synapse(cfg.read_width, self.heads, gain=cfg.read_synapse_gain, charge_bits=bits)
                nn.init.zeros_(self.clock.weight)
                # Spikes of charge a clock cell lacks before a beat becomes as likely as none, in quarter spikes.
                rest = math.log((1 - cfg.read_clock_rate) / cfg.read_clock_rate)
                self.clock_rest = nn.Parameter(torch.full((self.heads,), rest))
                self.tick = _ClockFire(bits + _READ_CLOCK_BITS, 1)
                # A clock synapse saturates where the sum of all of them at the largest tags would leave the
                # range of exact sums, so no step of training can carry a clock cell out of it.
                quarter = self.tick.threshold // (self.tick.quanta_per_threshold * _PACKET_UNITS)
                self.clock_rest_ticks = -(-_GRANULE_REST_LIMIT * _PACKET_UNITS * _PACKET_UNITS * quarter // cap)
                self.clock_limit = min(_WEIGHT_LIMIT << (bits - _CHARGE_BITS),
                                       (_FP32_EXACT // cap - self.clock_rest_ticks) // cfg.read_width)
                if self.clock_limit < 1:
                    raise ValueError("a clock cell cannot hear this many read channels within the range of exact "
                                     "sums; lower the read width or the spike resolution")
        # The read is silent until its output synapses grow.
        self.project = _Synapse(cfg.read_width, cfg.dim, gain=0.0, charge_bits=bits)
        self.emit = _IntegrateFire(cap, bits, False)
        if self.mode == "race":
            self.signed = cfg.read_signed
            # Larger synapses meet thresholds raised by the same whole factor, so the pulses start as they were.
            self.threshold_scale = round(cfg.read_synapse_gain)
            # A probe, tag or payload neuron fires up to a stream neuron's largest count, or up to its own cap. A
            # lower cap leaves the read's output synapses more room within the range of exact sums.
            self.fire = _IntegrateFire(cap if cfg.read_cap is None else cfg.read_cap, bits, False)
            self.pool = _IntegrateFire(_HIDDEN_CAP * _PACKET_UNITS, cfg.inhibition_bits, False)
            self.budget = cfg.read_budget
            if self.budget is None:
                # Cells fire once more for every ``spacing`` of charge they come closer to the most charged cell.
                self.spacing = nn.Parameter(torch.zeros(self.heads, 1, 1))
            # Each head has its own factor on the read resolution.
            self.grain = nn.Parameter(torch.zeros(self.heads, 1, 1))

    def _split(self, values):
        return values.unflatten(-1, (self.heads, -1)).transpose(1, 2)

    def _unit(self, drive):
        """Each head's part of a drive at unit root mean square, as [batch, heads, cells, channels]."""
        drive = self._split(drive)
        return drive * drive.square().mean(-1, keepdim=True).add(1e-6).rsqrt()

    def shares(self, spikes, tokens=None):
        """Each visible cell's share of every read and the payloads, as [batch, heads, cells, cells or channels]."""
        return self._address(spikes, tokens)[:2]

    def word_beats(self, tokens):
        """The bytes at which the fixed clock beats, as [batch, bytes]."""
        return self.beat_at(self.word_bytes[tokens])

    def word_ages(self, tokens):
        """The beats of the fixed clock after each cell up to each byte, as [batch, bytes, cells] counted up to the
        limit: the words that started since the cell, or with the clock at word ends the words that ended."""
        started = self.word_beats(tokens).cumsum(1)
        return (started[:, :, None] - started[:, None, :]).clamp(0, self.beats)

    def packets(self, spikes):
        """The probes, tags and payloads of all cells as [batch, heads, cells, channels]: whole pulses in a race and
        drives of unit size for a softmax. A read by age alone has payloads only."""
        if self.mode == "age":
            return None, None, self._unit(self.payload(spikes)) * self.read_resolution
        probe, tag = self.probe(spikes), self.tag(spikes)
        if self.phases:
            # The pacemakers charge the probe and tag neurons beside the stream.
            beats = self.phase_counts(spikes.shape[1]).to(spikes.dtype)
            probe, tag = probe + self.probe_phase(beats), tag + self.tag_phase(beats)
        if self.mode == "soft":
            probe, tag, payload = (self._unit(drive) for drive in (probe, tag, self.payload(spikes)))
            return probe, tag, payload * self.read_resolution
        # The inhibitory pool raises every firing threshold with the stream's activity, as it does in a block.
        quanta = _PACKET_UNITS + self.pool(spikes.abs().sum(-1, keepdim=True, dtype=spikes.dtype))
        quanta = quanta * self.threshold_scale
        if not self.signed:
            probe, tag = probe.clamp_min(0), tag.clamp_min(0)
        if self.turn:
            probe, tag = (self._split(self.turned(self.let_in(drive), quanta)) for drive in (probe, tag))
        else:
            probe, tag = self._split(self.fire(probe, quanta)), self._split(self.fire(tag, quanta))
        return probe, tag, self._split(self.fire(self.payload(spikes), quanta))

    def phase_counts(self, steps):
        """The pacemakers' spikes at the first ``steps`` bytes of a window, as [bytes, cells]."""
        if steps > self.pacemakers.shape[0]:
            longer = _pacemaker_counts(steps, self.phases, 2 * self.window)
            self.pacemakers = longer.to(self.pacemakers.device)
        return self.pacemakers[:steps]

    def turn_counts(self, steps):
        """The spikes of the pacemakers that gate the probes and tags at the first ``steps`` bytes of a window, as
        [bytes, cells]: the cells of the cosines, then those of the sines."""
        if steps > self.turners.shape[0]:
            longer = _pacemaker_counts(steps, self.turners.shape[1] // 2, self.turn_longest, _TURN_SHORTEST)
            self.turners = longer.to(self.turners.device)
        return self.turners[:steps]

    def let_in(self, drive):
        """The charge that the pacemakers let into the bodies of probe or tag neurons, as [batch, cells, channels]:
        the charges of each pair's two dendrites, turned by the phase of the pair's pacemakers."""
        # A dendrite holds no more charge than a neuron counts.
        limit = self.fire.threshold * self.fire.cap
        first, second = drive.clamp(-limit, limit).unflatten(-1, (self.heads, -1, 2)).unbind(-1)
        cosine, sine = self.turn_counts(drive.shape[1]).to(drive.dtype)[:, None].chunk(2, -1)
        return torch.stack((first * cosine - second * sine, first * sine + second * cosine), -1).flatten(-3)

    def clock_ticks(self):
        """The clock cells' synapses in weight ticks, each within the limit that keeps their sums exact."""
        return self.clock.weight_ticks().clamp(-self.clock_limit, self.clock_limit)

    def clock_beats(self, spikes, tag=None):
        """The bytes at which each head's clock cell beats as [batch, bytes, heads], and the chances of those beats
        where the clock cells learn. The cells hear the tags of the read, which are computed from the stream
        where they are not passed."""
        tag = self.packets(spikes)[1] if tag is None else tag
        # A softmax read has drives of unit size where a race has a few pulses on a channel.
        heard = tag.transpose(1, 2).flatten(2) * (1 if self.mode == "race" else _PACKET_UNITS)
        # The tags are shunted where they fire, so a clock cell's threshold stays as it is.
        quanta = heard.new_full((), _PACKET_UNITS)
        quarter = self.tick.threshold // (self.tick.quanta_per_threshold * _PACKET_UNITS)
        rest = self.clock_rest.clamp(-_GRANULE_REST_LIMIT, _GRANULE_REST_LIMIT) * _PACKET_UNITS
        rest = rest.detach().round() + (rest - rest.detach())
        charge = F.linear(heard, self.clock_ticks()) - rest * quanta * quarter
        fired = self.tick(charge, quanta).squeeze(-1)
        return fired.detach(), self.tick.rates(charge, quanta) if fired.requires_grad else None

    def beat_offsets(self, spikes, tokens=None, tag=None):
        """Each cell's offset for its age in beats as [batch, heads, bytes, cells], and the number of beats that
        clock cells spent on it."""
        charges = self.beat_age * self.rate
        if self.clock is None:
            if tokens is None:
                raise ValueError("a read by word age reads the bytes themselves")
            ages, rates, spent = self.word_ages(tokens)[:, None], None, 0
            marked = self.word_beats(tokens)[:, None, None]
        else:
            beats, rates = self.clock_beats(spikes, tag)
            started = beats.to(torch.int64).cumsum(1).transpose(1, 2)
            ages, spent = (started[..., :, None] - started[..., None, :]).clamp(0, self.beats), beats.sum()
            marked = beats.transpose(1, 2)[:, :, None] > 0
        # One term per age costs far less to differentiate than indexing the offsets by age.
        offset = sum((ages == age) * charges[:, age, None, None] for age in range(self.beats + 1))
        marks = None if self.beat_mark is None else self.beat_mark * self.rate
        if marks is not None:
            offset = offset + sum(((ages == age) & marked) * marks[:, age, None, None]
                                  for age in range(self.beats + 1))
        if rates is not None:
            # The whole beats decide the ages. The clock cells learn from the mean offsets over their chances.
            chances = _beat_age_chances(rates.transpose(1, 2), self.beats)
            spare = (charges[:, :-1] - charges[:, -1:]).detach()
            mean = sum(chance * spare[:, age, None, None] for age, chance in enumerate(chances))
            if marks is not None:
                # A cell is marked with the chance of a beat on it, whatever its age then is.
                spare, oldest = (marks[:, :-1] - marks[:, -1:]).detach(), marks[:, -1:, None].detach()
                by_age = sum(chance * spare[:, age, None, None] for age, chance in enumerate(chances)) + oldest
                mean = mean + rates.transpose(1, 2)[:, :, None] * by_age
            offset = offset + (mean - mean.detach())
        return offset, spent

    def _address(self, spikes, tokens=None):
        """The shares, the payloads and the number of cell spikes, which is none for a softmax."""
        steps = spikes.shape[1]
        distance = torch.arange(steps, device=spikes.device)
        distance = distance[:, None] - distance
        visible = (distance >= 0) & (distance < self.window)
        # Row t holds the offset of age t - j in column j. Padding one column and folding the rows shifts
        # each row by its position, which costs far less to differentiate than indexing by distance.
        ages = F.pad(self.age[:, :steps] * self.rate, (0, max(0, steps - self.window))).flip(-1)
        offset = F.pad(ages[:, None].expand(-1, steps, -1), (1, 0)).reshape(self.heads, steps + 1, steps)[:, 1:]
        probe, tag, payload = self.packets(spikes)
        beats = 0
        if self.beats:
            by_beat, beats = self.beat_offsets(spikes, tokens, tag)
            offset = offset + by_beat
        if self.mode == "age":
            shares = offset.masked_fill(~visible, -torch.inf).softmax(-1)
            shares = shares if shares.ndim == 4 else shares[None]
            return shares, payload, None
        if self.mode == "soft":
            gain = (self.gain * self.rate).exp()
            charge = probe @ tag.transpose(2, 3) * (gain / math.sqrt(probe.shape[-1])) + offset
            return charge.masked_fill(~visible, -torch.inf).softmax(-1), payload, None
        # Pulse pairs charge a cell. The unit keeps the spread of the charge over cells near one at the start.
        unit = self.resolution * math.sqrt(probe.shape[-1]) * (_RACE_UNIT if self.signed else 0.5)
        charge = probe @ tag.transpose(2, 3) * ((self.gain * self.rate).exp() / unit) + offset
        if self.budget is None:
            spacing = (self.spacing * self.rate).exp()
            behind = (charge.masked_fill(~visible, -torch.inf).amax(-1, keepdim=True) - charge) / spacing
            fired = (integer_spike(1 - behind / self.levels, self.levels) * self.levels).masked_fill(~visible, 0)
        else:
            rate = charge.masked_fill(~visible, -torch.inf).softmax(-1) * self.budget
            # The cells' membranes start the race at evenly spread phases, so a rate below one spike still fires
            # in a part of the cells. Each head shifts the phases by its own amount.
            cells = torch.arange(1, steps + 1, device=spikes.device)
            heads = torch.arange(self.heads, device=spikes.device)[:, None, None]
            phase = (cells * _GOLDEN + heads / self.heads) % 1
            fired = rate + ((rate + phase).floor() - rate).detach()
        return fired / fired.sum(-1, keepdim=True).clamp_min(1), payload, fired.sum() + beats

    def forward(self, spikes, pulse_counts=None, tokens=None):
        shares, payload, cell_spikes = self._address(spikes, tokens)
        packet = shares @ payload
        if self.mode == "race":
            # Every cell spike raises the threshold of its head's read neurons, so the shunted sum of the payload
            # pulses is their mean, counted in whole read pulses.
            packet = packet * (self.read_resolution * self.grain.exp())
            packet = packet.detach().round() + (packet - packet.detach())
            if pulse_counts is not None:
                pulse_counts.extend((cell_spikes, packet.abs().sum()))
        packet = packet.transpose(1, 2).flatten(2)
        current = spikes * self.emit.threshold + self.project(packet)
        return self.emit(current, reference=spikes) + spikes

    def accumulator_bounds(self):
        """The largest sums of weight ticks at a spike read's probe, tag, payload and read neurons.

        The softmax read and the read by age are floating-point references and have no such bound.
        """
        if self.mode != "race":
            return ()
        bounds = [synapse.weight_ticks().abs().sum(-1) for synapse in (self.probe, self.tag, self.payload)]
        if self.phases:
            # A pacemaker's largest count is a part of a stream neuron's, and its synapses weigh that part.
            part = _PHASE_SPIKES / self.emit.cap
            for index, synapse in enumerate((self.probe_phase, self.tag_phase)):
                bounds[index] = bounds[index] + (synapse.weight_ticks().abs().sum(-1) * part).ceil()
        bounds = [bound.amax() for bound in bounds]
        # A head's read neurons count up to a cell's largest payload, in read pulses of that head's grain. The
        # bound is in ticks per unit of a stream neuron's largest count.
        pulses = (self.read_resolution * self.grain.detach().exp().flatten() * self.fire.cap + 0.5) / self.emit.cap
        heads = self.project.weight_ticks().abs().unflatten(-1, (self.heads, -1)).sum(-1)
        bounds.append(((heads * pulses).sum(-1) + self.emit.threshold).amax())
        if self.clock is not None:
            # With a clock cell's largest resting charge, in ticks per unit of the largest tag.
            bounds.append(self.clock_ticks().abs().sum(-1).amax() + self.clock_rest_ticks)
        return tuple(bounds)

    @torch.no_grad()
    def scale_synapses(self, limit):
        """Shrink the synapses of each probe, tag, payload and read neuron whose summed weights would exceed
        ``limit`` ticks per unit of a stream neuron's largest count, and count those neurons.

        The sums are those of ``accumulator_bounds``. Rounding moves a synapse by at most half a tick, so the sums
        of the unrounded synapses are held that far below the limit.
        """
        if self.mode != "race":
            return 0
        bits, scaled = self.probe.charge_bits, 0
        spare = self.probe.weight.shape[-1] / 2 + 1
        for index, synapse in enumerate((self.probe, self.tag, self.payload)):
            allowed = limit - spare
            if self.phases and index < 2:
                phase = (self.probe_phase, self.tag_phase)[index]
                allowed = allowed - (phase.weight_ticks().abs().sum(-1) * (_PHASE_SPIKES / self.emit.cap)).ceil()
            sums = _unrounded_sums(synapse.weight, bits)
            scaled += _shrink_synapses(synapse.weight, allowed / sums.clamp_min(1), bits)
        # A read neuron of the stream sums each head's synapses at that head's largest count of read pulses.
        pulses = (self.read_resolution * self.grain.exp().flatten() * self.fire.cap + 0.5) / self.emit.cap
        weight = self.project.weight
        heads = (weight * (1 << bits)).clamp(-(_WEIGHT_LIMIT << (bits - _CHARGE_BITS)),
                                             _WEIGHT_LIMIT << (bits - _CHARGE_BITS)).abs()
        heads = heads.unflatten(-1, (self.heads, -1)).sum(-1)
        spare = (weight.shape[-1] / self.heads / 2 * pulses).sum() + 1
        allowed = limit - self.emit.threshold - spare
        return scaled + _shrink_synapses(weight, allowed / (heads * pulses).sum(-1).clamp_min(1), bits)


def _pacemaker_counts(steps, pairs, longest, shortest=_PHASE_SHORTEST):
    """The spikes of a bank of pacemaker cells at each byte, as [bytes, 2 * pairs] whole counts of either sign.

    The two cells of a pair fire with one period, a quarter of it apart: their counts follow the cosine and the
    sine of the pair's phase. The periods run in equal ratios from ``shortest`` to ``longest`` bytes. The phases
    at two bytes then tell how many bytes lie between them, at every scale from a few bytes to the whole window,
    whatever the bytes are.
    """
    ratio = (longest / shortest) ** (torch.arange(pairs, dtype=torch.float64) / max(pairs - 1, 1))
    phase = 2 * math.pi * torch.arange(steps, dtype=torch.float64)[:, None] / (shortest * ratio)
    return torch.cat((phase.cos(), phase.sin()), -1).mul(_PHASE_SPIKES).round().to(torch.float32)


def _beat_age_chances(rates, limit):
    """The chances that no beat, one beat and so on up to ``limit - 1`` beats fall after each cell up to each byte.

    ``rates`` are a clock's chances of a beat at every byte as [..., bytes], and each returned chance is
    [..., bytes, cells]. No beat falls after a cell up to a byte with the product of the chances of none at the
    bytes between them. A further beat is the last one up to the byte, at any byte after the cell, with one beat
    fewer before it, which one triangular matrix product per count gives. These are exact chances over clocks whose
    beats do not depend on each other. Running sums over the odds of the beats give the same chances in fewer
    operations, and at 512 bytes they took longer than the products on a processor and on a graphics card.
    """
    steps = rates.shape[-1]
    quiet = torch.log1p(-rates.clamp(max=1 - 1e-6)).cumsum(-1)
    visible = torch.ones(steps, steps, dtype=torch.bool, device=rates.device).tril()
    chance = (quiet[..., :, None] - quiet[..., None, :]).masked_fill(~visible, -torch.inf).exp()
    # last[t, s]: the last beat up to byte t falls at byte s.
    last = chance * rates[..., None, :]
    chances = [chance]
    for _ in range(limit - 1):
        chance = last @ F.pad(chance, (0, 0, 1, 0))[..., :steps, :]
        chances.append(chance)
    return chances


def _spread_channels(channels, clusters, fan, generator):
    """``fan`` distinct channels for every cluster, drawn so that all channels are used about equally often."""
    rows, pool = [], []
    for _ in range(clusters):
        row = []
        while len(row) < fan:
            if not pool:
                pool = torch.randperm(channels, generator=generator).tolist()
            channel = pool.pop()
            if channel not in row:
                row.append(channel)
        rows.append(sorted(row))
    return torch.tensor(rows)


class _ExcitatoryFire(_IntegrateFire):
    """Integrate-and-fire cells with one excitatory rail: a charge below zero emits nothing.

    A silent cell has no slope: nothing tells it which way its charge should
    move, so a cell that falls silent stays silent. With ``smooth`` the
    spikes are the same and the slope is that of a smooth threshold of the
    charge, which is small and positive just below the threshold. A silent
    cell then still learns from the bytes that bring it near its threshold.
    Where a population fires several fine spikes for a threshold of charge,
    ``smooth`` is that number, and the smooth threshold is as wide as it is
    for whole spikes.
    """

    def forward(self, current, quanta=None, *, reference=None, smooth=False):
        counts = super().forward(current.clamp_min(0), quanta, reference=reference)
        if not smooth or not current.is_floating_point():
            return counts
        threshold = self.threshold if quanta is None else quanta * (self.threshold // self.quanta_per_threshold)
        learned = smooth * F.gelu(current / (threshold * smooth))
        return counts.detach() + (learned - learned.detach())


class _GatedFire(_IntegrateFire):
    """The body of a gated cell: it fires for the charge that its gate lets in.

    The value dendrite of the cell collects charge and does not fire. Every
    spike of the gate dendrite lets the collected charge into the body once,
    and the body fires a spike on the rail of the charge's sign for every
    ``unit`` thresholds of what it was let in. So ``unit`` gate spikes pass the
    collected charge as a plain cell would count it, more gate spikes pass it
    more than once and a closed gate passes nothing. The collected charge is
    bounded as a plain cell's is, which keeps its product with the gate's
    count within the range of exact sums.
    """

    def __init__(self, cap: int, charge_bits: int, unit: int):
        super().__init__(cap, charge_bits, False)
        self.unit = unit

    def forward(self, current, quanta=None, *, reference=None, opened=None):
        bounded = current.clamp(-self.threshold * self.cap, self.threshold * self.cap)
        passed = bounded * opened
        units = quanta.to(torch.int32)
        threshold = units * (self.threshold // self.quanta_per_threshold * self.unit)
        charge = passed.to(torch.int32)
        magnitude = (charge.abs() + threshold // 2) // threshold
        counts = magnitude.minimum((self.cap * self.quanta_per_threshold) // units) * charge.sign()
        if not current.is_floating_point():
            return counts
        surrogate = passed / (quanta * (self.threshold // self.quanta_per_threshold * self.unit))
        return counts.to(current.dtype) + (surrogate - surrogate.detach())


class _RaceFire(_IntegrateFire):
    """Cells that race for a budget of spikes, as the cells of a state read do.

    Every threshold of charge multiplies a cell's rate by e, and the pool's
    inhibition ends the race after ``budget`` spikes. The cells' membranes
    start at evenly spread phases, so a rate below one spike still fires in a
    part of the cells. The spikes a race spends do not depend on how large
    its charges are. With ``races`` the cells run that many races side by
    side, each among the cells of one equal group and each for the budget.

    A cell far behind in a race has a rate near zero, and the rate's slope is
    then near zero too: the cell cannot learn that a larger charge would let
    it win. With ``soften`` above one the cells learn from a race whose
    charges are divided by it. The spikes are those of the race as it is.
    """

    def __init__(self, cap: int, charge_bits: int, budget: int, phase: torch.Tensor, races: int = 1,
                 soften: float = 1.0):
        super().__init__(cap, charge_bits, False)
        self.budget = budget
        self.races = races
        self.soften = soften
        self.register_buffer("phase", phase, persistent=False)

    def rates(self, charges):
        if self.races > 1:
            return charges.unflatten(-1, (self.races, -1)).softmax(-1).flatten(-2) * self.budget
        return charges.softmax(-1) * self.budget

    def forward(self, current, quanta=None, *, reference=None, sharpness=1.0):
        thresholds = current / (quanta * (self.threshold // self.quanta_per_threshold))
        charges = thresholds * sharpness
        rate = self.rates(charges)
        spikes = (rate + self.phase).floor()
        learned = rate if self.soften == 1 else self.rates(charges / self.soften)
        return learned + (spikes - learned).detach()


class _ClockFire(_IntegrateFire):
    """The clock cells of a granule cluster, each of which races a silent rival for one spike per byte.

    The cells of a cluster share their synapses, so they hold the same charge,
    and every threshold of it multiplies the odds of a beat by e. Their
    membranes start each byte at phases spread evenly over the cells, so a
    rate below one beats in that part of them, and in other cells at the next
    byte. Charges are [batch, bytes, clusters] and beats have a last axis of
    cells.
    """

    def __init__(self, charge_bits: int, cells: int):
        super().__init__(1, charge_bits, False)
        self.cells = cells

    def rates(self, current, quanta):
        return (current / (quanta * (self.threshold // self.quanta_per_threshold))).sigmoid()

    def forward(self, current, quanta=None, *, reference=None):
        rate = self.rates(current, quanta)[..., None]
        steps = torch.arange(1, current.shape[1] + 1, dtype=torch.float64)[:, None, None]
        clusters = torch.arange(current.shape[-1], dtype=torch.float64)[:, None] / current.shape[-1]
        cells = torch.arange(self.cells, dtype=torch.float64)
        phase = ((steps * _GOLDEN + (clusters + cells) / self.cells) % 1).to(rate)
        return rate + ((rate + phase).floor() - rate).detach()


def _round_away(values):
    """The nearest whole numbers, with halves away from zero as an integer division rounds them."""
    return values.sign() * (values.abs() + 0.5).floor()


def _beat_views(carried, beats, orders, hold, cap, rates=None):
    """What beat fibres bring to their cells, as [batch, bytes, clusters, fibres].

    ``carried`` is what each fibre's channel carries at every byte and
    ``beats`` marks the bytes at which a clock opens a new stretch, as
    [batch, bytes, clusters] for one clock per cluster or with a last axis of
    fibres where every fibre has a clock of its own. A fibre of order 0 brings
    the present stretch so far and a fibre of order k the k-th stretch before
    it, which is nothing while fewer stretches have closed. ``sum`` is the
    stretch's summed count up to a cell's largest count, ``mean`` its mean
    count per byte in whole pulses and ``last`` the count at its last byte.
    Everything follows from running sums and from what they were at the
    beats, without a loop over bytes.

    The beats are whole spikes and have no slope. With the ``rates`` of
    learned clocks, the sums, the lengths and the last counts take their
    slopes from their means over clocks that beat with those chances.
    """
    batch, steps, clusters, fibres = carried.shape
    # One clock serves all the fibres of its cluster, unless the beats come with an axis of fibres.
    opens = (beats if beats.ndim == 4 else beats[..., None]).to(torch.int64)
    # The stretch a byte lies in is the number of beats up to and including it.
    stretch = opens.cumsum(1)
    slot = (stretch * opens).expand(-1, -1, -1, fibres)
    opened = opens.to(carried.dtype)
    means, mean_lengths = ((None, None) if rates is None
                           else _expected_beat_views(carried.detach(), rates, orders, hold))

    def at_beats(values):
        """Slot s holds ``values`` at the byte of the s-th beat. Slot 0, before any beat, holds nothing."""
        return values.new_zeros(batch, steps + 1, clusters, fibres).scatter_add(1, slot, values * opened)

    def sloped(exact, expected):
        return exact if expected is None else exact + (expected - expected.detach())

    brought = stretch - orders
    held = brought >= 0
    lower, upper = brought.clamp_min(0), (brought + 1).clamp(0, steps)
    present = orders == 0
    if hold == "last":
        earlier = F.pad(carried, (0, 0, 0, 0, 1, 0))[:, :steps]
        return sloped(torch.where(present, carried, at_beats(earlier).gather(1, upper)) * held, means)
    if steps * cap >= _FP32_EXACT:
        raise ValueError("byte sequence exceeds the exact running sums of the beat fibres")
    total = carried.cumsum(1)
    # The running sum just before each beat closes the stretch that ends there.
    before = at_beats(total - carried)
    sums = sloped((torch.where(present, total, before.gather(1, upper)) - before.gather(1, lower)) * held, means)
    if hold == "sum":
        return sums.clamp(-cap, cap)
    position = torch.arange(steps, device=carried.device, dtype=carried.dtype)[None, :, None, None]
    starts = at_beats(position.expand(batch, -1, clusters, fibres))
    length = torch.where(present, position + 1, starts.gather(1, upper)) - starts.gather(1, lower)
    mean = sums / sloped(length, mean_lengths).clamp_min(1)
    return _round_away(mean).detach() + (mean - mean.detach())


def _expected_beat_views(carried, rates, orders, hold):
    """The means of what beat fibres bring when a clock beats at each byte with its own chance ``rates``.

    Two bytes lie in one stretch if no beat falls after the first up to the
    second, and that chance is a product over the bytes between them. So each
    view is a weighted sum over earlier bytes, which two triangular matrices
    per cluster give. Returned are the mean sums of the stretches and their
    mean lengths, or for ``last`` the mean last counts and nothing. These are
    exact means over clocks, and the slope of one with respect to a beat is
    what adding or removing that beat would change.
    """
    batch, steps, clusters, fibres = carried.shape
    rates, carried = rates.transpose(1, 2), carried.transpose(1, 2)
    quiet = torch.log1p(-rates.clamp(max=1 - 1e-6)).cumsum(-1)
    visible = torch.ones(steps, steps, dtype=torch.bool, device=carried.device).tril()
    # together[t, j]: no beat after byte j up to byte t. opened[t, s]: the present stretch of byte t opened at s.
    together = (quiet[..., :, None] - quiet[..., None, :]).masked_fill(~visible, -torch.inf).exp()
    opened = together * rates[..., None, :]
    most = int(orders.max())
    if hold == "last":
        levels = [carried]
    else:
        # A last column of ones gathers the bytes of a stretch, so the same sums give its length.
        ones = carried.new_ones(batch, clusters, steps, 1)
        levels = [together @ (torch.cat((carried, ones), dim=-1) if hold == "mean" else carried)]
    for _ in range(most):
        levels.append(opened @ F.pad(levels[-1], (0, 0, 1, 0))[..., :steps, :])
    levels = torch.stack(levels, dim=-1)
    chosen = orders[None, :, None, :, None].expand(batch, -1, steps, -1, -1)
    values = levels[..., :fibres, :].gather(-1, chosen).squeeze(-1).transpose(1, 2)
    if hold != "mean":
        return values, None
    lengths = levels[..., fibres:, :].expand(-1, -1, -1, fibres, -1).gather(-1, chosen).squeeze(-1).transpose(1, 2)
    return values, lengths


class _GranuleLayer(nn.Module):
    """Many cheap threshold cells on sparse synapses, after the granule cells of the cerebellum.

    A block has few neurons and each listens to the whole stream. Here a
    cluster of cells shares ``fan`` stream channels, as granule cells share
    the mossy fibres of a glomerulus, and every cell weighs those channels
    with its own synapses. A cell has a single excitatory rail and its own
    resting charge below the firing threshold, which the pool's inhibition
    raises with the stream's activity as in a block. So a cell stays silent
    until enough of its inputs arrive together, and its spikes count how far
    the charge passed the threshold. Its axon reaches ``fan`` other channels,
    where the spikes join the stream through a firing population like a
    block's correction. The layer is silent until its output synapses grow.
    With ``granule_reach`` the axons of a cluster reach that many channels
    instead, so that a cell can hear more of the stream than it writes to,
    or less.

    With a ``granule_budget`` the cells of a cluster race for that many
    spikes per byte instead, and a cell's rest is its handicap in the race.
    A cluster then holds learned cells that the stream addresses by what it
    carries, as a state read addresses the stream's own earlier cells, and
    no price on pulses can silence it. With ``granule_races`` the cells of a
    cluster race in that many equal groups, each for the budget, so the
    cluster answers with a cell of every group instead of a single cell.
    With ``granule_soften`` the cells learn from a softer race than the one
    they fire in, so that a cell far behind can still be drawn into use.
    With ``granule_slope`` cells that do not race learn below their
    thresholds as well: their spikes are unchanged and their slope is that
    of a smooth threshold, so that a silent cell is not lost.
    With ``granule_soft`` the cells neither race nor spike: each answers
    with a real number, a smooth threshold of its own charge, through the
    synapses and the gains of the race. That is a stand-in for diagnosis,
    which shows what the race costs. A decoder with soft cells is not a
    spiking network.

    With ``granule_delays`` the fibres of a cluster conduct at different
    speeds: each brings its channel as it was zero to ``granule_delays - 1``
    bytes ago. A cell then hears the last bytes side by side and can answer
    to a run of them, which a cell on the present stream cannot. A fan of two
    or more times the stream gives every channel that many fibres to each
    cluster: one brings the channel as it is now, and each of the others
    brings it after a delay of its own. A cell then hears the whole present
    stream and a part or all of the last bytes beside it.

    With ``granule_traces`` a cluster also has slow fibres. A slow synapse
    holds the charge it receives and passes a fixed share of it on with
    every byte: a quarter at the first kind of slow fibre, a sixteenth at
    the second, and so on. Through them a cell hears what the last words
    left behind, beside the exact bytes that the delayed fibres bring.

    With ``granule_cap`` a cell fires at most that many spikes at a byte, where it otherwise fires up to a stream
    neuron's largest count. A channel that hears many cells can then take larger synapses from each within the
    range of exact sums.

    With ``granule_gate`` every cell has two dendrites on the same fibres, each
    with its own synapses. The gate dendrite has the cell's resting charge
    and fires above it, ``granule_gate`` spikes for a threshold of charge. The
    value dendrite collects charge of either sign and does not fire. Every
    gate spike lets the collected charge into the cell body once, and the
    body fires on the rail of the charge's sign. A cell then answers with the
    product of two views of the stream: one says whether and how far the
    cell opens, the other what it passes on. A plain cell has one view for
    both. A gated cell has three rows of synapses where a plain cell has two.

    With ``granule_relays`` the fibres do not come from the stream itself.
    That many relay neurons each hear the whole stream and fire for it, and
    their axons are the fibres that the clusters share, as mossy fibres come
    from relay nuclei. A cell then weighs a learned view of the whole
    stream, where a fibre from the stream shows it a single channel.

    With ``granule_beats`` a cluster also has beat fibres, whose delay counts
    beats of a clock where a delayed fibre counts bytes. A beat opens a new
    stretch of bytes. The synapse of a beat fibre keeps what its channel
    brought since the last beat, and at a beat what it kept moves one place
    down a chain of that many held stretches. A cell hears the present
    stretch so far and the held ones side by side, whatever their lengths in
    bytes, and what it hears of a held stretch stays the same until the next
    beat. What is kept of a stretch is its summed count, its mean count or
    the count at its last byte. The ``word`` clock beats where a word starts
    and the ``end`` clock at the first byte after a word, which the bytes
    themselves tell. Seen from the gap after a word and from inside the next
    word, a held stretch of the ``end`` clock is the same word, which it is
    not under the ``word`` clock. A ``learned`` clock is a cell that hears
    the whole stream and races a silent rival for one spike per byte: its
    spike is the beat. Every beat fibre has such a clock cell of its own. The
    clock cells of a cluster share their synapses and differ in the phases
    their membranes start at, so while they are unsure a part of the fibres
    moves on and the cells hear a mixture, and once they are sure the whole
    cluster keeps one time. A clock that beats at ``every`` byte turns the
    beat fibres into delayed fibres.
    """

    def __init__(self, cfg: EventLMConfig, index: int):
        super().__init__()
        clusters, cells, fan = cfg.granule_clusters, cfg.granule_cells, cfg.granule_fan
        # A cell sums many small synapses, so its weights may sit on a finer grid than the blocks' weights.
        self.bits = cfg.router_charge_bits if cfg.granule_bits is None else cfg.granule_bits
        self.channels = cfg.dim
        cap = _HIDDEN_CAP * cfg.resolution
        # The wiring is fixed and belongs to the layer's place in the stack.
        generator = torch.Generator().manual_seed(7919 * (index + 1))
        self.relay, sources = None, cfg.dim
        if cfg.granule_relays:
            # A relay hears the stream as a block's neurons do and fires a spike for every threshold of charge.
            self.relay = _Synapse(cfg.dim, cfg.granule_relays, gain=1.0, charge_bits=cfg.router_charge_bits)
            self.relay_fire = _IntegrateFire(cap, cfg.router_charge_bits, False)
            sources = cfg.granule_relays
        self.delays, self.traces = cfg.granule_delays, cfg.granule_traces
        # Beat fibres come in one order for the present stretch and one for each held stretch.
        self.orders = cfg.granule_beats + 1 if cfg.granule_beats else 0
        plain = self.delays + self.traces
        copies = fan // sources
        if copies > 1:
            # A fan of several times the stream: every channel sends that many fibres to each cluster.
            self.register_buffer("inputs", torch.arange(sources).repeat(clusters, copies))
        else:
            self.register_buffer("inputs", _spread_channels(sources, clusters, fan, generator))
        reach = min(fan, cfg.dim) if cfg.granule_reach is None else cfg.granule_reach
        self.register_buffer("outputs", _spread_channels(cfg.dim, clusters, reach, generator))
        if copies > 1:
            # One fibre brings the channel as it is now. Each of the others brings it after a delay of its own,
            # and every delay is on about the same number of a cluster's fibres.
            taps = []
            for _ in range(clusters):
                first = torch.randperm(sources, generator=generator)
                later = [(first + copy) % (self.delays - 1) + 1 for copy in range(copies - 1)]
                taps.append(torch.cat([torch.zeros_like(first), *later]))
            self.register_buffer("taps", torch.stack(taps))
        elif plain + self.orders > 1:
            # Every cluster hears each delay, each slow synapse and each order of beat fibre on about the same
            # number of its fibres.
            self.register_buffer("taps", torch.stack([torch.randperm(fan, generator=generator)
                                                      % (plain + self.orders) for _ in range(clusters)]))
        self.clock = None
        if self.orders:
            self.clock_kind, self.hold = cfg.granule_clock, cfg.granule_hold
            on_beat = self.taps >= plain
            # A beat fibre takes its channel as it is now, and its cluster's clock makes a stretch of it. Every
            # cluster has the same number of fibres of each kind, so its beat fibres form a block.
            self.register_buffer("plain_taps", self.taps.masked_fill(on_beat, 0), persistent=False)
            self.register_buffer("beat_fibres", on_beat.nonzero()[:, 1].view(clusters, -1), persistent=False)
            self.register_buffer("beat_orders", (self.taps - plain).gather(1, self.beat_fibres), persistent=False)
            if self.clock_kind in ("word", "end"):
                self.register_buffer("word_bytes", _word_bytes(), persistent=False)
                self.beat_at = _word_starts if self.clock_kind == "word" else _word_ends
            elif self.clock_kind == "learned":
                # The clock cells of a cluster hear the whole stream through one set of synapses. They start deaf,
                # so that they first beat at their own pace.
                self.clock = _Synapse(cfg.dim, clusters, gain=cfg.granule_gain, charge_bits=cfg.router_charge_bits)
                nn.init.zeros_(self.clock.weight)
                # Spikes of charge the clock cell lacks before a beat becomes as likely as none, or has to spare
                # where the number is negative. Stored in quarter spikes as the cells' resting charges are.
                rest = math.log((1 - cfg.granule_clock_rate) / cfg.granule_clock_rate)
                self.clock_rest = nn.Parameter(torch.full((clusters,), rest))
                self.tick = _ClockFire(cfg.router_charge_bits + _CLOCK_BITS, self.beat_fibres.shape[1])
        self.receive = nn.Parameter(torch.empty(clusters, cells, fan))
        nn.init.normal_(self.receive, std=cfg.granule_gain / math.sqrt(fan))
        self.send = nn.Parameter(torch.zeros(clusters, reach, cells))
        # Spikes of charge a cell needs before its threshold counts, stored in quarter spikes.
        self.rest = nn.Parameter(torch.full((clusters, cells, 1), float(cfg.granule_rest)))
        self.budget = cfg.granule_budget
        # The most spikes of a cell at a byte with no inhibition from the pool.
        cell_cap = cap if cfg.granule_cap is None else cfg.granule_cap
        self.gated = cfg.granule_gate
        if self.gated:
            # ``receive`` holds the synapses of the value dendrite and ``gate`` those of the gate dendrite.
            self.gate = nn.Parameter(torch.empty(clusters, cells, fan))
            nn.init.normal_(self.gate, std=cfg.granule_gain / math.sqrt(fan))
            self.open = _ExcitatoryFire(_GATE_CAP, self.bits, False, quanta_per_threshold=_PACKET_UNITS * self.gated)
            self.fire = _GatedFire(cell_cap, self.bits, self.gated)
            self.open.price_share = self.fire.price_share = cfg.granule_price
        elif self.budget is None:
            self.fire = _ExcitatoryFire(cell_cap, self.bits, False)
            # By default the pulse budget counts the cells' own spikes and its price acts on the stream pulses
            # that they cause.
            self.fire.price_share = cfg.granule_price
        else:
            # The starting phases are spread evenly over a cluster's cells and shifted from cluster to cluster.
            phase = (torch.arange(1, cells + 1) * _GOLDEN + torch.arange(clusters)[:, None] / clusters) % 1
            self.fire = _RaceFire(cell_cap, self.bits, self.budget, phase, cfg.granule_races, cfg.granule_soften)
            # Each cluster has its own gain on its cells' charges.
            self.sharpness = nn.Parameter(torch.zeros(clusters, 1))
        self.soft, self.slope = cfg.granule_soft, cfg.granule_slope
        self.pool = _IntegrateFire(_HIDDEN_CAP * _PACKET_UNITS, cfg.inhibition_bits, False)
        self.emit = _IntegrateFire(cap, self.bits, False)

    def rest_quarters(self):
        quarters = self.rest.clamp(0, _GRANULE_REST_LIMIT) * _PACKET_UNITS
        return quarters.detach().round() + (quarters - quarters.detach())

    def fibres(self, spikes):
        """The spikes that the fibres carry: the stream's own, or those of the relays that hear the whole stream."""
        return spikes if self.relay is None else self.relay_fire(self.relay(spikes))

    def held(self, spikes):
        """What the slow synapses pass on at every byte: kind ``j`` a ``4 ** -j`` share of the charge it holds.

        The shares are computed in floating point, as the rates of a race are.
        """
        passed = []
        for kind in range(1, self.traces + 1):
            share = 0.25 ** kind
            parts, charge = [], 0.0
            # A synapse holds the sum of what arrived, each arrival shrunk once per byte since. The sum is taken in
            # stretches, so that the shrinking stays within the range of the numbers.
            for part in spikes.double().split(_TRACE_STRETCH, dim=1):
                steps = torch.arange(1, part.shape[1] + 1, device=spikes.device, dtype=torch.float64)[:, None]
                shrink = (1 - share) ** steps
                parts.append(((part / shrink).cumsum(1) + charge) * shrink)
                charge = parts[-1][:, -1:]
            passed.append((torch.cat(parts, 1) * share).to(spikes.dtype))
        return passed

    def beats(self, spikes, tokens=None, quanta=None):
        """The bytes at which a clock opens a new stretch, as [batch, bytes, clusters], and nothing. Learned clocks
        give the beats of every beat fibre's clock cell, with a last axis of fibres, and the clusters' rates."""
        batch, steps = spikes.shape[:2]
        clusters = self.inputs.shape[0]
        if self.clock_kind == "every":
            return spikes.new_ones(batch, steps, clusters), None
        if self.clock_kind in ("word", "end"):
            if tokens is None:
                raise ValueError("the word clock reads the bytes themselves")
            opens = self.beat_at(self.word_bytes[tokens])
            return opens[..., None].expand(-1, -1, clusters).to(spikes.dtype), None
        if quanta is None:
            quanta = _PACKET_UNITS + self.pool(spikes.abs().sum(-1, keepdim=True, dtype=spikes.dtype))
        quarter = self.tick.threshold // (self.tick.quanta_per_threshold * _PACKET_UNITS)
        rest = self.clock_rest.clamp(-_GRANULE_REST_LIMIT, _GRANULE_REST_LIMIT) * _PACKET_UNITS
        rest = rest.detach().round() + (rest - rest.detach())
        charge = self.clock(spikes) - rest * quanta * quarter
        fired = self.tick(charge, quanta)
        return fired.detach(), self.tick.rates(charge, quanta) if fired.requires_grad else None

    def heard(self, spikes, beats=None, rates=None):
        """What every cluster's fibres bring as [batch, bytes, clusters, fan]: their channels, each after its delay,
        through its slow synapse or as a stretch between beats of the cluster's clock."""
        plain = self.delays + self.traces
        if plain + self.orders == 1:
            return spikes[..., self.inputs]
        length = spikes.shape[1]
        views = [spikes] + [F.pad(spikes, (0, 0, delay, 0))[:, :length] for delay in range(1, self.delays)]
        views += self.held(spikes)
        taps = self.plain_taps if self.orders else self.taps
        heard = torch.stack(views, dim=-1).flatten(-2)[..., self.inputs * len(views) + taps]
        if not self.orders:
            return heard
        if beats is None:
            raise ValueError("beat fibres need the beats of their clusters' clocks")
        place = self.beat_fibres.expand(*heard.shape[:2], -1, -1)
        carried = heard.gather(-1, place)
        # The whole spikes of the beats decide what is heard. The clock cells learn from the means over beats.
        brought = _beat_views(carried, beats, self.beat_orders, self.hold, self.emit.cap,
                              rates if rates is not None and rates.requires_grad else None)
        return heard.scatter(-1, place, brought)

    def cells(self, spikes, tokens=None):
        """The spike counts of all cells as [batch, bytes, clusters, cells]."""
        quanta = _PACKET_UNITS + self.pool(spikes.abs().sum(-1, keepdim=True, dtype=spikes.dtype))
        beats, rates = self.beats(spikes, tokens, quanta) if self.orders else (None, None)
        quanta = quanta[..., None]
        heard = self.heard(self.fibres(spikes), beats, rates)
        charge = torch.einsum("btgk,gnk->btgn", heard, _weight_ticks(self.receive, charge_bits=self.bits))
        # A quarter spike of rest is a quarter of the cell's present threshold.
        quarter = self.fire.threshold // (self.fire.quanta_per_threshold * _PACKET_UNITS)
        rest = self.rest_quarters().squeeze(-1) * quanta * quarter
        if self.gated:
            # The gate dendrite fires above the rest, and its spikes let the value dendrite's charge into the body.
            opening = torch.einsum("btgk,gnk->btgn", heard, _weight_ticks(self.gate, charge_bits=self.bits))
            return self.fire(charge, quanta, opened=self.open(opening - rest, quanta, smooth=self.gated))
        charge = charge - rest
        if self.budget is None:
            return self.fire(charge, quanta, smooth=self.slope)
        if self.soft:
            # The charge in thresholds, as the race takes it, and no competition between the cells.
            thresholds = charge / (quanta * (self.fire.threshold // self.fire.quanta_per_threshold))
            return F.gelu(thresholds * self.sharpness.exp())
        return self.fire(charge, quanta, sharpness=self.sharpness.exp())

    def forward(self, spikes, tokens=None):
        sent = torch.einsum("btgn,gkn->btgk", self.cells(spikes, tokens),
                            _weight_ticks(self.send, charge_bits=self.bits))
        arriving = torch.zeros_like(spikes).index_add_(spikes.ndim - 1, self.outputs.flatten(), sent.flatten(-2))
        current = spikes * self.emit.threshold + arriving
        return self.emit(current, reference=spikes) + spikes

    def accumulator_bounds(self):
        """The largest sums of weight ticks that a cell, a stream channel and a relay can receive."""
        receive = _weight_ticks(self.receive, charge_bits=self.bits).abs().sum(-1).amax()
        # The resting charge at the strongest inhibition, in ticks per unit of the largest input count.
        quarter = self.fire.threshold // (self.fire.quanta_per_threshold * _PACKET_UNITS)
        most_quanta = _PACKET_UNITS + self.pool.cap
        rest = -(-_GRANULE_REST_LIMIT * _PACKET_UNITS * most_quanta * quarter // self.emit.cap)
        if self.gated:
            # The rest belongs to the gate dendrite. The value dendrite collects without it.
            gate = _weight_ticks(self.gate, charge_bits=self.bits).abs().sum(-1).amax()
            receive, rest = torch.maximum(receive, gate + rest), 0
        send = _weight_ticks(self.send, charge_bits=self.bits).abs().sum(-1)
        arriving = send.new_zeros(self.channels).index_add_(0, self.outputs.flatten(), send.flatten())
        # A cell brings at most its own largest count, which may be below a stream neuron's.
        arriving = (arriving.amax() * (self.fire.cap / self.emit.cap)).ceil()
        bounds = (receive + rest, arriving + self.emit.threshold)
        if self.relay is not None:
            bounds += (self.relay.weight_ticks().abs().sum(-1).amax(),)
        if self.clock is not None:
            quarter = self.tick.threshold // (self.tick.quanta_per_threshold * _PACKET_UNITS)
            rest = -(-_GRANULE_REST_LIMIT * _PACKET_UNITS * most_quanta * quarter // self.emit.cap)
            bounds += (self.clock.weight_ticks().abs().sum(-1).amax() + rest,)
        return bounds

    @torch.no_grad()
    def scale_synapses(self, limit):
        """Shrink the synapses of each cell, stream channel and relay whose summed weights would exceed ``limit``
        ticks per unit of a stream neuron's largest count, and count those neurons.

        The sums are those of ``accumulator_bounds``, with half a tick of rounding spared for every synapse. A
        stream channel sums the synapses that all the cells of all its clusters send to it, so those shrink
        together.
        """
        quarter = self.fire.threshold // (self.fire.quanta_per_threshold * _PACKET_UNITS)
        most_quanta = _PACKET_UNITS + self.pool.cap
        rest = -(-_GRANULE_REST_LIMIT * _PACKET_UNITS * most_quanta * quarter // self.emit.cap)
        spare = self.receive.shape[-1] / 2 + 1
        # The resting charge belongs to a gated cell's gate dendrite and to a plain cell's only dendrite.
        dendrites = [(self.receive, 0 if self.gated else rest)]
        if self.gated:
            dendrites.append((self.gate, rest))
        scaled = 0
        for weight, resting in dendrites:
            allowed = limit - resting - spare
            scaled += _shrink_synapses(weight, allowed / _unrounded_sums(weight, self.bits).clamp_min(1), self.bits)
        # What arrives at a channel is counted at a cell's largest count, in units of a stream neuron's.
        sums = _unrounded_sums(self.send, self.bits)
        targets = self.outputs.flatten()
        arriving = sums.new_zeros(self.channels).index_add_(0, targets, sums.flatten())
        fibres = sums.new_zeros(self.channels).index_add_(0, targets, torch.ones_like(sums.flatten()))
        ratio = self.fire.cap / self.emit.cap
        allowed = (limit - self.emit.threshold - 1) / ratio - fibres * self.send.shape[-1] / 2
        factor = (allowed / arriving.clamp_min(1))[self.outputs]
        scaled += _shrink_synapses(self.send, factor, self.bits)
        if self.relay is not None:
            weight, bits = self.relay.weight, self.relay.charge_bits
            allowed = limit - weight.shape[-1] / 2 - 1
            scaled += _shrink_synapses(weight, allowed / _unrounded_sums(weight, bits).clamp_min(1), bits)
        return scaled


class EventLM(nn.Module):
    """One-hot byte events in, signed output spike counts over 256 bytes out.

    Floating-point master weights use a straight-through training surrogate.
    The CPU export uses only integer tensors to execute the same circuit.
    """

    def __init__(self, cfg: EventLMConfig = EventLMConfig()):
        super().__init__()
        if min(cfg.dim, cfg.depth) < 1 or cfg.hidden < len(_DELAYS[0]):
            raise ValueError("invalid decoder dimensions")
        if cfg.delay_group_widths is not None:
            groups = cfg.group_widths
            if (len(groups) != len(_STEERED_DELAYS[0]) or any(not isinstance(width, int) or width < 1 for width in groups)
                    or sum(groups) != cfg.hidden):
                raise ValueError("delay group widths must be positive integers that partition the expansion")
        if cfg.route not in ("coincidence", "tonic", *_STEERED_ROUTES):
            raise ValueError("invalid spike packet route")
        if cfg.continuous and (cfg.route != "steered" or cfg.carry):
            raise ValueError("the continuous control uses the reset-per-byte steered topology")
        if cfg.stack and (cfg.route != "steered" or cfg.carry or cfg.continuous or cfg.clock_start is not None
                          or cfg.local_trace or cfg.compact_delays or cfg.activity_inhibition
                          or cfg.residual_correction_start is not None or cfg.learn_packet_thresholds
                          or cfg.output_reference or cfg.synapse_orders is not None or cfg.head_orders is not None
                          or (cfg.read_after is None and cfg.granule_after is None)):
            raise ValueError("a stack has no blocks: its levels are state reads and cell layers on a "
                             "reset-per-byte stream, with no clock and no option of a block")
        if cfg.route in _PREDICTION_ROUTES and cfg.carry:
            raise ValueError("prediction memory requires reset-per-byte local neurons (--no-carry)")
        if (cfg.forecast_key_channels or cfg.match_forecast_key) and cfg.route != "steered_correction_code":
            raise ValueError("forecast keys need the direct forecast bank")
        if not isinstance(cfg.forecast_key_channels, int) or (cfg.match_forecast_key and cfg.forecast_key_channels < 1):
            raise ValueError("matching needs a positive integer number of forecast key channels")
        if cfg.tag_phase_address and not cfg.match_forecast_key:
            raise ValueError("tag phase addresses need a forecast bank with matching tags")
        if cfg.tag_read_competition and not cfg.match_forecast_key:
            raise ValueError("read competition needs a forecast bank with matching tags")
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
        if cfg.trace_tap_read is not None and (cfg.trace_tap_read not in ("fixed", "race") or not cfg.local_trace):
            raise ValueError("trace tap reads need a local trace and a fixed or race selector")
        if cfg.trace_arrival is not None and (
            cfg.trace_arrival not in _TRACE_ARRIVAL_MODES or not cfg.local_trace or cfg.trace_tap_read is not None
        ):
            raise ValueError("arrival projections need a held trace and the scheduled fixed-delay packet")
        if (not isinstance(cfg.trace_arrival_groups, int)
                or not 1 <= cfg.trace_arrival_groups <= len(_STEERED_DELAYS[0])):
            raise ValueError("trace arrival groups must select a nonempty prefix of the fixed delay groups")
        if cfg.compact_trace_state and not cfg.local_trace:
            raise ValueError("compact trace state needs a local charge trace")
        if cfg.steering_transfer is not None and (
            cfg.steering_transfer not in ("present", "delayed") or not cfg.local_trace
            or max(cfg.group_widths) < 2
        ):
            raise ValueError("steering transfers need paired held-charge neurons and present or delayed input")
        if cfg.trace_coincidence_start is not None and (not cfg.local_trace
                or not isinstance(cfg.trace_coincidence_start, int) or not 0 < cfg.trace_coincidence_start < cfg.dim):
            raise ValueError("trace coincidence needs a local trace and two nonempty residual groups")
        if cfg.opponent_steering and not cfg.local_trace:
            raise ValueError("opponent steering needs a local charge trace")
        if cfg.trace_write_credit and not cfg.local_trace:
            raise ValueError("trace write credit needs a local charge trace")
        if cfg.trace_write_confirm and not cfg.local_trace:
            raise ValueError("delayed write confirmation needs a local charge trace")
        if cfg.trace_integration and not cfg.local_trace:
            raise ValueError("trace integration needs a local charge trace")
        if cfg.steering_gradient_at_zero and cfg.route not in ("steered", "steered_correction_code"):
            raise ValueError("the right steering gradient needs a rectified steering population")
        if cfg.bank_write_credit and cfg.route not in (*_CORRECTION_ROUTES, *_PREDICTION_ROUTES):
            raise ValueError("bank write credit needs a correction bank")
        if cfg.trace_read_veto and not cfg.local_trace:
            raise ValueError("trace read veto needs a local charge trace")
        if cfg.trace_read_drive is not None and (
            cfg.trace_read_drive not in _TRACE_READ_DRIVE_SOURCES or not cfg.local_trace
        ):
            raise ValueError("trace read drive needs a local charge trace and write or present spikes")
        if not isinstance(cfg.trace_read_drive_bits, int) or not _CHARGE_BITS <= cfg.trace_read_drive_bits < 31:
            raise ValueError("trace read drive precision must be an integer from 6 to 30")
        if cfg.trace_read_drive is None and (cfg.trace_read_drive_bits != _CHARGE_BITS or cfg.trace_read_drive_carry
                                             or cfg.trace_read_share_steering):
            raise ValueError("trace read precision, carry and steering sharing require a driven read")
        if cfg.residual_membrane and not cfg.local_trace:
            raise ValueError("residual membranes need chunkable local trace state")
        if cfg.residual_correction_start is not None and (
            cfg.carry or cfg.continuous or cfg.residual_membrane or not isinstance(cfg.residual_correction_start, int)
            or not 0 <= cfg.residual_correction_start < cfg.depth
        ):
            raise ValueError("residual correction needs reset-per-byte neurons and an in-range starting block")
        if cfg.spread_coincidence and (cfg.carry or cfg.continuous
                                       or cfg.route not in ("steered", "steered_correction_code")):
            raise ValueError("spread coincidence needs a reset-per-byte steered spike router")
        if cfg.late_steering and (cfg.carry or cfg.continuous or cfg.residual_membrane or cfg.spread_coincidence
                                 or cfg.trace_coincidence_start is not None
                                 or cfg.route not in ("steered", "steered_correction_code")):
            raise ValueError("late steering needs ordinary immediate coincidence and reset-per-byte residual neurons")
        if cfg.delay_filter and (not cfg.compact_delays or cfg.carry or cfg.continuous or cfg.delay_window != 1
                                 or cfg.departure_coincidence or cfg.local_trace):
            raise ValueError("delay filters need the fixed-delay coincidence path and reset-per-byte spike routing")
        if not isinstance(cfg.router_charge_bits, int) or not _CHARGE_BITS <= cfg.router_charge_bits < 31:
            raise ValueError("router_charge_bits must be an integer from 6 to 30")
        if (not isinstance(cfg.inhibition_threshold_bits, int)
                or not _PACKET_BITS <= cfg.inhibition_threshold_bits <= cfg.router_charge_bits):
            raise ValueError("inhibition threshold precision must fit the router's charge grid")
        if cfg.inhibition_threshold_bits != _PACKET_BITS and not cfg.learn_inhibition_thresholds:
            raise ValueError("inhibition threshold precision applies to learned group thresholds")
        extra_charge_bits = cfg.router_charge_bits - _CHARGE_BITS
        if extra_charge_bits and (cfg.carry or cfg.continuous or cfg.learn_packet_thresholds
                                  or cfg.route not in ("steered", "steered_correction_code")):
            raise ValueError("finer router charge needs reset-per-byte neurons with fixed firing thresholds")
        if not isinstance(cfg.resolution, int) or cfg.resolution < 1:
            raise ValueError("spike resolution must be a positive integer")
        if cfg.inhibition_by_expansion and not cfg.activity_inhibition:
            raise ValueError("an expansion-sized inhibitory pool needs activity inhibition")
        if cfg.learn_inhibition_thresholds and not cfg.activity_inhibition:
            raise ValueError("learned inhibition thresholds need an activity pool")
        if cfg.redistribute_inhibition and not cfg.activity_inhibition:
            raise ValueError("redistributing inhibition needs activity inhibition")
        if cfg.activity_inhibition:
            if (cfg.carry or cfg.continuous or cfg.learn_packet_thresholds
                    or cfg.route not in ("steered", "steered_correction_code")):
                raise ValueError("activity inhibition needs a reset-per-byte router with a fixed threshold floor")
            if cfg.dim * _HIDDEN_CAP * cfg.resolution >= _FP32_EXACT:
                raise ValueError("the inhibition pool would exceed the exact FP32 integer range")
        if cfg.clock_start is None:
            if cfg.clock_threshold or cfg.clock_leak or cfg.clock_source != "surprise":
                raise ValueError("clock settings need a first clocked block")
        else:
            if (cfg.carry or cfg.continuous or not cfg.local_trace
                    or cfg.route not in ("steered", "steered_correction_code")):
                raise ValueError("a surprise clock needs reset-per-byte routing with chunkable held charge")
            if not isinstance(cfg.clock_start, int) or not 0 < cfg.clock_start < cfg.depth:
                raise ValueError("the clock needs byte-rate blocks below it and at least one clocked block")
            if cfg.clock_source not in _CLOCK_SOURCES:
                raise ValueError("the clock integrates surprise or a constant drive")
            if (not isinstance(cfg.clock_threshold, int) or not isinstance(cfg.clock_leak, int)
                    or not 0 <= cfg.clock_leak < cfg.clock_threshold
                    or cfg.clock_threshold > torch.iinfo(torch.int16).max):
                raise ValueError("the clock needs a positive threshold above a nonnegative leak")
            if cfg.clock_source == "periodic" and cfg.clock_leak:
                raise ValueError("a periodic clock has no leak")
            if cfg.output_reference:
                raise ValueError("the clock's forecast population has no shared output reference")
        output_bits = 3
        if not isinstance(cfg.output_cap, int) or not 1 <= (cfg.output_cap * cfg.resolution) << output_bits <= _FP32_EXACT:
            raise ValueError("the output cap is a whole number of eighths of a bit, and an output neuron's largest "
                             "charge stays within the range of exact sums")
        output_cap = cfg.output_cap * cfg.resolution
        if cfg.output_reference:
            if cfg.carry or cfg.continuous or cfg.route not in ("steered", "steered_correction_code"):
                raise ValueError("output reference needs a reset-per-byte spike router")
            if _VOCAB * output_cap * (1 << output_bits) > _FP32_EXACT:
                raise ValueError("pooled output charge would exceed the exact FP32 integer range")
        projection_width = (4 if cfg.route in _STEERED_ROUTES else 3) * cfg.hidden + cfg.trace_arrival_width
        cap = _HIDDEN_CAP * cfg.resolution
        if cfg.trace_read_drive is not None:
            source_cap = 3 if cfg.trace_integration and cfg.trace_read_drive.startswith("write") else 2
            read_limit = (cap * (_TICKS + source_cap * _WEIGHT_LIMIT)) << (cfg.trace_read_drive_bits - _CHARGE_BITS)
            if read_limit > _FP32_EXACT:
                raise ValueError("trace read drive would exceed the exact FP32 integer range")
        if cfg.trace_tap_read is not None:
            count_scale = 2 if cfg.learn_packet_thresholds else 1
            ticks = 2 * cap * count_scale * _TICKS
            if cfg.trace_tap_read == "race":
                ticks *= 2 * max(cfg.group_widths)
            if ticks > _FP32_EXACT:
                raise ValueError("trace tap charge would exceed the exact FP32 integer range")
        residual_bits = cfg.router_charge_bits + (_PACKET_BITS if cfg.learn_packet_thresholds else 0)
        if cfg.residual_correction_start is not None and ((2 * cap + 1) << residual_bits) > torch.iinfo(torch.int32).max:
            raise ValueError("residual correction charge would exceed the int32 range")
        if cfg.opponent_steering and ((cap * _WEIGHT_LIMIT) << cfg.router_charge_bits) > _FP32_EXACT:
            raise ValueError("opponent steering charge would exceed the exact FP32 integer range")
        if cfg.late_steering and (cap << (cfg.router_charge_bits + _CHARGE_BITS)) > _FP32_EXACT:
            raise ValueError("late steering charge would exceed the exact FP32 integer range")
        # The additional read projection is bounded by one delay group's width.
        if cfg.route in _CORRECTION_ROUTES and cfg.route != "steered_correction":
            projection_width += cfg.group_widths[-1]
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
        # Probes can mute the clocked correction or displace its ticks at fixed weights.
        self.clock_intervention = None
        self.clock_trace = None
        self.input = _Synapse(_VOCAB, cfg.dim, gain=0.9, byte_input=True, continuous=cfg.continuous)
        taps = 2 if cfg.route == "convergent" else 0
        self.delayed_input = nn.ModuleList(_Synapse(_VOCAB, cfg.dim, gain=0.5, byte_input=True) for _ in range(taps))
        self.input_emit = _IntegrateFire(cap, _CHARGE_BITS, cfg.carry, continuous=cfg.continuous)
        self.blocks = nn.ModuleList(_Level() if cfg.stack else _DelayBlock(cfg, i) for i in range(cfg.depth))
        self.local_history = len(self.delayed_input) + sum(block.history_size for block in self.blocks)
        self.output = _Synapse(cfg.dim, _VOCAB, gain=0.8, continuous=cfg.continuous)
        self.output_emit = _IntegrateFire(output_cap, output_bits, cfg.carry, continuous=cfg.continuous)
        # A second vocabulary population reads the byte-rate stream through the same
        # synapses. Its higher threshold removes the finer spike resolution.
        coarse = cfg.resolution.bit_length() - 1
        self.forecast_resolution = cfg.resolution / (1 << coarse)
        forecast_cap = min(-(-output_cap >> coarse), torch.iinfo(torch.int8).max)
        self.forecast_emit = _IntegrateFire(forecast_cap, output_bits + coarse, False)
        self.fast_synapses = None
        if cfg.synapse_orders is not None:
            if cfg.carry or cfg.continuous or not cfg.local_trace or cfg.output_reference:
                raise ValueError("fast synapses need reset-per-byte routing with chunkable held charge")
            self.fast_synapses = FastSynapses(cfg.synapse_orders, cfg.synapse_groups, cfg.synapse_cells,
                                              cfg.synapse_levels, cfg.synapse_gate)
        self.read_heads = None
        if cfg.head_orders is not None:
            if cfg.carry or cfg.continuous or not cfg.local_trace or cfg.output_reference:
                raise ValueError("coasting heads need reset-per-byte routing with chunkable held charge")
            self.read_heads = CoastingHeads(cfg.head_orders, cfg.head_cells, cfg.head_gate, cfg.head_tape,
                                            cfg.head_lookup, cfg.head_miss, cfg.head_relock)
        self.output_reference = None
        if cfg.output_reference:
            self.output_reference = _IntegrateFire(output_cap, output_bits + _VOCAB.bit_length() - 1, False)
        self.prediction_memory = None
        if cfg.route in _PREDICTION_ROUTES:
            width = self.blocks[-1].widths[-1] if cfg.memory_width is None else cfg.memory_width
            self.prediction_memory = _SpikeCorrection(
                cfg.dim, width, cfg.memory_heads, cfg.memory_slots, cfg.resolution, bind_previous=True, observed=True,
                forecast=cfg.route == "steered_correction_code", key_channels=cfg.forecast_key_channels,
                match_key=cfg.match_forecast_key, tag_phase_address=cfg.tag_phase_address,
                write_credit=cfg.bank_write_credit, tag_read_competition=cfg.tag_read_competition,
            )
            self.recall_synapse = _Synapse(width - cfg.memory_heads * cfg.forecast_key_channels, cfg.dim, gain=0.15)
            self.recall_emit = _IntegrateFire(cap, _CHARGE_BITS, False)
        # The reads are created last, so the other weights start as they do without them.
        self.state_reads = None
        if cfg.read_after is not None:
            below = cfg.depth if cfg.clock_start is None else cfg.clock_start
            if (cfg.carry or cfg.continuous or cfg.read_mode not in _READ_MODES
                    or not cfg.read_after or len(set(cfg.read_after)) != len(cfg.read_after)
                    or any(not isinstance(index, int) or not 0 <= index < below for index in cfg.read_after)):
                raise ValueError("state reads follow distinct byte-rate blocks of a reset-per-byte decoder")
            if (min(cfg.read_width, cfg.read_heads, cfg.read_window, cfg.read_levels,
                    *(1 if size is None else size for size in (cfg.read_resolution, cfg.read_budget))) < 1
                    or min(cfg.read_address_rate, cfg.read_synapse_gain) <= 0
                    or (cfg.read_mode == "race" and cfg.read_synapse_gain != round(cfg.read_synapse_gain))
                    or cfg.read_width % cfg.read_heads):
                raise ValueError("state reads need positive sizes and a width that the heads divide")
            if cfg.read_cap is not None and (cfg.read_mode != "race" or not isinstance(cfg.read_cap, int)
                                             or not 1 <= cfg.read_cap <= _HIDDEN_CAP * cfg.resolution):
                raise ValueError("the neurons of a race read fire a whole number of pulses up to a stream neuron's "
                                 "largest count")
            if (not isinstance(cfg.read_beats, int) or not 0 <= cfg.read_beats <= _READ_BEATS
                    or not math.isfinite(cfg.read_beat_start) or (cfg.read_beat_start and not cfg.read_beats)):
                raise ValueError("a state read counts a cell's age in words up to sixteen, and a starting charge "
                                 "for a head's own age needs such ages")
            if (cfg.read_clock not in _READ_CLOCKS or not 0 < cfg.read_clock_rate < 1
                    or (cfg.read_clock != "word" and not cfg.read_beats)):
                raise ValueError("a state read counts ages in beats of a clock at the starts or the ends of words "
                                 "or of learned clock cells, which start at a rate between none and every byte; "
                                 "the clocks other than the default need such ages")
            if cfg.read_clock == "learned" and cfg.read_mode == "age":
                raise ValueError("a learned clock hears the tags of its read, which a read by age alone has not")
            if cfg.read_marks and not cfg.read_beats:
                raise ValueError("a beat marks its cell only where a state read counts ages in beats")
            if (not isinstance(cfg.read_phases, int) or not 0 <= cfg.read_phases <= _READ_PHASES
                    or (cfg.read_phases and cfg.read_mode == "age")
                    or _PHASE_SPIKES > _HIDDEN_CAP * cfg.resolution):
                raise ValueError("the probes and tags of a state read hear up to 64 pairs of pacemaker cells, whose "
                                 "counts stay within a stream neuron's; a read by age alone has no probes or tags")
            if cfg.read_turn:
                if (cfg.read_mode != "race" or not cfg.read_signed or cfg.read_budget is None
                        or cfg.read_width // cfg.read_heads % 2):
                    raise ValueError("pacemakers turn the probes and tags of a race read with a budget, whose "
                                     "neurons come in pairs and fire pulses of both signs")
                held = (_HIDDEN_CAP * cfg.resolution if cfg.read_cap is None else cfg.read_cap) << cfg.router_charge_bits
                if 2 * _PHASE_SPIKES * held > _FP32_EXACT:
                    raise ValueError("the charge that the pacemakers let into a turned probe or tag neuron exceeds "
                                     "the range of exact sums; lower the read cap")
            self.state_reads = nn.ModuleDict({str(index): _StateRead(cfg) for index in sorted(cfg.read_after)})
        self.granule_layers = None
        if cfg.granule_after is not None:
            below = cfg.depth if cfg.clock_start is None else cfg.clock_start
            if (cfg.carry or cfg.continuous or not cfg.granule_after
                    or len(set(cfg.granule_after)) != len(cfg.granule_after)
                    or any(not isinstance(index, int) or not 0 <= index < below for index in cfg.granule_after)):
                raise ValueError("granule layers follow distinct byte-rate blocks of a reset-per-byte decoder")
            copies = cfg.granule_fan // cfg.dim
            if (min(cfg.granule_clusters, cfg.granule_cells, cfg.granule_fan) < 1
                    or (cfg.granule_fan > cfg.dim and cfg.granule_fan % cfg.dim)
                    or cfg.granule_gain <= 0 or not 0 <= cfg.granule_rest <= _GRANULE_REST_LIMIT
                    or not 0 <= cfg.granule_price <= 1):
                raise ValueError("granule layers need positive sizes, a fan within the stream or of a whole number "
                                 "of streams, a rest within its limit and a price share from 0 to 1")
            if copies > 1 and (not isinstance(cfg.granule_delays, int) or copies > cfg.granule_delays
                               or cfg.granule_traces or cfg.granule_beats or cfg.granule_relays):
                raise ValueError("with a fan of several streams every channel has one fibre for the present and "
                                 "one for each of as many delays as the others need, and a cluster has no relays, "
                                 "slow fibres or beat fibres")
            if cfg.granule_soft and cfg.granule_budget is None:
                raise ValueError("soft granule cells stand in for racing cells and take their cluster gains")
            if cfg.granule_slope and cfg.granule_budget is not None:
                raise ValueError("the slope below the threshold belongs to granule cells that do not race")
            if cfg.granule_reach is not None and (
                    not isinstance(cfg.granule_reach, int) or not 1 <= cfg.granule_reach <= cfg.dim):
                raise ValueError("the axons of a granule cluster reach a whole number of the stream's channels")
            if cfg.granule_bits is not None and (
                    not isinstance(cfg.granule_bits, int) or cfg.granule_bits < _CHARGE_BITS
                    or (_HIDDEN_CAP * cfg.resolution) << cfg.granule_bits > _FP32_EXACT):
                raise ValueError("granule cells need charge bits from 6 that keep a full stream channel exact in FP32")
            if cfg.granule_cap is not None and (not isinstance(cfg.granule_cap, int)
                                                or not 1 <= cfg.granule_cap <= _HIDDEN_CAP * cfg.resolution):
                raise ValueError("a granule cell's largest count is a whole number of spikes up to a stream neuron's")
            if not isinstance(cfg.granule_gate, int) or cfg.granule_gate < 0 or (
                    cfg.granule_gate and (cfg.granule_budget is not None or cfg.granule_soft)):
                raise ValueError("gated granule cells have a whole number of gate spikes to a threshold and do "
                                 "not race")
            if cfg.granule_gate:
                # The largest charge a body collects, times the most spikes of its gate.
                collected = (cfg.granule_cap or _HIDDEN_CAP * cfg.resolution) << (
                    cfg.router_charge_bits if cfg.granule_bits is None else cfg.granule_bits)
                if collected * _GATE_CAP * cfg.granule_gate > _FP32_EXACT:
                    raise ValueError("the charge that the gate of a gated granule cell lets in at its largest "
                                     "count exceeds the range of exact sums")
            if cfg.granule_budget is not None and (
                    not isinstance(cfg.granule_budget, int)
                    or not 1 <= cfg.granule_budget <= (cfg.granule_cap or _HIDDEN_CAP * cfg.resolution)):
                raise ValueError("a race of granule cells spends a whole budget of spikes within a cell's largest count")
            if (not isinstance(cfg.granule_races, int) or cfg.granule_races < 1
                    or cfg.granule_cells % cfg.granule_races
                    or (cfg.granule_races > 1 and cfg.granule_budget is None)):
                raise ValueError("the cells of a granule cluster race in equal groups, each for the budget")
            if (not isinstance(cfg.granule_soften, (int, float)) or not 1 <= cfg.granule_soften < math.inf
                    or (cfg.granule_soften != 1 and cfg.granule_budget is None)):
                raise ValueError("racing granule cells learn from a race softened by a finite factor of at least one")
            if not isinstance(cfg.granule_delays, int) or not 1 <= cfg.granule_delays <= cfg.granule_fan:
                raise ValueError("granule fibres need at least one delay and at most one delay per fibre")
            if (not isinstance(cfg.granule_traces, int) or not 0 <= cfg.granule_traces <= _GRANULE_TRACES
                    or cfg.granule_delays + cfg.granule_traces > cfg.granule_fan):
                raise ValueError("granule clusters have up to four kinds of slow fibres, and a fibre for each kind "
                                 "and each delay")
            if (not isinstance(cfg.granule_beats, int) or not 0 <= cfg.granule_beats <= _GRANULE_BEATS
                    or cfg.granule_clock not in _GRANULE_CLOCKS or cfg.granule_hold not in _GRANULE_HOLDS
                    or not 0 < cfg.granule_clock_rate < 1):
                raise ValueError("beat fibres hold up to eight stretches between the beats of a word, word-end, "
                                 "every-byte or learned clock, as a sum, a mean or a last count, and a learned "
                                 "clock starts at a rate between zero and one beat per byte")
            if cfg.granule_beats and (cfg.granule_delays + cfg.granule_traces + cfg.granule_beats + 1
                                      > cfg.granule_fan):
                raise ValueError("granule clusters need a fibre for each delay, each slow synapse, the present "
                                 "stretch and each held stretch")
            if not isinstance(cfg.granule_relays, int) or (cfg.granule_relays != 0
                                                           and cfg.granule_relays < cfg.granule_fan):
                raise ValueError("granule relays are none or at least one for every fibre of a cluster")
            self.granule_layers = nn.ModuleDict({str(index): _GranuleLayer(cfg, index)
                                                 for index in sorted(cfg.granule_after)})

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
        for block in () if self.cfg.stack else self.blocks:
            bounds.append(row_sums(block.expand).amax())
            if block.inhibition_redistribution is not None:
                bounds.append(row_sums(block.inhibition_redistribution.synapse).amax())
            if block.delay_filters is not None:
                bounds.extend(2 * row_sums(kernel).amax() for kernel in block.delay_filters if kernel is not None)
            if block.trace is not None and block.trace.read_veto is not None:
                bounds.append(2 * block.trace.read_veto.gains().amax())
            if block.trace is not None and block.trace.input is not None:
                # The unshunted packet thresholds can emit up to twice the nominal count cap.
                count_scale = 2 if block.packet_thresholds is not None else 1
                bounds.append(count_scale * block.trace.input.weight_ticks().abs().sum(0).amax())
            if block.steering_transfer is not None:
                count_scale = 2 if block.packet_thresholds is not None else 1
                bounds.extend(count_scale * _weight_ticks(weight).abs().sum(-1).amax()
                              for weight in block.steering_transfer.weights if weight.shape[0])
            packet_units = _PACKET_UNITS if block.packet_thresholds is not None else 1
            projected = row_sums(block.project) * packet_units + block.emit.threshold
            if block.arrival_project is not None:
                projected = projected + row_sums(block.arrival_project) * packet_units
            if block.correction is not None:
                bounds.append(row_sums(block.correction.encode).amax())
            if block.memory_residual is not None:
                projected = projected + row_sums(block.memory_residual)
            bounds.append(projected.amax())
        if self.prediction_memory is not None:
            bounds.append(row_sums(self.prediction_memory.encode).amax())
            bounds.append(row_sums(self.prediction_memory.code).amax())
            bounds.append(row_sums(self.recall_synapse).amax() + _TICKS)
        if self.state_reads is not None:
            for read in self.state_reads.values():
                bounds.extend(read.accumulator_bounds())
        if self.granule_layers is not None:
            for layer in self.granule_layers.values():
                bounds.extend(layer.accumulator_bounds())
        limit = _FP32_EXACT // self.input_emit.cap
        if not torch.stack(bounds).amax().item() <= limit:
            raise ValueError("quantized weights exceed the exact FP32 accumulator range at this spike resolution")

    @torch.no_grad()
    def synaptic_scaling(self):
        """Keep every neuron of the state reads, the cell layers and the output within the range of exact sums.

        A neuron's membrane sums its synapses' charges exactly as long as the summed sizes of its synapses, at
        the largest count of what it hears, stay below 2 ** 24 ticks. Training can carry a neuron beyond that,
        and the decoder then refuses to run. Here every such neuron shrinks all of its synapses by one factor,
        which brings their sum back to the range and leaves their proportions as they are: the neuron keeps what
        it listens for and gives up size, as a neuron that scales its synapses to its own activity does. No other
        synapse changes. Returns the number of neurons that shrank, as a tensor. Called after every step of
        training, this is a rule of the learning and not of the network: the weights it leaves pass
        ``_check_accumulator``.
        """
        if not self.input.weight.is_floating_point():
            return 0
        limit = _FP32_EXACT // self.input_emit.cap
        weight, bits = self.output.weight, self.output.charge_bits
        allowed = limit - weight.shape[-1] / 2 - 1
        scaled = _shrink_synapses(weight, allowed / _unrounded_sums(weight, bits).clamp_min(1), bits)
        for layers in (self.state_reads, self.granule_layers):
            for layer in () if layers is None else layers.values():
                scaled += layer.scale_synapses(limit)
        return scaled

    def _forecast_counts(self, spikes):
        return self.forecast_emit(self.output(spikes))

    def _surprise(self, forecast, tokens, access):
        """Error spikes at each arriving byte: the winner's forecast spikes its own neuron did not match."""
        previous = None
        if access is not None and access.local_layers is not None:
            if self.cfg.clock_start is not None:
                previous = None if access.clock is None else access.clock.forecast
            elif access.synapses is not None:
                previous = access.synapses.forecast
            elif access.heads is not None:
                previous = access.heads.forecast
        with torch.no_grad():
            return _unmatched_winner_spikes(forecast.detach(), tokens, previous).to("cpu", torch.int64)

    def _add_recall_spikes(self, counts, votes, levels):
        """Count each recall neuron's spikes together with its vocabulary neuron's spikes.

        The recall neurons are a population of their own, so the vocabulary
        neurons' spike limit does not clip the sum.
        """
        if self.fast_synapses.muted:
            return counts
        return counts + self.fast_synapses.excitation(votes, levels).to(counts.device, counts.dtype)

    def _excite_vocabulary(self, counts, tokens, surprise, forecast, access):
        """Move the fast synapses on error spikes and let the agreeing ones fire the recall neurons."""
        carried = access is not None and access.local_layers is not None
        votes, levels, state = self.fast_synapses(tokens, surprise, access.synapses if carried else None, carry=carried)
        if carried:
            if self.cfg.clock_start is None:
                state.forecast = forecast[:, -1].detach().to("cpu", _spike_storage_dtype(self.forecast_emit.cap))
            access.synapses = state
        return self._add_recall_spikes(counts, votes, levels)

    def _read_tape(self, tokens, surprise, forecast, access):
        """Recall spikes of the read heads, which continue the stream when the trace is carried."""
        if access is None or access.local_layers is None:
            return self.read_heads(tokens, surprise)
        if access.heads is None:
            access.heads = self.read_heads.new_state(tokens.shape[0])
        if self.cfg.clock_start is None and self.fast_synapses is None:
            # Without a clock or fast synapses, the heads keep the forecast for the next chunk's error spikes.
            access.heads.forecast = forecast[:, -1].detach().to("cpu", _spike_storage_dtype(self.forecast_emit.cap))
        return self.read_heads(tokens, surprise, access.heads, access.stats)

    def _event_level(self, blocks, packed, lengths, layers, pulse_counts):
        if layers is None or packed.shape[0] == 1 or bool((lengths == lengths[0]).all()):
            # Padding follows every stream's last event and causal blocks never read it.
            for index, block in enumerate(blocks):
                packed = block(packed, pulse_counts=pulse_counts,
                               local_state=None if layers is None else layers[index])
            return packed
        # Carried delay lines and held charge stop at each stream's own last event.
        emitted = torch.zeros_like(packed)
        parts = [[] for _ in blocks]
        for row, length in enumerate(lengths.tolist()):
            states = [_state_rows(layer, slice(row, row + 1)) for layer in layers]
            if length:
                spikes = packed[row:row + 1, :length]
                for block, state in zip(blocks, states, strict=True):
                    spikes = block(spikes, pulse_counts=pulse_counts, local_state=state)
                emitted[row, :length] = spikes[0]
            for part, state in zip(parts, states, strict=True):
                part.append(state)
        for layer, part in zip(layers, parts, strict=True):
            _join_state_rows(layer, part)
        return emitted

    def _clocked_blocks(self, spikes, tokens, forecast, surprise, access, pulse_counts):
        """Advance the upper blocks only on bytes where the clock neuron fires.

        Their delays and held charge count events. The correction they
        add to the stream is held until the next event.
        """
        cfg = self.cfg
        carried = access is not None and access.local_layers is not None
        clock = access.clock if carried else None
        layers = access.local_layers[cfg.clock_start:] if carried else None
        with torch.no_grad():
            drive = torch.ones(tokens.shape, dtype=torch.int64) if cfg.clock_source == "periodic" else surprise
            ticks, membrane = _clock_ticks(drive, cfg.clock_threshold, cfg.clock_leak,
                                           None if clock is None else clock.membrane)
            if self.clock_intervention == "shuffle":
                ticks = ticks.roll(tokens.shape[1] // 2, dims=1)
            if self.clock_trace is not None:
                self.clock_trace.append(ticks.clone())
            ticks = ticks.to(spikes.device)
            slot = ticks.cumsum(1)
            lengths = slot[:, -1]
        batch, _, dim = spikes.shape
        held = spikes.new_zeros(batch, dim) if clock is None else clock.held.to(spikes.dtype)
        table = held[:, None]
        longest = int(lengths.max())
        if longest:
            rows, steps = ticks.nonzero(as_tuple=True)
            packed = spikes.new_zeros(batch, longest, dim)
            packed[rows, slot[rows, steps] - 1] = spikes[rows, steps]
            emitted = self._event_level(self.blocks[cfg.clock_start:], packed, lengths, layers, pulse_counts)
            table = torch.cat((table, emitted - packed), dim=1)
        correction = table.gather(1, slot.unsqueeze(-1).expand(-1, -1, dim))
        if carried:
            events = lengths.cpu() if clock is None else clock.events + lengths.cpu()
            access.clock = ClockState(membrane, forecast[:, -1].detach().clone(),
                                      correction[:, -1].detach().clone(), events)
        if self.clock_intervention == "mute":
            return spikes
        cap = self.input_emit.cap
        return (spikes + correction).clamp(-cap, cap)

    def forward(self, tokens, *, delay_limit=None, shuffle_phase=False, correction_access=None,
                return_prediction=False, corrected_prediction=False, memory_pulses=None, bank_read_losses=None,
                early_readout=None):
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("expected a nonempty [batch, bytes] tensor")
        if delay_limit is not None and (delay_limit < 1 or self.cfg.route not in _DELAY_ROUTES):
            raise ValueError("delay_limit needs a positive value and a selectable delay line")
        if shuffle_phase and self.cfg.route not in _PHASE_ROUTES:
            raise ValueError("shuffle_phase needs timed steering")
        if return_prediction and self.cfg.route != "steered_correction_code":
            raise ValueError("return_prediction needs a spike forecast")
        if corrected_prediction and not return_prediction:
            raise ValueError("a corrected forecast requires returning the prediction")
        if bank_read_losses is not None and (
            not self.training or not self.cfg.tag_read_competition or correction_access is not None or tokens.shape[1] < 2
        ):
            raise ValueError("read ranking needs a complete training window with forecast-tag competition")
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
        clock_start = self.cfg.clock_start
        # The forecast neurons read the stream under the clock, or the top stream of an unclocked decoder.
        forecast_tap = clock_start
        if forecast_tap is None and (self.fast_synapses is not None or self.read_heads is not None):
            forecast_tap = len(self.blocks)
        tap = forecast_tap if early_readout is None else early_readout
        if tap is not None and (not 0 < tap <= len(self.blocks) or (forecast_tap is not None and tap != forecast_tap)):
            raise ValueError("the early readout follows a byte-rate block and coincides with the surprise forecast")
        early = surprise = None
        for index, block in enumerate(self.blocks):
            if index == tap:
                early = self._forecast_counts(spikes)
                surprise = self._surprise(early, tokens, correction_access)
                if clock_start is not None:
                    spikes = self._clocked_blocks(spikes, tokens, early, surprise, correction_access, memory_pulses)
                    break
            spikes = block(spikes, delay_limit=delay_limit, shuffle_phase=shuffle_phase,
                           correction_access=correction_access, pulse_counts=memory_pulses,
                           local_state=None if local_layers is None else local_layers[index])
            if self.state_reads is not None and str(index) in self.state_reads:
                if local_layers is not None:
                    raise ValueError("state reads have no carried stream state yet")
                spikes = self.state_reads[str(index)](spikes, memory_pulses, tokens)
            if self.granule_layers is not None and str(index) in self.granule_layers:
                if local_layers is not None:
                    raise ValueError("granule layers are not wired into the carried stream yet")
                spikes = self.granule_layers[str(index)](spikes, tokens)
        if tap == len(self.blocks):
            early = self._forecast_counts(spikes)
            surprise = self._surprise(early, tokens, correction_access)
        if memory_pulses is not None and self.state_reads is not None:
            # One bank of pacemakers fires for all the reads that hear it, and one for all whose neurons it gates.
            read = next(iter(self.state_reads.values()))
            if self.cfg.read_phases:
                memory_pulses.append(read.phase_counts(tokens.shape[1]).abs().sum() * tokens.shape[0])
            if self.cfg.read_turn:
                memory_pulses.append(read.turn_counts(tokens.shape[1]).abs().sum() * tokens.shape[0])
        if self.prediction_memory is not None:
            prediction = self._prediction_code(spikes)
            recall = self.prediction_memory(
                spikes, correction_access, observed=observed, prediction_code=prediction, pulse_counts=memory_pulses,
                read_losses=bank_read_losses,
            )
            spikes = self._apply_recall(spikes, recall)
        current = self.output(spikes)
        counts = self.output_emit(current, reference=self._reference_counts(current))
        if self.fast_synapses is not None:
            counts = self._excite_vocabulary(counts, tokens, surprise, early, correction_access)
        if self.read_heads is not None:
            # The heads' recall spikes are counted like those of the fast synapses.
            counts = counts + self._read_tape(tokens, surprise, early, correction_access).to(counts.device, counts.dtype)
        if return_prediction:
            if corrected_prediction:
                # The bank emits corrections in the same signed count units as the forecast.
                # The tag participates in addressing and carries no correction payload.
                payload = recall.unflatten(-1, (self.cfg.memory_heads, -1))
                prediction = prediction + F.pad(payload, (self.cfg.forecast_key_channels, 0))
            outputs = (counts, prediction, self.prediction_memory.observation_code(observed))
            return outputs if early_readout is None else (*outputs, early)
        return counts if early_readout is None else (counts, early)

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
            position = 0 if state is None else state.position
            access = _CorrectionAccess(correction, 0, hide_age, stats, shuffle_keys, local_layers=layers,
                                       local_position=position, clock=None if state is None else state.clock,
                                       synapses=None if state is None else state.synapses,
                                       heads=None if state is None else state.heads)
            counts = self(tokens, correction_access=access)
            return counts, EventChunkState(None, access.state, layers, position + tokens.shape[1], access.clock,
                                           access.synapses, access.heads)
        if state is not None and state.local_prefix is None:
            raise ValueError("chunk state must contain the local byte prefix")
        prefix = 0 if state is None else state.local_prefix.shape[1]
        if state is not None:
            tokens = torch.cat((state.local_prefix, tokens), dim=1)
        access = _CorrectionAccess(correction, prefix, hide_age, stats, shuffle_keys)
        counts = self(tokens, correction_access=access)
        return counts[..., prefix:, :], EventChunkState(tokens[:, -self.local_history:].clone(), access.state)

    @torch.no_grad()
    def initialize_deeper(self, prior: EventLMConfig, previous: dict[str, torch.Tensor]):
        """Append silent spike branches alongside the existing residual axons."""
        start = prior.residual_correction_start if prior.residual_correction_start is not None else prior.depth
        cfg = self.cfg
        expected = replace(prior, depth=cfg.depth, residual_correction_start=start, clock_start=cfg.clock_start,
                           clock_threshold=cfg.clock_threshold, clock_leak=cfg.clock_leak,
                           clock_source=cfg.clock_source, synapse_orders=cfg.synapse_orders,
                           synapse_groups=cfg.synapse_groups, synapse_cells=cfg.synapse_cells,
                           synapse_levels=cfg.synapse_levels, synapse_gate=cfg.synapse_gate)
        if cfg.depth <= prior.depth or expected != cfg or prior.route in _CORRECTION_ROUTES:
            raise ValueError("depth growth needs the same spiking router with added residual correction blocks")
        if prior.clock_start is not None and (
            prior.clock_start, prior.clock_threshold, prior.clock_leak, prior.clock_source
        ) != (cfg.clock_start, cfg.clock_threshold, cfg.clock_leak, cfg.clock_source):
            raise ValueError("depth growth keeps an existing clock")
        if prior.clock_start is None and cfg.clock_start is not None and cfg.clock_start < prior.depth:
            raise ValueError("depth growth can put only the added blocks on the clock")

        new_blocks = self.blocks[prior.depth:]
        added = {f"blocks.{index}.{name}" for index, block in enumerate(new_blocks, prior.depth)
                 for name in block.state_dict()}
        if self.state_dict().keys() - added != previous.keys():
            raise ValueError("the checkpoint must supply all original blocks and synapses")
        self.load_state_dict(previous, strict=False)
        for block in new_blocks:
            block.expand.weight.zero_()

    @torch.no_grad()
    def initialize_wider_residual(self, prior: EventLMConfig, previous: dict[str, torch.Tensor]):
        """Load the trained neurons and start added residual neurons silent."""
        cfg = self.cfg
        expected = replace(prior, dim=cfg.dim, trace_write_credit=cfg.trace_write_credit,
                           bank_write_credit=cfg.bank_write_credit,
                           trace_coincidence_start=cfg.trace_coincidence_start,
                           steering_gradient_at_zero=cfg.steering_gradient_at_zero,
                           compact_axon_state=cfg.compact_axon_state, inhibition_by_expansion=cfg.inhibition_by_expansion,
                           router_charge_bits=cfg.router_charge_bits)
        if cfg.dim <= prior.dim or cfg.continuous or expected != cfg:
            raise ValueError("widening needs the same spiking routing and memory geometry")
        if cfg.trace_coincidence_start != prior.trace_coincidence_start and (
            prior.trace_coincidence_start is not None or cfg.trace_coincidence_start is None
            or cfg.trace_coincidence_start < prior.dim
        ):
            raise ValueError("widening can introduce trace coincidence only in added residual neurons")
        if cfg.activity_inhibition and prior.inhibition_bits != cfg.inhibition_bits:
            raise ValueError("widening must preserve the inhibitory threshold")
        if cfg.router_charge_bits < prior.router_charge_bits:
            raise ValueError("widening needs at least the checkpoint's charge precision")

        weights = self.state_dict()
        if previous.keys() != weights.keys():
            raise ValueError("the wider decoder and checkpoint must contain the same synapses")
        synapses = {f"{name}.weight": module for name, module in self.named_modules() if isinstance(module, _Synapse)}
        for name, target in weights.items():
            original = previous[name]
            synapse = synapses.get(name)
            if (cfg.router_charge_bits > prior.router_charge_bits and synapse is not None
                    and synapse.charge_bits == cfg.router_charge_bits):
                # Keep existing physical weights on the old grid at initialization.
                original = _weights_on_charge_grid(original, prior.router_charge_bits)
            if target.shape == original.shape:
                weights[name] = original
                continue
            if (synapse is None or original.ndim != 2 or target.ndim != 2
                    or any(new < old for new, old in zip(target.shape, original.shape, strict=True))):
                raise ValueError(f"incompatible synapse shape while widening: {name}")
            # New inputs keep their random synapses, so gradients can reach
            # the new neurons as soon as training starts.
            output_axis = int(synapse.byte_input)
            output_added = target.shape[output_axis] - original.shape[output_axis]
            widened = target.clone()
            if output_added:
                widened.narrow(output_axis, original.shape[output_axis], output_added).zero_()
            widened[:original.shape[0], :original.shape[1]].copy_(original)
            weights[name] = widened
        self.load_state_dict(weights)

    @torch.no_grad()
    def initialize_wider_expansion(self, prior: EventLMConfig, previous: dict[str, torch.Tensor]):
        """Add silent content and steering neurons inside each fixed-delay group."""
        cfg = self.cfg
        add_transfer = prior.steering_transfer is None and cfg.steering_transfer is not None
        expected = replace(prior, hidden=cfg.hidden, steering_gradient_at_zero=cfg.steering_gradient_at_zero,
                           compact_trace_state=cfg.compact_trace_state,
                           steering_transfer=cfg.steering_transfer if add_transfer else prior.steering_transfer,
                           delay_group_widths=cfg.delay_group_widths)
        if not cfg.local_trace or cfg.hidden <= prior.hidden or expected != cfg:
            raise ValueError("expansion growth needs the same held-charge router with a larger expansion")
        if cfg.activity_inhibition and cfg.inhibition_bits != prior.inhibition_bits:
            raise ValueError("expansion growth must preserve the inhibitory threshold")
        groups = self.blocks[0].widths
        old_groups = prior.group_widths
        if any(after < before for before, after in zip(old_groups, groups, strict=True)):
            raise ValueError("expansion growth needs room for every existing delay channel")
        if self.prediction_memory is not None and cfg.memory_width is None and groups[-1] != old_groups[-1]:
            raise ValueError("expansion growth needs a separate memory_width to preserve the forecast bank")
        weights = self.state_dict()
        new_names = {name for name in weights if add_transfer and ".steering_transfer." in name}
        if weights.keys() - new_names != previous.keys():
            raise ValueError("expansion growth needs the same synaptic populations")

        old_channels = []
        offset = 0
        device = self.blocks[0].expand.weight.device
        for before, after in zip(old_groups, groups, strict=True):
            old_channels.append(torch.arange(offset, offset + before, device=device))
            offset += after
        retained = torch.cat(old_channels)
        expansion_rows = torch.cat((retained, retained + cfg.hidden))
        projection_columns = torch.cat([retained + route * cfg.hidden for route in range(4)])
        widened = previous.copy()
        for index, block in enumerate(self.blocks):
            expansion_name = f"blocks.{index}.expand.weight"
            projection_name = f"blocks.{index}.project.weight"
            expansion = torch.zeros_like(weights[expansion_name])
            projection = weights[projection_name].clone()
            expansion.index_copy_(0, expansion_rows, previous[expansion_name].to(expansion))
            projection.index_copy_(1, projection_columns, previous[projection_name].to(projection))
            widened[expansion_name] = expansion
            widened[projection_name] = projection
            if block.arrival_project is not None:
                name = f"blocks.{index}.arrival_project.weight"
                arrival = torch.zeros_like(weights[name])
                arrival.index_copy_(1, retained[:prior.trace_arrival_width], previous[name].to(arrival))
                widened[name] = arrival
            if block.opponent_steering is not None:
                name = f"blocks.{index}.opponent_steering.weight"
                opponent = torch.zeros_like(weights[name])
                opponent.index_copy_(0, retained, previous[name].to(opponent))
                widened[name] = opponent
            if block.trace.input is not None:
                name = f"blocks.{index}.trace.input.weight"
                inputs = torch.zeros_like(weights[name])
                inputs.index_copy_(1, retained, previous[name].to(inputs))
                widened[name] = inputs
            if block.trace.read_drive is not None:
                name = f"blocks.{index}.trace.read_drive.weight"
                read = torch.zeros_like(weights[name])
                read.index_copy_(-1, retained, previous[name].to(read))
                widened[name] = read
            if block.trace_taps is not None:
                name = f"blocks.{index}.trace_taps.weight"
                gains = torch.zeros_like(weights[name])
                old_widths = tuple(width for width, count in zip(old_groups, block.trace_taps.choices, strict=True)
                                   if count > 1)
                for target, source in zip(gains.split(block.trace_taps.active_widths),
                                          previous[name].split(old_widths), strict=True):
                    target[:source.numel()].copy_(source.to(target))
                widened[name] = gains
            if block.steering_transfer is not None:
                for group, weights_in_group in enumerate(block.steering_transfer.weights):
                    name = f"blocks.{index}.steering_transfer.weights.{group}"
                    enlarged = torch.zeros_like(weights_in_group)
                    if not add_transfer:
                        original = previous[name]
                        enlarged[:original.shape[0], :original.shape[1]].copy_(original.to(enlarged))
                    widened[name] = enlarged

        if any(value.shape != weights[name].shape for name, value in widened.items()):
            raise ValueError("expansion growth would change a synaptic population outside the delay groups")
        self.load_state_dict(widened)

    @torch.no_grad()
    def preserve_router_charge(self, prior_bits):
        """Place the loaded routing synapses on their previous physical charge grid."""
        if self.cfg.router_charge_bits <= prior_bits:
            raise ValueError("preserving router charge requires a finer target grid")
        for block in self.blocks:
            for synapse in (block.expand, block.project, block.arrival_project):
                if synapse is not None:
                    synapse.weight.copy_(_weights_on_charge_grid(synapse.weight, prior_bits))

    @torch.no_grad()
    def to_integer(self):
        """Export an eval-mode CPU decoder with int32 synapses."""
        if self.cfg.continuous:
            raise ValueError("the continuous control has no integer export")
        model = copy.deepcopy(self).cpu().eval()
        for submodule in model.modules():
            if isinstance(submodule, (_Synapse, _DelaySelector, _PacketThresholds, _DelayCorrection, _DelayFilter,
                                      _OpponentSteering, _SpreadCoincidence, _TraceInput, _TraceTapRead,
                                      _SteeringTransfer, _TraceReadDrive)):
                submodule.freeze_integer()
        return model

    @torch.no_grad()
    def step(self, tokens, state: EventLMState | None = None):
        """Apply one byte per stream using bounded integer state."""
        if self.input.weight.is_floating_point() or self.training:
            raise ValueError("step requires an eval-mode integer export from to_integer()")
        if self.state_reads is not None or self.granule_layers is not None:
            raise ValueError("state reads and cell layers carry no state through a stream yet: "
                             "run this decoder on whole windows")
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
        forecast = surprise = None
        for index, (block, layer) in enumerate(zip(self.blocks, state.layers, strict=True)):
            if index == self.cfg.clock_start:
                forecast = self._forecast_counts(spikes)
                surprise = self._step_surprise(tokens, None if state.clock is None else state.clock.forecast)
                spikes = self._step_clocked(spikes, tokens, state, forecast, surprise)
                break
            spikes = block.step(spikes, layer, state.position)
        if self.fast_synapses is not None:
            if state.synapses is None:
                state.synapses = self.fast_synapses.new_state(tokens.shape[0])
            if self.cfg.clock_start is None:
                forecast = self._forecast_counts(spikes)
                surprise = self._step_surprise(tokens, state.synapses.forecast)
                state.synapses.forecast = forecast.to(_spike_storage_dtype(self.forecast_emit.cap))
        if self.read_heads is not None:
            if state.heads is None:
                state.heads = self.read_heads.new_state(tokens.shape[0])
            if self.cfg.clock_start is None and self.fast_synapses is None:
                forecast = self._forecast_counts(spikes)
                surprise = self._step_surprise(tokens, state.heads.forecast)
                state.heads.forecast = forecast.to(_spike_storage_dtype(self.forecast_emit.cap))
        if self.prediction_memory is not None:
            recall, state.prediction_memory = self.prediction_memory.step(
                spikes, state.prediction_memory, observed=observed, prediction_code=self._prediction_code(spikes),
            )
            spikes = self._apply_recall(spikes, recall)
        current = self.output(spikes)
        counts, state.output_phase = self.output_emit.step(
            current, state.output_phase, reference=self._reference_counts(current),
        )
        if self.fast_synapses is not None:
            votes, levels, state.synapses = self.fast_synapses.step(tokens, surprise, state.synapses)
            counts = self._add_recall_spikes(counts, votes, levels)
        if self.read_heads is not None:
            fired, state.heads = self.read_heads.step(tokens, surprise, state.heads)
            counts = counts + fired.to(counts.dtype)
        state.position += 1
        return counts, state

    @staticmethod
    def _step_surprise(tokens, previous):
        if previous is None:
            return torch.zeros(tokens.shape[0], dtype=torch.int32)
        previous = previous.to(torch.int32)
        return previous.amax(-1) - previous.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

    def _step_clocked(self, spikes, tokens, state, forecast, surprise):
        cfg = self.cfg
        batch = tokens.shape[0]
        clock = state.clock
        if clock is None:
            clock = state.clock = ClockState(
                torch.zeros(batch, dtype=torch.int16), None,
                torch.zeros((batch, cfg.dim), dtype=_spike_storage_dtype(2 * self.input_emit.cap)),
                torch.zeros(batch, dtype=torch.int64),
            )
        drive = torch.ones(batch, dtype=torch.int32) if cfg.clock_source == "periodic" else surprise
        membrane = (clock.membrane + drive.clamp(max=cfg.clock_threshold) - cfg.clock_leak).clamp_min(0)
        ticks = membrane >= cfg.clock_threshold
        clock.membrane = (membrane - cfg.clock_threshold * ticks).to(torch.int16)
        clock.forecast = forecast.to(_spike_storage_dtype(self.forecast_emit.cap))
        layers = state.layers[cfg.clock_start:]
        blocks = self.blocks[cfg.clock_start:]
        for count in clock.events[ticks].unique().tolist():
            # Streams at the same event index share the ring position of their delay lines.
            rows = (ticks & (clock.events == count)).nonzero().squeeze(-1)
            parts = [_state_rows(layer, rows) for layer in layers]
            emitted = spikes[rows]
            for block, part in zip(blocks, parts, strict=True):
                emitted = block.step(emitted, part, count)
            clock.held[rows] = (emitted - spikes[rows]).to(clock.held.dtype)
            for layer, part in zip(layers, parts, strict=True):
                _write_state_rows(layer, rows, part, batch)
        clock.events += ticks
        if self.clock_intervention == "mute":
            return spikes
        cap = self.input_emit.cap
        return (spikes + clock.held).clamp(-cap, cap)

    @staticmethod
    def byte(counts):
        """Decode the winning byte directly from its spike evidence."""
        return counts.argmax(dim=-1)
