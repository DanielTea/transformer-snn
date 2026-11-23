"""Telemetry as byte streams for a stack of reads and cells.

A sensor's samples become bytes in two steps. A ``Quantizer`` fitted on a stretch of normal operation maps every
value to one of a fixed number of levels, and an encoding writes every level as a fixed number of bytes: ``digits``
as a zero-padded decimal number and a newline, ``printable`` as one printable ASCII character, and ``delta`` the
change of level since the sample before as one printable character, so that a steady sensor writes the same byte
again and again and a jump writes a rare one. A stack trained on text has seen digits, newlines and printable
characters, so no encoding hands it a byte it never heard.

A ``StreamReader`` reads many such streams side by side, one per sensor, and keeps the state that the stack carries
for each of them: the tags and payloads of its reads, the fibres of its cell layers and, when the stack has fast
output synapses, what they have learned from that stream alone. That state is the stream's capsule. It can be saved
and resumed, and a stream read in pieces, or stopped, saved and resumed, is scored exactly as one pass over all of it.

The reader's measure is the surprise of every sample: the bits that the stack's predictions spent on its bytes, each
byte predicted before it arrived, from everything that the stream held before it. A stream that carries its state
adapts to what it reads, and a new state that holds steady soon costs few bits. A reader can therefore also
``mark`` the capsule of a stretch of normal operation and ``rewind`` to it before every short piece, so that each
piece is predicted from the normal and its surprise says how far the stream now lies from it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ENCODINGS = ("digits", "printable", "delta")
# One printable character per level: '!' (33) to '~' (126). The space is left out, a word separator in text.
_PRINTABLE_FIRST, _PRINTABLE_LEVELS = 33, 94
# The delta encoding writes changes from -46 to +47 levels; no change is the character 'O'.
_DELTA_REACH = 46


@dataclass(frozen=True)
class Quantizer:
    """Values to whole levels from 0 to ``levels - 1``, evenly between ``low`` and ``high``, clipped at both ends."""

    low: float
    high: float
    levels: int

    @classmethod
    def fit(cls, values, levels, *, quantiles=(0.005, 0.995), margin=0.5):
        """The range of normal values: the quantiles of ``values``, widened by ``margin`` of their span on each side,
        so that a value somewhat beyond normal still has a level of its own before the clip."""
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("a quantizer is fitted on at least one finite value")
        low, high = np.quantile(values, quantiles)
        span = high - low
        if span <= 0:
            span = max(abs(float(low)), 1.0) * 1e-3
        return cls(float(low - margin * span), float(high + margin * span), int(levels))

    def __call__(self, values):
        values = np.asarray(values, dtype=np.float64)
        scaled = (values - self.low) / (self.high - self.low) * (self.levels - 1)
        return np.clip(np.rint(np.nan_to_num(scaled, nan=0.0)), 0, self.levels - 1).astype(np.int64)

    def value(self, levels):
        """The value at the middle of each level."""
        return self.low + np.asarray(levels, dtype=np.float64) / (self.levels - 1) * (self.high - self.low)


def sample_width(encoding, levels):
    """The bytes that one sample takes in an encoding."""
    if encoding == "digits":
        return len(str(levels - 1)) + 1
    if encoding == "printable":
        if levels > _PRINTABLE_LEVELS:
            raise ValueError(f"the printable encoding has {_PRINTABLE_LEVELS} levels")
        return 1
    if encoding == "delta":
        return 1
    raise ValueError(f"unknown encoding {encoding!r}")


def encode(levels, encoding, count):
    """The bytes of every sample as [..., width] (int64), for levels out of ``count``."""
    levels = np.asarray(levels, dtype=np.int64)
    if levels.size and (levels.min() < 0 or levels.max() >= count):
        raise ValueError("a level lies outside the quantizer's range")
    width = sample_width(encoding, count)
    if encoding == "printable":
        return (levels + _PRINTABLE_FIRST)[..., None]
    if encoding == "delta":
        # The change along the last axis, the first sample's from nothing; changes beyond the 94 characters clip.
        change = np.diff(levels, axis=-1, prepend=levels[..., :1])
        return (np.clip(change, -_DELTA_REACH, _PRINTABLE_LEVELS - 1 - _DELTA_REACH) + _DELTA_REACH
                + _PRINTABLE_FIRST)[..., None]
    digits = width - 1
    out = np.empty(levels.shape + (width,), dtype=np.int64)
    rest = levels.copy()
    for place in range(digits - 1, -1, -1):
        out[..., place] = ord("0") + rest % 10
        rest //= 10
    out[..., digits] = ord("\n")
    return out


def decode(samples, encoding):
    """The levels of bytes that ``encode`` wrote, as [...]. For ``delta`` they are the changes, which add up to the
    levels after the first sample as long as no change was clipped."""
    samples = np.asarray(samples, dtype=np.int64)
    if encoding == "printable":
        return samples[..., 0] - _PRINTABLE_FIRST
    if encoding == "delta":
        return samples[..., 0] - _PRINTABLE_FIRST - _DELTA_REACH
    digits = samples[..., :-1] - ord("0")
    return (digits * 10 ** np.arange(digits.shape[-1] - 1, -1, -1)).sum(-1)


class StreamReader:
    """Streams of samples read side by side by one stack, each with its own carried state.

    ``read`` takes the next samples of every stream as [streams, samples, width] bytes and gives the surprise of
    each as [streams, samples] in bits. The prediction of a sample's first byte is the one that the stack made after
    the last byte before it, so a stream is scored the same however it is cut into reads.
    """

    def __init__(self, model, streams):
        self.model, self.streams = model, int(streams)
        self.state, self.last = None, None
        self.samples = 0

    @torch.inference_mode()
    def read(self, samples):
        samples = torch.as_tensor(samples, dtype=torch.long)
        if samples.ndim != 3 or samples.shape[0] != self.streams or samples.shape[1] == 0:
            raise ValueError("expected the next samples of every stream as [streams, samples, width]")
        streams, count, width = samples.shape
        device = next(self.model.parameters()).device
        tokens = samples.reshape(streams, count * width).to(device)
        counts, self.state = self.model.stream(tokens, self.state)
        logp = F.log_softmax(counts.float() * (math.log(2) / (8 * self.model.cfg.resolution)), dim=-1)
        first = (torch.full((streams, 1, logp.shape[-1]), -math.log(logp.shape[-1]), device=device)
                 if self.last is None else self.last[:, None])
        before = torch.cat((first, logp[:, :-1]), 1)
        bits = -before.gather(-1, tokens[..., None])[..., 0].double() / math.log(2)
        self.last = logp[:, -1]
        self.samples += count
        return bits.view(streams, count, width).sum(-1).cpu().numpy()

    def mark(self):
        """The reader's place: a later ``rewind`` to it reads on as if nothing had been read since. Reading never
        changes a state that the stack handed out, so a mark can be rewound to any number of times."""
        return self.state, self.last, self.samples

    def rewind(self, mark):
        self.state, self.last, self.samples = mark

    def save(self, path):
        """Write every stream's capsule: what the stack carries for it and its last prediction."""
        torch.save({"streams": self.streams, "samples": self.samples, "state": self.state, "last": self.last},
                   Path(path))

    @classmethod
    def resume(cls, model, path):
        """A reader that goes on where a saved one stopped. Only load capsules that you wrote yourself: they are
        pickled objects, not plain tensors."""
        saved = torch.load(Path(path), map_location=next(model.parameters()).device, weights_only=False)
        reader = cls(model, saved["streams"])
        reader.state, reader.last, reader.samples = saved["state"], saved["last"], saved["samples"]
        return reader


@dataclass(frozen=True)
class Event:
    """A stream leaving or regaining its normal: ``kind`` is ``watch``, ``alarm`` or ``clear``."""

    sample: int
    stream: int
    kind: str
    score: float


class SurpriseMonitor:
    """Events from the surprise of every stream, each measured against its own normal.

    ``calibrate`` takes the bits of a stretch of normal operation as [streams, samples]. A stream's surprise is then
    counted in robust standard deviations from its calibration median and smoothed by an exponential mean of span
    ``span``. The stream is on ``watch`` above the 99th percentile that this smoothed score reached in calibration,
    on ``alarm`` above ``alarm_factor`` times it, and ``clear`` again when it falls below the 90th percentile. A
    stream changes state only after the score has stayed on the new side for ``hold`` samples in a row, so that a
    single surprising sample raises nothing.
    """

    def __init__(self, streams, *, span=5, alarm_factor=2.0, hold=3):
        self.streams, self.alpha, self.alarm_factor = int(streams), 2 / (span + 1), float(alarm_factor)
        self.hold, self.pending = int(hold), [None] * int(streams)
        self.median = self.spread = self.watch = self.clear = None
        self.level = np.zeros(self.streams)
        self.state = ["normal"] * self.streams
        self.samples = 0

    def _scores(self, bits):
        return (np.asarray(bits, dtype=np.float64) - self.median[:, None]) / self.spread[:, None]

    def _smooth(self, scores):
        out = np.empty_like(scores)
        for t in range(scores.shape[1]):
            self.level = self.alpha * scores[:, t] + (1 - self.alpha) * self.level
            out[:, t] = self.level
        return out

    def calibrate(self, bits, warm=0):
        bits = np.asarray(bits, dtype=np.float64)
        if bits.shape[0] != self.streams or bits.shape[1] <= warm + 1:
            raise ValueError("calibrate on [streams, samples] bits, more samples than the warm-up")
        reference = bits[:, warm:]
        self.median = np.median(reference, 1)
        self.spread = 1.4826 * np.median(np.abs(reference - self.median[:, None]), 1) + 1e-9
        self.level = np.zeros(self.streams)
        smoothed = self._smooth(self._scores(bits))[:, warm:]
        self.watch = np.maximum(np.quantile(smoothed, 0.99, axis=1), 1.0)
        self.clear = np.quantile(smoothed, 0.90, axis=1)
        self.samples = bits.shape[1]

    def update(self, bits):
        """The events that the next samples [streams, samples] raise, in order."""
        if self.median is None:
            raise ValueError("calibrate the monitor before it watches")
        smoothed, events = self._smooth(self._scores(bits)), []
        for t in range(smoothed.shape[1]):
            for stream in range(self.streams):
                score, state = float(smoothed[stream, t]), self.state[stream]
                watch, clear = self.watch[stream], self.clear[stream]
                # One step at a time: normal to watch to alarm, and back to normal from either.
                if state == "normal":
                    kind = "watch" if score > watch else None
                elif score < clear:
                    kind = "clear"
                elif state == "watch" and score > self.alarm_factor * watch:
                    kind = "alarm"
                else:
                    kind = None
                run = self.pending[stream]
                run = None if kind is None else ((kind, run[1] + 1) if run and run[0] == kind else (kind, 1))
                self.pending[stream] = run
                if run is None or run[1] < self.hold:
                    continue
                self.pending[stream] = None
                self.state[stream] = "normal" if kind == "clear" else kind
                events.append(Event(self.samples + t, stream, kind, score))
        self.samples += smoothed.shape[1]
        return events

    def snapshot(self):
        """What the monitor needs to go on later, as plain lists."""
        return {"streams": self.streams, "alpha": self.alpha, "alarm_factor": self.alarm_factor,
                "median": self.median.tolist(), "spread": self.spread.tolist(), "watch": self.watch.tolist(),
                "clear": self.clear.tolist(), "level": self.level.tolist(), "state": list(self.state),
                "hold": self.hold, "pending": [None if run is None else list(run) for run in self.pending],
                "samples": self.samples}

    @classmethod
    def restore(cls, snapshot):
        monitor = cls(snapshot["streams"], alarm_factor=snapshot["alarm_factor"], hold=snapshot["hold"])
        monitor.pending = [None if run is None else tuple(run) for run in snapshot["pending"]]
        monitor.alpha = snapshot["alpha"]
        for name in ("median", "spread", "watch", "clear", "level"):
            setattr(monitor, name, np.asarray(snapshot[name], dtype=np.float64))
        monitor.state, monitor.samples = list(snapshot["state"]), snapshot["samples"]
        return monitor
