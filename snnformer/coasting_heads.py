"""Coasting read heads on a tape of the byte stream.

The tape is a ring of cells that holds the last bytes of the stream. A head
is one neuron sitting on a cell. Every byte it moves one cell on and fires the
recall neuron of the byte stored there. A confirmed byte adds one unit of
charge and a wrong byte kills the head, so its charge counts the bytes it has
followed without an error. Key neurons address the tape: each keeps one
synapse onto a tape cell. A byte that arrives with error spikes moves the
synapse of the previous context's key neuron onto its own cell. When no head
is alive, the key neuron of the current context places one on the cell its
synapse reaches.

The fast synapses of v11 are the case in which every head dies after one byte
and a synapse reaches a recall neuron instead of a cell. Nothing here is
trained by gradient descent. One integer strength per head and charge level
is fitted afterwards.

``coast`` replays one head over a recorded stream. ``CoastingHeads`` runs all
heads inside the decoder, over windows that start from an empty tape or over
one stream that arrives in chunks or single bytes with a bounded state.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_VOCAB = 256
# Charge levels: 0, 1, 2, 3, 4-5, 6-7, 8-11, 12-15, 16-23, 24-31, 32-63, 64-127, 128 and more confirmed bytes.
CHARGE_STEPS = np.array([0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128])
LEVELS = len(CHARGE_STEPS)
SEED = 0x9E3779B1
_LIBRARY = None


def library():
    """The compiled replay. ``CC`` chooses the compiler and ``SNN_BUILD_DIR`` where the library is kept."""
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    source = Path(__file__).with_name("coasting_heads.c")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    work = Path(os.environ.get("SNN_BUILD_DIR") or Path(tempfile.gettempdir()) / f"snnformer-{os.getuid()}")
    work.mkdir(parents=True, exist_ok=True)
    target = work / f"coasting_heads_{digest}.so"
    if not target.exists():
        compilers = [shlex.split(os.environ["CC"])] if os.environ.get("CC") else [["cc"], ["gcc"]]
        partial = target.with_suffix(f".{os.getpid()}.tmp")
        for compiler in compilers:
            built = subprocess.run([*compiler, "-O3", "-shared", "-fPIC", "-o", str(partial), str(source)],
                                   capture_output=True)
            if built.returncode == 0 and partial.exists():
                os.replace(partial, target)
                break
        else:
            raise RuntimeError(f"no C compiler built {source.name}: {built.stderr.decode()[-400:]}")
    lib = ctypes.CDLL(str(target))
    lib.coast.restype = None
    lib.coast.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int64, ctypes.c_int,
                          ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32, ctypes.c_int,
                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.coast_chunk.restype = None
    lib.coast_chunk.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int64, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_void_p]
    _LIBRARY = lib
    return lib


def _context_key(window, seed):
    key = seed
    for byte in window:
        key = ((key ^ int(byte)) * 16777619) & 0xFFFFFFFF
    key ^= key >> 15
    key = (key * 2246822519) & 0xFFFFFFFF
    return key ^ (key >> 13)


def coast_reference(data, surprise, order, cells, gate, tape, lookup, miss, relock, seed, lifetime=0):
    """The replay of ``coasting_heads.c`` in plain Python, for tests."""
    rows = len(data) - 1
    pred = np.full(rows, -1, dtype=np.int16)
    charge = np.zeros(rows, dtype=np.int32)
    synapse = {}
    pointer, held, emitted = -1, 0, 0
    for t in range(rows):
        if pointer >= 0:
            if lifetime > 0 and emitted >= lifetime:
                pointer, held = -1, 0
            elif data[pointer] == data[t]:
                held, pointer = min(held + 1, 1000000), pointer + 1
            elif miss > 0 and held >> miss > 0:
                held, pointer = held >> miss, pointer + 1
            else:
                pointer, held = -1, 0
        if t >= order and surprise[t] >= gate:
            synapse[_context_key(data[t - order:t], seed) & (cells - 1)] = t
        wants = pointer < 0 or (relock > 0 and held < relock and surprise[t] >= 1)
        if wants and t + 1 >= order and surprise[t] >= lookup:
            cell = synapse.get(_context_key(data[t + 1 - order:t + 1], seed) & (cells - 1), -1)
            if cell >= 0:
                if tape > 0:
                    cell = t - ((t - cell) % tape)
                pointer, held, emitted = cell, 0, 0
        if pointer >= 0:
            pred[t], charge[t] = data[pointer], held
            emitted += 1
    return pred, charge


def coast(data, surprise, order, cells, gate, tape, lookup, miss, relock, seed=SEED, lifetime=0):
    """Predicted byte (-1 for none) and charge of one head for every target of a continuous stream.

    ``data`` holds the bytes and ``surprise[t]`` the error spikes left when byte ``t`` arrived.
    Target ``t`` is byte ``t + 1``. The last byte of ``data`` is never read.
    """
    data = np.ascontiguousarray(data, dtype=np.uint8)
    surprise = np.ascontiguousarray(surprise, dtype=np.int16)
    rows = len(data) - 1
    if len(surprise) < rows:
        raise ValueError("every arriving byte needs its error spikes")
    pred = np.full(rows, -1, dtype=np.int16)
    charge = np.zeros(rows, dtype=np.int32)
    counts = np.zeros(4, dtype=np.int64)
    library().coast(data.ctypes.data, surprise.ctypes.data, len(data), order, cells, gate, tape, lookup, miss, relock,
                    (seed + order) & 0xFFFFFFFF, lifetime, pred.ctypes.data, charge.ctypes.data, counts.ctypes.data)
    return pred, charge, counts


def charge_level(charge):
    """1 for a head that was just placed, up to ``LEVELS`` for 128 and more confirmed bytes."""
    return np.searchsorted(CHARGE_STEPS, charge, side="right")


@dataclass
class HeadState:
    """What the read heads carry from one byte to the next.

    ``tape`` is the ring of the last bytes. ``synapse`` holds the tape cell
    that each key neuron's synapse reaches as 16 bits, and ``written`` one bit
    per key neuron that has a synapse. ``pointer`` is the cell under each head,
    -1 for none, and ``charge`` the bytes it has confirmed.
    """

    tape: torch.Tensor
    synapse: torch.Tensor
    written: torch.Tensor
    pointer: torch.Tensor
    charge: torch.Tensor
    cell: int = 0
    seen: int = 0
    forecast: torch.Tensor | None = None


class CoastingHeads(nn.Module):
    """One coasting head per context length, with a fitted number of recall spikes per charge level."""

    def __init__(self, orders, cells, gate, tape, lookup=0, miss=0, relock=0):
        super().__init__()
        orders = tuple(int(order) for order in orders)
        if not orders or min(orders) < 1 or len(set(orders)) != len(orders):
            raise ValueError("coasting heads need distinct positive context lengths")
        if cells < 2 or cells & (cells - 1) or gate < 0 or tape < 0 or lookup < 0 or miss < 0 or relock < 0:
            raise ValueError("coasting heads need a power-of-two number of key neurons and non-negative settings")
        self.orders, self.cells, self.gate, self.tape = orders, cells, gate, tape
        self.lookup, self.miss, self.relock = lookup, miss, relock
        # Recall spikes fired by each charge level of each head. Level 0 is a silent head.
        self.register_buffer("readout", torch.zeros(len(orders), LEVELS + 1, dtype=torch.int32))
        self.capture = None
        self.muted = False

    @property
    def longest(self):
        return max(self.orders)

    @property
    def state_bytes(self):
        """Persistent bytes of one stream: the tape, the key synapses with their marks, and the heads."""
        return self.tape + len(self.orders) * (2 * self.cells + (self.cells + 7) // 8 + 8)

    def new_state(self, batch, forecast=None):
        if not self.longest <= self.tape <= 1 << 16:
            raise ValueError("carried heads need a tape of at most 65,536 cells that holds the longest context")
        heads = len(self.orders)
        return HeadState(torch.zeros((batch, self.tape), dtype=torch.uint8),
                         torch.zeros((batch, heads, self.cells), dtype=torch.int16),
                         torch.zeros((batch, heads, (self.cells + 7) // 8), dtype=torch.uint8),
                         torch.full((batch, heads), -1, dtype=torch.int32),
                         torch.zeros((batch, heads), dtype=torch.int32), 0, 0, forecast)

    def _carried(self, tokens, surprise, state):
        """Predicted byte and charge ``[batch, heads, bytes]`` of every head, advancing the carried state."""
        batch, steps = tokens.shape
        heads = len(self.orders)
        orders = np.asarray(self.orders, dtype=np.int32)
        seeds = ((SEED + orders.astype(np.int64)) & 0xFFFFFFFF).astype(np.uint32)
        pred = np.empty((batch, heads, steps), dtype=np.int16)
        charge = np.empty((batch, heads, steps), dtype=np.int32)
        # The state tensors are contiguous CPU storage, so the replay advances them in place.
        fields = [getattr(state, name).numpy() for name in ("tape", "synapse", "written", "pointer", "charge")]
        if not all(field.flags.c_contiguous for field in fields):
            raise ValueError("head state must be contiguous CPU storage")
        tape, synapse, written, pointer, held = fields
        coast_chunk = library().coast_chunk
        for row in range(batch):
            coast_chunk(tokens[row].ctypes.data, surprise[row].ctypes.data, steps, heads, orders.ctypes.data,
                        seeds.ctypes.data, self.cells, self.gate, self.tape, self.lookup, self.miss, self.relock,
                        tape[row].ctypes.data, state.cell, state.seen, synapse[row].ctypes.data,
                        written[row].ctypes.data, pointer[row].ctypes.data, held[row].ctypes.data,
                        pred[row].ctypes.data, charge[row].ctypes.data)
        state.cell = (state.cell + steps) % self.tape
        state.seen = min(state.seen + steps, self.longest)
        return pred, charge

    def excitation(self, pred, charge):
        """Recall spikes ``[batch, bytes, 256]`` of the heads' predictions ``[batch, heads, bytes]``."""
        batch, _, steps = pred.shape
        readout = self.readout.to("cpu", torch.int64).numpy()
        spikes = np.zeros((batch, steps, _VOCAB), dtype=np.int64)
        for head in range(len(self.orders)):
            row, position = np.nonzero(pred[:, head] >= 0)
            strength = readout[head, charge_level(charge[:, head][row, position])]
            np.add.at(spikes, (row, position, pred[:, head][row, position]), strength)
        return torch.from_numpy(spikes)

    def activity(self, pred, charge):
        """Spikes of the head circuit for predictions ``[batch, heads, bytes]`` of windows from an empty tape.

        A key neuron fires in every head whose context is complete. The tape
        cell under a live head fires its byte, and the head fires the recall
        neuron of that byte with the strength of its charge level.
        """
        steps = pred.shape[-1]
        live = pred >= 0
        strength = np.abs(self.readout.to("cpu", torch.int64).numpy())
        recall = sum(int(strength[head, charge_level(charge[:, head][live[:, head]])].sum())
                     for head in range(len(self.orders)))
        return {"key": pred.shape[0] * sum(max(steps - order + 1, 0) for order in self.orders),
                "tape": int(live.sum()), "recall": recall}

    @torch.no_grad()
    def forward(self, tokens, surprise, state=None, stats=None):
        """Recall spikes ``[batch, bytes, 256]``.

        Without a state every window starts from an empty tape. With one, the
        heads continue the stream and the state advances in place. ``stats``
        receives the live heads and the recall spikes they fired.
        """
        tokens = np.ascontiguousarray(tokens.to("cpu", torch.uint8).numpy())
        surprise = np.ascontiguousarray(surprise.to("cpu", torch.int16).numpy())
        batch, steps = tokens.shape
        if state is not None:
            pred, charge = self._carried(tokens, surprise, state)
        else:
            pred = np.empty((batch, len(self.orders), steps), dtype=np.int16)
            charge = np.empty((batch, len(self.orders), steps), dtype=np.int32)
            for row in range(batch):
                # One spare byte keeps the last target inside the replay. It is never read.
                data = np.concatenate((tokens[row], np.zeros(1, dtype=np.uint8)))
                for head, order in enumerate(self.orders):
                    pred[row, head], charge[row, head], _ = coast(
                        data, surprise[row], order, self.cells, self.gate, self.tape, self.lookup, self.miss,
                        self.relock)
        if self.capture is not None:
            self.capture.append((pred, charge, surprise))
        if stats is not None:
            fired = self.activity(pred, charge)
            stats["live_tape_heads"] = torch.tensor(fired["tape"])
            stats["tape_recall_spikes"] = torch.tensor(fired["recall"])
        if self.muted:
            return torch.zeros((batch, steps, _VOCAB), dtype=torch.int64)
        return self.excitation(pred, charge)

    @torch.no_grad()
    def step(self, tokens, surprise, state):
        """One byte per stream: recall spikes ``[batch, 256]`` and the advanced state."""
        return self(tokens[:, None], surprise[:, None], state)[:, 0], state
