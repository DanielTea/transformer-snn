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
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shlex
import subprocess
import tempfile
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
        self.muted = False

    @torch.no_grad()
    def forward(self, tokens, surprise):
        """Recall spikes ``[batch, bytes, 256]`` for windows that each start from an empty tape."""
        tokens = tokens.to("cpu", torch.uint8).numpy()
        surprise = surprise.to("cpu", torch.int16).numpy()
        batch, steps = tokens.shape
        readout = self.readout.to("cpu", torch.int64).numpy()
        spikes = np.zeros((batch, steps, _VOCAB), dtype=np.int64)
        if self.muted:
            return torch.from_numpy(spikes)
        position = np.arange(steps)
        for row in range(batch):
            # One spare byte keeps the last target inside the replay. It is never read.
            data = np.concatenate((tokens[row], np.zeros(1, dtype=np.uint8)))
            for head, order in enumerate(self.orders):
                pred, charge, _ = coast(data, surprise[row], order, self.cells, self.gate, self.tape, self.lookup,
                                        self.miss, self.relock)
                live = pred >= 0
                np.add.at(spikes[row], (position[live], pred[live]), readout[head, charge_level(charge[live])])
        return torch.from_numpy(spikes)
