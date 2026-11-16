"""Surprise-written fast synapses from context key neurons to recall neurons.

Each head watches the last few bytes. A fixed random projection lets one key
neuron win in each of its groups, so a context is a small set of active key
neurons. Every key neuron keeps exactly one potentiated synapse onto 256
recall neurons, one per byte value. When a byte arrives and the forecast
neurons left error spikes, the key neurons that were active just before it
move their synapse to that byte. On later visits the active key neurons vote
through their synapses and the agreeing votes fire one recall neuron. The
decoder counts a recall neuron's spikes together with the spikes of the
vocabulary neuron for the same byte. Without error spikes no synapse moves.

Nothing here is trained by gradient descent. The context codes are fixed and
only one integer strength per head and agreement level is fitted afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

_VOCAB = 256
_PRIME = (1 << 31) - 1


@dataclass
class SynapseState:
    """What the fast synapses carry from one byte to the next.

    ``successor`` holds one byte per key neuron: the recall neuron its
    potentiated synapse reaches, or 0 for none. Byte 0 is therefore never stored.
    """

    successor: torch.Tensor
    history: torch.Tensor
    seen: int = 0
    forecast: torch.Tensor | None = None

    @property
    def written(self):
        return self.successor != 0


class FastSynapses(nn.Module):
    """Key neurons with one potentiated synapse each onto the recall neurons.

    ``orders`` gives the context length of every head in bytes. Each head has
    ``groups`` groups of ``cells`` key neurons. ``levels`` is the number of
    agreement levels that fire a recall neuron: fewer than
    ``groups - levels + 1`` agreeing synapses stay below threshold. A byte is
    stored only when it arrives with at least ``gate`` error spikes, so gate 0
    stores every byte.
    """

    def __init__(self, orders, groups, cells, levels, gate, *, seed=0):
        super().__init__()
        orders = tuple(int(order) for order in orders)
        if not orders or min(orders) < 1 or len(set(orders)) != len(orders):
            raise ValueError("fast synapses need distinct positive context lengths")
        if groups < 1 or cells < 2 or cells & (cells - 1) or not 1 <= levels <= groups or gate < 0:
            raise ValueError("fast synapses need groups, a power-of-two group size, levels within groups and a gate")
        self.orders, self.groups, self.cells, self.levels, self.gate = orders, groups, cells, levels, gate
        self.longest = max(orders)
        generator = torch.Generator().manual_seed(seed)
        weights = torch.randint(1, _PRIME, (len(orders), groups, self.longest), generator=generator)
        for head, order in enumerate(orders):
            weights[head, :, order:] = 0
        # The fixed random projection from the delayed bytes onto each group. Index 0 is the newest byte.
        self.register_buffer("projection", weights)
        # Recall spikes fired by each agreement level of each head. Level 0 is silent.
        self.register_buffer("readout", torch.zeros(len(orders), levels + 1, dtype=torch.int32))
        self.capture = None
        self.muted = False

    @property
    def state_cells(self):
        return len(self.orders) * self.groups * self.cells

    def new_state(self, batch, forecast=None):
        shape = (batch, len(self.orders), self.groups * self.cells)
        return SynapseState(torch.zeros(shape, dtype=torch.uint8),
                            torch.zeros((batch, self.longest), dtype=torch.uint8), 0, forecast)

    def _winners(self, windows):
        """The winning key neuron of every group for byte windows ordered oldest to newest."""
        drive = (windows.flip(-1).to(torch.int64) + 1) @ self.projection.cpu().flatten(0, 1).T
        winner = (drive % _PRIME) & (self.cells - 1)
        offsets = torch.arange(self.groups, dtype=torch.int64) * self.cells
        return winner.unflatten(-1, (len(self.orders), self.groups)) + offsets

    def _levels(self, votes):
        same = votes.unsqueeze(-1) == votes.unsqueeze(-2)
        agreeing = same.sum(-1)
        first = same.to(torch.uint8).argmax(-1) == torch.arange(self.groups)
        level = (agreeing - (self.groups - self.levels)).clamp(0, self.levels)
        return level * (first & (votes >= 0))

    def excitation(self, votes, levels):
        """Recall spikes from the agreeing synapses of all heads."""
        strength = self.readout.to("cpu", torch.int64).unsqueeze(-1).expand(-1, -1, self.groups)
        strength = strength.expand(*levels.shape[:-2], -1, -1, -1).gather(-2, levels.unsqueeze(-2)).squeeze(-2)
        spikes = torch.zeros((*votes.shape[:-2], _VOCAB), dtype=torch.int64)
        return spikes.scatter_add_(-1, votes.clamp_min(0).flatten(-2), strength.flatten(-2))

    @torch.no_grad()
    def forward(self, tokens, surprise, state=None, *, carry=False):
        """Votes and agreement levels for ``[batch, bytes]`` and the carried state.

        ``surprise[:, t]`` holds the error spikes left when byte ``t`` arrived.
        The returned votes are ``[batch, bytes, heads, groups]`` with -1 where a
        key neuron has no potentiated synapse yet.
        """
        tokens = tokens.to("cpu", torch.int64)
        surprise = surprise.to("cpu", torch.int64)
        batch, steps = tokens.shape
        if state is None and carry:
            state = self.new_state(batch)
        seen = 0 if state is None else state.seen
        history = tokens.new_zeros((batch, self.longest)) if state is None else state.history.to(torch.int64)
        padded = torch.cat((history, tokens), dim=1)
        # Window q ends just before byte q of this chunk, so q = t writes and q = t + 1 reads.
        winners = self._winners(padded.unfold(1, self.longest, 1))
        consumed = seen + torch.arange(steps + 1)
        # Byte 0 marks a key neuron without a synapse and is never stored.
        present = tokens != 0
        stores = (surprise >= self.gate) & present
        span, period = self.groups * self.cells, 2 * steps + 2
        rows = torch.arange(batch).view(-1, 1, 1)
        clock = torch.arange(steps).view(1, -1, 1)
        votes = torch.full((batch, steps, len(self.orders), self.groups), -1, dtype=torch.int64)
        for head, order in enumerate(self.orders):
            valid = consumed >= order
            before, after = winners[:, :-1, head], winners[:, 1:, head]
            # A visit is a byte that arrives with error spikes after a complete context.
            visit = (stores & valid[:-1]).unsqueeze(-1).expand(-1, -1, self.groups)
            visit_key = ((rows * span + before) * period + 2 * clock)[visit]
            visit_value = tokens.unsqueeze(-1).expand(-1, -1, self.groups)[visit]
            read_key = ((rows * span + after) * period + 2 * clock + 1).flatten()
            keys, order_index = torch.cat((visit_key, read_key)).sort()
            is_visit = order_index < len(visit_key)
            position = torch.arange(len(keys))
            group = keys // period
            # The last visit sorted before each entry, if it reached the same key neuron.
            last = torch.where(is_visit, position, -1).cummax(0).values
            visited = last >= 0
            visited[visited.clone()] = group[last[visited]] == group[visited]
            value = torch.full((len(keys),), -1, dtype=torch.int64)
            value[visited] = visit_value[order_index[last[visited]]]
            if state is not None:
                # Key neurons untouched in this chunk still hold their carried synapse.
                carried = state.successor[:, head].flatten()[group].to(torch.int64)
                kept = ~visited & (carried != 0)
                value[kept] = carried[kept]
                visits = is_visit.nonzero().squeeze(-1)
                final = torch.ones(len(visits), dtype=torch.bool)
                final[:-1] = group[visits[:-1]] != group[visits[1:]]
                final = visits[final]
                row, cell = group[final] // span, group[final] % span
                state.successor[row, head, cell] = visit_value[order_index[final]].to(torch.uint8)
            read = torch.empty(len(read_key), dtype=torch.int64)
            read[order_index[~is_visit] - len(visit_key)] = value[~is_visit]
            read = read.view(batch, steps, self.groups)
            votes[:, :, head] = torch.where(valid[1:].view(1, -1, 1), read, -1)
        if state is not None:
            state.history = padded[:, -self.longest:].to(torch.uint8)
            state.seen = min(seen + steps, self.longest)
        levels = self._levels(votes)
        if self.capture is not None:
            self.capture.append((votes, levels, surprise))
        return votes, levels, state

    @torch.no_grad()
    def step(self, tokens, surprise, state):
        """One byte per stream: move the synapses of the previous context, then read the new one."""
        batch = tokens.shape[0]
        rows = torch.arange(batch).unsqueeze(-1)
        before = self._winners(state.history)
        # Byte 0 marks a key neuron without a synapse and is never stored.
        stored = ((surprise >= self.gate) & (tokens != 0)).nonzero().squeeze(-1)
        for head, order in enumerate(self.orders):
            if state.seen >= order:
                state.successor[stored[:, None], head, before[stored, head]] = tokens[stored, None].to(torch.uint8)
        state.history = torch.cat((state.history[:, 1:], tokens[:, None].to(torch.uint8)), dim=1)
        state.seen = min(state.seen + 1, self.longest)
        after = self._winners(state.history)
        votes = torch.full((batch, len(self.orders), self.groups), -1, dtype=torch.int64)
        for head, order in enumerate(self.orders):
            if state.seen >= order:
                cells = after[:, head]
                stored = state.successor[rows, head, cells].to(torch.int64)
                votes[:, head] = torch.where(stored != 0, stored, -1)
        levels = self._levels(votes)
        return votes, levels, state
