"""Causal byte successor memory with bounded tables and exact tag checks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def successor_slot(tag, slots: int):
    """Map int64 tags to slots identically for Torch and NumPy arrays."""
    mixed = (tag ^ (tag >> 33)) * -49064778989728563
    mixed = (mixed ^ (mixed >> 33)) * -4265267296055464877
    return (mixed ^ (mixed >> 33)) & (slots - 1)


@dataclass
class SuccessorMemoryState:
    """Direct-mapped tables and the current packed byte history."""

    tags: torch.Tensor
    successors: torch.Tensor
    observed_at: torch.Tensor
    history: torch.Tensor
    length: int = 0


class SuccessorMemory(nn.Module):
    """Retrieve the last observed successor of an exact byte suffix.

    The input is a tensor of byte IDs in [0, 255]. At position t, the
    byte x[t] first becomes the observed successor of the suffix ending at
    t-1. The current suffix then queries the table. Each suffix length has
    its own table with `slots` entries. Collisions overwrite a slot, but
    the full packed suffix is checked before it can be read.

    Predictions are byte IDs, and ages count positions since the matching
    successor was observed. Both are -1 on a miss. The batched path sorts
    suffix appearances by (slot, position) and finds their predecessors.
    Streaming keeps only the bounded tables and the rolling suffix.
    """

    def __init__(self, orders: tuple[int, ...] = (3, 5, 8), slots: int = 1024) -> None:
        super().__init__()
        self.orders = tuple(orders)
        if (
            not self.orders
            or any(not 1 <= order <= 8 for order in self.orders)
            or len(set(self.orders)) != len(self.orders)
        ):
            raise ValueError("orders must be distinct byte lengths from 1 to 8")
        if slots < 1 or slots & (slots - 1):
            raise ValueError("slots must be a positive power of two")
        self.slots = slots
        self.register_buffer("_orders", torch.tensor(self.orders, dtype=torch.long), persistent=False)
        self.register_buffer(
            "_masks",
            torch.tensor([(1 << (8 * order)) - 1 if order < 8 else -1 for order in self.orders],
                         dtype=torch.long),
            persistent=False,
        )

    @staticmethod
    def _bytes(tokens: torch.Tensor, dimensions: int) -> torch.Tensor:
        if tokens.ndim != dimensions or tokens.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
        ):
            raise ValueError(f"expected {dimensions}-dimensional integer byte IDs")
        return tokens.to(torch.long)

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return prediction and age tensors of shape [batch, time, orders]."""
        tokens = self._bytes(tokens, 2)
        batch, length = tokens.shape
        if not length:
            empty = tokens.new_empty((batch, 0, len(self.orders)))
            return empty, empty.clone()

        max_order = max(self.orders)
        padded = F.pad(tokens, (max_order - 1, 0))
        packed = torch.zeros_like(tokens)
        # The newest byte occupies the low eight bits. Left shifts preserve
        # the full signed 64-bit bit pattern for an eight-byte suffix.
        for offset in range(max_order):
            start = max_order - offset - 1
            packed |= padded[:, start : start + length] << (8 * offset)
        tags = packed[:, None, :] & self._masks.to(tokens.device)[None, :, None]
        positions = torch.arange(length, device=tokens.device)[None, None, :]
        ready = positions >= self._orders.to(tokens.device)[None, :, None] - 1
        slots = torch.where(ready, successor_slot(tags, self.slots), self.slots)

        # The preceding valid context in the same slot is exactly the final
        # write visible at this query. Its successor was observed one step
        # after the prior context, including at this query's current byte.
        order = (slots * length + positions).argsort(dim=-1)
        prior_sorted = F.pad(order[..., :-1], (1, 0), value=-1)
        prior = torch.empty_like(order).scatter_(-1, order, prior_sorted)
        prior_index = prior.clamp_min(0)
        prior_tag = tags.gather(-1, prior_index)
        hit = ready & (prior >= 0) & (slots.gather(-1, prior_index) == slots) & (prior_tag == tags)
        observed = (prior + 1).clamp_max(length - 1)
        predicted = tokens[:, None, :].expand_as(tags).gather(-1, observed)
        return (
            torch.where(hit, predicted, -1).transpose(1, 2),
            torch.where(hit, positions - observed, -1).transpose(1, 2),
        )

    @torch.no_grad()
    def step(
        self, tokens: torch.Tensor, state: SuccessorMemoryState | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, SuccessorMemoryState]:
        """Consume a single byte per stream and return prediction, age and state.

        The state is updated in place. Pass `None` to start a new sequence.
        """
        tokens = self._bytes(tokens, 1)
        batch = tokens.shape[0]
        orders = self._orders.to(tokens.device)
        masks = self._masks.to(tokens.device)
        if state is None:
            shape = (batch, len(self.orders), self.slots)
            state = SuccessorMemoryState(
                tags=tokens.new_zeros(shape),
                successors=tokens.new_zeros(shape),
                observed_at=tokens.new_full(shape, -1),
                history=tokens.new_zeros(batch),
            )
        elif state.history.shape != tokens.shape or state.history.device != tokens.device:
            raise ValueError("successor memory state must stay on the same device and batch")

        position = state.length
        prior = state.history[:, None] & masks
        eligible = (position >= orders)[None, :, None]
        slot = successor_slot(prior, self.slots).unsqueeze(-1)
        for table, update in (
            (state.tags, prior.unsqueeze(-1)),
            (state.successors, tokens[:, None, None].expand(batch, len(self.orders), 1)),
            (state.observed_at, torch.full_like(slot, position)),
        ):
            table.scatter_(-1, slot, torch.where(eligible, update, table.gather(-1, slot)))

        state.history.copy_((state.history << 8) | tokens)
        tag = state.history[:, None] & masks
        query = successor_slot(tag, self.slots).unsqueeze(-1)
        observed = state.observed_at.gather(-1, query).squeeze(-1)
        matched = state.tags.gather(-1, query).squeeze(-1) == tag
        hit = (position >= orders - 1) & (observed >= 0) & matched
        predicted = state.successors.gather(-1, query).squeeze(-1)
        state.length += 1
        return torch.where(hit, predicted, -1), torch.where(hit, position - observed, -1), state
