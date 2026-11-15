"""Local byte language model with a verified successor cache and spike readout."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .neurons import integer_spike
from .successor_memory import SuccessorMemory, SuccessorMemoryState
from .transformer_reference import TransformerLM

if TYPE_CHECKING:
    from .transformer_reference import LocalTransformerState


class SuccessorReadout(nn.Module):
    """Allocate probability mass to verified successors using graded spikes.

    Each order votes for one byte when its exact tag matches. The readout
    conditions on age, model confidence, cross-order agreement, and online
    prediction accuracy observed so far. A miss contributes zero mass.
    """

    def __init__(self, orders: tuple[int, ...], levels: int = 16, *, scaled_votes: bool = False):
        super().__init__()
        self.levels = levels
        self.register_buffer("orders", torch.tensor(orders, dtype=torch.float32), persistent=False)
        vote_scale = (self.orders / self.orders.max()).square() if scaled_votes else torch.ones_like(self.orders)
        self.register_buffer("vote_scale", 2 * vote_scale, persistent=False)
        self.gate = nn.Sequential(nn.Linear(8, 24), nn.SiLU(), nn.Linear(24, 1))
        # Start with low cache confidence. A verified hit still has a
        # nonzero surrogate derivative when its initial pulse count is zero.
        nn.init.constant_(self.gate[-1].bias, -2.5)

    def forward(self, log_probs, successors, ages, tokens, *, quantized=True):
        hit = successors >= 0
        # At position t the model has already observed the outcome of its
        # prediction at t-1. Current and future outcomes are absent.
        observed = F.pad(hit[:, :-1], (0, 0, 1, 0)).cumsum(dim=1)
        correct = (successors[:, :-1] == tokens[:, 1:, None]) & hit[:, :-1]
        correct = F.pad(correct, (0, 0, 1, 0)).cumsum(dim=1)
        return self.strength(log_probs, successors, ages, observed, correct, quantized=quantized)

    def strength(self, log_probs, successors, ages, observed, correct, *, quantized):
        hit = successors >= 0
        candidate_logp = log_probs.gather(-1, successors.clamp_min(0))
        peak_logp = log_probs.amax(-1, keepdim=True).expand_as(candidate_logp)
        agreement = ((successors.unsqueeze(-1) == successors.unsqueeze(-2)) & hit.unsqueeze(-2))
        evidence = agreement.sum(-1).to(log_probs.dtype) / len(self.orders)
        max_order = torch.where(hit, self.orders, 0.).amax(-1, keepdim=True)
        longest = self.orders == max_order

        success_rate = (correct + 1.) / (observed + 2.)
        features = torch.stack((
            (self.orders / 8.).expand_as(candidate_logp),
            ages.clamp_min(0).to(log_probs.dtype).log1p() / 8.,
            candidate_logp.clamp_min(-24.) / 8.,
            peak_logp / 8.,
            evidence,
            longest.to(log_probs.dtype),
            success_rate,
            observed.to(log_probs.dtype).log1p() / 8.,
        ), dim=-1)
        strength = torch.sigmoid(self.gate(features.to(self.gate[0].weight.dtype)).squeeze(-1))
        if quantized:
            strength = integer_spike(strength, self.levels)
        # The base model contributes unit mass. Short matches get finer
        # increments when their maximum possible vote is smaller.
        return (self.vote_scale * strength * hit).to(log_probs.dtype)


def mix_successors(log_probs, successors, strength):
    """Add predictive mass to sparse candidates and normalize in log space."""
    assigned = torch.zeros_like(log_probs).scatter_add(-1, successors.clamp_min(0), strength)
    numerator = log_probs.exp() + assigned
    return numerator.clamp_min(torch.finfo(numerator.dtype).tiny).log() - strength.sum(-1, keepdim=True).log1p()


@dataclass
class SuccessorLMState:
    local: LocalTransformerState
    memory: SuccessorMemoryState
    previous: torch.Tensor
    observed: torch.Tensor
    correct: torch.Tensor


class SuccessorDendriteLM(TransformerLM):
    """Local Transformer with a bounded exact-context byte prediction path."""

    def __init__(self, cfg, vocab=256, dim=128, depth=4, heads=4, ctx=128):
        if vocab != 256:
            raise ValueError("successor memory expects the 256-byte vocabulary")
        super().__init__(replace(cfg, attn="transformer_local"), vocab, dim, depth, heads, ctx)
        self.cfg = cfg
        self.successors = SuccessorMemory(cfg.successor_orders, cfg.successor_slots)
        self.readout = SuccessorReadout(cfg.successor_orders, scaled_votes=cfg.successor_scaled_votes)

    def local_log_probs(self, idx, *, position_offset=0):
        return F.log_softmax(super().forward(idx, position_offset=position_offset), dim=-1, dtype=torch.float32)

    def evidence(self, idx):
        log_probs = self.local_log_probs(idx)
        successors, ages = self.successors(idx)
        return log_probs, successors, ages

    def forward(self, idx):
        log_probs, successors, ages = self.evidence(idx)
        if not idx.shape[1]:
            return log_probs
        strength = self.readout(log_probs, successors, ages, idx, quantized=self.cfg.successor_spikes)
        return mix_successors(log_probs, successors, strength)

    @torch.no_grad()
    def step(self, idx, state: SuccessorLMState | None = None):
        """Consume one byte per stream and update the bounded inference state."""
        logits, local = super().step(idx, None if state is None else state.local)
        successors, ages, memory = self.successors.step(idx, None if state is None else state.memory)
        if state is None:
            state = SuccessorLMState(local, memory, successors,
                                     torch.zeros_like(successors), torch.zeros_like(successors))
        else:
            state.observed.add_(state.previous >= 0)
            state.correct.add_(state.previous == idx[:, None])
            state.local, state.memory, state.previous = local, memory, successors
        log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        strength = self.readout.strength(log_probs, successors, ages, state.observed, state.correct,
                                         quantized=self.cfg.successor_spikes)
        return mix_successors(log_probs, successors, strength), state
