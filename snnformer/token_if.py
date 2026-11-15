"""Fixed-point integrate-and-fire neurons on the token axis."""

import torch
from torch import nn


# One spike releases 256 ticks of charge. The initial half-charge makes the
# first pulse use the same half-threshold as the reset-every-token control.
_CHARGE_BITS = 8
_THRESHOLD = 1 << _CHARGE_BITS
_INITIAL_PHASE = _THRESHOLD // 2
_INT32_MAX = torch.iinfo(torch.int32).max


class TokenIF(nn.Module):
    """Nonleaky integrate-and-fire with subtractive reset on each token.

    Bounded input current is measured in multiples of the firing threshold.
    Internally, charge is quantized to ticks and carried as an integer phase.
    Reset mode starts each token at half-threshold instead of carrying phase.
    Training uses a local straight-through surrogate for emitted spike counts.
    """

    def __init__(self, max_spikes: int = 4, carry: bool = True) -> None:
        super().__init__()
        if not 1 <= max_spikes <= (_INT32_MAX - _INITIAL_PHASE) // _THRESHOLD:
            raise ValueError("max_spikes must be positive and fit in int32 charge")
        self.max_spikes = max_spikes
        self.carry = carry

    def _charge(self, currents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bounded = currents.clamp(0, self.max_spikes)
        charge = (bounded.float() * _THRESHOLD).round().to(torch.int32)
        return charge, bounded

    def forward(self, currents: torch.Tensor) -> torch.Tensor:
        """Emit spike counts for currents of shape [batch, tokens, channels]."""
        if self.carry:
            max_charge = currents.shape[1] * self.max_spikes * _THRESHOLD
            if max_charge > _INT32_MAX - _INITIAL_PHASE:
                raise ValueError("token sequence can overflow the int32 charge scan")
        charge, bounded = self._charge(currents)
        if self.carry:
            # The number of thresholds crossed since the start is cumulative.
            # Consecutive counts differ by the spikes released at this token.
            total = charge.cumsum(dim=1, dtype=torch.int32) + _INITIAL_PHASE
            cumulative_spikes = total >> _CHARGE_BITS
            spikes = cumulative_spikes.diff(dim=1, prepend=torch.zeros_like(cumulative_spikes[:, :1]))
        else:
            spikes = (charge + _INITIAL_PHASE) >> _CHARGE_BITS
        return spikes.to(bounded.dtype) + (bounded - bounded.detach())

    def step(
        self, currents: torch.Tensor, phase: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Emit spikes for [batch, channels] and return post-reset int32 phase."""
        charge, bounded = self._charge(currents)
        total = charge + (phase if self.carry and phase is not None else _INITIAL_PHASE)
        spikes = total >> _CHARGE_BITS
        phase = total - (spikes << _CHARGE_BITS)
        return spikes.to(bounded.dtype) + (bounded - bounded.detach()), phase
