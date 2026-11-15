from __future__ import annotations

import pytest
import torch

from snnformer.pulse_timing import PHASE_THRESHOLD, PhaseRead


def test_legacy_first_pulse_matches_comparator_for_every_phase_and_cut():
    read = PhaseRead(2)
    phase = torch.arange(PHASE_THRESHOLD, dtype=torch.float32)[:, None]
    cuts = torch.arange(PHASE_THRESHOLD + 1, dtype=torch.float32)[None, :, None]
    phase = phase.expand(-1, cuts.shape[-2])
    gate = torch.ones_like(phase)
    expected = (phase + cuts.squeeze(-1) >= PHASE_THRESHOLD).unsqueeze(-1).float()

    streamed, restored = read.step(phase, cuts, gate)
    assert torch.equal(read(phase, cuts, gate), expected)
    assert torch.equal(streamed, expected)
    assert torch.equal(restored, phase)


def test_timing_code_replays_ordered_and_zero_width_pulses():
    read = PhaseRead(4)
    cuts = torch.tensor([[0, 0, 0], [0, 64, 64], [21, 140, 221], [64, 128, 192],
                         [87, 87, 87], [0, 256, 256], [256, 256, 256]], dtype=torch.float32)
    phase = torch.arange(PHASE_THRESHOLD, dtype=torch.float32)[:, None].expand(-1, len(cuts))
    gate = torch.ones_like(phase)
    gate[::5, ::2] = 0

    pulses = read.pulses(cuts)
    assert (pulses >= 0).all()
    assert (pulses.sum(-1) == PHASE_THRESHOLD).all()
    expected_time = (phase.unsqueeze(-1) + cuts < PHASE_THRESHOLD).sum(-1)
    observed = read(phase, cuts, gate)
    streamed, restored = read.step(phase, cuts, gate)
    assert torch.equal(observed, streamed)
    assert torch.equal(observed.sum(-1), gate)
    assert torch.equal(observed.argmax(-1)[gate.bool()], expected_time[gate.bool()])
    assert torch.equal(restored, phase)


def test_fixed_cuts_ignore_queries_and_learned_cuts_keep_distinct_gradients():
    logits = torch.tensor([[1.3, -0.8, 0.1], [-10.0, 2.0, 10.0]], requires_grad=True)
    fixed = PhaseRead(4, fixed_cuts=True)
    assert torch.equal(fixed.cuts(logits), torch.tensor([[64.0, 128.0, 192.0]]).expand_as(logits))

    read = PhaseRead(4)
    cuts = read.cuts(logits)
    assert torch.equal(cuts, cuts.round())
    assert ((cuts >= 0) & (cuts <= PHASE_THRESHOLD)).all()
    assert (cuts.diff(dim=-1) >= 0).all()

    phase = torch.tensor([113., 113.])
    events = read(phase, cuts, torch.ones_like(phase))
    coefficients = events.new_tensor([1., 2., 3., 5.])
    (events * coefficients).sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[0] < 0).all()


def test_timing_configuration_validates_bin_count_and_projected_width():
    with pytest.raises(ValueError, match="2 or 4"):
        PhaseRead(3)
    with pytest.raises(ValueError, match="each cut"):
        PhaseRead(4).cuts(torch.zeros(1, 2))
