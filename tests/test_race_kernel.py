"""The fused race kernels (snnformer/race_kernel.py) against the read by age they replace: the same spikes and payload
pulses, and the same gradient up to single-precision rounding. They run where Triton runs: on a CUDA device, or on the
CPU with TRITON_INTERPRET=1 (set before Python starts)."""
import os

import pytest
import torch

triton = pytest.importorskip("triton")
if not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("the kernels need a CUDA device or TRITON_INTERPRET=1", allow_module_level=True)

import snnformer.event_lm as event_lm  # noqa: E402
from snnformer import race_kernel  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _inputs(seed, steps, kept, reach, knowledge, batch=2, heads=2, channels=16, payload_channels=16, position=None):
    generator = torch.Generator().manual_seed(seed)

    def pulses(*shape, low=0, high=8):
        return torch.randint(low, high + 1, shape, generator=generator).float()

    cells = kept + steps
    probe, tag = pulses(batch, heads, steps, channels), pulses(batch, heads, cells, channels)
    payload = pulses(batch, heads, cells, payload_channels, low=-8)
    scale = (torch.rand(heads, 1, 1, generator=generator) * 0.08 + 0.02)
    offset = torch.randn(heads, reach, generator=generator) * 0.5
    position = kept + 3 if position is None else position
    order = torch.arange(heads)[:, None, None]
    phase = ((event_lm._spread_phases(position - kept, cells, "cpu", torch.float32) + order / heads) % 1)[:, 0]
    known = None
    if knowledge:
        known = (pulses(heads, knowledge, channels), pulses(heads, knowledge, payload_channels, low=-8),
                 torch.randn(heads, knowledge, generator=generator) * 0.5,
                 ((event_lm._spread_phases(-knowledge - 1, knowledge, "cpu", torch.float32) + order / heads) % 1))
    tensors = dict(probe=probe, tag=tag, payload=payload, scale=scale, offset=offset, phase=phase)
    if known:
        tensors.update(kn_tag=known[0], kn_payload=known[1], kn_off=known[2], kn_phase=known[3])
    return {name: value.to(DEVICE) for name, value in tensors.items()}


def _reference(t, kept, reach, budget):
    """The read by age, as _StateRead._address_band computes it: payload pulses, spikes per byte, knowledge spikes."""
    steps = t["probe"].shape[2]
    back = (kept + torch.arange(steps, device=DEVICE))[:, None] - torch.arange(reach, device=DEVICE)
    valid = back >= 0
    phase = t["phase"][:, back.clamp_min(0)]
    charge = event_lm._band_dots(t["probe"], t["tag"], kept, reach) * t["scale"] + t["offset"][:, None]
    charge = charge.masked_fill(~valid, -torch.inf)
    known = kn_phase = None
    if "kn_tag" in t:
        known = t["probe"] @ t["kn_tag"].transpose(1, 2) * t["scale"] + t["kn_off"][:, None]
        kn_phase = t["kn_phase"]
    fired, kn_fired = event_lm._BudgetedRace.apply(charge, known, valid, phase, kn_phase, float(budget))
    packet = event_lm._band_read(fired, t["payload"], kept, reach)
    total = fired.sum(-1)
    if known is not None:
        packet = packet + kn_fired @ t["kn_payload"]
        total = total + kn_fired.sum(-1)
    return packet, total, kn_fired


def _fused(t, kept, reach, budget):
    return race_kernel.race(t["probe"], t["tag"], t["payload"], t["scale"], t["offset"], t.get("kn_tag"),
                            t.get("kn_payload"), t.get("kn_off"), t["phase"], t.get("kn_phase"), kept, reach, budget)


CASES = [(40, 0, 24, 16), (40, 23, 24, 16), (7, 23, 24, 16), (1, 10, 24, 0), (33, 5, 24, 0), (50, 31, 32, 16)]


@pytest.mark.parametrize("steps,kept,reach,knowledge", CASES)
def test_the_fused_race_fires_what_the_read_by_age_fires(steps, kept, reach, knowledge):
    t = _inputs(1, steps, kept, reach, knowledge)
    with torch.no_grad():
        want = _reference(t, kept, reach, 16.0)
        got = _fused(t, kept, reach, 16.0)
    assert torch.equal(got[1], want[1]), (got[1] - want[1]).abs().max()
    assert torch.equal(got[0], want[0])
    if knowledge:
        assert torch.equal(got[2], want[2])
    assert want[1].sum() > steps  # the race fires


@pytest.mark.parametrize("steps,kept,reach,knowledge", CASES[:4])
def test_the_fused_race_learns_with_the_read_by_age_s_gradient(steps, kept, reach, knowledge):
    grads = []
    for run in (_reference, _fused):
        t = _inputs(2, steps, kept, reach, knowledge)
        for name in ("probe", "tag", "payload", "scale", "offset", "kn_tag", "kn_payload", "kn_off"):
            if name in t:
                t[name].requires_grad_(True)
        packet, total, _ = run(t, kept, reach, 16.0)
        weights = torch.linspace(-1, 1, packet.numel(), device=DEVICE).reshape(packet.shape)
        ((packet * weights).sum() + (total * torch.linspace(1, 2, total.numel(), device=DEVICE).reshape(
            total.shape)).sum()).backward()
        grads.append({name: value.grad for name, value in t.items() if value.requires_grad})
    want, got = grads
    for name, grad in want.items():
        scale = grad.abs().max().clamp_min(1e-12)
        assert ((got[name] - grad).abs().max() / scale) < 1e-4, (name, ((got[name] - grad).abs().max() / scale))


import test_knowledge_gap_cells  # noqa: E402

_DESIGNS = [dict(), dict(read_knowledge=16), dict(read_knowledge=16, read_forget=True),
            dict(read_knowledge=16, stack_loop=(0, 1, 2), granule_power=8)]


def _stack(**design):
    # Four heads of 16 channels, as the kernels need: powers of two from 16.
    return test_knowledge_gap_cells._stack(0, **dict(read_width=64, read_reach=96, **design)).to(DEVICE)


def _stream(model, tokens, cuts, fused):
    saved = event_lm._READ_FUSED
    event_lm._READ_FUSED = fused
    try:
        counts, state, start = [], None, 0
        for length in cuts:
            fired, state = model.stream(tokens[:, start:start + length], state)
            counts.append(fired)
            start += length
        return torch.cat(counts, 1), state
    finally:
        event_lm._READ_FUSED = saved


@pytest.mark.parametrize("design", _DESIGNS)
@pytest.mark.parametrize("cuts", [(160,), (40, 40, 40, 40), (7, 96, 57)])
def test_a_stack_streams_the_same_spikes_through_fused_reads(design, cuts, monkeypatch):
    calls = []
    original = race_kernel.race
    monkeypatch.setattr(race_kernel, "race", lambda *a: calls.append(1) or original(*a))
    model, tokens = _stack(**design).eval(), test_knowledge_gap_cells._bytes(length=160).to(DEVICE)
    with torch.no_grad():
        by_age, age_state = _stream(model, tokens, cuts, False)
        assert not calls
        fused, fused_state = _stream(model, tokens, cuts, True)
    assert calls and torch.equal(fused, by_age)
    for name, tape in age_state.reads.items():
        assert torch.equal(fused_state.reads[name].payload, tape.payload)
        assert torch.equal(fused_state.reads[name].tag, tape.tag)


@pytest.mark.parametrize("design", _DESIGNS[2:])
def test_a_stack_learns_with_the_same_gradient_through_fused_reads(design):
    grads = []
    for fused in (False, True):
        model, tokens = _stack(**design).train(), test_knowledge_gap_cells._bytes(length=120).to(DEVICE)
        saved = event_lm._READ_FUSED
        event_lm._READ_FUSED = fused
        try:
            with torch.no_grad():
                _, state = model.stream(tokens[:, :64], None)
            counts, _ = model.stream(tokens[:, 64:], state)
        finally:
            event_lm._READ_FUSED = saved
        weights = torch.linspace(-1, 1, counts.numel(), device=DEVICE).reshape(counts.shape)
        (counts * weights).sum().backward()
        grads.append({name: p.grad for name, p in model.named_parameters() if p.grad is not None})
    by_age, fused = grads
    assert set(by_age) == set(fused)
    reads = 0
    for name, grad in by_age.items():
        scale = grad.abs().max()
        if scale > 0:
            assert (fused[name] - grad).abs().max() / scale < 1e-4, name
            reads += "state_reads" in name
    assert reads > 0
