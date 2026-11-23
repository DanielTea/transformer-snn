"""Quantal release: while training, synapses round to their charge grid at random (weight_dither)."""
import torch

from snnformer.event_lm import EventLM, EventLMConfig, _weight_ticks, weight_dither

_LEVELS = (0, 1)
# The small stack of the stream-form tests.
_STACK = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=_LEVELS, read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, granule_after=_LEVELS, granule_clusters=1, granule_fan=64,
              granule_cells=96, granule_slope=True, read_delays=4, granule_delays=4, read_cap=128, granule_cap=128)


def _stack(seed=0):
    torch.manual_seed(seed)
    model = EventLM(EventLMConfig(**_STACK))
    # Reads and cell layers start silent; give them output synapses so that their weights shape the stream.
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for read in model.state_reads.values():
            read.project.weight.copy_(torch.randn(read.project.weight.shape, generator=generator) * 0.05)
        for layer in model.granule_layers.values():
            layer.send.copy_(torch.randn(layer.send.shape, generator=generator) * 0.05)
    return model


def _bytes(seed=2, length=96, streams=2):
    return torch.randint(0, 256, (streams, length), generator=torch.Generator().manual_seed(seed))


def test_off_by_default_and_without_a_gradient():
    weight = torch.randn(64, 32) * 0.3
    nearest = _weight_ticks(weight, charge_bits=8)
    assert torch.equal(nearest, (weight * 256).clamp(-508, 508).round())
    with weight_dither(), torch.no_grad():
        assert torch.equal(_weight_ticks(weight, charge_bits=8), nearest)


def test_whole_ticks_next_to_the_stored_size_and_unbiased():
    torch.manual_seed(0)
    weight = (torch.randn(4096) * 0.3).requires_grad_()
    exact = (weight.detach() * 256).clamp(-508, 508)
    draws = []
    with weight_dither():
        for _ in range(400):
            draws.append(_weight_ticks(weight, charge_bits=8).detach())
    draws = torch.stack(draws)
    assert torch.equal(draws, draws.round())
    assert bool(((draws == exact.floor()) | (draws == exact.ceil())).all())
    # The mean over draws is the stored size: the error of 400 draws is below a tenth of a tick everywhere.
    assert (draws.mean(0) - exact).abs().max() < 0.1
    assert (draws.mean(0) - exact).mean().abs() < 0.005


def test_gradient_passes_straight_through_as_before():
    weight = (torch.randn(32, 16) * 0.3).requires_grad_()
    _weight_ticks(weight, charge_bits=8).sum().backward()
    plain = weight.grad.clone()
    weight.grad = None
    with weight_dither():
        _weight_ticks(weight, charge_bits=8).sum().backward()
    assert torch.equal(weight.grad, plain)


def test_the_context_restores_what_was_set_before():
    weight = (torch.randn(8, 8) * 0.3).requires_grad_()
    with weight_dither():
        with weight_dither(False):
            assert torch.equal(_weight_ticks(weight).detach(), (weight.detach() * 64).clamp(-127, 127).round())
    assert torch.equal(_weight_ticks(weight).detach(), (weight.detach() * 64).clamp(-127, 127).round())


def test_a_stack_trains_with_random_release_and_scores_as_before():
    model = _stack()
    tokens = _bytes()
    model.eval()
    with torch.no_grad():
        reference, _ = model.stream(tokens, None)
        with weight_dither():
            scored, _ = model.stream(tokens, None)
    # Scoring takes no gradient, so release is the nearest tick and every spike is as before.
    assert torch.equal(scored, reference)
    model.train()
    with weight_dither():
        first, _ = model.stream(tokens, None)
        second, _ = model.stream(tokens, None)
    plain, _ = model.stream(tokens, None)
    # Training with release draws other charges each pass; without it the pass is the deterministic one.
    assert not torch.equal(first, second)
    assert torch.equal(plain.detach(), reference)
    loss = first.float().square().mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
