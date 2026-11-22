"""Stochastic firing: while training, threshold cells fire their last spike at random (cell_dither)."""
import torch

from snnformer.event_lm import EventLM, EventLMConfig, _ExcitatoryFire, cell_dither, weight_dither

_LEVELS = (0, 1)
# The small stack of the stream-form tests.
_STACK = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=_LEVELS, read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, granule_after=_LEVELS, granule_clusters=1, granule_fan=64,
              granule_cells=96, granule_slope=True, read_delays=4, granule_delays=4, read_cap=128, granule_cap=128)


def _stack(seed=0):
    torch.manual_seed(seed)
    model = EventLM(EventLMConfig(**_STACK))
    # Reads and cell layers start silent; give them output synapses so that the cells shape the stream.
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for read in model.state_reads.values():
            read.project.weight.copy_(torch.randn(read.project.weight.shape, generator=generator) * 0.05)
        for layer in model.granule_layers.values():
            layer.send.copy_(torch.randn(layer.send.shape, generator=generator) * 0.05)
    return model


def _bytes(seed=2, length=96, streams=2):
    return torch.randint(0, 256, (streams, length), generator=torch.Generator().manual_seed(seed))


def _cells():
    fire = _ExcitatoryFire(128, 8, False, quanta_per_threshold=64, capped=True)
    fire.dither = True
    return fire


def test_only_the_threshold_cells_may_fire_at_random():
    model = _stack()
    assert all(layer.fire.dither for layer in model.granule_layers.values())
    assert not any(getattr(read.fire, "dither", False) for read in model.state_reads.values() if hasattr(read, "fire"))


def test_counts_are_whole_spikes_next_to_the_charge_and_unbiased():
    torch.manual_seed(0)
    fire = _cells()
    quanta = torch.full((1,), 64)
    current = (torch.rand(4096) * 40 * 256).requires_grad_()  # charges up to 40 thresholds, below the cap
    exact = current.detach() / 256
    with cell_dither():
        draws = torch.stack([fire(current, quanta).detach() for _ in range(400)])
    assert torch.equal(draws, draws.round())
    assert bool(((draws == exact.floor()) | (draws == exact.floor() + 1)).all())
    assert (draws.mean(0) - exact).abs().max() < 0.12
    assert (draws.mean(0) - exact).mean().abs() < 0.005


def test_off_by_default_without_a_gradient_and_for_other_neurons():
    fire = _cells()
    quanta = torch.full((1,), 64)
    current = torch.rand(512) * 20 * 256
    nearest = fire(current, quanta)
    with cell_dither(), torch.no_grad():
        assert torch.equal(fire(current, quanta), nearest)
    plain = _ExcitatoryFire(128, 8, False, quanta_per_threshold=64, capped=True)
    with cell_dither():
        assert torch.equal(plain(current.requires_grad_(), quanta).detach(), nearest)


def test_the_slope_is_the_same_as_without_random_firing():
    fire = _cells()
    quanta = torch.full((1,), 64)
    current = (torch.rand(256) * 20 * 256).requires_grad_()
    fire(current, quanta, smooth=1).sum().backward()
    plain = current.grad.clone()
    current.grad = None
    with cell_dither():
        fire(current, quanta, smooth=1).sum().backward()
    assert torch.equal(current.grad, plain)


def test_a_stack_trains_with_random_firing_and_scores_as_before():
    model = _stack()
    tokens = _bytes()
    model.eval()
    with torch.no_grad():
        reference, _ = model.stream(tokens, None)
        with cell_dither(), weight_dither():
            scored, _ = model.stream(tokens, None)
    assert torch.equal(scored, reference)
    model.train()
    with cell_dither():
        first, _ = model.stream(tokens, None)
        second, _ = model.stream(tokens, None)
    plain, _ = model.stream(tokens, None)
    assert not torch.equal(first, second)
    assert torch.equal(plain.detach(), reference)
    first.float().square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
