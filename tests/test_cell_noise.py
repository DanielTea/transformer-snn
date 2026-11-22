"""Noisy thresholds: while training, threshold cells add logistic membrane noise to their charge and learn from the
slope of the count they fire on average (cell_noise)."""
import torch
from torch.nn import functional as F

from snnformer.event_lm import EventLM, EventLMConfig, _ExcitatoryFire, cell_dither, cell_noise

_LEVELS = (0, 1)
# The small stack of the stream-form tests.
_STACK = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=_LEVELS, read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, granule_after=_LEVELS, granule_clusters=1, granule_fan=64,
              granule_cells=96, granule_slope=True, read_delays=4, granule_delays=4, read_cap=128, granule_cap=128)
# Quanta that make one whole threshold of 256 ticks.
_QUANTA = torch.full((1,), 64)
_WHOLE = 256


def _stack(seed=0):
    torch.manual_seed(seed)
    model = EventLM(EventLMConfig(**_STACK))
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


def _expected(charges, width):
    """The mean count under logistic noise of ``width`` thresholds: the chance of passing every half threshold."""
    halves = (torch.arange(1, 200, dtype=torch.float64)[:, None] - 0.5) * _WHOLE
    return torch.sigmoid((charges.double() - halves) / (width * _WHOLE)).sum(0)


def test_counts_are_whole_spikes_whose_mean_passes_each_half_threshold_by_chance():
    torch.manual_seed(0)
    fire = _cells()
    charges = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 12.0]) * _WHOLE
    current = charges.repeat(40000, 1).requires_grad_()
    with cell_noise(0.5):
        draws = fire(current, _QUANTA).detach()
    assert torch.equal(draws, draws.round()) and bool((draws >= 0).all())
    assert (draws.double().mean(0) - _expected(charges, 0.5)).abs().max() < 0.015
    # The mean is close to the softplus that the cells learn from.
    assert (_expected(charges, 0.5) - 0.5 * F.softplus(charges.double() / (0.5 * _WHOLE))).abs().max() < 0.025


def test_a_cell_near_its_threshold_fires_by_chance_on_either_side():
    torch.manual_seed(1)
    fire = _cells()
    current = torch.tensor([0.4, 0.6]).mul(_WHOLE).repeat(4000, 1).requires_grad_()
    nearest = fire(current, _QUANTA).detach()[0]
    assert nearest.tolist() == [0.0, 1.0]
    with cell_noise(0.25):
        draws = fire(current, _QUANTA).detach()
    below, above = (draws[:, 0] > 0).double().mean(), (draws[:, 1] == 0).double().mean()
    assert 0.2 < below < 0.5 and 0.2 < above < 0.5


def test_the_slope_is_that_of_the_mean_count_at_the_peak_whatever_the_noise_now():
    fire = _cells()
    current = ((torch.rand(512) - 0.3) * 8 * _WHOLE).requires_grad_()
    expected = torch.sigmoid(current.detach() / (0.5 * _WHOLE)) / _WHOLE
    for share in (1.0, 0.3, 0.0):
        current.grad = None
        with cell_noise(0.5, share):
            fire(current, _QUANTA, smooth=1).sum().backward()
        assert torch.allclose(current.grad, expected, rtol=1e-5, atol=1e-9)
    # With no noise left the counts are the plain ones.
    with cell_noise(0.5, 0.0):
        assert torch.equal(fire(current, _QUANTA, smooth=1).detach(), fire(current.detach(), _QUANTA))


def test_off_without_a_gradient_by_default_and_for_other_neurons():
    fire = _cells()
    current = (torch.rand(512) - 0.2) * 20 * _WHOLE
    nearest = fire(current, _QUANTA)
    with cell_noise(0.5), torch.no_grad():
        assert torch.equal(fire(current, _QUANTA), nearest)
    plain = _ExcitatoryFire(128, 8, False, quanta_per_threshold=64, capped=True)
    with cell_noise(0.5):
        assert torch.equal(plain(current.clone().requires_grad_(), _QUANTA).detach(), nearest)
    # Without the context the slope is the plain one.
    current.requires_grad_()
    fire(current, _QUANTA, smooth=1).sum().backward()
    assert torch.allclose(current.grad, torch.autograd.grad(F.gelu(current / _WHOLE).sum(), current)[0])


def test_the_context_restores_what_was_set_before():
    with cell_noise(0.5, 0.5):
        with cell_noise(1.0):
            pass
        fire = _cells()
        current = torch.full((20000,), 0.0).requires_grad_()
        draws = fire(current, _QUANTA).detach()
        # Noise of a quarter threshold: a charge of nothing passes half a threshold with the chance sigmoid(-2).
        assert abs((draws > 0).double().mean().item() - torch.sigmoid(torch.tensor(-2.0)).item()) < 0.01
    with cell_dither():
        assert torch.equal(fire(current, _QUANTA).detach(), torch.zeros(20000))


def test_a_stack_trains_with_noisy_thresholds_and_scores_as_before():
    model = _stack()
    tokens = _bytes()
    model.eval()
    with torch.no_grad():
        reference, _ = model.stream(tokens, None)
        with cell_noise(0.5):
            scored, _ = model.stream(tokens, None)
    assert torch.equal(scored, reference)
    model.train()
    with cell_noise(0.5):
        first, _ = model.stream(tokens, None)
        second, _ = model.stream(tokens, None)
    plain, _ = model.stream(tokens, None)
    assert not torch.equal(first, second)
    assert torch.equal(plain.detach(), reference)
    with cell_noise(0.5):
        counts, _ = model.stream(tokens, None)
    counts.float().square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
