"""Pools that hear the summed counts but train with the energy's gradient (--pool-energy-gradient), and the float
control that divides by the mean absolute activity but trains with the root mean square's gradient (gpool)."""
from dataclasses import replace

import pytest
import torch

from snnformer.config import SNNConfig
from snnformer.event_lm import EventLM, EventLMConfig, _heard_activity
from snnformer.transformer_reference import TransformerLM, _EnergyGradientPoolNorm, _PoolNorm
from test_stack_loop import _stack
from test_teaching_spikes import _STACK, _bytes


def test_the_pool_hears_the_summed_counts_and_passes_the_energys_gradient():
    torch.manual_seed(0)
    counts = torch.randint(0, 9, (2, 5, 32)).float().requires_grad_()
    heard = _heard_activity(counts, "gradient")
    torch.testing.assert_close(heard.detach(), counts.detach().sum(-1, keepdim=True), rtol=0, atol=0)
    heard.sum().backward()
    x = counts.detach()
    summed = x.sum(-1, keepdim=True)
    smooth = (x.square().sum(-1, keepdim=True) * 32 + 1e-6).sqrt()
    torch.testing.assert_close(counts.grad, summed / smooth * 32 * x / smooth)
    # Like the summed counts' own gradient, it gives back the summed counts along the stream (Euler's theorem),
    # but it points along the stream, not along its sign pattern.
    torch.testing.assert_close((counts.grad * x).sum(-1, keepdim=True), summed, rtol=1e-4, atol=1e-4)
    assert not torch.allclose(counts.grad, torch.ones_like(counts.grad))
    # Without a gradient to pass it is the summed counts.
    assert torch.equal(_heard_activity(x, "gradient"), x.sum(-1, keepdim=True))


def test_the_stack_fires_the_same_and_learns_differently():
    tokens = _bytes()
    plain, trained = _stack(), _stack(pool_energy_gradient=True)
    assert all(read.energy == "gradient" for read in trained.state_reads.values())
    with torch.no_grad():
        torch.testing.assert_close(trained(tokens), plain(tokens), rtol=0, atol=0)
        streamed, _ = trained.stream(tokens)
        torch.testing.assert_close(streamed, plain(tokens), rtol=0, atol=0)
    for model in (plain, trained):
        model.zero_grad()
        model(tokens).float().square().mean().backward()
    grads = [{n: p.grad for n, p in m.named_parameters() if p.grad is not None} for m in (plain, trained)]
    assert grads[0].keys() == grads[1].keys()
    assert any(not torch.allclose(grads[0][n], grads[1][n]) for n in grads[0])


def test_pools_that_train_with_the_energys_gradient_belong_to_a_stack_with_plain_pools():
    with pytest.raises(ValueError, match="energy's gradient"):
        EventLM(EventLMConfig(**dict(_STACK, pool_energy_gradient=True, pool_energy=True)))
    with pytest.raises(ValueError, match="energy's gradient"):
        EventLM(EventLMConfig(**dict(_STACK, stack=False, granule_after=None, read_after=None,
                                     pool_energy_gradient=True)))


def test_the_float_control_computes_the_pool_and_trains_with_the_root_mean_squares_gradient():
    torch.manual_seed(5)
    pool, control = _PoolNorm(16), _EnergyGradientPoolNorm(16)
    with torch.no_grad():
        pool.weight.copy_(torch.rand(16) + 0.5)
        control.weight.copy_(pool.weight)
    x = (torch.randn(3, 4, 16) * 2).requires_grad_()
    y = x.detach().clone().requires_grad_()
    out_pool, out_control = pool(x), control(y)
    torch.testing.assert_close(out_control, out_pool, rtol=0, atol=0)
    upstream = torch.randn_like(out_pool)
    (out_pool * upstream).sum().backward()
    (out_control * upstream).sum().backward()
    assert not torch.allclose(x.grad, y.grad)
    # Both are blind to the stream's scale: their gradients are orthogonal to the stream itself, up to the rest.
    for grad in (x.grad, y.grad):
        assert (grad * x.detach()).sum(-1).abs().max() < 1e-2
    model = TransformerLM(replace(SNNConfig(neuron="ann", attn="transformer", mlp_ratio=4.0, position_embeddings=True),
                                  norm="gpool"), vocab=32, dim=16, depth=2, heads=4, ctx=24)
    assert len([m for m in model.modules() if isinstance(m, _EnergyGradientPoolNorm)]) == 2 * len(model.blocks) + 1
