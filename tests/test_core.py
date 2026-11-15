import math

import pytest
import torch
import torch.nn as nn

from snnformer import SNNConfig, SpikingLM, SpikingViT, count_ops
from snnformer.attention import ATTENTION, CompetitiveShuntingAttention
from snnformer.neurons import SpikingNeuron

torch.manual_seed(0)


@pytest.mark.parametrize("levels", [1, 2, 4])
@pytest.mark.parametrize("v_th", [1.0, 4.0])
def test_neuron_emits_quantised_counts(levels, v_th):
    sn = SpikingNeuron(levels=levels, v_th=v_th)
    x = torch.randn(3, 64, 32) * 4
    s = sn(x)
    c = s / sn.unit
    assert s.shape == x.shape
    assert torch.allclose(c, c.round(), atol=1e-5)
    assert c.min() >= 0 and c.max() <= levels


def test_neuron_gradients_nonzero():
    for levels in (1, 4):
        x = torch.randn(2, 128, requires_grad=True)
        SpikingNeuron(levels=levels)(x).sum().backward()
        assert x.grad.abs().sum() > 0


SPIKE_ATTN = ["csa", "ssa", "ssa_norm", "sdsa", "softmax", "wta", "dendritic", "dendritic_event"]


@pytest.mark.parametrize("attn", SPIKE_ATTN)
@pytest.mark.parametrize("neuron", ["ilif", "lif"])
def test_lm_is_causal(attn, neuron):
    cfg = SNNConfig(neuron=neuron, attn=attn, T=2 if neuron == "lif" else 1)
    m = SpikingLM(cfg, vocab=50, dim=32, depth=2, heads=2, ctx=24).eval()
    idx = torch.randint(0, 50, (3, 24))
    out1 = m(idx)
    idx2 = idx.clone()
    idx2[:, 15:] = torch.randint(0, 50, (3, 9))
    out2 = m(idx2)
    assert torch.allclose(out1[:, :15], out2[:, :15], atol=1e-5)
    assert not torch.allclose(out1[:, 15:], out2[:, 15:])


@pytest.mark.parametrize("attn", ["csa", "ssa", "ssa_norm", "sdsa", "wta", "dendritic"])
@pytest.mark.parametrize("mlp", ["cdglu", "mlp"])
@pytest.mark.parametrize("v_th", [1.0, 4.0])
def test_weight_layers_only_receive_spikes(attn, mlp, v_th):
    """Spike-driven property: every Linear/Conv except the input encoder receives
    non-negative integer multiples of the spike unit v_th/D, at most D of them."""
    cfg = SNNConfig(attn=attn, mlp=mlp, levels=4, v_th=v_th, qk_levels=1 if attn == "csa" else None)
    m = SpikingViT(cfg, dim=32, depth=2, heads=2).eval()
    bad = []

    def hook(mod, inp, out):
        x = inp[0]
        if getattr(mod, "real_input", False):
            return
        c = x / cfg.spike_unit
        if not torch.allclose(c, c.round(), atol=1e-4) or c.min() < 0 or c.max() > 4 + 1e-4:
            bad.append(mod)

    for mod in m.modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            mod.register_forward_hook(hook)
    m(torch.randn(2, 3, 32, 32))
    assert not bad, bad


@pytest.mark.parametrize("causal", [False, True])
def test_csa_output_is_bounded_fraction(causal):
    cfg = SNNConfig(attn="csa")
    attn = CompetitiveShuntingAttention(32, 2, cfg, causal=causal)
    T, B, H, N, d = 1, 2, 2, 40, 16
    q, k, v = (torch.randint(0, 5, (T, B, H, N, d)).float() / 4 for _ in range(3))
    y = attn.mix(q, k, v)
    assert y.min() >= 0 and y.max() <= 1 + 1e-6


def test_csa_selects_matching_key():
    """With orthogonal binary keys, a query equal to one key attends only to it."""
    cfg = SNNConfig(attn="csa", levels=1)
    attn = CompetitiveShuntingAttention(8, 1, cfg)
    keys = torch.eye(8)[None, None, None]  # [1,1,1,8,8]
    vals = torch.rand(1, 1, 1, 8, 8).round()
    q = keys[..., 3:4, :]
    y = attn.mix(q, keys, vals)
    assert torch.allclose(y[..., 0, :], vals[..., 3, :])


def test_csa_threshold_params_get_gradients():
    cfg = SNNConfig(attn="csa")
    m = SpikingLM(cfg, vocab=40, dim=32, depth=1, heads=2, ctx=16)
    idx = torch.randint(0, 40, (4, 16))
    nn.functional.cross_entropy(m(idx).flatten(0, 1), idx.flatten()).backward()
    a = m.blocks[0].attn
    assert a.gamma.grad.abs().sum() > 0 and a.bias.grad.abs().sum() > 0


def test_ann_reference_and_op_counter():
    for cfg in (SNNConfig(neuron="ann", attn="softmax", mlp="mlp"), SNNConfig()):
        m = SpikingViT(cfg, dim=32, depth=1, heads=2).eval()
        with torch.no_grad(), count_ops(m, cfg.spike_levels, cfg.spike_unit) as c:
            m(torch.randn(2, 3, 32, 32))
            c.samples += 2
        s = c.summary()
        assert s["energy_mJ_per_sample"] > 0
        if cfg.neuron == "ann":
            assert s["ac_per_sample"] == 0
        else:
            assert 0 < s["firing_rate"] < 1 and s["ac_per_sample"] > s["mac_per_sample"]


def test_all_attention_registered():
    assert set(SPIKE_ATTN) == set(ATTENTION)


@pytest.mark.parametrize("attn", SPIKE_ATTN)
def test_causal_gradients_are_finite(attn):
    cfg = SNNConfig(attn=attn, v_th=4.0)
    m = SpikingLM(cfg, vocab=40, dim=32, depth=2, heads=2, ctx=16)
    idx = torch.randint(0, 40, (4, 16))
    nn.functional.cross_entropy(m(idx).flatten(0, 1), idx.flatten()).backward()
    for name, p in m.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), name


def test_graded_csa_counts_and_bounds():
    cfg = SNNConfig(attn="csa", csa_levels=4)
    attn = CompetitiveShuntingAttention(32, 2, cfg, causal=True)
    q, k, v = (torch.randint(0, 5, (1, 2, 2, 20, 16)).float() / 4 for _ in range(3))
    y = attn.mix(q, k, v)
    assert y.min() >= 0 and y.max() <= 1 + 1e-6  # still a weighted average of values in [0, 1]
    # attention counts are integers in [0, K]
    n = q.shape[-2]
    scores = (q @ k.transpose(-2, -1)) / 4.0
    kbar = k.cumsum(-2) / torch.arange(1, n + 1).float().unsqueeze(-1)
    theta = attn.gamma * (q * kbar).sum(-1, keepdim=True) / 4.0 + attn.bias
    from snnformer.neurons import integer_spike
    a = integer_spike(((scores - theta) + 0.5) / 4, 4) * 4
    assert torch.allclose(a, a.round()) and a.min() >= 0 and a.max() <= 4
