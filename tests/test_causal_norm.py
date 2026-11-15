import copy

import pytest
import torch

from snnformer import SNNConfig, SpikingLM
from snnformer.config import SeqBN


@pytest.mark.parametrize("attn,mlp", [("csa", "mlp"), ("csa", "cdglu"), ("dendritic", "cdglu"),
                                      ("dendritic_event", "cdglu")])
def test_causal_normalization_preserves_prefix_during_training(attn, mlp):
    cfg = SNNConfig(attn=attn, csa_levels=4, mlp=mlp, causal_norm=True, v_th=4.0, memory_block=4)
    model = SpikingLM(cfg, vocab=32, dim=24, depth=2, heads=2, ctx=12)
    gen = torch.Generator().manual_seed(10)
    model(torch.randint(0, 32, (3, 12), generator=gen))
    same_state = copy.deepcopy(model)

    tokens = torch.randint(0, 32, (3, 12), generator=gen)
    changed_suffix = tokens.clone()
    changed_suffix[:, 7:] = (changed_suffix[:, 7:] + 1) % 32
    logits = model(tokens)
    changed_logits = same_state(changed_suffix)

    torch.testing.assert_close(logits[:, :7], changed_logits[:, :7])
    assert not torch.equal(logits[:, 7:], changed_logits[:, 7:])
    logits[:, :7].square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_causal_normalization_matches_running_stats():
    causal = SeqBN(3, causal=True)
    regular = SeqBN(3)
    x = torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3)

    expected_initial = regular.eval()(x)
    torch.testing.assert_close(causal(x), expected_initial)
    regular.train()(x)
    torch.testing.assert_close(causal.bn.running_mean, regular.bn.running_mean)
    torch.testing.assert_close(causal.bn.running_var, regular.bn.running_var)

    regular.eval()
    causal.eval()
    torch.testing.assert_close(causal(x), regular(x))
