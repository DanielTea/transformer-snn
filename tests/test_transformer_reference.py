import pytest
import torch
import torch.nn.functional as F

from snnformer.config import SNNConfig
from snnformer.transformer_reference import TransformerLM, _sliding_window_attention


def test_blocked_attention_matches_dense_window_and_gradients():
    window = 4
    length = 2 * window + 1
    generator = torch.Generator().manual_seed(62)
    q, k, v = (
        torch.randn((2, 3, length, 5), generator=generator, dtype=torch.float64, requires_grad=True)
        for _ in range(3)
    )
    positions = torch.arange(length)
    distance = positions[:, None] - positions
    causal_window = (distance >= 0) & (distance < window)

    dense = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_window)
    blocked = _sliding_window_attention(q, k, v, window)
    torch.testing.assert_close(blocked, dense)
    dense_grad = torch.autograd.grad(dense.square().sum(), (q, k, v))
    blocked_grad = torch.autograd.grad(blocked.square().sum(), (q, k, v))
    for actual, expected in zip(blocked_grad, dense_grad, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("window,rope", [(1, False), (4, True), (12, True)])
def test_local_transformer_step_matches_batch_across_ring_wraps(window, rope):
    model = TransformerLM(SNNConfig(neuron="ann", attn="transformer_local", local_window=window,
                                    position_embeddings=rope),
                          vocab=32, dim=20, depth=2, heads=4, ctx=8).eval()
    tokens = torch.randint(0, 32, (2, 27), generator=torch.Generator().manual_seed(88))
    with torch.no_grad():
        batched = model(tokens)
    state, steps = None, []
    for position, token in enumerate(tokens.unbind(dim=1)):
        logits, state = model.step(token, state)
        steps.append(logits)
        if position == 0:
            rings = [(layer.keys.data_ptr(), layer.values.data_ptr()) for layer in state.layers]
    torch.testing.assert_close(torch.stack(steps, dim=1), batched)
    assert state.length == tokens.shape[1]
    assert [(layer.keys.data_ptr(), layer.values.data_ptr()) for layer in state.layers] == rings
    assert all(layer.keys.shape == layer.values.shape == (2, 4, window, 5) for layer in state.layers)
