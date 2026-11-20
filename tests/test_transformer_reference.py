from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from snnformer.config import SNNConfig
from snnformer.transformer_reference import TransformerLM, _sliding_window_attention


def test_feed_forward_growth_preserves_logits_and_new_outputs_can_learn():
    torch.manual_seed(45)
    cfg = SNNConfig(neuron="ann", attn="transformer", mlp_ratio=2.0, position_embeddings=True)
    options = dict(vocab=32, dim=16, depth=2, heads=4, ctx=32)
    parent = TransformerLM(cfg, **options)
    grown = TransformerLM(replace(cfg, mlp_ratio=5.0), **options)
    width = grown.initialize_wider_feed_forward(cfg, parent.state_dict())
    tokens = torch.randint(32, (2, 23))
    expected, actual = parent(tokens), grown(tokens)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    actual.square().mean().backward()
    assert all(bool(block.mlp_out.weight.grad[:, width:].any()) for block in grown.blocks)


def test_transformer_depth_growth_can_also_widen_the_original_feed_forward_layers():
    torch.manual_seed(45)
    cfg = SNNConfig(neuron="ann", attn="transformer", mlp_ratio=2.0, position_embeddings=True)
    options = dict(vocab=32, dim=16, heads=4, ctx=32)
    parent = TransformerLM(cfg, depth=2, **options)
    grown = TransformerLM(replace(cfg, mlp_ratio=3.0), depth=3, **options)
    width = grown.initialize_deeper(cfg, parent.state_dict(), prior_depth=2)
    tokens = torch.randint(32, (2, 23))
    output = grown(tokens)
    torch.testing.assert_close(output, parent(tokens), atol=1e-7, rtol=1e-6)
    output.square().mean().backward()
    assert width == parent.blocks[0].mlp_out.in_features
    assert all(bool(block.mlp_out.weight.grad[:, width:].any()) for block in grown.blocks[:2])
    assert bool(grown.blocks[2].attn.out.weight.grad.any())
    assert bool(grown.blocks[2].mlp_out.weight.grad.any())


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


_FULL = SNNConfig(neuron="ann", attn="transformer", mlp_ratio=4.0, position_embeddings=True)
_SMALL = dict(vocab=32, dim=16, depth=2, heads=4, ctx=24)


def test_plain_units_spend_the_same_weights_on_more_units_with_a_threshold():
    torch.manual_seed(3)
    gated, plain = TransformerLM(_FULL, **_SMALL), TransformerLM(replace(_FULL, feed_forward="relu"), **_SMALL)
    count = lambda model: sum(parameter.numel() for parameter in model.parameters())  # noqa: E731
    # Three rows of weights per gated unit, two per plain unit.
    assert gated.blocks[0].mlp_out.in_features == 43 and plain.blocks[0].mlp_out.in_features == 64
    assert plain.blocks[0].mlp_in.out_features == 64 and abs(count(plain) - count(gated)) <= 2 * 16
    block = plain.blocks[0]
    x = torch.randn(2, 5, 16)
    hidden = block.mlp_in(block.mlp_norm(x))
    torch.testing.assert_close(block._feed_forward(x), x + block.mlp_out(hidden.clamp_min(0)))
    tokens = torch.randint(32, (2, 23))
    plain(tokens).square().mean().backward()
    assert all(bool(block.mlp_in.weight.grad.any()) and bool(block.mlp_out.weight.grad.any()) for block in plain.blocks)


def test_age_offsets_address_by_distance_in_place_of_rotary_phases():
    torch.manual_seed(4)
    model = TransformerLM(replace(_FULL, age_offsets=8.0), **_SMALL)
    assert model.rotary_dim == 0 and model._rope(torch.arange(4.0)) is None
    attention = model.blocks[0].attn
    assert attention.age.shape == (4, 24) and not attention.age.any() and attention.age_rate == 8.0
    # Level offsets leave a head with what the keys hold: the model without positions.
    bare = TransformerLM(replace(_FULL, position_embeddings=False), **_SMALL)
    bare.load_state_dict(model.state_dict(), strict=False)
    tokens = torch.randint(32, (2, 23))
    torch.testing.assert_close(model(tokens), bare(tokens), atol=1e-6, rtol=1e-5)
    # A head whose offset for one distance is far above the others reads the key that far back.
    x = torch.randn(2, 9, 16)
    with torch.no_grad():
        attention.age[:, 2] = 5.0
    values = attention.qkv(x).reshape(2, 9, 3, 4, 4)[:, :, 2].flatten(2)
    torch.testing.assert_close(attention(x, None)[:, 2:], attention.out(values[:, :-2]), atol=1e-5, rtol=1e-4)
    # No later token is seen, and a distance beyond the table takes its last offset.
    longer, changed = torch.randint(32, (1, 40)), None
    changed = longer.clone()
    changed[:, 30:] = (changed[:, 30:] + 1) % 32
    first, second = model(longer), model(changed)
    torch.testing.assert_close(first[:, :30], second[:, :30], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(first[:, 30:], second[:, 30:])
    model(tokens).square().mean().backward()
    assert all(bool(block.attn.age.grad.any()) for block in model.blocks)


def test_a_unit_is_gated_or_plain_and_opens_smoothly_or_above_a_threshold():
    torch.manual_seed(5)
    count = lambda model: sum(parameter.numel() for parameter in model.parameters())  # noqa: E731
    models = {name: TransformerLM(replace(_FULL, feed_forward=name), **_SMALL) for name in ("swiglu", "reglu", "gelu", "relu")}
    # The two gated kinds have the same weights, and so have the two plain kinds.
    assert count(models["reglu"]) == count(models["swiglu"]) and count(models["gelu"]) == count(models["relu"])
    x = torch.randn(2, 5, 16)
    for name, opened in (("swiglu", F.silu), ("reglu", F.relu)):
        block = models[name].blocks[0]
        gate, value = block.mlp_in(block.mlp_norm(x)).chunk(2, dim=-1)
        torch.testing.assert_close(block._feed_forward(x), x + block.mlp_out(opened(gate) * value))
    for name, fired in (("gelu", F.gelu), ("relu", F.relu)):
        block = models[name].blocks[0]
        torch.testing.assert_close(block._feed_forward(x), x + block.mlp_out(fired(block.mlp_in(block.mlp_norm(x)))))
    tokens = torch.randint(32, (2, 23))
    for model in models.values():
        model(tokens).square().mean().backward()
        assert all(bool(block.mlp_in.weight.grad.any()) and bool(block.mlp_out.weight.grad.any()) for block in model.blocks)
    # A gated unit that opens only above zero passes nothing on while its gate is shut, whatever its value row hears.
    block = models["reglu"].blocks[0]
    with torch.no_grad():
        block.mlp_in.weight[:block.mlp_in.out_features // 2] = 0
    torch.testing.assert_close(block._feed_forward(x), x)


def test_an_output_layer_of_its_own_learns_apart_from_the_embedding():
    torch.manual_seed(6)
    count = lambda model: sum(parameter.numel() for parameter in model.parameters())  # noqa: E731
    tied, untied = TransformerLM(_FULL, **_SMALL), TransformerLM(replace(_FULL, untied_output=True), **_SMALL)
    assert tied.head.weight is tied.tok.weight and untied.head.weight is not untied.tok.weight
    # The output layer adds a weight from every channel to every byte, drawn as the embedding's are.
    assert count(untied) - count(tied) == untied.head.weight.numel() == 32 * 16
    assert not torch.equal(untied.head.weight, untied.tok.weight)
    assert abs(untied.head.weight.std() / 0.02 - 1) < 0.2
    tokens = torch.randint(32, (2, 23))
    untied(tokens).square().mean().backward()
    assert untied.head.weight.grad.any() and untied.tok.weight.grad.any()
    assert not torch.equal(untied.head.weight.grad, untied.tok.weight.grad)
    # With the embedding's weights in its output layer the model is the tied one.
    with torch.no_grad():
        untied.head.weight.copy_(untied.tok.weight)
    tied.load_state_dict(untied.state_dict())
    torch.testing.assert_close(tied(tokens), untied(tokens))


@pytest.mark.parametrize("setting", [dict(feed_forward="tanh"), dict(age_offsets=-1.0),
                                     dict(attn="transformer_local", feed_forward="relu"),
                                     dict(attn="transformer_local", feed_forward="reglu"),
                                     dict(attn="transformer_local", age_offsets=8.0),
                                     dict(attn="transformer_local", untied_output=True)])
def test_the_controls_belong_to_the_full_transformer(setting):
    with pytest.raises(ValueError):
        TransformerLM(replace(_FULL, **setting), **_SMALL)
