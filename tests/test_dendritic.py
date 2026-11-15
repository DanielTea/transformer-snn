import pytest
import torch
from torch.nn import functional as F

from snnformer import SNNConfig, SpikingLM
from snnformer.dendritic import DendriticDelayAttention, address_events, prefix_read


def test_prefix_sums_match_dense_oracle_and_gradient():
    """Check partial, complete, and multiple blocks against an independent oracle."""
    rng = torch.Generator().manual_seed(123)
    q = torch.rand(2, 3, 15, 8, generator=rng, dtype=torch.float64, requires_grad=True)
    k = torch.rand(q.shape, generator=rng, dtype=q.dtype, requires_grad=True)
    v = torch.rand(2, 3, 15, 5, generator=rng, dtype=torch.float64, requires_grad=True)
    expected = (q @ k.transpose(-2, -1)).tril() @ v
    expected_grads = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for block in (1, 4, 15, 32):
        observed = prefix_read(q, k, v, block)
        torch.testing.assert_close(observed, expected)
        for grad, expected_grad in zip(torch.autograd.grad(observed.square().sum(), (q, k, v)), expected_grads):
            torch.testing.assert_close(grad, expected_grad)


def test_address_and_delay_bind_later_values_to_earlier_keys():
    cfg = SNNConfig(attn="dendritic", address_bits=2, memory_delay=1, memory_block=2)
    mixer = DendriticDelayAttention(2, 2, cfg, causal=True)
    # Address word 0 at tokens 0 and 3; only tokens 1 and 2 allow writes.
    words = torch.tensor([0, 1, 2, 0]).expand(1, 1, 2, 4)
    bits = ((words.unsqueeze(-1) >> torch.arange(2)) & 1).float()
    events = address_events(bits)
    torch.testing.assert_close(events, F.one_hot(words, 4).float())
    writes = torch.tensor([0, 1, 1, 0]).reshape(1, 1, 1, 4, 1).expand(1, 1, 2, 4, 1)
    values = torch.tensor([9.0, 3.0, 7.0, 5.0]).reshape(1, 1, 1, 4, 1).expand_as(writes)
    reads = prefix_read(events, writes * mixer._shift(events),
                        torch.cat((values, torch.ones_like(values)), -1), block=2)
    # The delayed head retrieves value 3 at token 3. The undelayed head has
    # never written at address 0, and the delayed head couldn't read token 1 early.
    torch.testing.assert_close(reads[0, 0, :, 3], torch.tensor([[0., 0.], [3., 1.]]))
    torch.testing.assert_close(reads[0, 0, :, 0], torch.zeros(2, 2))


@pytest.mark.parametrize("delay,timesteps,relay", [(0, 1, False), (1, 1, True), (3, 2, True), (3, 2, False)])
def test_stream_matches_batched_model_with_fixed_state(delay, timesteps, relay):
    cfg = SNNConfig(attn="dendritic", v_th=4.0, T=timesteps, memory_delay=delay,
                    memory_block=4, address_bits=3, position_embeddings=False,
                    causal_norm=True, memory_relay=relay)
    model = SpikingLM(cfg, vocab=24, dim=16, heads=2, depth=2).eval()
    idx = torch.randint(0, 24, (3, 15))
    state, first_storage = None, None
    outputs = []
    for token in idx.unbind(dim=-1):
        out, state = model.step(token, state)
        storage = [(layer.table.shape, layer.history.shape, layer.table.data_ptr()) for layer in state.layers]
        if first_storage is None:
            first_storage = storage
        assert storage == first_storage
        outputs.append(out)
    torch.testing.assert_close(torch.stack(outputs, dim=1), model(idx))
