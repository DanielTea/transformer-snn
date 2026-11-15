import torch
from torch import nn

from snnformer import SNNConfig, SpikingLM
from snnformer.dendritic_event import DendriticEventAttention, _EventScan


def test_event_scan_matches_unrolled_hard_recurrence_and_its_gradient():
    gen = torch.Generator().manual_seed(91)
    reset = (torch.rand(2, 17, 1, generator=gen, dtype=torch.float64) < 0.3).double().requires_grad_()
    raw = torch.randn(2, 17, 4, generator=gen, dtype=torch.float64)
    update = (raw * reset.detach()).requires_grad_()
    state = torch.zeros_like(update[:, 0])
    states = []
    for r, u in zip(reset.unbind(dim=1), update.unbind(dim=1)):
        state = (1 - r) * state + u
        states.append(state)

    expected = torch.stack(states, dim=1)
    observed = _EventScan.apply(reset, update)
    torch.testing.assert_close(observed, expected)
    expected_grads = torch.autograd.grad(expected.square().sum(), (reset, update))
    observed_grads = torch.autograd.grad(observed.square().sum(), (reset, update))
    for got, want in zip(observed_grads, expected_grads):
        torch.testing.assert_close(got, want)


def test_pending_address_survives_a_long_gap_and_is_consumed_once():
    class CaptureToken(nn.Module):
        def forward(self, spikes):
            return spikes[..., :1]

    mixer = DendriticEventAttention(8, 1, SNNConfig(address_bits=2), causal=True).eval()
    mixer.capture = CaptureToken()
    mixer.capture_bn = nn.Identity()
    addresses = torch.zeros(1, 1, 1, 1027, 4)
    addresses[..., 0] = 1
    addresses[..., 1, 0] = 0
    addresses[..., 1, 3] = 1  # key address 3
    spikes = torch.zeros(1, 1, 1027, 8)
    spikes[..., 1, 0] = 1  # capture once
    write = torch.zeros_like(addresses[..., :1])
    write[..., [0, 1025, 1026], 0] = 1  # no pending, mature value, duplicate close
    keys = mixer._write_events(addresses, write, spikes)
    expected = torch.zeros_like(keys)
    expected[..., 1025, 3] = 1
    torch.testing.assert_close(keys, expected)


def test_event_lm_stream_matches_batched_with_fixed_pending_buffer():
    cfg = SNNConfig(attn="dendritic_event", v_th=4.0, T=2, causal_norm=True,
                    position_embeddings=False, memory_block=7)
    model = SpikingLM(cfg, vocab=16, dim=16, depth=2, heads=2).eval()
    idx = torch.randint(0, 16, (2, 31), generator=torch.Generator().manual_seed(8))
    state = None
    outputs = []
    for token in idx.unbind(-1):
        logits, state = model.step(token, state)
        outputs.append(logits)
    torch.testing.assert_close(torch.stack(outputs, dim=1), model(idx))
    assert all(layer.pending_address.shape == (2, 2, 2) for layer in state.layers)
