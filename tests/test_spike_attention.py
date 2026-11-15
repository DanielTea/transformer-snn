import pytest
import torch

from snnformer.spike_attention import SpikeCoincidence


@pytest.mark.parametrize("length", [3, 4, 11])
def test_blocked_coincidence_matches_dense_causal_oracle(length):
    window = 4
    mixer = SpikeCoincidence(heads=3, head_dim=4, window=window).double()
    generator = torch.Generator().manual_seed(42)
    q, k, v = (
        torch.randint(-3, 4, (2, 3, length, 4), generator=generator).double()
        for _ in range(3)
    )
    with torch.no_grad():
        mixer.age_bias.copy_(torch.randn(3, window, dtype=q.dtype, generator=generator))

    positions = torch.arange(length)
    ages = positions[:, None] - positions
    visible = (0 <= ages) & (ages < window)
    scores = (q @ k.transpose(-1, -2)) / 2 + mixer.age_bias[:, ages.clamp(0, window - 1)]
    best = scores.masked_fill(~visible, -torch.inf).amax(dim=-1, keepdim=True)
    gaps = (best - scores).round().clamp(0, 5).long()
    weights = torch.tensor([16., 8., 4., 2., 1., 0.], dtype=q.dtype)[gaps] * visible
    expected = (weights @ v) / weights.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(mixer(q, k, v), expected)


def test_coincidence_discrete_vote_cutoff():
    mixer = SpikeCoincidence(heads=1, head_dim=1, window=3)
    with torch.no_grad():
        mixer.age_bias.zero_()
    q = torch.ones(1, 1, 3, 1)
    k = torch.tensor([0., 1., 5.]).reshape(1, 1, 3, 1)
    v = torch.tensor([999., 17., 0.]).reshape(1, 1, 3, 1)
    # The oldest key is five bins below the winner and gets no vote.
    # The second key gets one vote and the winner gets sixteen.
    torch.testing.assert_close(mixer(q, k, v)[0, 0, -1, 0], torch.tensor(1.))


@pytest.mark.parametrize("window", [1, 4, 9])
def test_coincidence_step_matches_batch_and_keeps_ring_storage(window):
    mixer = SpikeCoincidence(heads=2, head_dim=4, window=window)
    generator = torch.Generator().manual_seed(24)
    q, k, v = (
        torch.randint(-3, 4, (2, 2, 22, 4), generator=generator).float()
        for _ in range(3)
    )
    expected = mixer(q, k, v)
    state = None
    outputs = []
    for position, events in enumerate(zip(q.unbind(2), k.unbind(2), v.unbind(2), strict=True)):
        output, state = mixer.step(*events, state=state)
        outputs.append(output)
        if position == 0:
            backing = (state.keys.data_ptr(), state.values.data_ptr())
    torch.testing.assert_close(torch.stack(outputs, dim=2), expected)
    assert state.keys.shape == state.values.shape == (2, 2, window, 4)
    assert (state.keys.data_ptr(), state.values.data_ptr()) == backing


def test_coincidence_surrogate_gradients_are_finite():
    mixer = SpikeCoincidence(heads=2, head_dim=4, window=4)
    generator = torch.Generator().manual_seed(91)
    q, k, v = (
        torch.randint(0, 3, (2, 2, 9, 4), generator=generator).float().requires_grad_()
        for _ in range(3)
    )
    mixer(q, k, v).square().mean().backward()
    for tensor in (q, k, v, mixer.age_bias):
        assert tensor.grad is not None
        assert tensor.grad.isfinite().all()
        assert tensor.grad.abs().sum() > 0
