import pytest
import torch

from snnformer.token_if import TokenIF


def test_charge_history_changes_spiking_after_silence():
    currents = torch.zeros(2, 1027, 1)
    currents[0, 0] = 0.375
    currents[:, -1] = 0.125

    carried = TokenIF(max_spikes=1)(currents)
    reset = TokenIF(max_spikes=1, carry=False)(currents)

    expected = torch.zeros_like(currents)
    expected[0, -1] = 1
    torch.testing.assert_close(carried, expected)
    torch.testing.assert_close(reset, torch.zeros_like(currents))


_DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("carry", [True, False])
def test_scan_matches_streaming_phase_and_emits_bounded_counts(device, carry):
    neuron = TokenIF(max_spikes=4, carry=carry)
    rng = torch.Generator().manual_seed(7)
    currents = (torch.rand(2, 129, 7, generator=rng) * 5 - 0.5).to(device)

    batched = neuron(currents)
    streamed = []
    phase = None
    for current in currents.unbind(dim=1):
        spikes, phase = neuron.step(current, phase)
        streamed.append(spikes)

    assert torch.equal(batched, torch.stack(streamed, dim=1))
    assert torch.equal(batched, batched.round())
    assert torch.all((0 <= batched) & (batched <= 4))
    assert phase.dtype == torch.int32
    assert torch.all((0 <= phase) & (phase < 256))


@pytest.mark.parametrize("device", _DEVICES)
def test_gradients_use_a_local_surrogate(device):
    currents = torch.tensor([[[0.125], [0.375], [0.25]]], device=device, requires_grad=True)
    neuron = TokenIF(max_spikes=1)
    neuron(currents)[:, 1].sum().backward()
    torch.testing.assert_close(currents.grad.cpu(), torch.tensor([[[0.0], [1.0], [0.0]]]))
