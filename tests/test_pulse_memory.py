import pytest
import torch

from snnformer.pulse_memory import PulseMemory, PulseMemoryState


def _routed_memory(carry_reads=True, **kwargs):
    memory = PulseMemory(3, 2, carry_reads=carry_reads, **kwargs)
    with torch.no_grad():
        weights = memory.project.weight.view(memory.read_bins + 2, 2, 3)
        weights.zero_()
        weights[0, 0, 0] = weights[2, 0, 0] = 0.75
        weights[0, 1, 1] = weights[2, 1, 1] = 0.75
        weights[3, :, 2] = 0.75
    return memory


def test_complementary_pulse_restores_all_charges_and_erase_requires_a_probe():
    pulses = torch.arange(257)
    memory = PulseMemory(1, len(pulses))
    with torch.no_grad():
        memory.project.weight.zero_()
        memory.bias[0].fill_(1)
        memory.bias[1].copy_(torch.logit(pulses.float() / 256, eps=1e-6))
    phase = torch.arange(256).float()[:, None].expand(-1, len(pulses)).clone()
    spikes = torch.zeros(256, 1)
    probe = torch.arange(256).remainder(2) == 0
    expected = (phase + pulses >= 256).float()

    read, restored = memory.step(spikes, phase)
    assert torch.equal(read, expected)
    assert torch.equal(restored, phase)

    memory.carry_reads = False
    read, cleared = memory.step(spikes, phase, probe=probe)
    assert torch.equal(read, expected * probe[:, None])
    assert torch.equal(cleared, phase.masked_fill(probe[:, None], 0))


@pytest.mark.parametrize("carry", [True, False])
def test_cell_routing_read_before_overwrite_and_external_clear_precedence(carry):
    memory = _routed_memory(carry)
    spikes = torch.tensor([[[1., 0., 1.], [0., 1., 1.], [1., 0., 0.], [1., 0., -1.],
                            [0., 1., 0.], [1., 0., 0.], [0., 1., 1.], [0., 1., 0.]]])
    probe = torch.tensor([[True, False, True, True, True, True, True, True]])
    write = torch.tensor([[True, True, False, True, False, False, True, False]])
    clear = torch.tensor([[False, False, False, False, False, False, True, False]])
    stats = {}

    observed = memory(spikes, probe=probe, write=write, clear=clear, stats=stats)

    expected = torch.zeros(1, 8, 2)
    expected[0, 2, 0] = expected[0, 4, 1] = expected[0, 7, 1] = 1
    if carry:
        expected[0, 3, 0] = 1
    assert torch.equal(observed, expected)
    assert stats["probe_events"] == 7
    assert stats["write_events"] == 4
    assert stats["read_spikes"] == expected.sum()
    assert all(not stat.requires_grad for stat in stats.values())


_DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("carry", [True, False])
@pytest.mark.parametrize("device", _DEVICES)
def test_scan_matches_streaming_with_learned_events(device, carry):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        memory = PulseMemory(11, 8, carry_reads=carry).to(device)
    generator = torch.Generator().manual_seed(36)
    spikes = torch.randint(-4, 5, (2, 49, 11), generator=generator).float().to(device)
    probe, write, clear = (torch.rand((2, 49), generator=generator).to(device) < rate
                           for rate in (0.8, 0.7, 0.1))

    with torch.no_grad():
        expected = memory(spikes, probe=probe, write=write, clear=clear)
        phase, streamed = None, []
        for index, token in enumerate(spikes.unbind(1)):
            read, phase = memory.step(token, phase, probe=probe[:, index],
                                      write=write[:, index], clear=clear[:, index])
            streamed.append(read)
    assert torch.equal(torch.stack(streamed, 1), expected)
    assert torch.any(expected > 0) and torch.any(expected == 0)
    assert phase.shape == (2, 8)
    assert torch.all((phase >= 0) & (phase < 256) & (phase == phase.round()))


@pytest.mark.parametrize("device", _DEVICES)
def test_gradient_reaches_write_across_long_silence(device):
    memory = _routed_memory().to(device)
    spikes = torch.zeros(1, 1027, 3, device=device)
    spikes[0, 0] = torch.tensor([1., 0., 1.], device=device)
    spikes[0, -1, 0] = 1
    spikes.requires_grad_()
    write = torch.zeros(1, 1027, dtype=torch.bool, device=device)
    write[:, 0] = True
    probe = torch.zeros_like(write)
    probe[:, -1] = True

    observed = memory(spikes, probe=probe, write=write)
    assert observed.sum() == observed[0, -1, 0] == 1
    observed[0, -1, 0].backward()

    assert torch.isfinite(spikes.grad).all()
    assert spikes.grad[0, 0, 2] > 0
    assert torch.isfinite(memory.project.weight.grad).all()
    weight_grad = memory.project.weight.grad.view(4, 2, 3)
    assert weight_grad[2, 0, 0] > 0
    assert weight_grad[3, 0, 2] > 0


@pytest.mark.parametrize("clear_bank", [False, True])
@pytest.mark.parametrize("device", _DEVICES)
def test_parallel_write_holds_match_greedy_cooldown(device, clear_bank):
    holds = (0, 1, 2, 3, 17, 64, 128, 2**31)
    memory = PulseMemory(8, len(holds), write_holds=holds).to(device)
    generator = torch.Generator().manual_seed(41)
    proposals = (torch.rand((2, 257, len(holds)), generator=generator) > 0.38).to(device)
    clear = (torch.rand((2, 257, 1), generator=generator) < 0.02).to(device) if clear_bank else None
    observed = memory._accepted_writes(proposals.float(), clear).bool()

    cooldown = torch.zeros_like(proposals[:, 0], dtype=torch.long)
    streamed = []
    for t, proposal in enumerate(proposals.unbind(1)):
        if clear is not None:
            cooldown = cooldown.masked_fill(clear[:, t], 0)
        accepted = proposal & (cooldown == 0)
        cooldown = torch.where(accepted, memory._write_holds, cooldown).sub(1).clamp_min(0)
        streamed.append(accepted)

    assert torch.equal(observed, torch.stack(streamed, dim=1))


@pytest.mark.parametrize("read_bins", [2, 4])
@pytest.mark.parametrize("device", _DEVICES)
def test_protected_reads_and_selective_ablation_leave_writes_unchanged(device, read_bins):
    memory = _routed_memory(write_holds=(3, 0), read_bins=read_bins, fixed_cuts=read_bins == 4).to(device)
    spikes = torch.tensor([[[1., 0., 1.], [1., 0., -1.], [1., 0., -1.], [1., 0., -1.],
                            [1., 0., -1.], [1., 0., 1.], [1., 0., -1.]]], device=device)
    clear = torch.tensor([[False, False, False, False, False, True, False]], device=device)
    stats, ablated_stats, training_stats = {}, {}, {}
    read = memory(spikes, clear=clear, stats=stats, training_stats=training_stats)
    ablated = memory(spikes, clear=clear, stats=ablated_stats, ablate_min_age=2)

    state, observations = None, []
    for current, reset in zip(spikes.unbind(1), clear.unbind(1)):
        emitted, state = memory.step(current, state, clear=reset)
        observations.append(emitted)
    assert torch.equal(read, torch.stack(observations, 1))
    assert isinstance(state, PulseMemoryState)
    assert state.phase.shape == state.cooldown.shape == (1, 2)
    assert torch.equal(state.cooldown, torch.tensor([[1, 0]], device=device))

    assert stats["proposed_write_events"] == ablated_stats["proposed_write_events"] == 7
    assert stats["write_events"] == stats["accepted_write_events"] == ablated_stats["write_events"] == 3
    assert stats["rejected_write_events"] == ablated_stats["rejected_write_events"] == 4
    assert stats["probe_events"] == ablated_stats["probe_events"] == 7
    assert ablated_stats["old_probe_events"] == ablated_stats["altered_probe_events"] == 2
    assert memory.read_width == read.shape[-1] == (2 if read_bins == 2 else 8)
    if read_bins == 2:
        assert read[0, :, 0].tolist() == [0, 1, 1, 1, 0, 0, 1]
        assert ablated[0, :, 0].tolist() == [0, 1, 0, 0, 0, 0, 1]
    else:
        assert stats["read_bin_counts"].tolist() == [0, 4, 1, 2]
        assert ablated_stats["read_bin_counts"].tolist() == [0, 2, 1, 4]

    assert not training_stats["write_drive"].requires_grad
    assert training_stats["write_drive"][0, 0, 0] == 0.25
    proposals = training_stats["proposed_write_counts"]
    accepted = training_stats["accepted_write_counts"]
    assert proposals.tolist() == [7, 0] and accepted.tolist() == [3, 0]
    (proposals + accepted).sum().backward()
    assert torch.isfinite(memory.bias.grad).all()
    assert memory.bias.grad[2, 0] > 0


def test_unrestricted_sparse_control_loads_legacy_weights_and_preserves_reads():
    legacy = PulseMemory(4, 3)
    control = PulseMemory(4, 3, write_holds=(0, 0, 1))
    control.load_state_dict(legacy.state_dict(), strict=True)
    token = torch.ones(1, 5, 4)
    assert torch.equal(control(token), legacy(token))
    _, phase = legacy.step(token[:, 0])
    assert isinstance(phase, torch.Tensor)


@pytest.mark.parametrize("read_bins", [2, 4])
def test_counterfactual_read_matches_streamed_payload_under_frozen_schedule(read_bins):
    memory = _routed_memory(write_holds=(1, 1), read_bins=read_bins, fixed_cuts=read_bins == 4)
    spikes = torch.ones(4, 520, 3)
    spikes[:, 0, 2] = -1
    write = torch.zeros(4, 520, dtype=torch.bool)
    write[:, :2] = True
    write[0, 350] = True
    probe = torch.ones_like(write)
    probe[:, 140] = False
    clear = torch.zeros_like(write)
    clear[2, 245] = True
    trace = {}
    factual = memory(spikes, probe=probe, write=write, clear=clear, trace=trace)
    positions = torch.tensor([1, 1, 1, -1])
    cells = torch.tensor([0, 0, 0, -1])

    read, eligible = memory.counterfactual_read(trace, positions, cells)
    assert all(not item.requires_grad for item in trace.values())
    assert torch.equal(trace["reads"], factual)
    assert torch.all(trace["pre_write_phase"][:3, 1, 0] < trace["write_charge"][:3, 1, 0])
    assert trace["latest_accepted_write"][0, 350, 0] == 1
    assert trace["latest_accepted_write"][0, 351, 0] == 350
    assert trace["latest_accepted_write"][2, 245, 0] == 1
    assert not trace["live_phase"][2, 245, 0]
    expected = torch.zeros_like(eligible)
    expected[0, 129:351] = True
    expected[1, 129:514] = True
    expected[2, 129:245] = True
    expected[:, 140] = False
    assert torch.equal(eligible, expected)
    assert torch.equal((read != factual).any(dim=-1), expected)
    assert torch.equal(read[..., memory.read_width // 2:], factual[..., memory.read_width // 2:])

    state, readings = None, []
    for t in range(spikes.shape[1]):
        current, state = memory.step(spikes[:, t], state, probe=probe[:, t], write=write[:, t], clear=clear[:, t])
        if t == 1:
            state.phase[:3, 0] = trace["pre_write_phase"][:3, t, 0]
        readings.append(current)
    streamed = torch.stack(readings, dim=1)
    assert torch.equal(read, torch.where(eligible.unsqueeze(-1), streamed, factual))


def test_counterfactual_trace_tracks_destructive_erasure():
    memory = _routed_memory(carry_reads=False)
    spikes = torch.ones(1, 5, 3)
    write = torch.tensor([[True, False, False, False, False]])
    probe = torch.tensor([[False, False, True, True, True]])
    trace = {}
    factual = memory(spikes, probe=probe, write=write, trace=trace)
    read, eligible = memory.counterfactual_read(trace, torch.tensor([0]), torch.tensor([0]), min_age=1, max_age=5)
    assert eligible.tolist() == [[False, False, True, False, False]]
    assert trace["pre_write_phase"][0, 2, 0] == 0
    assert trace["latest_accepted_write"][0, -1, 0] == 0
    assert not trace["live_phase"][0, -1, 0]
    assert factual[0, 2, 0] == 1 and read[0, 2, 0] == 0
    with pytest.raises(ValueError, match="accepted write"):
        memory.counterfactual_read(trace, torch.tensor([1]), torch.tensor([0]))


@pytest.mark.parametrize("device", _DEVICES)
def test_global_timing_cuts_use_trainable_query_independent_biases(device):
    memory = PulseMemory(4, 2, read_bins=4, global_cuts=True).to(device)
    spikes = torch.tensor([[[0., 1., -2., 3.], [3., -2., 1., 0.]]], device=device)
    _, cuts, _, _ = memory._events(spikes, None, None)
    assert torch.equal(cuts, torch.tensor([64., 128., 192.], device=device).expand_as(cuts))
    cuts.sum().backward()
    assert torch.all(memory.bias.grad[[1, 4, 5]] > 0)
    assert memory.project.weight.grad is None
