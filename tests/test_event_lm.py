from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from snnformer.event_lm import (
    EventLM,
    EventLMConfig,
    _DelaySelector,
    _IntegrateFire,
    _balanced_steer,
    _phase_swap,
    _route_packets,
    _steer_packet,
    _steering_start,
)


_SELECTED_DELAYS = ("steered_static", "steered_feedback", "steered_margin")
_STEERED_VARIANTS = ("steered", "steered_local", "convergent", "steered_balanced", "steered_timed", "steered_phase_swap",
                     "steered_phase_delay", *_SELECTED_DELAYS)
_ROUTES = ("coincidence", "tonic", *_STEERED_VARIANTS)


@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
@pytest.mark.parametrize("route", ["coincidence", "tonic"])
def test_routing_partitions_every_slot_in_each_signed_packet(dtype, route):
    counts = torch.arange(-32, 33, dtype=dtype)
    present, delayed = torch.meshgrid(counts, counts, indexing="ij")
    slots = torch.arange(32)
    present_slots = present[..., None].abs() > slots
    delayed_slots = delayed[..., None].abs() > slots
    if route == "coincidence":
        shared_slots = present_slots & delayed_slots & (present.sign() == delayed.sign())[..., None]
    else:
        shared_slots = delayed_slots & (slots % 2 == 0)
    expected = shared_slots.sum(-1) * delayed.sign()
    paired, unpaired = _route_packets(present, delayed, route)
    assert torch.equal(paired, expected)
    assert torch.equal(paired + unpaired, delayed)
    assert torch.equal(paired.abs() + unpaired.abs(), delayed.abs())
    assert torch.any((paired.abs() > 1) & (unpaired.abs() > 1))


@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
def test_steering_spikes_partition_positive_and_negative_packets(dtype):
    packet, steering = torch.meshgrid(torch.arange(-32, 33, dtype=dtype),
                                      torch.arange(33, dtype=dtype), indexing="ij")
    slots = torch.arange(32)
    opened = (packet[..., None].abs() > slots) & (steering[..., None] > slots)
    selected, rest = _steer_packet(packet, steering)
    assert torch.equal(selected, opened.sum(-1) * packet.sign())
    assert torch.equal(selected + rest, packet)
    assert torch.equal(selected.abs() + rest.abs(), packet.abs())


@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
def test_timing_intersects_bursts_and_preserves_all_content(dtype):
    packet, steering, start = torch.meshgrid(torch.arange(-32, 33, dtype=dtype), torch.arange(33, dtype=dtype),
                                            torch.arange(-2, 2, dtype=dtype), indexing="ij")
    slots = torch.arange(32)
    matched = ((slots < packet[..., None].abs()) & (slots >= start[..., None])
               & (slots < (start + steering)[..., None]))
    selected, rest = _steer_packet(packet, steering, start)
    assert torch.equal(selected, matched.sum(-1) * packet.sign())
    assert torch.equal(selected + rest, packet)
    assert torch.equal(selected.abs() + rest.abs(), packet.abs())


def test_subthreshold_timing_varies_without_changing_the_gate_count():
    drive = torch.arange(32, 96, dtype=torch.int32)[None, :, None]
    count = _IntegrateFire(cap=32, charge_bits=6, carry=False)(drive)
    assert torch.equal(count, torch.ones_like(count))
    assert _steering_start(drive)[0, :, 0].tolist() == [-2] * 16 + [-1] * 16 + [0] * 16 + [1] * 16


def test_phase_swap_preserves_aligned_groups():
    packet = torch.tensor([0, 8, -4, -9, 12, 3])
    steering = torch.tensor([1, 3, 1, 5, 7, 2])
    matched, rest = _steer_packet(packet, steering)
    for state, expected in ((-1, matched), (0, rest)):
        selected, other = _phase_swap(packet, steering, torch.full_like(steering, state))
        assert torch.equal(selected, expected)
        assert torch.equal(selected + other, packet)


@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
def test_signed_steering_transfers_pulses_in_both_directions(dtype):
    packet = torch.tensor([6, -6, 6, -6, 6, -6, 0], dtype=dtype)
    steering = torch.tensor([0, 0, 2, 2, -9, -9, 32], dtype=dtype)
    selected, rest = _balanced_steer(packet, steering)
    assert selected.tolist() == [3, -3, 5, -5, 0, 0, 0]
    assert torch.equal(selected.abs() + rest.abs(), packet.abs())
    assert torch.equal(selected + rest, packet)


@pytest.mark.parametrize("margin", [0, 8])
def test_feedback_selects_using_only_prior_packet_errors(margin):
    selector = _DelaySelector((1, 2), (2, 2), 4, feedback=True, margin=margin).eval()
    pattern = torch.tensor([[8, -4, 7, 8], [-8, 4, -7, -8]], dtype=torch.int32)
    packets = pattern.repeat(10, 1)
    bound = selector.delays[-1]
    priors = (selector.prior.detach() * 64).round().to(torch.int32).tolist()
    expected = []
    for t in range(len(packets)):
        available = [d for d in selector.delays if d <= max(t, 1)]
        anchor = selector.prior[:, :len(available)].argmax(-1).tolist()
        chosen = []
        for channel in range(packets.shape[1]):
            def at(time):
                return int(packets[time, channel]) if time >= 0 else 0

            prior = priors[channel]
            credit = [sum(abs(at(i)) - abs(at(i) - at(i - d)) for i in range(t - 8, t))
                      for d in available]
            if margin:
                credit = [max(value - credit[anchor[channel]] - margin, 0) for value in credit]
            best = max(range(len(available)), key=lambda i: prior[i] + credit[i])
            chosen.append(at(t - available[best]))
        expected.append(chosen)
    observed = selector(packets.unsqueeze(0), delay_limit=bound)
    torch.testing.assert_close(observed[0], torch.tensor(expected, dtype=packets.dtype), atol=0, rtol=0)
    assert torch.equal(observed[0, -1], packets[-3])


@pytest.mark.parametrize("carry", [False, True])
def test_subtractive_reset_conserves_charge_with_multiple_signed_spikes(carry):
    neuron = _IntegrateFire(cap=3, charge_bits=6, carry=carry)
    current = torch.tensor([240, 94, 31, 1, -250, -158, -3, 0, 64], dtype=torch.int32)[None, :, None]
    expected = []
    carried_phase = torch.full((1, 1, 2), 32, dtype=torch.int32)
    state = None
    for drive in current.unbind(1):
        drive_magnitude = drive.abs().clamp_max(3 * 64)
        incoming = torch.stack((torch.where(drive > 0, drive_magnitude, 0),
                                torch.where(drive < 0, drive_magnitude, 0)), dim=-1)
        charged = incoming + (carried_phase if carry else 32)
        emitted = torch.div(charged, 64, rounding_mode="floor")
        carried_phase = charged - emitted * 64
        signed = emitted[..., 0] - emitted[..., 1]
        counts, state = neuron.step(drive, state)
        torch.testing.assert_close(counts, signed, rtol=0, atol=0)
        if carry:
            torch.testing.assert_close(state, carried_phase, rtol=0, atol=0)
            assert ((state >= 0) & (state < 64)).all()
        else:
            assert state is None
        expected.append(signed)
    expected = torch.stack(expected, dim=1)
    assert expected.max().item() == 3 and expected.min().item() == -3
    torch.testing.assert_close(neuron(current), expected, rtol=0, atol=0)


def _seeded_fixture(route="coincidence", carry=True):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(214)
        model = EventLM(EventLMConfig(dim=8, hidden=13, depth=3, route=route, carry=carry))
    tokens = torch.randint(256, (2, 139), generator=torch.Generator().manual_seed(510))
    return model, tokens


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("carry", [False, True])
def test_trained_fake_integer_export_matches_integer_batch_and_stream(route, carry):
    model, tokens = _seeded_fixture(route, carry)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = F.cross_entropy(model(tokens[:, :30]).flatten(0, 1) / 8, tokens[:, 1:31].flatten())
    loss.backward()
    assert all(parameter.grad is not None and parameter.grad.isfinite().all() and parameter.grad.abs().sum() > 0
               for parameter in model.parameters())
    optimizer.step()

    model.eval()
    integer = model.to_integer()
    assert all(weight.is_floating_point() for weight in model.parameters())
    assert not integer.training
    assert all(weight.dtype == torch.int32 and not weight.requires_grad for weight in integer.parameters())
    with torch.no_grad():
        fake = model(tokens)
        batched = integer(tokens)
    assert batched.dtype == torch.int32
    assert fake.is_floating_point()
    torch.testing.assert_close(fake, batched.to(fake.dtype), rtol=0, atol=0)
    assert fake.min() < -1 and fake.max() > 1
    assert fake.abs().max() <= 128
    torch.testing.assert_close(model.byte(fake), integer.byte(batched), rtol=0, atol=0)

    state, streamed = None, []
    history_storage = None
    if route in _SELECTED_DELAYS:
        delays = (24, 72, 136)
    else:
        delays = (3, 8, 27) if route in _STEERED_VARIANTS else (8, 27, 64)
    for i, frame in enumerate(tokens.unbind(1), start=1):
        counts, state = integer.step(frame, state)
        assert state.position == i
        assert counts.dtype == torch.int32
        if carry:
            assert state.input_phase.shape == (2, 8, 2)
            assert state.output_phase.shape == (2, 256, 2)
            assert state.input_phase.dtype == state.output_phase.dtype == torch.int32
        else:
            assert state.input_phase is state.output_phase is None
        addresses = []
        if route == "convergent":
            assert state.previous_bytes.shape == (2, 2)
            assert state.previous_bytes.dtype == tokens.dtype
            assert torch.equal(state.previous_bytes[:, (i - 1) % 2], frame)
            addresses.append(state.previous_bytes.data_ptr())
        else:
            assert state.previous_bytes is None
        for layer, delay in zip(state.layers, delays, strict=True):
            assert layer.history.shape == (2, delay, 13)
            assert layer.history.dtype == torch.int32
            addresses.append(layer.history.data_ptr())
            if carry:
                assert layer.relay_phase.shape == (2, 13, 2)
                assert layer.output_phase.shape == (2, 8, 2)
                assert layer.relay_phase.dtype == layer.output_phase.dtype == torch.int32
            else:
                assert layer.relay_phase is layer.output_phase is None
            if carry and route in _STEERED_VARIANTS:
                assert layer.steering_phase.shape == (2, 13, 2)
                assert layer.steering_phase.dtype == torch.int32
            else:
                assert layer.steering_phase is None
        if history_storage is None:
            history_storage = addresses
        assert addresses == history_storage
        streamed.append(counts)
    torch.testing.assert_close(batched, torch.stack(streamed, dim=1), rtol=0, atol=0)


@pytest.mark.parametrize("route", _ROUTES)
def test_prefix_is_independent_of_future_bytes_and_other_batch_rows(route):
    model, tokens = _seeded_fixture(route)
    prefix_length = 45
    changed = (tokens + 1) % 256
    changed[0, :prefix_length] = tokens[0, :prefix_length]
    with torch.no_grad():
        for decoder in (model.eval(), model.to_integer()):
            original = decoder(tokens)
            altered = decoder(changed)
            for length in (1, 2, prefix_length):
                shortened = decoder(tokens[:1, :length])
                torch.testing.assert_close(shortened[0], original[0, :length], rtol=0, atol=0)
            torch.testing.assert_close(altered[0, :prefix_length], original[0, :prefix_length], rtol=0, atol=0)
            assert not torch.equal(altered[0, prefix_length:], original[0, prefix_length:])


@pytest.mark.parametrize(("settings", "message"), [
    ({"dim": 0}, "decoder dimensions"),
    ({"hidden": 3}, "decoder dimensions"),
    ({"depth": 0}, "decoder dimensions"),
    ({"route": "unknown"}, "invalid spike packet route"),
    ({"hidden": 1376}, "exact FP32 integer range"),
    ({"dim": 4128}, "exact FP32 integer range"),
    ({"route": "steered", "hidden": 1032}, "exact FP32 integer range"),
])
def test_invalid_decoder_configuration(settings, message):
    with pytest.raises(ValueError, match=message):
        EventLM(replace(EventLMConfig(), **settings))


@torch.no_grad()
def test_fine_spike_resolution_checks_the_accumulator_with_current_weights():
    model = EventLM(EventLMConfig(hidden=198, route="steered", carry=False, resolution=32)).eval()
    tokens = torch.arange(48).unsqueeze(0)
    counts = model.to_integer()(tokens)
    torch.testing.assert_close(model(tokens), counts.float(), rtol=0, atol=0)

    model.blocks[-1].project.weight.fill_(2)
    with pytest.raises(ValueError, match="exact FP32 accumulator range"):
        model(tokens)
    assert model.to_integer()(tokens).dtype == torch.int32


def test_minimum_dimensions_and_stream_contract():
    model = EventLM(EventLMConfig(dim=1, hidden=4, depth=1)).eval()
    tokens = torch.tensor([[0, 255]])
    with pytest.raises(ValueError, match="nonempty"):
        model(tokens[:, :0])
    with pytest.raises(ValueError, match="integer export"):
        model.step(tokens[:, 0])
    integer = model.to_integer()
    assert integer(tokens).shape == (1, 2, 256)
    with pytest.raises(ValueError, match="shape"):
        integer.step(tokens)
    _, state = integer.step(tokens[:, 0])
    with pytest.raises(ValueError, match="batch"):
        integer.step(torch.tensor([0, 0]), state)
    with pytest.raises(ValueError, match="charge scan"):
        _IntegrateFire(cap=1, charge_bits=30, carry=True)(torch.zeros(1, 2, 1))
