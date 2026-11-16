from dataclasses import replace

import pytest
import torch

from snnformer.correction_memory import _interval_scan, _write_credit_scan, correction_reads, correction_step
from snnformer.event_lm import (
    EventLM, EventLMConfig, _DelayCorrection, _DelayFilter, _DelayState, _OpponentSteering,
    _ResidualMembraneFire, _SpikeCorrection, _SpikeTrace, _SpreadCoincidence, _steer_packet,
)


def test_correction_moves_only_addressed_cells_to_nearest_consistent_value():
    address = torch.tensor([[[[1, 0]], [[0, 1]], [[1, 0]], [[1, 0]], [[0, 1]], [[0, 0]]]], dtype=torch.int32)
    target = torch.tensor([[[[5]], [[-6]], [[3]], [[-4]], [[-4]], [[7]]]], dtype=torch.int32)
    tolerance = torch.tensor([[[[2]], [[2]], [[1]], [[1]], [[2]], [[0]]]], dtype=torch.int32)
    read, state = correction_reads(address, target, tolerance, 8)
    assert read.flatten().tolist() == [0, 0, 3, 3, -4, 0]
    assert state.charge.flatten().tolist() == [-3, -4]

    sequential_reads = []
    streaming = None
    for a, t, r in zip(address.unbind(1), target.unbind(1), tolerance.unbind(1), strict=True):
        pulse, streaming = correction_step(a, t, r, 8, streaming)
        sequential_reads.append(pulse)
    torch.testing.assert_close(torch.stack(sequential_reads, 1), read)
    torch.testing.assert_close(streaming.charge, state.charge)


def test_read_ablation_uses_age_of_effective_change_and_keeps_state_intact():
    address = torch.ones((1, 5, 1, 1), dtype=torch.int32)
    target = torch.tensor([4, 4, 4, -4, -4], dtype=torch.int32).reshape(1, 5, 1, 1)
    tolerance = torch.zeros_like(target)
    complete, original = correction_reads(address, target, tolerance, 8, stats={})
    erased, masked = correction_reads(address, target, tolerance, 8, hide_age=2)
    assert complete.flatten().tolist() == [0, 4, 4, 4, -4]
    assert erased.flatten().tolist() == [0, 4, 0, 0, -4]
    torch.testing.assert_close(original.charge, masked.charge)
    torch.testing.assert_close(original.last_change, masked.last_change)


@pytest.mark.parametrize(("route", "key_channels", "match_key"), [
    ("steered_correction", 0, False),
    ("steered_correction_residual", 0, False),
    ("steered_correction_binding", 0, False),
    ("steered_correction_observed", 0, False),
    ("steered_correction_error", 0, False),
    ("steered_correction_code", 0, False),
    ("steered_correction_code", 1, False),
    ("steered_correction_code", 1, True),
])
def test_correction_decoder_chunk_stream_and_batch_match_with_finer_spikes(route, key_channels, match_key):
    torch.manual_seed(59)
    model = EventLM(EventLMConfig(dim=16, hidden=24, route=route, carry=False,
                                 memory_heads=2, memory_slots=4, resolution=4,
                                 forecast_key_channels=key_channels, match_forecast_key=match_key)).eval()
    tokens = torch.randint(0, 256, (2, 71), generator=torch.Generator().manual_seed(29))
    expected = model(tokens)
    integer = model.to_integer()
    torch.testing.assert_close(integer(tokens), expected.to(torch.int32), rtol=0, atol=0)

    batched_state = None
    chunks = []
    for chunk in (tokens[:, :9], tokens[:, 9:26], tokens[:, 26:30], tokens[:, 30:]):
        counts, batched_state = integer.forward_chunk(chunk, batched_state)
        chunks.append(counts)
    streaming_state = None
    stream = []
    for byte in tokens.unbind(1):
        counts, streaming_state = integer.step(byte, streaming_state)
        stream.append(counts)
    torch.testing.assert_close(torch.cat(chunks, dim=1), torch.stack(stream, dim=1), rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(chunks, dim=1), expected.to(torch.int32), rtol=0, atol=0)


def test_previous_address_receives_the_observed_continuation_after_the_read():
    address = torch.tensor([[[[1, 0]], [[0, 1]], [[1, 0]], [[0, 1]]]], dtype=torch.int32)
    target = torch.tensor([2, -3, 5, 6], dtype=torch.int32).reshape(1, 4, 1, 1)
    read, state = correction_reads(address, target, torch.zeros_like(target), 8, bind_previous=True)
    assert read.flatten().tolist() == [0, 0, -3, 5]
    assert state.charge.flatten().tolist() == [6, 5]


@torch.no_grad()
def test_prediction_error_writes_to_the_old_address_and_uses_the_old_guess():
    memory = _SpikeCorrection(2, 2, 1, 2, 4, bind_previous=True, observed=True).eval()
    memory.encode.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [0., 0.]]))
    memory.code.weight.copy_(torch.eye(2))
    memory.encode.freeze_integer()
    memory.code.freeze_integer()

    context = torch.tensor([[[1, 0], [0, 1], [1, 0], [0, 1]]], dtype=torch.int32)
    observed = torch.tensor([[[9, 0], [6, 1], [2, 5], [7, 2]]], dtype=torch.int32)
    guesses = torch.tensor([[[2, 3], [1, 3], [7, 2], [4, 4]]], dtype=torch.int32)
    expected = torch.tensor([[[0, 0], [0, 0], [4, -2], [1, 2]]], dtype=torch.int32)
    prediction = memory._value_code(guesses)
    actual = memory(context, None, observed=observed, prediction_code=prediction)
    torch.testing.assert_close(actual, expected)
    state = None
    for step in range(context.shape[1]):
        read, state = memory.step(context[:, step], state, observed=observed[:, step],
                                  prediction_code=prediction[:, step])
        torch.testing.assert_close(read, expected[:, step])
    assert state.charge[0, 0].tolist() == [[0, 0], [1, 2]]


@torch.no_grad()
@pytest.mark.parametrize("keyed", (False, True))
def test_multiple_read_masks_share_the_same_chunked_prediction_state(keyed):
    torch.manual_seed(31)
    route = "steered_correction_code" if keyed else "steered_correction_error"
    decoder = EventLM(EventLMConfig(dim=8, hidden=16, route=route, carry=False,
                                    memory_heads=2, memory_slots=4, resolution=4,
                                    forecast_key_channels=int(keyed), match_forecast_key=keyed)).to_integer()
    stream = torch.randint(0, 256, (2, 63), generator=torch.Generator().manual_seed(55))
    warmup, tokens = stream.split((24, 39), dim=1)
    cases = (None, 8, 1)
    _, before = decoder.forward_chunk(warmup, hide_age=cases)
    simultaneous, after = decoder.forward_chunk(tokens, before, hide_age=cases)
    for age, counts in zip(cases, simultaneous, strict=True):
        separate, state = decoder.forward_chunk(tokens, before, hide_age=age)
        torch.testing.assert_close(counts, separate)
        torch.testing.assert_close(after.correction.charge, state.correction.charge)


@torch.no_grad()
def test_spike_code_prediction_uses_one_vocabulary_pass_and_integer_codes():
    model = EventLM(EventLMConfig(dim=8, hidden=16, route="steered_correction_code", carry=False,
                                 memory_heads=2, memory_slots=4, memory_width=4, resolution=4)).to_integer()
    tokens = torch.tensor([[8, 7, 4, 2]])
    calls = []
    handle = model.output.register_forward_hook(lambda *_: calls.append(True))
    counts, prediction, observed = model(tokens, return_prediction=True)
    handle.remove()
    expected = model.input_emit(model.input(tokens) * model.cfg.resolution)[..., :4]
    torch.testing.assert_close(observed.flatten(-2), expected)
    assert calls == [True]
    assert counts.dtype == prediction.dtype == torch.int32
    assert prediction.abs().max() <= 2 * model.cfg.resolution


@torch.no_grad()
def test_corrected_forecast_adds_only_the_delivered_error_spikes():
    torch.manual_seed(244)
    model = EventLM(EventLMConfig(dim=8, hidden=16, route="steered_correction_code", carry=False,
                                 memory_heads=2, memory_slots=4, memory_width=8, resolution=4,
                                 forecast_key_channels=2, match_forecast_key=True)).to_integer()
    tokens = torch.randint(256, (1, 256))
    counts, prediction, observed = model(tokens, return_prediction=True)
    delivered = []
    handle = model.prediction_memory.register_forward_hook(lambda _m, _i, out: delivered.append(out))
    corrected_counts, corrected, corrected_observed = model(
        tokens, return_prediction=True, corrected_prediction=True,
    )
    handle.remove()
    assert bool(delivered[0].any())
    torch.testing.assert_close(counts, corrected_counts)
    torch.testing.assert_close(observed, corrected_observed)
    torch.testing.assert_close(corrected[..., :2], prediction[..., :2])
    torch.testing.assert_close(corrected[..., 2:], prediction[..., 2:] + delivered[0].unflatten(-1, (2, 2)))


def test_forecast_mismatch_cancels_coincident_pulses_on_both_sign_rails():
    bank = _SpikeCorrection(8, 6, 1, 2, 4, bind_previous=True, observed=True, forecast=True,
                            key_channels=2, match_key=True)
    read = torch.tensor([[[[10, -3, 5, -5, 1, -1]]]], dtype=torch.int32).expand(1, 3, 1, 6)
    query = torch.tensor([[[[10, -3]], [[14, 1]], [[18, -6]]]], dtype=torch.int32)
    tolerance = torch.full_like(query[..., :1], 2)
    address = torch.ones_like(query[..., :1])
    stats = {}
    payload = bank._read_payload(read, query, tolerance, address, stats)
    assert payload[0, :, 0].tolist() == [[5, -5, 1, -1], [3, -3, 0, 0], [0, 0, 0, 0]]
    assert stats["veto_spikes"].item() == 8
    assert stats["muted_payload_spikes"].item() == 18


def test_rotating_keys_and_hiding_old_payload_preserve_previous_context_writes():
    address = torch.tensor([[[[1, 0]], [[0, 1]], [[1, 0]], [[0, 1]]]], dtype=torch.int32)
    target = torch.tensor([[[[0, 0]], [[4, 5]], [[-6, 7]], [[4, 5]]]], dtype=torch.int32)
    options = dict(bind_previous=True, key_channels=1, tolerance=torch.zeros_like(target), limit=8)
    normal, clean = correction_reads(address, target, **options)
    shuffled, rotated = correction_reads(address, target, shuffle_keys=True, **options)
    hidden, masked = correction_reads(address, target, hide_age=1, **options)
    assert normal[0, :, 0].tolist() == [[0, 0], [0, 0], [4, 5], [-6, 7]]
    assert shuffled[0, :, 0].tolist() == [[0, 0], [0, 0], [0, 5], [4, 7]]
    assert hidden[0, :, 0].tolist() == [[0, 0], [0, 0], [4, 0], [-6, 0]]
    torch.testing.assert_close(rotated.charge, clean.charge, rtol=0, atol=0)
    torch.testing.assert_close(masked.charge, clean.charge, rtol=0, atol=0)


def test_shared_thresholds_preserve_neutral_outputs_and_export_unequal_groups():
    torch.manual_seed(37)
    cfg = EventLMConfig(dim=16, hidden=20, route="steered_correction_code", carry=False,
                        memory_heads=2, memory_slots=4, memory_width=8, forecast_key_channels=2,
                        learn_context_keys=True, match_forecast_key=True, resolution=16)
    uniform = EventLM(cfg).eval()
    learned = EventLM(replace(cfg, learn_packet_thresholds=True)).eval()
    missing, unexpected = learned.load_state_dict(uniform.state_dict(), strict=False)
    assert missing == [f"blocks.{index}.packet_thresholds.relative" for index in range(cfg.depth)]
    assert unexpected == []
    tokens = torch.randint(256, (2, 45))
    with torch.no_grad():
        torch.testing.assert_close(learned(tokens), uniform(tokens), rtol=0, atol=0)
        factors = torch.tensor([0.5, 0.75, 1.25, 2.0])
        for block in learned.blocks:
            block.packet_thresholds.relative.copy_(factors)
        expected = learned(tokens)

    integer = learned.to_integer()
    pulse_counts = []
    counts = integer(tokens, memory_pulses=pulse_counts)
    torch.testing.assert_close(counts, expected.to(torch.int32), rtol=0, atol=0)
    assert not torch.equal(counts, uniform.to_integer()(tokens))
    assert len(pulse_counts) == 3 and all(value.dtype == torch.int64 for value in pulse_counts)
    state = None
    streamed = []
    for token in tokens.unbind(1):
        output, state = integer.step(token, state)
        streamed.append(output)
    torch.testing.assert_close(torch.stack(streamed, dim=1), counts, rtol=0, atol=0)
    first, state = integer.forward_chunk(tokens[:, :19])
    last, _ = integer.forward_chunk(tokens[:, 19:], state)
    torch.testing.assert_close(torch.cat((first, last), dim=1), counts, rtol=0, atol=0)


def test_shared_output_reference_preserves_every_vocabulary_margin():
    torch.manual_seed(81)
    cfg = EventLMConfig(dim=16, hidden=20, route="steered_correction_code", carry=False,
                        memory_heads=2, memory_slots=4, memory_width=8, resolution=26)
    plain = EventLM(cfg).eval()
    referenced = EventLM(replace(cfg, output_reference=True)).eval()
    referenced.load_state_dict(plain.state_dict())
    tokens = torch.randint(256, (2, 41))
    native = referenced.to_integer()
    with torch.no_grad():
        before = plain.to_integer()(tokens)
        after = native(tokens)
        difference = before - after
        assert torch.equal(difference, difference[..., :1].expand_as(difference))
        assert difference.abs().amax().item() > 0
        torch.testing.assert_close(referenced(tokens), after.to(torch.float32), rtol=0, atol=0)
    state = None
    streamed = []
    for token in tokens.unbind(1):
        output, state = native.step(token, state)
        streamed.append(output)
    torch.testing.assert_close(torch.stack(streamed, dim=1), after, rtol=0, atol=0)


@torch.no_grad()
@pytest.mark.parametrize("marked", (False, True))
def test_packed_delay_rings_keep_only_live_packets(marked):
    torch.manual_seed(52)
    cfg = EventLMConfig(dim=16, hidden=21, route="steered_correction_code", carry=False,
                        memory_heads=2, memory_slots=4, memory_width=8, resolution=16, output_reference=True)
    saved = EventLM(cfg).eval()
    compact = EventLM(replace(cfg, compact_delays=True, departure_coincidence=marked)).eval()
    compact.load_state_dict(saved.state_dict())
    tokens = torch.randint(256, (2, 113))
    native = compact.to_integer()
    expected = native(tokens)
    torch.testing.assert_close(compact(tokens), expected.to(torch.float32), rtol=0, atol=0)
    if marked:
        assert not torch.equal(expected, saved.to_integer()(tokens))
    else:
        torch.testing.assert_close(expected, saved.to_integer()(tokens), rtol=0, atol=0)
    state = None
    for offset, token in enumerate(tokens.unbind(1)):
        output, state = native.step(token, state)
        torch.testing.assert_close(output, expected[:, offset], rtol=0, atol=0)
    for block, layer in zip(compact.blocks, state.layers, strict=True):
        counts = sum(width * delay for width, delay in zip(block.widths, block.delays, strict=True))
        assert layer.history.shape == (2, counts * (1 + marked))
        assert layer.history.untyped_storage().nbytes() == 2 * counts * (1 + marked) * 4


def test_delay_choice_uses_signed_coincidence_and_skips_unavailable_taps():
    cfg = EventLMConfig(dim=8, hidden=16, route="steered", carry=False, compact_delays=True, delay_window=4)
    block = EventLM(cfg).to_integer().blocks[2]
    history = torch.tensor([[[50, -2, 3], [-1, -3, -4]]], dtype=torch.int32)
    query = torch.tensor([[2, -3]], dtype=torch.int32)
    aperture = query.abs()
    chosen = block._match_delay(history, query, aperture, torch.tensor([True, True, True]))
    torch.testing.assert_close(chosen, history[..., 2], rtol=0, atol=0)
    chosen = block._match_delay(history, query, aperture, torch.tensor([True, True, False]))
    torch.testing.assert_close(chosen, history[..., 0], rtol=0, atol=0)
    chosen = block._match_delay(history, torch.zeros_like(query), torch.zeros_like(aperture),
                                torch.tensor([False, True, True]))
    torch.testing.assert_close(chosen, history[..., 1], rtol=0, atol=0)


@torch.no_grad()
def test_delay_matching_preserves_native_batch_chunk_and_stream_parity():
    torch.manual_seed(91)
    cfg = EventLMConfig(dim=16, hidden=21, route="steered_correction_code", carry=False,
                        memory_heads=2, memory_slots=4, memory_width=8, resolution=16, output_reference=True,
                        compact_delays=True, delay_window=4)
    model = EventLM(cfg).eval()
    tokens = torch.randint(256, (2, 99))
    native = model.to_integer()
    expected = native(tokens)
    torch.testing.assert_close(model(tokens), expected.to(torch.float32), rtol=0, atol=0)
    state = None
    for i, token in enumerate(tokens.unbind(1)):
        output, state = native.step(token, state)
        torch.testing.assert_close(output, expected[:, i], rtol=0, atol=0)
    chunk_state = None
    outputs = []
    for chunk in tokens.split((1, 13, 2, 20, 63), dim=1):
        counts, chunk_state = native.forward_chunk(chunk, chunk_state)
        outputs.append(counts)
    torch.testing.assert_close(torch.cat(outputs, dim=1), expected, rtol=0, atol=0)
    for block, layer in zip(native.blocks, state.layers, strict=True):
        assert layer.history.shape[-1] == sum(width * delay for width, delay in zip(block.widths, block.delays))


@torch.no_grad()
def test_finer_charge_keeps_firing_when_weights_lie_on_the_original_grid():
    torch.manual_seed(89)
    cfg = EventLMConfig(dim=16, hidden=21, route="steered_correction_code", carry=False, resolution=16,
                        memory_heads=2, memory_slots=4, memory_width=8, compact_delays=True, delay_window=4)
    original = EventLM(cfg).eval()
    for block in original.blocks:
        for synapse in (block.expand, block.project):
            synapse.weight.copy_(synapse.weight_ticks() / block.relay.threshold)
    fine = EventLM(replace(cfg, router_charge_bits=8)).eval()
    fine.load_state_dict(original.state_dict())
    tokens = torch.randint(256, (2, 41))
    old_native = original.to_integer()
    fine_native = fine.to_integer()
    torch.testing.assert_close(old_native(tokens), fine_native(tokens), rtol=0, atol=0)
    torch.testing.assert_close(fine_native.blocks[0].project.weight, old_native.blocks[0].project.weight * 4)

    fine.blocks[0].project.weight.add_(1 / 256)
    fine_native = fine.to_integer()
    target = fine_native(tokens)
    assert not torch.equal(old_native(tokens), target)
    torch.testing.assert_close(fine(tokens), target.to(torch.float32), rtol=0, atol=0)
    state = None
    for position, token in enumerate(tokens.unbind(1)):
        counts, state = fine_native.step(token, state)
        torch.testing.assert_close(counts, target[:, position], rtol=0, atol=0)


@torch.no_grad()
def test_delay_correction_is_neutral_at_zero_and_matches_streaming_after_learning():
    torch.manual_seed(43)
    cfg = EventLMConfig(dim=16, hidden=21, route="steered_correction_code", carry=False, resolution=16,
                        memory_heads=2, memory_slots=4, memory_width=8, compact_delays=True)
    fixed = EventLM(cfg).eval()
    corrected = EventLM(replace(cfg, delay_window=4, delay_residual=True)).eval()
    missing, unexpected = corrected.load_state_dict(fixed.state_dict(), strict=False)
    assert missing == [f"blocks.{i}.delay_correction.gain" for i in range(cfg.depth)]
    assert not unexpected
    tokens = torch.randint(256, (2, 65))
    before = fixed.to_integer()(tokens)
    torch.testing.assert_close(corrected.to_integer()(tokens), before, rtol=0, atol=0)

    for block in corrected.blocks:
        block.delay_correction.gain.copy_(torch.tensor([0., .25, .5, 1.]))
    native = corrected.to_integer()
    after = native(tokens)
    assert not torch.equal(after, before)
    torch.testing.assert_close(corrected(tokens), after.to(torch.float32), rtol=0, atol=0)
    state = None
    for position, token in enumerate(tokens.unbind(1)):
        counts, state = native.step(token, state)
        torch.testing.assert_close(counts, after[:, position], rtol=0, atol=0)


def test_zero_delay_correction_gains_still_receive_emission_gradients():
    correction = _DelayCorrection((2, 1), cap=32)
    before = torch.tensor([[-3., 0., 2.]])
    after = torch.tensor([[-1., 3., 0.]])
    spikes = correction(before, after)
    torch.testing.assert_close(spikes, torch.zeros_like(spikes), rtol=0, atol=0)
    gradient, = torch.autograd.grad(spikes.sum(), correction.gain)
    torch.testing.assert_close(gradient, torch.tensor([5., -2.]), rtol=0, atol=0)


@torch.no_grad()
@pytest.mark.parametrize("bits", (6, 8))
def test_population_inhibition_raises_both_firing_thresholds_and_keeps_native_parity(bits):
    cfg = EventLMConfig(dim=16, hidden=21, route="steered_correction_code", carry=False, resolution=16,
                        memory_heads=2, memory_slots=4, memory_width=8, compact_delays=True,
                        activity_inhibition=True, router_charge_bits=bits)
    torch.manual_seed(17)
    model = EventLM(cfg).eval()
    integer = model.to_integer()
    block = integer.blocks[0]
    incoming = torch.tensor([0, 4, 16, -16], dtype=torch.int32)[:, None].expand(-1, cfg.dim)
    firing = block._firing_quanta(incoming, None)
    torch.testing.assert_close(firing[:, 0], torch.tensor([4, 5, 8, 8], dtype=torch.int32))
    current = torch.full((4, cfg.hidden), 8 * (1 << bits), dtype=torch.int32)
    licensed = torch.tensor([8, 6, 4, 4], dtype=torch.int32)
    torch.testing.assert_close(block.relay(-current, firing)[:, 0], -licensed)
    torch.testing.assert_close(block.steer(current, firing)[:, 0], licensed)

    tokens = torch.randint(256, (2, 60))
    expected = integer(tokens)
    torch.testing.assert_close(model(tokens), expected.to(torch.float32), rtol=0, atol=0)
    state = None
    for position, byte in enumerate(tokens.unbind(1)):
        counts, state = integer.step(byte, state)
        torch.testing.assert_close(counts, expected[:, position], rtol=0, atol=0)
    first, chunk = integer.forward_chunk(tokens[:, :23])
    second, _ = integer.forward_chunk(tokens[:, 23:], chunk)
    torch.testing.assert_close(torch.cat((first, second), dim=1), expected, rtol=0, atol=0)


def test_trace_steering_reads_old_charge_and_resists_writes_within_its_interval():
    trace = _SpikeTrace(cap=12)
    present = torch.tensor([5, 5, 5, 0, 4, -2], dtype=torch.int32).reshape(1, -1, 1)
    steering = torch.tensor([0, 9, 9, 9, 1, 2], dtype=torch.int32).reshape_as(present)
    pulses = []
    read, charge = trace(present, steering, None, pulses)
    assert read.flatten().tolist() == [0, 5, 5, 5, 1, 2]
    assert charge.item() == 0
    assert [int(count) for count in pulses] == [10, 18]


def test_trace_credit_follows_actual_moves_on_an_odd_length_scan_with_ties():
    lower = torch.tensor([[[2., -3.], [2., -5.], [-1., 4.], [3., 0.], [3., 2.], [0., 1.], [5., -1.]]],
                         requires_grad=True)
    upper = torch.tensor([[[2., -1.], [5., -3.], [1., 4.], [3., 4.], [6., 4.], [2., 4.], [8., 0.]]],
                         requires_grad=True)
    initial = torch.tensor([[0., 3.]], requires_grad=True)
    final = initial
    states = []
    for lo, hi in zip(lower.unbind(1), upper.unbind(1), strict=True):
        final = torch.where(final < lo, lo, torch.where(final > hi, hi, final))
        states.append(final)
    expected = torch.stack(states, dim=1)
    actual = _write_credit_scan(lower, upper, initial)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, _interval_scan(lower, upper, initial), rtol=0, atol=0)

    weights = torch.arange(actual.numel()).view_as(actual)
    reference_grads = torch.autograd.grad((expected * weights).sum(), (lower, upper, initial))
    for observed, desired in zip(torch.autograd.grad((actual * weights).sum(), (lower, upper, initial)),
                                 reference_grads, strict=True):
        torch.testing.assert_close(observed, desired, rtol=0, atol=0)
    assert actual[0, 1, 0].item() == lower[0, 1, 0].item()
    assert reference_grads[0][0, 1, 0].item() == 0


def test_opponent_steering_protects_charge_through_a_silent_read():
    trace = _SpikeTrace(cap=12)
    opponent = _OpponentSteering(width=1, cap=12, charge_bits=6)
    current = torch.tensor([0., -384., 512.]).reshape(1, 3, 1)
    protection = opponent(current, None)
    torch.testing.assert_close(protection, torch.zeros_like(protection), rtol=0, atol=0)
    gradient, = torch.autograd.grad(protection.sum(), opponent.weight)
    torch.testing.assert_close(gradient, torch.tensor([6.]), rtol=0, atol=0)

    with torch.no_grad():
        opponent.weight.fill_(1)
    opponent.freeze_integer()
    protection = opponent(current.to(torch.int32), None)
    assert protection.flatten().tolist() == [0, 6, 0]
    present = torch.tensor([5, 0, 5], dtype=torch.int32).reshape_as(protection)
    steering = torch.tensor([0, 0, 8], dtype=torch.int32).reshape_as(protection)
    original, _ = trace(present, steering, None, None)
    protected, state = trace(present, steering, None, None, write_radius=steering + protection)
    assert original.flatten().tolist() == [0, 0, 0]
    assert protected.flatten().tolist() == [0, 0, 5]
    assert state.item() == 5


@torch.no_grad()
def test_opponent_steering_warm_start_and_integer_chunk_and_step_parity():
    torch.manual_seed(71)
    cfg = EventLMConfig(dim=16, hidden=21, depth=4, route="steered_correction_code", carry=False,
                        resolution=16, memory_heads=2, memory_slots=4, memory_width=8, compact_delays=True,
                        activity_inhibition=True, local_trace=True)
    base_master = EventLM(cfg).eval()
    model = EventLM(replace(cfg, opponent_steering=True)).eval()
    missing, unexpected = model.load_state_dict(base_master.state_dict(), strict=False)
    assert not unexpected and set(missing) == {f"blocks.{i}.opponent_steering.weight" for i in range(cfg.depth)}
    tokens = torch.randint(256, (2, 67))
    base = base_master.to_integer()
    original = base(tokens)
    torch.testing.assert_close(model.to_integer()(tokens), original, rtol=0, atol=0)

    for block in model.blocks:
        block.opponent_steering.weight.fill_(0.85)
    native = model.to_integer()
    expected = native(tokens)
    assert not torch.equal(expected, original)
    torch.testing.assert_close(model(tokens), expected.to(torch.float32), rtol=0, atol=0)
    state = None
    for i, byte in enumerate(tokens.unbind(1)):
        count, state = native.step(byte, state)
        torch.testing.assert_close(count, expected[:, i], rtol=0, atol=0)
    chunk = None
    parts = []
    for part in tokens.split((1, 13, 42, 11), dim=1):
        count, chunk = native.forward_chunk(part, chunk)
        parts.append(count)
    torch.testing.assert_close(torch.cat(parts, dim=1), expected, rtol=0, atol=0)
    for streamed, batched in zip(state.layers, chunk.local_layers, strict=True):
        torch.testing.assert_close(streamed.trace_charge, batched.trace_charge, rtol=0, atol=0)


@torch.no_grad()
@pytest.mark.parametrize("route_options", (
    {}, {"trace_read_veto": True}, {"spread_coincidence": True},
    {"residual_membrane": True, "compact_axon_state": True},
))
@pytest.mark.parametrize("route", ("steered", "steered_correction_code"))
def test_trace_preserves_exact_history_across_steps_and_differently_sized_chunks(route, route_options):
    torch.manual_seed(97)
    cfg = EventLMConfig(dim=16, hidden=21, route=route, carry=False, resolution=16,
                        memory_heads=2, memory_slots=4, memory_width=8 if route.endswith("code") else None,
                        compact_delays=True, activity_inhibition=True, local_trace=True, **route_options)
    model = EventLM(cfg).eval()
    for block in model.blocks:
        if cfg.trace_read_veto:
            block.trace.read_veto.gain.fill_(0.5)
        if cfg.spread_coincidence:
            block.spread_coincidence.mix.fill_(0.625)
    native = model.to_integer()
    tokens = torch.randint(256, (2, 93))
    expected = native(tokens)
    torch.testing.assert_close(expected, model(tokens).to(torch.int32), rtol=0, atol=0)
    state = None
    for i, byte in enumerate(tokens.unbind(1)):
        counts, state = native.step(byte, state)
        torch.testing.assert_close(counts, expected[:, i], rtol=0, atol=0)
    chunk = None
    cursor = 0
    for part in tokens.split((1, 3, 13, 35, 41), dim=1):
        scores, chunk = native.forward_chunk(part, chunk)
        torch.testing.assert_close(scores, expected[:, cursor:cursor + part.shape[1]], rtol=0, atol=0)
        cursor += part.shape[1]
        assert chunk.local_prefix is None
    for streamed, batched in zip(state.layers, chunk.local_layers, strict=True):
        torch.testing.assert_close(streamed.trace_charge, batched.trace_charge, rtol=0, atol=0)
        if cfg.residual_membrane:
            assert streamed.output_phase.dtype == torch.uint8
            torch.testing.assert_close(streamed.output_phase, batched.output_phase, rtol=0, atol=0)
        if cfg.compact_axon_state:
            assert streamed.history.dtype == batched.history.dtype == torch.int16


def test_signed_residual_membrane_conserves_bounded_charge():
    neuron = _ResidualMembraneFire(cap=3, bits=2)
    current = torch.tensor([-2, 2, 1, 1, -3, -3, 20, -20], dtype=torch.int32).reshape(1, -1, 1)
    counts = neuron(current)
    assert counts.flatten().tolist() == [0, 0, 0, 1, -1, -1, 3, -3]

    phase = None
    charge = 0
    pulses = 0
    for i, value in enumerate(current.unbind(1)):
        emitted, phase = neuron.step(value, phase)
        torch.testing.assert_close(emitted, counts[:, i])
        charge += value.clamp(-12, 12).item()
        pulses += emitted.item()
        assert charge == 4 * pulses + phase.item() - 2
        assert 0 <= phase.item() < 4

    differentiable = current.float().requires_grad_()
    neuron(differentiable).sum().backward()
    torch.testing.assert_close(differentiable.grad, (current.abs() <= 12).float() / 4)


def test_compact_axon_storage_preserves_signed_maximum_packets():
    cfg = EventLMConfig(dim=16, hidden=8, route="steered", carry=False, resolution=16, compact_delays=True)
    dense = EventLM(cfg).blocks[0]
    compact = EventLM(replace(cfg, compact_axon_state=True)).blocks[0]
    full_state, compact_state = _DelayState(), _DelayState()
    packet = torch.tensor([[-512, 512] * 4], dtype=torch.int32)
    for position in range(15):
        steering = packet.abs()
        full, *_ = dense._step_compact(packet, steering, full_state, position)
        packed, *_ = compact._step_compact(packet, steering, compact_state, position)
        torch.testing.assert_close(packed, full)
        packet = -packet
    assert compact_state.history.untyped_storage().nbytes() * 2 == full_state.history.untyped_storage().nbytes()


def test_compact_axons_preserve_training_gradients_across_chunks():
    torch.manual_seed(177)
    cfg = EventLMConfig(dim=16, hidden=12, route="steered", carry=False, resolution=16,
                        compact_delays=True, local_trace=True)
    standard = EventLM(cfg)
    compact = EventLM(replace(cfg, compact_axon_state=True))
    compact.load_state_dict(standard.state_dict())
    stream = torch.randint(256, (1, 25))
    for model in (standard, compact):
        _, state = model.forward_chunk(stream[:, :11])
        assert state.local_layers[0].history.requires_grad
        outputs, _ = model.forward_chunk(stream[:, 11:], state)
        outputs.square().mean().backward()
    for reference, optimized in zip(standard.parameters(), compact.parameters(), strict=True):
        torch.testing.assert_close(reference.grad, optimized.grad, rtol=0, atol=0)


def test_tag_polarities_change_phase_without_adding_an_address_pulse():
    bank = _SpikeCorrection(8, 4, 1, 3, 16, bind_previous=True, observed=True, forecast=True,
                            key_channels=2, match_key=True, tag_phase_address=True)
    address = torch.tensor([[[0, 1, 0]]] * 4 + [[[0, 0, 0]]], dtype=torch.int32)
    tags = torch.tensor([[[5, 0, 0, 0]], [[-6, 0, 0, 0]], [[0, -6, 0, 0]],
                         [[-6, -6, 0, 0]], [[-6, -6, 0, 0]]], dtype=torch.int32)
    emitted = bank._phase_address(address, tags)
    assert emitted[:4].argmax(-1).flatten().tolist() == [4, 5, 6, 7]
    torch.testing.assert_close(emitted.sum(-1), address.sum(-1))

    prediction = tags.float().requires_grad_()
    phases = bank._phase_address(address.float(), prediction)
    (phases * torch.arange(12)).sum().backward()
    assert bool((prediction.grad[:4, :, :2] != 0).all())
    assert not bool(prediction.grad[:, :, 2:].any())
    assert not bool(prediction.grad[4].any())


@torch.no_grad()
def test_tag_phase_bank_matches_integer_steps_and_streaming_chunks():
    torch.manual_seed(13)
    cfg = EventLMConfig(dim=16, hidden=21, route="steered_correction_code", carry=False, resolution=16,
                        memory_heads=2, memory_slots=4, memory_width=8, forecast_key_channels=2,
                        match_forecast_key=True, learn_context_keys=True, tag_phase_address=True,
                        compact_delays=True, compact_axon_state=True, activity_inhibition=True, local_trace=True)
    model = EventLM(cfg).eval()
    native = model.to_integer()
    tokens = torch.randint(256, (2, 49))
    expected = native(tokens)
    torch.testing.assert_close(expected, model(tokens).to(torch.int32), rtol=0, atol=0)
    state = None
    for i, byte in enumerate(tokens.unbind(1)):
        emitted, state = native.step(byte, state)
        torch.testing.assert_close(emitted, expected[:, i], rtol=0, atol=0)
    assert state.prediction_memory.charge.shape == (2, 2, 16, 4)

    chunk = None
    parts = []
    for part in tokens.split((1, 19, 29), dim=1):
        emitted, chunk = native.forward_chunk(part, chunk)
        parts.append(emitted)
    torch.testing.assert_close(torch.cat(parts, dim=1), expected, rtol=0, atol=0)
    torch.testing.assert_close(chunk.correction.charge, state.prediction_memory.charge)


@torch.no_grad()
def test_spread_coincidence_matches_clock_slots_and_conserves_each_signed_packet():
    spread = _SpreadCoincidence((6,), cap=128, resolution=4)
    spread.mix.fill_(0.5)
    spread.freeze_integer()
    packet = torch.tensor([-17, -4, 0, 9, 12, 32], dtype=torch.int32)
    steering = torch.tensor([0, 2, 5, 3, 11, 4], dtype=torch.int32)
    coincident, bypass = spread(packet, steering, _steer_packet(packet, steering)[0])
    expected = []
    for content, window in zip(packet.tolist(), steering.tolist(), strict=True):
        count = abs(content)
        timed = (count + 1) // 2
        # Place event j at the center of its equal-width sector in the clock.
        observed = sum((2 * j - 1) * spread.clock_slots <= 2 * timed * min(window, spread.clock_slots)
                       for j in range(1, timed + 1))
        signed = min(count - timed, window) + observed
        expected.append(-signed if content < 0 else signed)
    assert coincident.tolist() == expected
    torch.testing.assert_close(coincident + bypass, packet, rtol=0, atol=0)
    assert bool((coincident.abs() <= packet.abs()).all())


def test_silent_spread_coincidence_preserves_router_gradients_and_can_learn():
    spread = _SpreadCoincidence((4, 2), cap=128, resolution=4)
    packet = torch.tensor([-7., -4., 0., 6., 8., 12.], requires_grad=True)
    steering = torch.tensor([0., 2., 4., 6., 3., 11.], requires_grad=True)
    packed, _ = _steer_packet(packet, steering)
    candidate, bypass = spread(packet, steering, packed)
    torch.testing.assert_close(candidate, packed, rtol=0, atol=0)
    torch.testing.assert_close(candidate + bypass, packet, rtol=0, atol=0)
    expected = torch.autograd.grad(packed.sum(), (packet, steering), retain_graph=True)
    dp, ds, mix = torch.autograd.grad(candidate.sum(), (packet, steering, spread.mix))
    torch.testing.assert_close(dp, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(ds, expected[1], rtol=0, atol=0)
    assert bool((mix != 0).all())


def test_trace_correction_vetoes_mismatched_old_charge_without_altering_the_write():
    trace = _SpikeTrace(cap=12, veto_widths=(1, 1, 1))
    present = torch.tensor([[3., -3., 5.]])
    steering = torch.ones_like(present)
    before = torch.tensor([[5., -5., 5.]])
    emitted, after = trace(present[:, None], steering[:, None], before, [])
    assert emitted.flatten().tolist() == [1, -1, 1]
    gradient, = torch.autograd.grad(emitted.abs().sum(), trace.read_veto.gain)
    # At zero veto the unused sign rail is tied in the coincidence minimum.
    torch.testing.assert_close(gradient, torch.tensor([-0.5, -0.5, 0.]), rtol=0, atol=0)
    with torch.no_grad():
        trace.read_veto.gain.fill_(1)
    trace.read_veto.freeze_integer()
    read, charge = trace.step(present.to(torch.int32), steering.to(torch.int32), before.to(torch.int32))
    assert read.flatten().tolist() == [0, 0, 1]
    torch.testing.assert_close(charge, after.to(torch.int32), rtol=0, atol=0)


def test_delay_filter_is_neutral_at_initialization_and_preserves_constant_packets():
    kernel = _DelayFilter(width=2, delay=4, cap=16)
    anchor = torch.tensor([[3., -2.]])
    halfway = torch.tensor([[8., -5.]])
    recent = torch.tensor([[4., 2.]])
    unchanged = kernel(anchor, [halfway, recent])
    torch.testing.assert_close(unchanged, anchor, rtol=0, atol=0)
    gradient, = torch.autograd.grad(unchanged.sum(), kernel.weight)
    torch.testing.assert_close(gradient, torch.tensor([[5., 1.], [-3., 4.]]), rtol=0, atol=0)
    with torch.no_grad():
        kernel.weight.copy_(torch.tensor([[1., -1.], [1., 0.5]]))
    kernel.freeze_integer()
    integer = anchor.to(torch.int32)
    torch.testing.assert_close(kernel(integer, [integer, integer]), integer, rtol=0, atol=0)
    torch.testing.assert_close(kernel(integer, [halfway.to(torch.int32), recent.to(torch.int32)]),
                               torch.tensor([[7, -3]], dtype=torch.int32), rtol=0, atol=0)


@torch.no_grad()
def test_delay_filter_warm_start_and_streaming_parity():
    torch.manual_seed(61)
    cfg = EventLMConfig(dim=16, hidden=21, route="steered_correction_code", carry=False, resolution=16,
                        memory_heads=2, memory_slots=4, memory_width=8, compact_delays=True, activity_inhibition=True)
    base = EventLM(cfg).eval()
    filtered = EventLM(replace(cfg, delay_filter=True)).eval()
    missing, unexpected = filtered.load_state_dict(base.state_dict(), strict=False)
    assert missing and all(".delay_filters." in name for name in missing) and not unexpected
    tokens = torch.randint(256, (2, 73))
    baseline = base.to_integer()(tokens)
    torch.testing.assert_close(filtered.to_integer()(tokens), baseline, rtol=0, atol=0)

    for block in filtered.blocks:
        for kernel in block.delay_filters:
            if kernel is not None:
                kernel.weight.fill_(0.5)
    native = filtered.to_integer()
    expected = native(tokens)
    assert not torch.equal(expected, baseline)
    torch.testing.assert_close(filtered(tokens), expected.to(torch.float32), rtol=0, atol=0)
    state = None
    for i, byte in enumerate(tokens.unbind(1)):
        counts, state = native.step(byte, state)
        torch.testing.assert_close(counts, expected[:, i], rtol=0, atol=0)
    first, chunk = native.forward_chunk(tokens[:, :3])
    second, chunk = native.forward_chunk(tokens[:, 3:21], chunk)
    third, _ = native.forward_chunk(tokens[:, 21:], chunk)
    torch.testing.assert_close(torch.cat((first, second, third), dim=1), expected, rtol=0, atol=0)
