from dataclasses import replace

import pytest
import torch

from snnformer.correction_memory import correction_reads, correction_step
from snnformer.event_lm import EventLM, EventLMConfig, _DelayCorrection, _SpikeCorrection, _SpikeTrace


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


@torch.no_grad()
@pytest.mark.parametrize("route", ("steered", "steered_correction_code"))
def test_trace_preserves_exact_history_across_steps_and_differently_sized_chunks(route):
    torch.manual_seed(97)
    cfg = EventLMConfig(dim=16, hidden=21, route=route, carry=False, resolution=16,
                        memory_heads=2, memory_slots=4, memory_width=8 if route.endswith("code") else None,
                        compact_delays=True, activity_inhibition=True, local_trace=True)
    model = EventLM(cfg).eval()
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
