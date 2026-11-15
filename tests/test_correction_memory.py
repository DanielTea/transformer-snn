from dataclasses import replace

import pytest
import torch

from snnformer.correction_memory import correction_reads, correction_step
from snnformer.event_lm import EventLM, EventLMConfig, _SpikeCorrection


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
