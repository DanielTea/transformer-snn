from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from snnformer import SNNConfig, TemporalSpikingLM


def _model_and_tokens(carry=True):
    cfg = SNNConfig(attn="temporal_spike", levels=8, local_window=5,
                    position_embeddings=False, causal_norm=True, temporal_carry=carry)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        model = TemporalSpikingLM(cfg, vocab=37, dim=24, depth=2, heads=3, ctx=16)
    tokens = torch.randint(37, (2, 31), generator=torch.Generator().manual_seed(61))
    return model, tokens


def test_all_linear_synapses_read_bounded_spike_counts():
    model, tokens = _model_and_tokens()
    model.eval()
    seen = {}

    def capture(module, args):
        seen[module] = args[0].detach()

    linears = {name: module for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    hooks = [module.register_forward_pre_hook(capture) for module in linears.values()]
    try:
        with torch.no_grad():
            model(tokens)
    finally:
        for hook in hooks:
            hook.remove()

    assert set(seen) == set(linears.values())
    for name, module in linears.items():
        spikes = seen[module]
        assert torch.equal(spikes, spikes.round())
        assert torch.all(spikes.abs() <= model.cfg.spike_levels)
        assert torch.any(spikes > 0)
        if name.endswith(".contract"):
            assert torch.all(spikes >= 0)
        else:
            assert torch.any(spikes < 0)


@pytest.mark.parametrize("training", [False, True])
def test_current_sequence_suffix_cannot_affect_earlier_predictions(training):
    model, tokens = _model_and_tokens()
    model.train(training)
    same_initial_model = deepcopy(model)
    changed = tokens.clone()
    prefix_length = 12
    changed[:, prefix_length:] = (changed[:, prefix_length:] + 1) % 37

    original_logits = model(tokens)
    changed_logits = same_initial_model(changed)

    torch.testing.assert_close(original_logits[:, :prefix_length], changed_logits[:, :prefix_length],
                               atol=0, rtol=0)
    assert not torch.equal(original_logits[:, prefix_length:], changed_logits[:, prefix_length:])


@pytest.mark.parametrize("carry", [True, False])
def test_streaming_logits_match_batched_through_ring_wraps(carry):
    model, tokens = _model_and_tokens(carry)
    model.eval()
    with torch.no_grad():
        batched = model(tokens)

    state, streamed = None, []
    buffer_addresses = None
    for length, token in enumerate(tokens.unbind(1), start=1):
        logits, state = model.step(token, state)
        streamed.append(logits)
        assert state.length == length
        assert state.input_phase.shape == (2, 48)
        assert state.input_phase.dtype == torch.int32
        addresses = []
        for layer in state.layers:
            attention = layer.attention
            assert attention.length == length
            assert attention.keys.shape == attention.values.shape == (2, 3, 5, 8)
            assert layer.delay_history.shape == (2, 3, 24)
            addresses.extend([attention.keys.data_ptr(), attention.values.data_ptr(),
                              layer.delay_history.data_ptr()])
        if buffer_addresses is None:
            buffer_addresses = addresses
        assert buffer_addresses == addresses

    torch.testing.assert_close(torch.stack(streamed, dim=1), batched)


_DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", _DEVICES)
def test_streaming_logits_match_wider_batched_projection(device):
    cfg = SNNConfig(attn="temporal_spike", levels=8, local_window=32,
                    position_embeddings=False, causal_norm=True)
    with torch.random.fork_rng(devices=[]):
        # This shape and seed previously exposed a charge rounding mismatch
        # between the batched and single-token matrix multiplications.
        torch.manual_seed(3)
        model = TemporalSpikingLM(cfg, dim=96, depth=3, heads=4, ctx=128)
        tokens = torch.randint(256, (2, 129))
    model = model.to(device).eval()
    tokens = tokens.to(device)

    state, streamed = None, []
    with torch.no_grad():
        batched = model(tokens)
        for token in tokens.unbind(1):
            logits, state = model.step(token, state)
            streamed.append(logits)

    assert state.length == tokens.shape[1]
    torch.testing.assert_close(torch.stack(streamed, dim=1), batched, atol=0, rtol=0)


def test_training_gradients_reach_all_hidden_paths():
    model, tokens = _model_and_tokens()
    logits = model(tokens[:, :-1])
    F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten()).backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("preserve", [True, False])
def test_pulse_decoder_matches_stream_and_accepts_a_causal_bank_intervention(device, preserve):
    original, tokens = _model_and_tokens(carry=False)
    cfg = replace(original.cfg, pulse_memory_cells=8, pulse_preserve_reads=preserve)
    model = TemporalSpikingLM(cfg, vocab=37, dim=24, depth=2, heads=3, ctx=16).to(device).eval()
    tokens = tokens.to(device)
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[:, 15] = True

    with torch.inference_mode():
        full = model(tokens)
        cleared = model(tokens, memory_clear_at=mask)
        torch.testing.assert_close(cleared[:, :15], full[:, :15], atol=0, rtol=0)
        for expected, intervene in ((full, False), (cleared, True)):
            state, output = None, []
            for index, token in enumerate(tokens.unbind(1)):
                if intervene and index == 15:
                    state.layers[-1].memory_phase.zero_()
                logits, state = model.step(token, state)
                output.append(logits)
                assert state.layers[0].memory_phase is None
                assert state.layers[-1].memory_phase.shape == (tokens.shape[0], 8)
            torch.testing.assert_close(torch.stack(output, dim=1), expected, atol=0, rtol=0)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("fixed_cuts", [False, True])
def test_protected_timing_decoder_matches_streaming(device, fixed_cuts):
    original, tokens = _model_and_tokens(carry=False)
    cfg = replace(original.cfg, pulse_memory_cells=4, pulse_write_holds=(0, 4, 8, 16),
                  pulse_read_bins=4, pulse_fixed_cuts=fixed_cuts)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21)
        model = TemporalSpikingLM(cfg, vocab=37, dim=24, depth=2, heads=3, ctx=16).to(device).eval()

    tokens = tokens.to(device)
    with torch.inference_mode():
        full = model(tokens)
        state, streamed = None, []
        for token in tokens.unbind(1):
            logits, state = model.step(token, state)
            streamed.append(logits)
            memory = state.layers[-1].memory_phase
            assert memory.phase.shape == memory.cooldown.shape == (tokens.shape[0], 4)
    torch.testing.assert_close(torch.stack(streamed, dim=1), full, atol=0, rtol=0)
