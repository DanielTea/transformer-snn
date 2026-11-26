"""The loudness diagnostic (experiments/stream_loudness.py): how loud a stream's loudest channels are where a norm or
a pool hears it."""
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch

from snnformer.config import SNNConfig
from snnformer import event_lm
from snnformer.transformer_reference import TransformerLM
from test_stack_loop import _stack
from test_teaching_spikes import _bytes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from stream_loudness import Places, loudness, measure_stack, measure_transformer  # noqa: E402


def test_channels_that_fire_alike_are_not_loud_and_one_loud_channel_is():
    alike = torch.zeros(1, 64)
    alike[0, :16] = 3.0
    ratio, loud, active, silent = loudness(alike)
    # Sixteen of 64 channels alike: an energy pool hears twice the summed counts, all through the silent channels.
    assert silent == 0 and math.isclose(ratio[0], 2.0) and math.isclose(loud[0], 1.0) and active[0] == 0.25
    one = torch.zeros(1, 64)
    one[0, 0], one[0, 1:9] = 40.0, 1.0
    ratio, loud, active, _ = loudness(one)
    assert loud[0] > 2.5 and math.isclose(ratio[0], loud[0] * math.sqrt(64 / 9))
    # A silent token is counted and left out; signed values count by their size.
    ratio, loud, active, silent = loudness(torch.stack([torch.zeros(8), -torch.ones(8)]))
    assert silent == 1 and ratio.tolist() == [1.0] and loud.tolist() == [1.0] and active.tolist() == [1.0]


def test_the_summary_gives_quantiles_over_tokens_for_every_place():
    places = Places()
    places.add("a", torch.ones(4, 10, 8))
    places.add("a", torch.zeros(2, 8))
    summary = places.summary()["a"]
    assert summary["tokens"] == 40 and summary["silent_tokens"] == 2 and summary["width"] == 8
    assert summary["ratio"]["q50"] == 1.0 and summary["loudness"]["q99_over_q50"] == 1.0


def test_every_norm_of_a_transformer_is_a_place():
    torch.manual_seed(3)
    model = TransformerLM(replace(SNNConfig(neuron="ann", attn="transformer", mlp_ratio=4.0,
                                            position_embeddings=True), norm="pool"),
                          vocab=256, dim=16, depth=2, heads=4, ctx=32).eval()
    places = measure_transformer(model, torch.randint(0, 256, (200,), dtype=torch.uint8), window=32)
    summary = places.summary()
    assert len(summary) == 2 * 2 + 1
    assert all(entry["tokens"] == 6 * 32 and entry["width"] == 16 for entry in summary.values())
    # The hooks are gone afterwards.
    assert not any(module._forward_pre_hooks for module in model.modules())


def test_every_pool_of_a_stack_is_a_place_and_the_pools_are_unchanged():
    model, tokens = _stack(), _bytes()
    original = event_lm._heard_activity
    with torch.no_grad():
        plain, _ = model.stream(tokens)
    data = tokens.flatten()
    places = measure_stack(model, data, streams=tokens.shape[0], piece=5)
    assert event_lm._heard_activity is original
    summary = places.summary()
    names = dict(model.named_modules())
    assert summary and all(place in names or place == "stack output" for place in summary)
    assert all(entry["tokens"] + entry["silent_tokens"] > 0 for entry in summary.values())
    with torch.no_grad():
        again, _ = model.stream(tokens)
    torch.testing.assert_close(again, plain, rtol=0, atol=0)
