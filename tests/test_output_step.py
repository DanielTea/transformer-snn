"""An output spike that stands for several fine steps of evidence: the same score for far fewer output spikes."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from snnformer.event_lm import EventLM, EventLMConfig, _IntegrateFire, _ShuntedFire, emitted_pulses

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from eval_stream import coarsen_output  # noqa: E402
from gate_state import gate_state  # noqa: E402

_LEVELS = (0, 1)
_STACK = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=_LEVELS, read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, granule_after=_LEVELS, granule_clusters=1, granule_fan=64,
              granule_cells=96, granule_slope=True)
# Sixty-four quanta to a threshold with four at rest, and a pool at the output.
_FINE = dict(pool_fine=16, pool_rest=4, pool_rate=32, output_pool=128)


def _stack(**settings):
    torch.manual_seed(0)
    model = EventLM(EventLMConfig(**dict(_STACK, **settings)))
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        # Reads and cell layers start silent. Give them output synapses, and the output neurons a wide say.
        for read in model.state_reads.values():
            read.project.weight.copy_(torch.randn(read.project.weight.shape, generator=generator) * 0.05)
        for layer in model.granule_layers.values():
            layer.send.copy_(torch.randn(layer.send.shape, generator=generator) * 0.05)
        model.output.weight.mul_(6)
    return model.eval()


def _bytes(length=96, streams=2):
    return torch.randint(0, 256, (streams, length), generator=torch.Generator().manual_seed(2))


def _output_charge(model, tokens):
    """The charge of the output neurons, what divides it, and their spikes, in one pass."""
    seen = {}
    handle = model.output_emit.register_forward_hook(
        lambda _fire, args, fired: seen.update(charge=args[0], quanta=args[1] if len(args) > 1 else None, fired=fired))
    with torch.no_grad():
        counts = model(tokens)
    handle.remove()
    return counts, seen


def test_the_step_is_one_unless_told_otherwise_and_adds_no_synapse():
    assert EventLMConfig.output_step == 1
    fine, coarse = _stack(), _stack(output_step=8)
    assert fine.output_step == 1 and coarse.output_step == 8
    # The membrane holds the same evidence: a threshold eight times higher and a largest count eight times smaller.
    assert coarse.output_emit.threshold == 8 * fine.output_emit.threshold
    assert coarse.output_emit.cap * 8 == fine.output_emit.cap == 128 * 16
    assert set(fine.state_dict()) == set(coarse.state_dict())
    assert all(torch.equal(value, coarse.state_dict()[name]) for name, value in fine.state_dict().items())


@pytest.mark.parametrize("pool", [dict(), _FINE])
@pytest.mark.parametrize("step", [2, 8, 32])
def test_coarse_spikes_round_the_same_charge_and_are_scored_in_fine_steps(pool, step):
    fine, coarse = _stack(**pool), _stack(output_step=step, **pool)
    assert type(coarse.output_emit) is (_ShuntedFire if pool else _IntegrateFire)
    tokens = _bytes()
    fine_counts, fine_seen = _output_charge(fine, tokens)
    counts, seen = _output_charge(coarse, tokens)
    # The stream below the output is the same, so the output neurons hold the same charge.
    assert torch.equal(seen["charge"], fine_seen["charge"])
    # The counts are whole multiples of the step: every spike stands for that many fine steps.
    assert torch.equal(counts, seen["fired"] * step) and torch.equal(counts % step, torch.zeros_like(counts))
    # A coarse spike is the charge over the higher threshold, rounded half away from zero.
    threshold = float(coarse.output_emit.threshold)
    if pool:
        threshold = threshold * seen["quanta"].double() / coarse.output_emit.quanta_per_threshold
    exact = seen["charge"].double() / threshold
    rounded = (exact.abs() + 0.5).floor() * exact.sign()
    assert torch.equal(seen["fired"].double(), rounded.clamp(-coarse.output_emit.cap, coarse.output_emit.cap))
    # So it stands within half a step of the fine count, and the two differ somewhere.
    assert (counts - fine_counts).abs().max() <= step / 2
    assert not torch.equal(counts, fine_counts)
    assert fine_counts.abs().max() > 4 * step


def test_a_budget_counts_the_coarse_spikes():
    fine, coarse = _stack(), _stack(output_step=8)
    tokens = _bytes()
    counts, seen = _output_charge(coarse, tokens)
    assert float(emitted_pulses(coarse.output_emit, seen["fired"])) == float(counts.abs().sum()) / 8
    measured = gate_state(coarse, tokens, 32, torch.device("cpu"))["pulses_by_population"]
    before = gate_state(fine, tokens, 32, torch.device("cpu"))["pulses_by_population"]
    assert measured["output"] == pytest.approx(float(counts.abs().sum()) / 8 / tokens.numel())
    # About an eighth of the output's spikes, and every other population fires as before.
    assert measured["output"] == pytest.approx(before["output"] / 8, rel=0.05)
    assert {label: value for label, value in measured.items() if label != "output"} == pytest.approx(
        {label: value for label, value in before.items() if label != "output"})


@pytest.mark.parametrize("pool", [dict(), _FINE])
def test_a_stream_fires_the_same_coarse_counts_however_it_is_cut(pool):
    model = _stack(output_step=8, **pool)
    tokens = _bytes()
    with torch.no_grad():
        whole = model(tokens)
        pieces, state = [], None
        for start in range(0, tokens.shape[1], 23):
            counts, state = model.stream(tokens[:, start:start + 23], state)
            pieces.append(counts)
    assert torch.equal(torch.cat(pieces, 1), whole)


def test_coarse_spikes_learn_as_fine_ones_do():
    fine, coarse = _stack().train(), _stack(output_step=8).train()
    tokens = _bytes(48)
    gradients = []
    for model in (fine, coarse):
        counts = model(tokens)
        counts.square().sum().backward()
        gradients.append(model.output.weight.grad.clone())
    # The slope of a count in fine steps is the same; only the rounded counts that it multiplies differ a little.
    assert torch.allclose(gradients[0], gradients[1], rtol=0.1, atol=0.05 * gradients[0].abs().max().item())
    assert gradients[1].abs().sum() > 0


def test_trained_weights_take_a_coarser_output_at_scoring():
    fine, built = _stack(), _stack(output_step=8)
    tokens = _bytes()
    with torch.no_grad():
        expected = built(tokens)
    coarsen_output(fine, 8)
    assert fine.output_step == 8 and fine.cfg.output_step == 8
    with torch.no_grad():
        assert torch.equal(fine(tokens), expected)
    # A stack built with a step takes a coarser one, and no finer one.
    coarsen_output(built, 32)
    assert built.output_step == 32 and built.output_emit.threshold == 4 * fine.output_emit.threshold
    with pytest.raises(ValueError, match="power of two"):
        coarsen_output(built, 8)
    with pytest.raises(ValueError, match="power of two"):
        coarsen_output(_stack(), 3)
    with pytest.raises(ValueError, match="power of two"):
        coarsen_output(_stack(), 128 * 16 * 2)


@pytest.mark.parametrize("settings", [dict(output_step=3), dict(output_step=0), dict(output_step=True),
                                      dict(output_step=128 * 16 * 2), dict(output_step=8, stack=False)])
def test_the_step_is_a_power_of_two_in_a_stack(settings):
    if settings.get("stack") is False:
        settings = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8,
                        output_step=8)
        with pytest.raises(ValueError, match="power of two"):
            EventLM(EventLMConfig(**settings))
        return
    with pytest.raises(ValueError, match="power of two"):
        EventLM(EventLMConfig(**dict(_STACK, **settings)))


def test_the_trainer_hands_the_step_to_the_model_and_the_budget_counts_it(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import train_event_lm

    data = np.random.default_rng(0).integers(0, 256, 30000, dtype=np.uint8)
    monkeypatch.setattr(train_event_lm, "load_splits", lambda path: (data[:20000], data[20000:25000], data[25000:]))
    flags = """--variant event_steered --no-carry --stack --dim 32 --hidden 4 --depth 1 --spike-resolution 16
    --router-charge-bits 8 --read-after 0 --read-width 16 --read-heads 2 --read-window 16 --read-mode race
    --read-budget 16 --read-resolution 4 --granule-after 0 --granule-clusters 1 --granule-fan 32 --granule-cells 16
    --granule-slope --output-step 8 --pulse-budget 4000
    --ctx 16 --bs 2 --steps 2 --warmup 1 --lr 0.003 --seed 3 --eval-every 2 --eval-bs 2 --eval-batches 1
    --final-eval-batches 1 --validation-only""".split()
    monkeypatch.setattr(sys, "argv", ["train_event_lm.py", *flags, "--checkpoint", str(tmp_path / "model.pt"),
                                      "--results-dir", str(tmp_path / "results")])
    train_event_lm.main()
    model, checkpoint = train_event_lm.load_checkpoint(tmp_path / "model.pt", torch.device("cpu"))
    assert model.cfg.output_step == 8 and checkpoint["cfg"]["output_step"] == 8
    assert model.output_emit.cap == 128 * 16 // 8
