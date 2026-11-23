"""The unit-density diagnostic (experiments/unit_density.py): how many of a Transformer's units are open at a byte."""
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from snnformer.config import SNNConfig
from snnformer.transformer_reference import TransformerLM, _TransformerBlock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from unit_density import density, passed_on, units_for  # noqa: E402

_CFG = SNNConfig(neuron="ann", attn="transformer", position_embeddings=True)


def _transformer(cfg=_CFG, seed=0):
    torch.manual_seed(seed)
    return TransformerLM(cfg, vocab=256, dim=32, depth=2, heads=4, ctx=32).eval()


def _windows(count=6, length=32, seed=1):
    return torch.randint(0, 256, (count, length), generator=torch.Generator().manual_seed(seed))


def _blocks(model):
    return [module for module in model.modules() if isinstance(module, _TransformerBlock)]


def test_a_gated_unit_is_open_where_its_gate_is_above_zero():
    gate, value = torch.tensor([1.0, -1.0, 2.0, -3.0]), torch.tensor([1.0, 2.0, -1.0, 1.0])
    drive = torch.cat([gate, value])
    is_open, smooth = passed_on(drive, plain=0, smooth=True)
    assert is_open.tolist() == [True, False, True, False]
    assert torch.allclose(smooth, F.silu(gate) * value)
    is_open, hard = passed_on(drive, plain=0, smooth=False)
    assert is_open.tolist() == [True, False, True, False]
    assert hard.tolist() == [1.0, 0.0, -2.0, 0.0]


def test_a_plain_unit_is_open_where_its_own_drive_is_above_zero():
    drive = torch.tensor([1.0, -1.0, 0.5])
    is_open, hard = passed_on(drive, plain=3, smooth=False)
    assert is_open.tolist() == [True, False, True]
    assert hard.tolist() == [1.0, 0.0, 0.5]
    assert torch.allclose(passed_on(drive, plain=3, smooth=True)[1], F.gelu(drive))


def test_the_share_of_units_that_pass_nine_tenths():
    # Of 1.0, 0.0, 2.0 and 0.0 the two open units pass everything: two of four.
    assert units_for(torch.tensor([[1.0, 0.0, -2.0, 0.0]])).tolist() == [0.5]
    # 5, 3, 1 and 1 of 10: the first two reach 0.8, the third reaches 0.9.
    assert units_for(torch.tensor([[1.0, 5.0, -3.0, 1.0]])).tolist() == [0.75]
    # Ten equal units: nine reach nine tenths.
    assert units_for(torch.ones(1, 10)).tolist() == pytest.approx([0.9])


def test_a_silent_byte_counts_every_unit_without_dividing_by_zero():
    assert units_for(torch.zeros(2, 4)).tolist() == [1.0, 1.0]


def test_density_counts_the_second_half_of_every_window():
    model, windows = _transformer(), _windows()
    drives = []
    handles = [block.mlp_in.register_forward_hook(lambda _m, _i, out: drives.append(out.detach()))
               for block in _blocks(model)]
    with torch.no_grad():
        model(windows)
    for handle in handles:
        handle.remove()
    rows = density(model, windows, batch=4)
    assert len(rows) == 2
    for row, drive in zip(rows, drives):
        gate = drive[:, 16:].chunk(2, dim=-1)[0]
        assert row["units"] == gate.shape[-1]
        assert row["unit"] == "gated, smooth"
        assert row["share_of_units_open_at_a_byte"] == pytest.approx((gate > 0).float().mean().item(), abs=1e-6)
        per_unit = (gate > 0).float().mean(dim=(0, 1))
        assert row["share_of_bytes_a_unit_is_open_at"]["median"] == pytest.approx(per_unit.median().item(), abs=1e-6)
        assert 0 < row["share_of_units_passing_nine_tenths"] <= 1


def test_a_block_with_silent_gates_has_no_open_unit():
    model = _transformer()
    with torch.no_grad():
        _blocks(model)[0].mlp_in.weight.zero_()
    first, second = density(model, _windows())
    assert first["share_of_units_open_at_a_byte"] == 0
    assert first["share_of_bytes_a_unit_is_open_at"] == {"median": 0.0, "p10": 0.0, "p90": 0.0}
    assert first["share_of_units_passing_nine_tenths"] == 1.0
    assert 0 < second["share_of_units_open_at_a_byte"] < 1


@pytest.mark.parametrize("units, name", [("gelu", "plain, smooth"), ("relu", "plain, hard"), ("reglu", "gated, hard")])
def test_other_units_are_named_and_counted(units, name):
    model = _transformer(replace(_CFG, feed_forward=units))
    rows = density(model, _windows())
    block = _blocks(model)[0]
    assert rows[0]["unit"] == name
    assert rows[0]["units"] == block.mlp_out.in_features
    assert 0 < rows[0]["share_of_units_open_at_a_byte"] < 1


def test_the_model_is_left_as_it_was():
    model = _transformer().train()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    density(model, _windows())
    assert model.training
    assert all(not block.mlp_in._forward_hooks for block in _blocks(model))
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
