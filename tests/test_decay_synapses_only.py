"""Weight decay on synapses only: a stack's charges, offsets and gains are left out of it."""

from pathlib import Path
import sys

import numpy as np
import torch

import test_lifted_ceiling
from snnformer.event_lm import CONTROL_PARAMETERS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from common import make_optimizer  # noqa: E402


def _decays(optimizer):
    """Every parameter's weight decay, by identity."""
    return {id(p): group["weight_decay"] for group in optimizer.param_groups for p in group["params"]}


def test_by_default_every_parameter_of_two_dimensions_decays():
    model = test_lifted_ceiling._stack(0, stack_loop=(0, 1, 2), stack_loop_phases=True)
    decays = _decays(make_optimizer(model, 1e-3, 0.05))
    for name, p in model.named_parameters():
        assert decays[id(p)] == (0.05 if p.ndim >= 2 else 0.0), name


def test_synapses_decay_and_charges_offsets_and_gains_do_not():
    model = test_lifted_ceiling._stack(0, stack_loop=(0, 1, 2), stack_loop_phases=True, read_forget=True)
    decays = _decays(make_optimizer(model, 1e-3, 0.05, exempt=CONTROL_PARAMETERS))
    controls = synapses = 0
    for name, p in model.named_parameters():
        parts = set(name.split("."))
        if parts & CONTROL_PARAMETERS:
            controls += 1
            assert decays[id(p)] == 0.0, name
        elif p.ndim >= 2:
            synapses += 1
            assert decays[id(p)] == 0.05, name
    # The read gains, grains and age offsets, the cells' rests and the phases are all among the controls.
    names = {part for name, _ in model.named_parameters() for part in name.split(".")} & CONTROL_PARAMETERS
    assert {"gain", "grain", "age", "rest", "phase_rests", "phase_gains"} <= names
    assert controls > 0 and synapses > 0


def test_the_trainer_takes_the_option(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import train_event_lm

    data = np.random.default_rng(0).integers(0, 256, 30000, dtype=np.uint8)
    monkeypatch.setattr(train_event_lm, "load_splits", lambda path: (data[:20000], data[20000:25000], data[25000:]))
    seen = {}
    real = train_event_lm.make_optimizer

    def spy(*args, **kwargs):
        seen["exempt"] = kwargs.get("exempt")
        return real(*args, **kwargs)

    monkeypatch.setattr(train_event_lm, "make_optimizer", spy)
    flags = """--variant event_steered --no-carry --stack --dim 32 --hidden 4 --depth 1 --spike-resolution 16
    --router-charge-bits 8 --read-after 0 --read-width 16 --read-heads 2 --read-window 16 --read-mode race
    --read-budget 16 --read-resolution 4 --granule-after 0 --granule-clusters 1 --granule-fan 32 --granule-cells 16
    --granule-slope --pool-fine 16 --pool-rest 4 --pool-rate 32 --output-pool 128 --pool-lifts-ceiling --ctx 16
    --bs 2 --steps 2 --warmup 1 --lr 0.003 --seed 3 --eval-every 2 --eval-bs 2 --eval-batches 1
    --final-eval-batches 1 --validation-only --decay-synapses-only""".split()
    monkeypatch.setattr(sys, "argv", ["train_event_lm.py", *flags, "--checkpoint", str(tmp_path / "model.pt"),
                                      "--results-dir", str(tmp_path / "results")])
    train_event_lm.main()
    assert seen["exempt"] == CONTROL_PARAMETERS
