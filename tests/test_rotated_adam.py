from pathlib import Path

import pytest
import torch


@pytest.fixture
def experiments(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))


def _correlated_problem():
    """Least squares whose inputs share one strong direction."""
    generator = torch.Generator().manual_seed(0)
    mix = torch.randn(96, 96, generator=generator) * 0.2 + 1
    inputs = torch.randn(4096, 96, generator=generator) @ mix
    target = torch.randn(24, 96, generator=generator) * 0.1
    return inputs, inputs @ target.T


def _fit(optimizer, weight, inputs, outputs, steps=600):
    generator = torch.Generator().manual_seed(1)
    for _ in range(steps):
        rows = torch.randint(0, inputs.shape[0], (32,), generator=generator)
        loss = (inputs[rows] @ weight.T - outputs[rows]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (inputs @ weight.T - outputs).square().mean().item()


def test_rotation_takes_a_shared_input_direction_as_one_coordinate(experiments):
    from rotated_adam import RotatedAdam

    inputs, outputs = _correlated_problem()
    plain = torch.zeros(24, 96, requires_grad=True)
    rotated = torch.zeros(24, 96, requires_grad=True)
    adam = _fit(torch.optim.AdamW([plain], lr=3e-3, betas=(0.9, 0.99), weight_decay=0.0), plain, inputs, outputs)
    optimizer = RotatedAdam([rotated], lr=3e-3, refresh=10)
    turned = _fit(optimizer, rotated, inputs, outputs)
    assert turned < 0.25 * adam
    for name, size in (("row_basis", 24), ("column_basis", 96)):
        basis = optimizer.state[rotated][name]
        assert torch.allclose(basis.T @ basis, torch.eye(size), atol=1e-4)


def test_tensors_without_a_basis_take_adam_steps(experiments):
    from rotated_adam import RotatedAdam

    generator = torch.Generator().manual_seed(2)
    start = [torch.randn(12, generator=generator), torch.randn(4, 1, 1, generator=generator),
             torch.randn(2, 3, 12, generator=generator)]
    ours = [value.clone().requires_grad_() for value in start]
    theirs = [value.clone().requires_grad_() for value in start]
    optimizers = (RotatedAdam(ours, lr=1e-2, weight_decay=0.1), torch.optim.AdamW(theirs, lr=1e-2, betas=(0.9, 0.99),
                                                                                  weight_decay=0.1))
    for step in range(20):
        noise = torch.randn(12, generator=generator)
        for values, optimizer in zip((ours, theirs), optimizers, strict=True):
            loss = ((values[0] * values[1].exp().mean() + values[2].sum((0, 1)) - noise).square().sum()
                    + (values[0] * step).sum())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    for mine, reference in zip(ours, theirs, strict=True):
        assert torch.allclose(mine, reference, atol=1e-5)
    assert all(optimizers[0].state[value]["row_basis"] is None for value in ours)


def test_a_stack_of_matrices_is_rotated_matrix_by_matrix(experiments):
    from rotated_adam import RotatedAdam

    generator = torch.Generator().manual_seed(4)
    problems = []
    for _ in range(3):
        mix = torch.randn(40, 40, generator=generator) * 0.2 + 1
        inputs = torch.randn(1024, 40, generator=generator) @ mix
        problems.append((inputs, inputs @ (torch.randn(12, 40, generator=generator) * 0.1).T))
    stack = torch.zeros(3, 12, 40, requires_grad=True)
    single = [torch.zeros(12, 40, requires_grad=True) for _ in problems]
    plain = torch.zeros(3, 12, 40, requires_grad=True)
    stacked = RotatedAdam([stack], lr=3e-3, refresh=10, stacks=True)
    alone = [RotatedAdam([weight], lr=3e-3, refresh=10) for weight in single]
    unrotated = RotatedAdam([plain], lr=3e-3, refresh=10)
    for step in range(120):
        rows = torch.randint(0, 1024, (32,), generator=generator)
        for weights, optimizers in ((stack, [stacked]), (single, alone), (plain, [unrotated])):
            loss = sum((inputs[rows] @ weight.T - outputs[rows]).square().mean()
                       for weight, (inputs, outputs) in zip(weights, problems, strict=True))
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
    # Every matrix of the stack moves as it would alone, and not as Adam moves it.
    for index, weight in enumerate(single):
        assert torch.allclose(stack[index], weight, atol=2e-4)
        assert not torch.allclose(plain[index], weight, atol=1e-2)
    assert stacked.state[stack]["row_basis"].shape == (3, 12, 12)
    assert stacked.state[stack]["column_basis"].shape == (3, 40, 40)
    assert unrotated.state[plain]["row_basis"] is None
    # A stack of single lines or single weights has no basis to turn into.
    lines = [torch.zeros(4, 1, 1, requires_grad=True), torch.zeros(3, 5, 1, requires_grad=True)]
    optimizer = RotatedAdam(lines, lr=1e-2, stacks=True)
    for _ in range(3):
        optimizer.zero_grad()
        (lines[0].sum() + (lines[1] - 1).square().sum()).backward()
        optimizer.step()
    assert optimizer.state[lines[0]]["row_basis"] is None and optimizer.state[lines[0]]["column_basis"] is None
    assert optimizer.state[lines[1]]["row_basis"].shape == (3, 5, 5) and optimizer.state[lines[1]]["column_basis"] is None


def test_a_side_above_the_limit_keeps_its_own_coordinates(experiments):
    from rotated_adam import RotatedAdam

    inputs, outputs = _correlated_problem()
    weight = torch.zeros(24, 96, requires_grad=True)
    optimizer = RotatedAdam([weight], lr=3e-3, refresh=10, largest=64)
    loss = _fit(optimizer, weight, inputs, outputs, steps=300)
    state = optimizer.state[weight]
    assert state["column_basis"] is None and state["columns"] is None
    assert state["row_basis"].shape == (24, 24)
    assert loss < (outputs.square().mean().item())
    with pytest.raises(ValueError, match="positive"):
        RotatedAdam([weight], lr=3e-3, refresh=0)


def test_averaged_weights_are_lent_to_the_model_and_taken_back(experiments):
    from train_event_lm import averaged_weights

    model = torch.nn.Linear(3, 2)
    own = [parameter.detach().clone() for parameter in model.parameters()]
    average = [torch.full_like(parameter, 0.5) for parameter in model.parameters()]
    with averaged_weights(model, average):
        assert all(torch.equal(parameter, mean) for parameter, mean in zip(model.parameters(), average, strict=True))
    assert all(torch.equal(parameter, saved) for parameter, saved in zip(model.parameters(), own, strict=True))
    with pytest.raises(RuntimeError), averaged_weights(model, average):
        raise RuntimeError("the weights still come back")
    assert all(torch.equal(parameter, saved) for parameter, saved in zip(model.parameters(), own, strict=True))


def test_named_weights_step_with_their_own_share_of_the_learning_rate(experiments):
    from common import make_optimizer
    from train_event_lm import _lr_scales

    scales = _lr_scales("0.=0.5,1.weight=3")
    assert scales == (("0.", 0.5), ("1.weight", 3.0))
    for text in ("0.=0", "=2", "0.", "0.=nan"):
        with pytest.raises(ValueError):
            _lr_scales(text)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    names = {parameter: name for name, parameter in model.named_parameters()}
    for rotated in (None, 5):
        optimizer = make_optimizer(model, 1e-2, 0.1, rotated=rotated, scales=scales)
        found = {names[parameter]: (group["lr_scale"], group["weight_decay"], group["lr"])
                 for group in optimizer.param_groups for parameter in group["params"]}
        assert found == {"0.weight": (0.5, 0.1, 5e-3), "0.bias": (0.5, 0.0, 5e-3), "1.weight": (3.0, 0.1, 3e-2),
                         "1.bias": (1.0, 0.0, 1e-2)}
    plain = make_optimizer(model, 1e-2, 0.1)
    assert [(group["lr_scale"], group["weight_decay"]) for group in plain.param_groups] == [(1.0, 0.1), (1.0, 0.0)]
    # A scaled weight moves that much farther in one step from the same gradient.
    torch.manual_seed(0)
    twins = [torch.nn.Linear(6, 5) for _ in range(2)]
    twins[1].load_state_dict(twins[0].state_dict())
    start = twins[0].weight.detach().clone()
    for twin, pairs in zip(twins, ((), (("weight", 3.0),)), strict=True):
        optimizer = make_optimizer(twin, 1e-3, 0.0, rotated=5, scales=pairs)
        twin(torch.ones(2, 6)).square().sum().backward()
        optimizer.step()
    assert torch.allclose(twins[1].weight - start, 3 * (twins[0].weight - start), atol=1e-7)
    assert torch.allclose(twins[1].bias, twins[0].bias)
