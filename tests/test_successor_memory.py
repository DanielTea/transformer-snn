import pytest
import torch

from snnformer.successor_memory import SuccessorMemory


def test_collision_overwrites_previous_address_without_false_hit():
    cache = SuccessorMemory(orders=(1,), slots=1)
    # After the second 1 arrives, observing 2 -> 1 has just evicted 1 -> 2.
    # At the last 3, its immediately preceding occurrence becomes a hit.
    prediction, age = cache(torch.tensor([[1, 2, 1, 3, 3]]))
    assert prediction.flatten().tolist() == [-1, -1, -1, -1, 3]
    assert age.flatten().tolist() == [-1, -1, -1, -1, 0]


def test_eight_byte_tag_read_uses_only_observed_successors():
    cache = SuccessorMemory(orders=(8,), slots=64)
    # The top byte sets the sign bit of the packed tag. The first complete
    # suffix has no known successor until the next byte has been consumed.
    motif = [255] * 8
    tokens = torch.tensor([motif + [2] + motif + [9]])
    prediction, age = cache(tokens)
    assert prediction[0, :16].eq(-1).all()
    assert prediction[0, 16, 0] == 2
    assert age[0, 16, 0] == 8

    changed = tokens.clone()
    changed[0, -1] = 8
    changed_prediction, changed_age = cache(changed)
    assert torch.equal(changed_prediction[:, :-1], prediction[:, :-1])
    assert torch.equal(changed_age[:, :-1], age[:, :-1])


@pytest.mark.parametrize("slots", [1, 4, 64])
def test_step_matches_batched_retrieval(slots):
    cache = SuccessorMemory(orders=(8, 1, 3, 2), slots=slots)
    rng = torch.Generator().manual_seed(100)
    tokens = torch.randint(0, 5, (3, 97), generator=rng) * 63
    predicted, age = cache(tokens)
    state, predicted_steps, age_steps = None, [], []
    for token in tokens.unbind(dim=1):
        pred, step_age, state = cache.step(token, state)
        predicted_steps.append(pred)
        age_steps.append(step_age)
    assert torch.equal(predicted, torch.stack(predicted_steps, dim=1))
    assert torch.equal(age, torch.stack(age_steps, dim=1))
