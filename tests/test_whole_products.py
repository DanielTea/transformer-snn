"""Products of whole numbers on the tensor cores (snnformer/whole_products.py): the same charges and the same gradient
to the bit, and the switch of cuBLAS left as it was found. The tests that need the tensor cores themselves run on a
CUDA device only; elsewhere the switch changes no number, and the tests are that every product goes through it."""
import pytest
import torch
from torch.nn import functional as F

import snnformer.event_lm as event_lm
from snnformer import whole_products
import test_knowledge_gap_cells

CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"
cuda = pytest.mark.skipif(not CUDA, reason="needs the tensor cores of a CUDA device")
PRECISION = torch.backends.cuda.matmul


@pytest.fixture(autouse=True)
def _as_found():
    """Every test starts with the switch off and leaves it off."""
    saved = whole_products.ON, whole_products.AUDIT
    whole_products.ON, whole_products.AUDIT = False, False
    yield
    whole_products.ON, whole_products.AUDIT = saved


def _whole(*shape, largest, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-largest, largest + 1, shape, generator=generator).float().to(DEVICE)


def test_switched_off_the_products_are_pytorch_s():
    counts, ticks = _whole(3, 5, 24, largest=128, seed=1), _whole(7, 24, largest=127, seed=2)
    before = whole_products.COUNT
    assert torch.equal(whole_products.linear(counts, ticks), F.linear(counts, ticks))
    grouped, by_cluster = counts[:, :, None], ticks[None]
    assert torch.equal(whole_products.by_cluster(grouped, by_cluster),
                       torch.einsum("btgk,gnk->btgn", grouped, by_cluster))
    assert torch.equal(whole_products.matmul(counts, ticks.t()), counts @ ticks.t())
    assert whole_products.COUNT == before


def test_the_tensor_cores_are_switched_on_for_a_product_and_off_again(monkeypatch):
    monkeypatch.setattr(whole_products, "_cublas", lambda tensor: True)
    whole_products.ON, whole_products.AUDIT = True, True
    counts, ticks = _whole(3, 5, 24, largest=128, seed=1), _whole(7, 24, largest=127, seed=2)
    found = PRECISION.fp32_precision
    seen = []
    plain = F.linear

    def watched(*args):
        seen.append(PRECISION.fp32_precision)
        return plain(*args)

    monkeypatch.setattr(whole_products.F, "linear", watched)
    before = whole_products.COUNT
    assert torch.equal(whole_products.linear(counts, ticks), plain(counts, ticks))
    # The product itself under the tensor cores' precision, the audit's under the precision found.
    assert seen == ["tf32", found]
    assert PRECISION.fp32_precision == found
    grouped, by_cluster = counts[:, :, None], ticks[None]
    assert torch.equal(whole_products.by_cluster(grouped, by_cluster),
                       torch.einsum("btgk,gnk->btgn", grouped, by_cluster))
    assert torch.equal(whole_products.matmul(counts, ticks.t()), counts @ ticks.t())
    assert whole_products.COUNT == before + 3
    assert PRECISION.fp32_precision == found


def test_the_precision_is_left_as_found_when_a_product_fails(monkeypatch):
    monkeypatch.setattr(whole_products, "_cublas", lambda tensor: True)
    whole_products.ON = True
    found = PRECISION.fp32_precision
    with pytest.raises(RuntimeError):
        whole_products.linear(torch.ones(2, 3), torch.ones(4, 5))
    assert PRECISION.fp32_precision == found


def test_on_a_machine_without_tensor_cores_the_switch_changes_nothing():
    if CUDA:
        pytest.skip("tensor cores are here")
    counts, ticks = _whole(3, 5, 24, largest=128, seed=1), _whole(7, 24, largest=127, seed=2)
    before = whole_products.COUNT
    whole_products.ON = True
    assert torch.equal(whole_products.linear(counts, ticks), F.linear(counts, ticks))
    assert whole_products.COUNT == before


@cuda
def test_on_the_tensor_cores_whole_products_are_exact_to_the_edge_of_the_range():
    whole_products.ON, whole_products.AUDIT = True, True
    for seed, (rows, inner, columns) in enumerate([(4096, 512, 2048), (4096, 1024, 1188), (512, 4, 64)]):
        largest = min(whole_products.LARGEST, int(((1 << 24) / inner) ** 0.5))
        counts = _whole(rows, inner, largest=largest, seed=20 + seed)
        ticks = _whole(columns, inner, largest=largest, seed=30 + seed)
        counts[0, 0], ticks[0, 0] = largest, largest
        want = (counts.double() @ ticks.double().t()).float()
        assert torch.equal(whole_products.linear(counts, ticks), want)
        assert torch.equal(whole_products.matmul(counts, ticks.t()), want)
    assert PRECISION.fp32_precision != "tf32"


@cuda
def test_the_audit_catches_a_factor_that_the_tensor_cores_round():
    counts, ticks = _whole(4096, 512, largest=64, seed=40), _whole(2048, 512, largest=64, seed=41)
    counts[::7, ::5] = 4097.0  # 13 significant bits: more than the tensor cores' format holds
    whole_products.ON, whole_products.AUDIT = True, True
    with pytest.raises(RuntimeError, match="differs from the single-precision product"):
        whole_products.linear(counts, ticks)


@cuda
def test_the_phases_of_a_race_are_the_same_numbers_on_the_device():
    """_spread_phases works on a CUDA device itself (a CUDA graph cannot hold a copy from the host): the same phases
    as on the CPU, for cells before a stream's start and far into it."""
    for first, count in ((0, 4096), (-1000, 2048), (-129, 128), (2 ** 31 - 100, 300), (10 ** 12, 512)):
        for dtype in (torch.float32, torch.float64):
            on_cpu = event_lm._spread_phases(first, count, "cpu", dtype)
            on_device = event_lm._spread_phases(first, count, DEVICE, dtype)
            assert on_device.device.type == "cuda" and torch.equal(on_device.cpu(), on_cpu), (first, count, dtype)


_DESIGNS = [dict(), dict(read_knowledge=16, read_forget=True),
            dict(read_knowledge=16, stack_loop=(0, 1, 2), granule_power=8)]


def _run(model, tokens, stream):
    """The model's counts on windows or as a stream in two pieces, and its parameters' gradients."""
    model.zero_grad(set_to_none=True)
    if stream:
        with torch.no_grad():
            _, state = model.stream(tokens[:, :64], None)
        counts, _ = model.stream(tokens[:, 64:], state)
    else:
        counts = model(tokens)
    weights = torch.linspace(-1, 1, counts.numel(), device=counts.device).reshape(counts.shape)
    (counts * weights).sum().backward()
    return counts.detach(), {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("design", _DESIGNS)
@pytest.mark.parametrize("stream", [False, True])
def test_a_stack_fires_the_same_spikes_with_its_products_on_the_tensor_cores(design, stream, monkeypatch):
    if not CUDA:
        # Without tensor cores the switch changes no number: the test is that every product goes through it.
        monkeypatch.setattr(whole_products, "_cublas", lambda tensor: True)
    model = test_knowledge_gap_cells._stack(0, **design).to(DEVICE).train()
    tokens = test_knowledge_gap_cells._bytes(length=120).to(DEVICE)
    want_counts, want = _run(model, tokens, stream)
    whole_products.ON, whole_products.AUDIT = True, True
    before = whole_products.COUNT
    counts, grads = _run(model, tokens, stream)
    assert whole_products.COUNT > before + 4 * len(model.granule_layers)
    assert torch.equal(counts, want_counts)
    assert set(grads) == set(want)
    for name, grad in want.items():
        # The backward pass is PyTorch's own, on the same charges.
        assert torch.equal(grads[name], grad), name
