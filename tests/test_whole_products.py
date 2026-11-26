"""Products of whole numbers on the tensor cores (snnformer/whole_products.py): the same charges to the bit, the same
gradient or one equal to single precision's rounding, and the switch of cuBLAS left as it was found.

The kernel's tests run where Triton runs: on a CUDA device, or on the CPU with TRITON_INTERPRET=1 (set before Python
starts). The tests that need the tensor cores themselves run on a CUDA device only."""
import os

import pytest
import torch
from torch.nn import functional as F

import snnformer.event_lm as event_lm
from snnformer import whole_products
import test_knowledge_gap_cells

CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"
KERNEL = whole_products.triton is not None and (CUDA or os.environ.get("TRITON_INTERPRET") == "1")
kernel = pytest.mark.skipif(not KERNEL, reason="the kernel needs a CUDA device or TRITON_INTERPRET=1")
cuda = pytest.mark.skipif(not CUDA, reason="needs the tensor cores of a CUDA device")
PRECISION = torch.backends.cuda.matmul


@pytest.fixture(autouse=True)
def _as_found():
    """Every test starts without a mode and leaves none behind."""
    saved = whole_products.MODE, whole_products.SPLIT, whole_products.AUDIT
    whole_products.MODE, whole_products.SPLIT, whole_products.AUDIT = None, False, False
    yield
    whole_products.MODE, whole_products.SPLIT, whole_products.AUDIT = saved


def _whole(*shape, largest, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-largest, largest + 1, shape, generator=generator).float().to(DEVICE)


def _gradient(*shape, seed):
    """Numbers as a gradient has them: of every size over many binades, some of them zero."""
    generator = torch.Generator().manual_seed(seed)
    size = torch.exp2(torch.randint(-40, 20, shape, generator=generator).float())
    values = torch.randn(shape, generator=generator) * size
    return values.masked_fill(torch.rand(shape, generator=generator) < 0.1, 0.0).to(DEVICE)


def test_without_a_mode_the_products_are_pytorch_s():
    counts, ticks = _whole(3, 5, 24, largest=128, seed=1), _whole(7, 24, largest=127, seed=2)
    before = dict(whole_products.COUNTS)
    assert torch.equal(whole_products.linear(counts, ticks), F.linear(counts, ticks))
    grouped, by_cluster = counts[:, :, None], ticks[None]
    assert torch.equal(whole_products.by_cluster(grouped, by_cluster),
                       torch.einsum("btgk,gnk->btgn", grouped, by_cluster))
    assert torch.equal(whole_products.matmul(counts, ticks.t()), counts @ ticks.t())
    assert whole_products.COUNTS == before


def test_tf32_switches_the_tensor_cores_on_for_a_product_and_off_again(monkeypatch):
    monkeypatch.setattr(whole_products, "_cublas", lambda tensor: True)
    whole_products.MODE, whole_products.AUDIT = "tf32", True
    counts, ticks = _whole(3, 5, 24, largest=128, seed=1), _whole(7, 24, largest=127, seed=2)
    found = PRECISION.fp32_precision
    seen = []
    plain = F.linear

    def watched(*args):
        seen.append(PRECISION.fp32_precision)
        return plain(*args)

    monkeypatch.setattr(whole_products.F, "linear", watched)
    before = whole_products.COUNTS["tf32"]
    assert torch.equal(whole_products.linear(counts, ticks), plain(counts, ticks))
    # The product itself under the tensor cores' precision, the audit's under the precision found.
    assert seen == ["tf32", found]
    assert PRECISION.fp32_precision == found
    grouped, by_cluster = counts[:, :, None], ticks[None]
    assert torch.equal(whole_products.by_cluster(grouped, by_cluster),
                       torch.einsum("btgk,gnk->btgn", grouped, by_cluster))
    assert torch.equal(whole_products.matmul(counts, ticks.t()), counts @ ticks.t())
    assert whole_products.COUNTS["tf32"] == before + 3
    assert PRECISION.fp32_precision == found


def test_tf32_leaves_the_precision_as_found_when_a_product_fails(monkeypatch):
    monkeypatch.setattr(whole_products, "_cublas", lambda tensor: True)
    whole_products.MODE = "tf32"
    found = PRECISION.fp32_precision
    with pytest.raises(RuntimeError):
        whole_products.linear(torch.ones(2, 3), torch.ones(4, 5))
    assert PRECISION.fp32_precision == found


def test_on_a_machine_without_tensor_cores_the_modes_change_nothing():
    if CUDA or KERNEL:
        pytest.skip("tensor cores or the kernel's interpreter are here")
    counts, ticks = _whole(3, 5, 24, largest=128, seed=1), _whole(7, 24, largest=127, seed=2)
    before = dict(whole_products.COUNTS)
    for mode in ("tf32", "half"):
        whole_products.MODE = mode
        assert torch.equal(whole_products.linear(counts, ticks), F.linear(counts, ticks))
    assert whole_products.COUNTS == before


_SHAPES = [(40, 48, 24), (1, 16, 1), (33, 7, 50), (130, 300, 129), (16, 2048, 16)]


@kernel
@pytest.mark.parametrize("rows,inner,columns", _SHAPES)
def test_the_kernel_s_product_of_whole_numbers_is_exact(rows, inner, columns):
    # The largest factors the formats hold, and sums up to the edge of single precision's exact range.
    largest = min(whole_products.LARGEST, int(((1 << 24) / inner) ** 0.5))
    a, b = _whole(rows, inner, largest=largest, seed=3), _whole(inner, columns, largest=largest, seed=4)
    a[0, 0], b[0, 0] = largest, largest
    want = (a.double() @ b.double())
    assert want.abs().max() < 1 << 24
    assert torch.equal(whole_products.product(a, b), want.float())


@kernel
def test_the_kernel_takes_views_by_their_strides():
    a, b = _whole(40, 48, largest=128, seed=5), _whole(24, 48, largest=127, seed=6)
    want = (a.double() @ b.double().t()).float()
    assert torch.equal(whole_products.product(a, b.t()), want)
    wide = _whole(40, 96, largest=128, seed=7)
    assert torch.equal(whole_products.product(wide[:, ::2], b.t()), (wide[:, ::2].double() @ b.double().t()).float())
    turned = _whole(48, 40, largest=128, seed=16)
    assert torch.equal(whole_products.product(turned.t(), b.t()), (turned.double().t() @ b.double().t()).float())


@kernel
@pytest.mark.parametrize("rows,inner,columns", _SHAPES[:4])
def test_the_kernel_s_split_product_is_a_gradient_s_to_single_precision(rows, inner, columns):
    gradient, whole = _gradient(rows, inner, seed=8), _whole(inner, columns, largest=2048, seed=9)
    want = gradient.double() @ whole.double()
    got = whole_products.product(gradient, whole, split=True).double()
    # Each entry against the sizes that it sums, single precision's own rounding of such a sum, and for the entries
    # far below their row's largest against that: 11 bits of an entry 2 ** -27 below it.
    sizes = gradient.double().abs() @ whole.double().abs()
    small = gradient.double().abs().amax(1, keepdim=True) * whole.double().abs().sum(0, keepdim=True) * 2.0 ** -37
    assert ((got - want).abs() <= sizes * 2e-6 + small).all()
    plain = (gradient @ whole).double()
    assert (got - want).abs().max() <= 4 * (plain - want).abs().max() + sizes.max() * 1e-7


@kernel
def test_a_split_product_keeps_a_small_entry_beside_a_large_one():
    gradient, whole = torch.zeros(40, 48, device=DEVICE), _whole(48, 24, largest=2048, seed=17)
    # A column of the whole numbers that hears the small entries alone.
    whole[0, 0], whole[1:, 0] = 0.0, 2048.0
    for row, ratio in enumerate((2.0 ** -20, 2.0 ** -30, 2.0 ** -37, 2.0 ** -50)):
        gradient[row, 0], gradient[row, 1] = 2.0 ** 20, 3 * 2.0 ** 20 * ratio
    got = whole_products.product(gradient, whole, split=True)
    want = (gradient.double() @ whole.double())
    assert (want[:4, 0] != 0).all()
    # At least 11 bits of an entry down to 2 ** -38 of its row's largest; below that the bound on every entry holds.
    assert ((got[:3, 0].double() - want[:3, 0]).abs() <= want[:3, 0].abs() * 1e-3).all()
    assert ((got.double() - want).abs() <= (gradient.double().abs() @ whole.double().abs()) * 2e-6
            + 2.0 ** 20 * 2048 * 48 * 2.0 ** -37).all()


@kernel
def test_a_split_product_keeps_zero_rows_and_whole_numbers():
    gradient, whole = _gradient(40, 48, seed=10), _whole(48, 24, largest=2048, seed=11)
    gradient[3] = 0
    gradient[5] = torch.arange(48, device=DEVICE) - 20.0
    got = whole_products.product(gradient, whole, split=True)
    assert torch.equal(got[3], torch.zeros_like(got[3]))
    assert torch.equal(got[5], (gradient[5].double() @ whole.double()).float())
    assert torch.isfinite(got).all()


def _both(run):
    """A function that runs ``run`` on the same whole counts and ticks under the mode set when it is called, and
    gives the charges and the gradients of the counts and the ticks."""
    def once():
        counts, ticks = (_whole(4, 9, 48, largest=128, seed=12).requires_grad_(True),
                         _whole(24, 48, largest=127, seed=13).requires_grad_(True))
        charge = run(counts, ticks)
        weights = torch.linspace(-1, 1, charge.numel(), device=DEVICE).reshape(charge.shape)
        (charge * weights).sum().backward()
        return charge.detach(), counts.grad, ticks.grad
    return once


@kernel
@pytest.mark.parametrize("split", [False, True])
def test_half_gives_linear_s_charges_and_its_gradient(split):
    once = _both(whole_products.linear)
    want = once()
    whole_products.MODE, whole_products.SPLIT, whole_products.AUDIT = "half", split, True
    before = whole_products.COUNTS["half"]
    got = once()
    assert whole_products.COUNTS["half"] == before + 1
    assert torch.equal(got[0], want[0])
    for mine, theirs in zip(got[1:], want[1:]):
        assert (mine - theirs).abs().max() <= theirs.abs().max() * 1e-6


@kernel
def test_half_gives_a_cluster_s_charges_and_leaves_several_clusters_to_pytorch():
    def clustered(counts, ticks):
        return whole_products.by_cluster(counts[:, :, None], ticks[None])
    once = _both(clustered)
    want = once()
    whole_products.MODE, whole_products.SPLIT, whole_products.AUDIT = "half", True, True
    before = whole_products.COUNTS["half"]
    got = once()
    assert whole_products.COUNTS["half"] == before + 1
    assert torch.equal(got[0], want[0])
    for mine, theirs in zip(got[1:], want[1:]):
        assert (mine - theirs).abs().max() <= theirs.abs().max() * 1e-6
    counts, ticks = _whole(2, 5, 3, 24, largest=128, seed=14), _whole(3, 7, 24, largest=127, seed=15)
    assert torch.equal(whole_products.by_cluster(counts, ticks), torch.einsum("btgk,gnk->btgn", counts, ticks))
    assert whole_products.COUNTS["half"] == before + 1


@cuda
def test_on_the_tensor_cores_whole_products_are_exact_to_the_edge_of_the_range():
    whole_products.MODE, whole_products.AUDIT = "tf32", True
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
    counts[::7, ::5] = 4097.0  # 13 significant bits: more than the tensor cores' formats hold
    whole_products.MODE, whole_products.AUDIT = "tf32", True
    with pytest.raises(RuntimeError, match="differs from the single-precision product"):
        whole_products.linear(counts, ticks)
    if KERNEL:
        whole_products.MODE = "half"
        with pytest.raises(RuntimeError, match="differs from the single-precision product"):
            whole_products.linear(counts, ticks)


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


def _modes():
    modes = [("tf32", False)]
    if KERNEL:
        modes += [("half", False), ("half", True)]
    return modes


@pytest.mark.parametrize("design", _DESIGNS)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mode,split", _modes())
def test_a_stack_fires_the_same_spikes_with_its_products_on_the_tensor_cores(design, stream, mode, split, monkeypatch):
    if not CUDA:
        # Without tensor cores the switch changes no number: the test is that every product goes through it.
        monkeypatch.setattr(whole_products, "_cublas", lambda tensor: True)
    model = test_knowledge_gap_cells._stack(0, **design).to(DEVICE).train()
    tokens = test_knowledge_gap_cells._bytes(length=120).to(DEVICE)
    want_counts, want = _run(model, tokens, stream)
    whole_products.MODE, whole_products.SPLIT, whole_products.AUDIT = mode, split, True
    before = whole_products.COUNTS[mode]
    counts, grads = _run(model, tokens, stream)
    assert whole_products.COUNTS[mode] > before + 4 * len(model.granule_layers)
    assert torch.equal(counts, want_counts)
    assert set(grads) == set(want)
    for name, grad in want.items():
        size = grad.abs().max()
        if mode == "tf32" and not split:
            # The backward pass is PyTorch's own, on the same charges.
            assert torch.equal(grads[name], grad), name
        elif size > 0:
            assert (grads[name] - grad).abs().max() / size < 1e-4, name
