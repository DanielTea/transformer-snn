"""Products of whole numbers on a GPU's tensor cores, to the bit.

Every charge in the model is a sum of whole spike counts times whole synapse sizes that stays inside the range in
which single precision counts exactly (below 2 ** 24). Such a sum is the same number in whatever order a device adds
it and in whatever precision, as long as it holds every factor and every partial sum exactly. PyTorch works these
products out in single precision on a GPU's general cores: the faster formats of the tensor cores are off by default,
because they round other numbers. The factors here are small whole numbers, at most ``LARGEST`` in size, which those
formats hold exactly, and the tensor cores add in single precision:

  "tf32"  PyTorch's own products (cuBLAS) with the tensor cores' single precision (TF32: 11 significant bits of each
          factor, single-precision sums), switched on for these products alone and off again after each. The backward
          pass is PyTorch's in full single precision, as without: the gradient is the same.
  "half"  A kernel of our own (Triton): half-precision factors (11 significant bits), single-precision sums. With
          ``SPLIT`` the gradient's products take the tensor cores as well. A gradient is not whole, so the kernel
          scales each row of a block of it by a power of two and splits it into two half-precision parts, the
          leading 11 bits and the next 11, and adds both products. An entry within 2 ** -27 of the largest in its
          row of the block keeps 22 bits, which is single precision's own rounding of such a sum; a smaller one
          keeps at least 11 bits down to 2 ** -38 of the largest, fewer below, and nothing below 2 ** -49.
  None    PyTorch's products in single precision (the default).

``AUDIT`` works every product out both ways and compares them to the bit. The trainer switches it on for an update
and an evaluation batch now and then, so that a factor beyond ``LARGEST`` does not go unseen.

On a machine without a CUDA device the modes change nothing: "tf32" is a switch of cuBLAS, and the kernel needs
Triton, which runs on the CPU only in its interpreter (TRITON_INTERPRET=1, for tests; there the kernel's factors take
single precision).
"""
from __future__ import annotations

import os

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # no Triton: "half" has no kernel and leaves the products as they are
    triton = None

_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"

# None, "tf32" or "half" (see above), and whether "half" splits the gradient's products too.
MODE = None
SPLIT = False
# Every product both ways, compared to the bit (slow: for a pass now and then).
AUDIT = False
# The largest factor that 11 significant bits hold exactly whatever it is: every whole number up to here.
LARGEST = 2048
# Products that took the tensor cores since the module was loaded, by mode: for tests and reports.
COUNTS = {"tf32": 0, "half": 0}

# The kernel's blocks: rows and columns of the product and the stretch of the inner sum that one step takes. Blocks
# of rows are taken in groups, so that neighbouring programs share what they load.
BLOCKS = dict(BM=32, BN=32, BK=16, GROUP=2) if _INTERPRET else dict(BM=128, BN=128, BK=32, GROUP=8)
WARPS, STAGES = 4, 3
# A gradient's row is scaled so that its largest entry lies in [2 ** _TOP, 2 ** (_TOP + 1)): two binades below
# half precision's largest number, with 27 binades below it in which an entry keeps 22 bits.
_TOP = 13

_MATMUL = torch.backends.cuda.matmul


class _TensorCorePrecision:
    """cuBLAS products in the tensor cores' single precision (TF32) inside the block, and as before after it."""

    __slots__ = ("before",)

    def __enter__(self):
        self.before = _MATMUL.fp32_precision
        _MATMUL.fp32_precision = "tf32"

    def __exit__(self, *error):
        _MATMUL.fp32_precision = self.before


def _cublas(tensor: torch.Tensor) -> bool:
    """Whether this tensor's products are cuBLAS's: on a CUDA device, in single precision."""
    return tensor.is_cuda and tensor.dtype == torch.float32


def kernel_available(tensor: torch.Tensor) -> bool:
    """Whether the kernel can take this tensor: Triton on a CUDA device, or its interpreter on the CPU."""
    return triton is not None and tensor.dtype == torch.float32 and (
        tensor.is_cuda or (_INTERPRET and tensor.device.type == "cpu"))


def _same(product, plain, what):
    if not torch.equal(product, plain):
        worst = (product - plain).abs().max().item()
        raise RuntimeError(f"a {what} on the tensor cores ({MODE}) differs from the single-precision product by up to "
                           f"{worst:g}: a factor is not a whole number of at most {LARGEST}")


if triton is not None:

    @triton.jit
    def _dot(a, b, acc, HALF: tl.constexpr):
        """acc + a @ b: half-precision factors and single-precision sums on the GPU; in the interpreter, which has
        no tensor cores, the same factors in single precision."""
        if HALF:
            return tl.dot(a.to(tl.float16), b.to(tl.float16), acc, out_dtype=tl.float32)
        else:
            return acc + tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="ieee", out_dtype=tl.float32)

    @triton.jit
    def _product_kernel(a_ptr, b_ptr, c_ptr, M, N, K, a_row, a_col, b_row, b_col, c_row, c_col,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
                        TOP: tl.constexpr, SPLIT: tl.constexpr, HALF: tl.constexpr):
        """c[M, N] = a[M, K] @ b[K, N], each given by its first entry and the strides of its rows and columns.

        ``b`` holds whole numbers of at most 2,048. Without SPLIT ``a`` does too, and every sum is exact. With SPLIT
        ``a`` is a gradient: each row of a block of it is scaled by a power of two that brings its largest entry to
        2 ** TOP, and split into its leading 11 bits and, 2 ** 11 times larger so that half precision holds what is
        left of small entries too, the next 11: two half-precision numbers whose products with ``b`` are added in
        single precision."""
        pid = tl.program_id(0)
        rows = tl.cdiv(M, BM)
        cols = tl.cdiv(N, BN)
        in_group = GROUP * cols
        first = (pid // in_group) * GROUP
        size = tl.minimum(rows - first, GROUP)
        m = (first + (pid % in_group) % size) * BM + tl.arange(0, BM)
        n = ((pid % in_group) // size) * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        a_ptrs = a_ptr + m.to(tl.int64)[:, None] * a_row + k.to(tl.int64)[None, :] * a_col
        b_ptrs = b_ptr + k.to(tl.int64)[:, None] * b_row + n.to(tl.int64)[None, :] * b_col
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for start in range(0, K, BK):
            inside = start + k < K
            a = tl.load(a_ptrs, mask=(m < M)[:, None] & inside[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=inside[:, None] & (n < N)[None, :], other=0.0)
            if SPLIT:
                a = a.to(tl.float32)
                largest = tl.max(tl.abs(a), 1)
                binade = tl.where(largest > 0, tl.floor(tl.log2(largest)), TOP)
                scale = tl.exp2(tl.minimum(tl.maximum(TOP - binade, -100.0), 100.0))
                scaled = a * scale[:, None]
                lead = scaled.to(tl.float16)
                # What the leading bits leave is at most 2 ** (TOP + 1 - 11): 2,048 times that fits half precision.
                rest = ((scaled - lead.to(tl.float32)) * 2048.0).to(tl.float16)
                part = _dot(lead, b, tl.zeros((BM, BN), dtype=tl.float32), HALF)
                part += _dot(rest, b, tl.zeros((BM, BN), dtype=tl.float32), HALF) * (1.0 / 2048.0)
                acc += part / scale[:, None]
            else:
                acc = _dot(a, b, acc, HALF)
            a_ptrs += BK * a_col
            b_ptrs += BK * b_row
        c_ptrs = c_ptr + m.to(tl.int64)[:, None] * c_row + n.to(tl.int64)[None, :] * c_col
        tl.store(c_ptrs, acc, mask=(m < M)[:, None] & (n < N)[None, :])


def product(a: torch.Tensor, b: torch.Tensor, split: bool = False) -> torch.Tensor:
    """``a [M, K] @ b [K, N]`` through the kernel, as a new single-precision tensor. ``b`` holds whole numbers of at
    most ``LARGEST``; ``a`` does too, or with ``split`` it is a gradient. Either may be a view with strides of its
    own (a transpose costs nothing)."""
    rows, inner = a.shape
    columns = b.shape[1]
    if b.shape[0] != inner:
        raise ValueError(f"a product of {tuple(a.shape)} and {tuple(b.shape)}")
    out = torch.empty((rows, columns), dtype=torch.float32, device=a.device)
    if rows == 0 or columns == 0:
        return out
    if inner == 0:
        return out.zero_()
    grid = (triton.cdiv(rows, BLOCKS["BM"]) * triton.cdiv(columns, BLOCKS["BN"]),)
    options = {} if _INTERPRET else {"enable_fp_fusion": False, "num_warps": WARPS, "num_stages": STAGES}
    _product_kernel[grid](a, b, out, rows, columns, inner, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                          out.stride(0), out.stride(1), TOP=float(_TOP), SPLIT=split, HALF=not _INTERPRET,
                          **BLOCKS, **options)
    return out


class _Linear(torch.autograd.Function):
    """``F.linear(counts, ticks)`` of whole counts [positions, fan] and whole ticks [cells, fan] through the kernel.
    The gradient is PyTorch's in single precision, or with ``SPLIT`` the kernel's from two half-precision parts."""

    @staticmethod
    def forward(ctx, counts, ticks):
        ctx.save_for_backward(counts, ticks)
        return product(counts, ticks.t())

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        counts, ticks = ctx.saved_tensors
        to_counts = to_ticks = None
        if ctx.needs_input_grad[0]:
            to_counts = product(grad, ticks, split=True) if SPLIT else grad @ ticks
        if ctx.needs_input_grad[1]:
            to_ticks = product(grad.t(), counts, split=True) if SPLIT else grad.t() @ counts
        return to_counts, to_ticks


def _kernel_takes(counts, ticks):
    return (kernel_available(counts) and ticks.dtype == torch.float32 and ticks.device == counts.device
            and counts.ndim >= 2 and counts.numel() > 0 and ticks.numel() > 0)


def linear(counts: torch.Tensor, ticks: torch.Tensor) -> torch.Tensor:
    """``F.linear(counts, ticks)``: whole counts [..., fan] through whole ticks [cells, fan], both of at most
    ``LARGEST`` in size, with every sum inside single precision's exact range."""
    if MODE == "tf32" and _cublas(counts):
        with _TensorCorePrecision():
            charge = F.linear(counts, ticks)
    elif MODE == "half" and _kernel_takes(counts, ticks):
        charge = _Linear.apply(counts.reshape(-1, counts.shape[-1]), ticks).unflatten(0, counts.shape[:-1])
    else:
        return F.linear(counts, ticks)
    COUNTS[MODE] += 1
    if AUDIT:
        _same(charge, F.linear(counts, ticks), "product of counts and ticks")
    return charge


def by_cluster(counts: torch.Tensor, ticks: torch.Tensor) -> torch.Tensor:
    """``torch.einsum("btgk,gnk->btgn", counts, ticks)``: every cluster's whole counts [batch, bytes, clusters, fan]
    through its cells' whole ticks [clusters, cells, fan], as ``linear`` takes them."""
    if MODE == "tf32" and _cublas(counts):
        with _TensorCorePrecision():
            charge = torch.einsum("btgk,gnk->btgn", counts, ticks)
    elif MODE == "half" and ticks.shape[0] == 1 and _kernel_takes(counts, ticks):
        charge = _Linear.apply(counts.reshape(-1, counts.shape[-1]), ticks[0]).unflatten(0, counts.shape[:-1])
    else:
        return torch.einsum("btgk,gnk->btgn", counts, ticks)
    COUNTS[MODE] += 1
    if AUDIT:
        _same(charge, torch.einsum("btgk,gnk->btgn", counts, ticks), "product of a cluster's counts and ticks")
    return charge


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` of whole numbers of at most ``LARGEST`` whose sums stay inside the exact range: a read's pulses.
    They take the tensor cores under "tf32" alone: they are tables that memory holds back, not the adding."""
    if MODE == "tf32" and _cublas(a):
        with _TensorCorePrecision():
            dots = a @ b
        COUNTS["tf32"] += 1
        if AUDIT:
            _same(dots, a @ b, "product of pulses")
        return dots
    return a @ b
