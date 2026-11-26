"""Products of whole numbers on a GPU's tensor cores, to the bit.

Every charge in the model is a sum of whole spike counts times whole synapse sizes that stays inside the range in
which single precision counts exactly (below 2 ** 24). Such a sum is the same number in whatever order a device adds
it and in whatever precision, as long as it holds every factor and every partial sum exactly. PyTorch works these
products out in single precision on a GPU's general cores: the faster format of the tensor cores (TF32: 11
significant bits of each factor, single-precision sums) is off by default, because it rounds other numbers. The
factors here are small whole numbers, at most ``LARGEST`` in size, which that format holds exactly.

With ``ON``, PyTorch's own products (cuBLAS) of such factors take the tensor cores' single precision, switched on for
these products alone and off again after each. The backward pass is PyTorch's in full single precision, as without:
the spikes and the gradient are the same to the bit, and so is a training run.

``AUDIT`` works every product out both ways and compares them to the bit. The trainer switches it on for an update
and an evaluation batch now and then, so that a factor beyond ``LARGEST`` does not go unseen.

On a machine without a CUDA device the switch changes nothing: it is a switch of cuBLAS.

Measured on an RTX PRO 6000 (2026-11-26, the full-size stack): a product takes a third to a quarter of its time
(4,096 x 2,048 -> 2,048: 184 microseconds for 593), an update on windows 0.176 s for 0.178 s and a stream's update
0.404 s for 0.415 s, because an update spends most of its time elsewhere. Tried beside it and dropped: a kernel of
our own with half-precision factors (the same speed as TF32; its hand-written gradient products were PyTorch's to the
bit at full size but not on the test bed), and the gradient's products split into two half-precision parts (2 to 9
times slower than the single-precision products they replace).
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

# Whether products of whole numbers take the tensor cores, and whether each is checked against single precision.
ON = False
AUDIT = False
# The largest whole number a factor may be: 11 significant bits.
LARGEST = 2048
# Products taken on the tensor cores so far.
COUNT = 0

_MATMUL = torch.backends.cuda.matmul


class _TensorCorePrecision:
    """cuBLAS's products with TF32 inside the block; the setting that stood before, after it."""

    __slots__ = ("before",)

    def __enter__(self):
        self.before = _MATMUL.fp32_precision
        _MATMUL.fp32_precision = "tf32"

    def __exit__(self, *error):
        _MATMUL.fp32_precision = self.before


def _cublas(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and tensor.dtype == torch.float32


def _taken(product: torch.Tensor, plain, what: str) -> torch.Tensor:
    """Count a product taken on the tensor cores; under AUDIT, compare it with the single-precision one."""
    global COUNT
    COUNT += 1
    if AUDIT:
        single = plain()
        if not torch.equal(product, single):
            worst = (product - single).abs().max().item()
            raise RuntimeError(f"a {what} on the tensor cores differs from the single-precision product by up to "
                               f"{worst:g}: a factor is not a whole number of at most {LARGEST}")
    return product


def linear(counts: torch.Tensor, ticks: torch.Tensor) -> torch.Tensor:
    """F.linear(counts, ticks) of whole counts [..., fan] and whole ticks [cells, fan]."""
    if not (ON and _cublas(counts)):
        return F.linear(counts, ticks)
    with _TensorCorePrecision():
        charge = F.linear(counts, ticks)
    return _taken(charge, lambda: F.linear(counts, ticks), "product of counts and ticks")


def by_cluster(counts: torch.Tensor, ticks: torch.Tensor) -> torch.Tensor:
    """torch.einsum("btgk,gnk->btgn", counts, ticks) of whole counts [batch, bytes, clusters, fan] and whole ticks
    [clusters, cells, fan]."""
    if not (ON and _cublas(counts)):
        return torch.einsum("btgk,gnk->btgn", counts, ticks)
    with _TensorCorePrecision():
        charge = torch.einsum("btgk,gnk->btgn", counts, ticks)
    return _taken(charge, lambda: torch.einsum("btgk,gnk->btgn", counts, ticks),
                  "product of a cluster's counts and ticks")


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b of whole pulses (a read's probes against its tags, its spikes against its payloads)."""
    if not (ON and _cublas(a)):
        return a @ b
    with _TensorCorePrecision():
        dots = a @ b
    return _taken(dots, lambda: a @ b, "product of pulses")
