"""A stream's budgeted race read in fused GPU kernels (Triton): the spikes of _StateRead._address_band exactly, without
the tables of every byte's cells by age that the read by age builds.

The read by age (snnformer/event_lm.py) works out, for each byte of a window and each of its cells within reach, a
charge, the race's growth, rate and spikes, and the payload pulses those spikes pass on, each as a tensor of
[batch, heads, bytes, reach]: at full size 268 MB apiece, written and read again a dozen times in each read. These
kernels take a block of bytes against a block of cells at a time, as flash attention does for attention: only the
race's growth is written, once, for its total, and the backward pass works its charges out again where it needs them.

Exactness. The forward pass gives the read by age's values to the bit:
  - a charge is a dot product of whole pulses (at most 128 on a channel, 64 channels), exact in half-precision inputs
    with single-precision sums in any order; it is then multiplied by its head's scale and added to its age's offset,
    rounded after each as PyTorch does (fused multiply-adds are switched off);
  - the growth exp(charge - peak) is CUDA's own expf (libdevice), which PyTorch's exp is;
  - the total, the one sum whose order matters, is PyTorch's own sum over the growth in the read by age's layout;
  - a rate is divided with IEEE rounding (div_rn), and the spikes and the payload pulses they pass on are whole numbers
    whose sums are exact.
The backward pass is the read by age's gradient (the rates' own derivative, as _BudgetedRace takes it) to rounding.
Its products of a gradient's real numbers with whole pulses run on the tensor cores, each real factor in two parts
(_real_dot), right to 2 ** -22 of each term: in one part they were 1.2e-3 off at full size, and in single precision
they were the kernels' slowest step. At full size one update's gradients are within 7e-6 of each tensor's largest
entry of the read by age's, and the same from run to run.

Measured on an RTX PRO 6000 (2026-11-26, the full-size stack alone on the GPU): a read with its backward pass takes
3.2 ms for the read by age's 15.9 ms, a stream's update 0.179 s for 0.416 s, and the update's memory peaks at about
9 GiB for 41 GiB. A run trained with these reads fires the read by age's spikes at every weight, but it is another
run: gradients that differ in their rounding move the weights apart within two updates (after 14 updates by as much
as the updates moved them), as reads by age do against reads by cell.

On a machine without Triton (the laptop) the read by age stays. With TRITON_INTERPRET=1 the kernels run on the CPU, for
tests: there exp is NumPy's and the dot products take single precision, so a spike may differ where a rate falls within
a rounding of a whole number.
"""
from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    from triton.runtime.errors import OutOfResources
except ImportError:  # no Triton: the read by age stays
    triton = None

_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
# For each kernel: the bytes and the cells in a block (a block of bytes takes its cells within reach a block at a
# time, a block of cells its bytes), its warps, and the stages of its loop's pipeline. Timed on an RTX PRO 6000 at the
# full-size read's shape (2 streams, 8 heads of 64 channels, 2,048 bytes against 4,096 cells within a reach of 2,048,
# 128 knowledge cells): 0.40, 0.24, 1.44 and 0.62 ms a kernel. Triton keeps the tiles that a loop loads in the GPU's
# shared memory, one copy for each stage, and the larger a block's tiles, the fewer blocks a multiprocessor runs at a
# time: blocks of 64 by 64 were up to ten times slower in the backward pass, or did not fit (122,880 bytes of
# 101,376). The arithmetic and its order do not depend on the warps or the stages; the blocks set the order in which
# the backward pass adds, to rounding.
TUNING = {"grown": (32, 64, 4, 1), "fire": (32, 64, 4, 1), "back_rows": (16, 64, 4, 1), "back_cells": (32, 32, 4, 1)}
if _INTERPRET:  # small blocks, so that the tests' small reads cross block borders
    TUNING = {name: (16, 16, 4, 1) for name in TUNING}
# Whether a backward kernel takes its gradient's products on the tensor cores, each in two parts (_real_dot).
SPLIT = {"back_rows": True, "back_cells": True}


def available(tensor: torch.Tensor) -> bool:
    """Whether the kernels can read this tensor's stream: Triton on a CUDA device, or its interpreter on the CPU."""
    return triton is not None and tensor.dtype == torch.float32 and (
        tensor.is_cuda or (_INTERPRET and tensor.device.type == "cpu"))


def fits(channels: int, payload_channels: int, knowledge: int) -> bool:
    """Whether a read's shape suits the kernels: probes, tags, payloads and knowledge cells in powers of two."""
    def power(n):
        return n >= 16 and n & (n - 1) == 0
    return power(channels) and power(payload_channels) and (knowledge == 0 or power(knowledge))


if triton is not None:

    @triton.jit
    def _grow(x, EXACT: tl.constexpr):
        if EXACT:
            return libdevice.exp(x)
        else:
            return tl.exp(x)

    @triton.jit
    def _pulse_dot(a, b, HALF: tl.constexpr):
        """a @ b of whole pulses: exact in half-precision inputs (on the GPU) or in single precision (interpreter)."""
        if HALF:
            return tl.dot(a.to(tl.float16), b.to(tl.float16), out_dtype=tl.float32)
        else:
            return tl.dot(a, b, input_precision="ieee", out_dtype=tl.float32)

    @triton.jit
    def _charges(probe16, tag_ptr, offset_ptr, scale, bh, h, t, rows, c0, cells, kept, R,
                 D: tl.constexpr, BC: tl.constexpr, HALF: tl.constexpr):
        """A block of bytes against a block of cells: the dot products, the charges (-inf where a cell is out of
        reach) and each pair's age and whether it is in reach."""
        c = c0 + tl.arange(0, BC)
        d = tl.arange(0, D)
        tag = tl.load(tag_ptr + (bh * cells + c)[:, None] * D + d[None, :], mask=(c < cells)[:, None], other=0.0)
        dots = _pulse_dot(probe16, tl.trans(tag), HALF)
        age = kept + t[:, None] - c[None, :]
        valid = (age >= 0) & (age < R) & rows[:, None] & (c < cells)[None, :]
        offset = tl.load(offset_ptr + h * R + age, mask=valid, other=0.0)
        charge = tl.where(valid, dots * scale + offset, float("-inf"))
        return c, tag, dots, age, valid, charge

    @triton.jit
    def _grown_kernel(probe_ptr, tag_ptr, offset_ptr, scale_ptr, kn_tag_ptr, kn_off_ptr,
                      grown_ptr, kn_grown_ptr, peak_ptr,
                      H, T, cells, kept, R,
                      D: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
                      KNOW: tl.constexpr, EXACT: tl.constexpr, HALF: tl.constexpr):
        """Each byte's peak charge and every cell's growth exp(charge - peak), by age, as _BudgetedRace has them."""
        bh = tl.program_id(0)
        h = bh % H
        t0 = tl.program_id(1) * BT
        t = t0 + tl.arange(0, BT)
        rows = t < T
        d = tl.arange(0, D)
        probe = tl.load(probe_ptr + (bh * T + t)[:, None] * D + d[None, :], mask=rows[:, None], other=0.0)
        scale = tl.load(scale_ptr + h)
        first = tl.maximum(kept + t0 - R + 1, 0)
        last = tl.minimum(kept + t0 + BT, cells)
        peak = tl.full([BT], float("-inf"), tl.float32)
        for c0 in range(first, last, BC):
            c, tag, dots, age, valid, charge = _charges(probe, tag_ptr, offset_ptr, scale, bh, h, t, rows, c0, cells,
                                                        kept, R, D, BC, HALF)
            peak = tl.maximum(peak, tl.max(charge, 1))
        if KNOW:
            k = tl.arange(0, K)
            kn_tag = tl.load(kn_tag_ptr + (h * K + k)[:, None] * D + d[None, :])
            kn_charge = _pulse_dot(probe, tl.trans(kn_tag), HALF) * scale + tl.load(kn_off_ptr + h * K + k)[None, :]
            peak = tl.maximum(peak, tl.max(kn_charge, 1))
        tl.store(peak_ptr + bh * T + t, peak, mask=rows)
        for c0 in range(first, last, BC):
            c, tag, dots, age, valid, charge = _charges(probe, tag_ptr, offset_ptr, scale, bh, h, t, rows, c0, cells,
                                                        kept, R, D, BC, HALF)
            grown = tl.where(valid, _grow(charge - peak[:, None], EXACT), 0.0)
            tl.store(grown_ptr + (bh * T + t)[:, None] * R + age, grown, mask=valid)
        if KNOW:
            kn_grown = _grow(kn_charge - peak[:, None], EXACT)
            tl.store(kn_grown_ptr + (bh * T + t)[:, None] * K + k[None, :], kn_grown, mask=rows[:, None])

    @triton.jit
    def _fire_kernel(grown_ptr, total_ptr, payload_ptr, phase_ptr, kn_grown_ptr, kn_payload_ptr, kn_phase_ptr,
                     packet_ptr, fired_total_ptr, kn_fired_ptr, budget,
                     H, T, cells, kept, R,
                     P: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
                     KNOW: tl.constexpr, HALF: tl.constexpr):
        """Every cell's spikes floor(grown / total * budget + phase) and the payload pulses that each byte's cells
        pass on, with each byte's number of spikes."""
        bh = tl.program_id(0)
        h = bh % H
        t0 = tl.program_id(1) * BT
        t = t0 + tl.arange(0, BT)
        rows = t < T
        p = tl.arange(0, P)
        total = tl.load(total_ptr + bh * T + t, mask=rows, other=1.0)
        first = tl.maximum(kept + t0 - R + 1, 0)
        last = tl.minimum(kept + t0 + BT, cells)
        packet = tl.zeros([BT, P], tl.float32)
        count = tl.zeros([BT], tl.float32)
        for c0 in range(first, last, BC):
            c = c0 + tl.arange(0, BC)
            age = kept + t[:, None] - c[None, :]
            valid = (age >= 0) & (age < R) & rows[:, None] & (c < cells)[None, :]
            grown = tl.load(grown_ptr + (bh * T + t)[:, None] * R + age, mask=valid, other=0.0)
            rate = tl.math.div_rn(grown, total[:, None]) * budget
            phase = tl.load(phase_ptr + h * cells + c, mask=c < cells, other=0.0)
            fired = tl.where(valid, tl.floor(rate + phase[None, :]), 0.0)
            payload = tl.load(payload_ptr + (bh * cells + c)[:, None] * P + p[None, :], mask=(c < cells)[:, None],
                              other=0.0)
            packet += _pulse_dot(fired, payload, HALF)
            count += tl.sum(fired, 1)
        if KNOW:
            k = tl.arange(0, K)
            kn_grown = tl.load(kn_grown_ptr + (bh * T + t)[:, None] * K + k[None, :], mask=rows[:, None], other=0.0)
            kn_rate = tl.math.div_rn(kn_grown, total[:, None]) * budget
            kn_fired = tl.where(rows[:, None], tl.floor(kn_rate + tl.load(kn_phase_ptr + h * K + k)[None, :]), 0.0)
            tl.store(kn_fired_ptr + (bh * T + t)[:, None] * K + k[None, :], kn_fired, mask=rows[:, None])
            kn_payload = tl.load(kn_payload_ptr + (h * K + k)[:, None] * P + p[None, :])
            packet += _pulse_dot(kn_fired, kn_payload, HALF)
            count += tl.sum(kn_fired, 1)
        tl.store(packet_ptr + (bh * T + t)[:, None] * P + p[None, :], packet, mask=rows[:, None])
        tl.store(fired_total_ptr + bh * T + t, count, mask=rows)

    @triton.jit
    def _real_dot(real, pulses, SPLIT: tl.constexpr):
        """real @ pulses: a gradient's real numbers against whole pulses. With SPLIT on the tensor cores, whose format
        (TF32) holds 11 significant bits of a factor: the pulses to the bit, and the real numbers in two parts that
        add up to them (Veltkamp's split: the leading 11 bits and the rest, of which that format drops one bit at
        most), so that each term is right to 2 ** -22 of itself. Without, in single precision."""
        if SPLIT:
            spread = real * 8193.0
            high = spread + (real - spread)
            return (tl.dot(high, pulses, input_precision="tf32")
                    + tl.dot(real - high, pulses, input_precision="tf32"))
        else:
            return tl.dot(real, pulses, input_precision="ieee")

    @triton.jit
    def _dot_real(pulses, real, SPLIT: tl.constexpr):
        """pulses @ real: as _real_dot, with the whole pulses on the left."""
        if SPLIT:
            spread = real * 8193.0
            high = spread + (real - spread)
            return (tl.dot(pulses, high, input_precision="tf32")
                    + tl.dot(pulses, real - high, input_precision="tf32"))
        else:
            return tl.dot(pulses, real, input_precision="ieee")

    @triton.jit
    def _rates(probe, tag_ptr, offset_ptr, payload_ptr, scale, peak, total, grad_packet, grad_total, budget,
               bh, h, t, rows, c0, cells, kept, R,
               D: tl.constexpr, P: tl.constexpr, BC: tl.constexpr, EXACT: tl.constexpr, HALF: tl.constexpr,
               SPLIT: tl.constexpr):
        """For a block of bytes against a block of cells: the forward pass's rates again, and the gradient that
        reaches each cell's spikes: from the payload pulses they pass on and from the bytes' spike counts."""
        c, tag, dots, age, valid, charge = _charges(probe, tag_ptr, offset_ptr, scale, bh, h, t, rows, c0, cells,
                                                    kept, R, D, BC, HALF)
        grown = tl.where(valid, _grow(charge - peak[:, None], EXACT), 0.0)
        rate = tl.math.div_rn(grown, total[:, None]) * budget
        p = tl.arange(0, P)
        payload = tl.load(payload_ptr + (bh * cells + c)[:, None] * P + p[None, :], mask=(c < cells)[:, None],
                          other=0.0)
        grad_fired = _real_dot(grad_packet, tl.trans(payload), SPLIT) + grad_total[:, None]
        return c, tag, dots, age, valid, rate, payload, grad_fired

    @triton.jit
    def _known(probe, kn_tag_ptr, kn_off_ptr, kn_payload_ptr, grad_kn_fired_ptr, scale, peak, total, grad_packet,
               grad_total, budget, bh, h, t, rows, k, T,
               D: tl.constexpr, P: tl.constexpr, K: tl.constexpr, GRAD_KN_FIRED: tl.constexpr, EXACT: tl.constexpr,
               HALF: tl.constexpr, SPLIT: tl.constexpr):
        """For a block of bytes against a block of knowledge cells: the cells' tags, dot products and rates, and the
        gradient that reaches each cell's spikes."""
        d = tl.arange(0, D)
        p = tl.arange(0, P)
        tag = tl.load(kn_tag_ptr + (h * K + k)[:, None] * D + d[None, :])
        dots = _pulse_dot(probe, tl.trans(tag), HALF)
        charge = dots * scale + tl.load(kn_off_ptr + h * K + k)[None, :]
        rate = tl.math.div_rn(_grow(charge - peak[:, None], EXACT), total[:, None]) * budget
        rate = tl.where(rows[:, None], rate, 0.0)
        payload = tl.load(kn_payload_ptr + (h * K + k)[:, None] * P + p[None, :])
        grad_fired = _real_dot(grad_packet, tl.trans(payload), SPLIT) + grad_total[:, None]
        if GRAD_KN_FIRED:
            grad_fired += tl.load(grad_kn_fired_ptr + (bh * T + t)[:, None] * K + k[None, :], mask=rows[:, None],
                                  other=0.0)
        return tag, dots, rate, grad_fired

    @triton.jit
    def _back_rows_kernel(probe_ptr, tag_ptr, payload_ptr, offset_ptr, scale_ptr, kn_tag_ptr, kn_off_ptr,
                          kn_payload_ptr, kn_phase_ptr, peak_ptr, total_ptr, grad_packet_ptr, grad_total_ptr,
                          grad_kn_fired_ptr, budget,
                          grad_probe_ptr, grad_charge_ptr, share_ptr, scale_part_ptr, kn_tag_part_ptr,
                          kn_off_part_ptr, kn_pay_part_ptr,
                          H, T, cells, kept, R, NT,
                          D: tl.constexpr, P: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
                          BK: tl.constexpr, KNOW: tl.constexpr, GRAD_KN_FIRED: tl.constexpr, EXACT: tl.constexpr,
                          HALF: tl.constexpr, SPLIT: tl.constexpr):
        """The backward pass for a block of bytes: each byte's share (the rates' derivative), the gradients of its
        charges (by age, for the age offsets), of its probe, of the heads' scale, and this block's parts of the
        knowledge cells' gradients. The knowledge cells are taken a block at a time, as the cells within reach are:
        taken whole, their tiles are the kernel's largest, and fewer blocks of bytes then run at a time."""
        bh = tl.program_id(0)
        h = bh % H
        b = bh // H
        tb = tl.program_id(1)
        t0 = tb * BT
        t = t0 + tl.arange(0, BT)
        rows = t < T
        d = tl.arange(0, D)
        p = tl.arange(0, P)
        probe = tl.load(probe_ptr + (bh * T + t)[:, None] * D + d[None, :], mask=rows[:, None], other=0.0)
        scale = tl.load(scale_ptr + h)
        peak = tl.load(peak_ptr + bh * T + t, mask=rows, other=0.0)
        total = tl.load(total_ptr + bh * T + t, mask=rows, other=1.0)
        grad_packet = tl.load(grad_packet_ptr + (bh * T + t)[:, None] * P + p[None, :], mask=rows[:, None], other=0.0)
        grad_total = tl.load(grad_total_ptr + bh * T + t, mask=rows, other=0.0)
        first = tl.maximum(kept + t0 - R + 1, 0)
        last = tl.minimum(kept + t0 + BT, cells)
        share = tl.zeros([BT], tl.float32)
        for c0 in range(first, last, BC):
            c, tag, dots, age, valid, rate, payload, grad_fired = _rates(
                probe, tag_ptr, offset_ptr, payload_ptr, scale, peak, total, grad_packet, grad_total, budget,
                bh, h, t, rows, c0, cells, kept, R, D, P, BC, EXACT, HALF, SPLIT)
            share += tl.sum(tl.where(valid, grad_fired * rate, 0.0), 1)
        if KNOW:
            for k0 in range(0, K, BK):
                k = k0 + tl.arange(0, BK)
                kn_tag, kn_dots, kn_rate, kn_grad = _known(
                    probe, kn_tag_ptr, kn_off_ptr, kn_payload_ptr, grad_kn_fired_ptr, scale, peak, total,
                    grad_packet, grad_total, budget, bh, h, t, rows, k, T, D, P, K, GRAD_KN_FIRED, EXACT, HALF, SPLIT)
                share += tl.sum(kn_grad * kn_rate, 1)
        share = share / budget
        tl.store(share_ptr + bh * T + t, share, mask=rows)
        grad_probe = tl.zeros([BT, D], tl.float32)
        scale_sum = tl.zeros([BT], tl.float32)
        for c0 in range(first, last, BC):
            c, tag, dots, age, valid, rate, payload, grad_fired = _rates(
                probe, tag_ptr, offset_ptr, payload_ptr, scale, peak, total, grad_packet, grad_total, budget,
                bh, h, t, rows, c0, cells, kept, R, D, P, BC, EXACT, HALF, SPLIT)
            grad_charge = tl.where(valid, rate * (grad_fired - share[:, None]), 0.0)
            tl.store(grad_charge_ptr + (bh * T + t)[:, None] * R + age, grad_charge, mask=valid)
            grad_probe += _real_dot(grad_charge * scale, tag, SPLIT)
            scale_sum += tl.sum(grad_charge * dots, 1)
        if KNOW:
            part = (b * NT + tb) * H + h
            for k0 in range(0, K, BK):
                k = k0 + tl.arange(0, BK)
                kn_tag, kn_dots, kn_rate, kn_grad = _known(
                    probe, kn_tag_ptr, kn_off_ptr, kn_payload_ptr, grad_kn_fired_ptr, scale, peak, total,
                    grad_packet, grad_total, budget, bh, h, t, rows, k, T, D, P, K, GRAD_KN_FIRED, EXACT, HALF, SPLIT)
                kn_grad_charge = kn_rate * (kn_grad - share[:, None])
                grad_probe += _real_dot(kn_grad_charge * scale, kn_tag, SPLIT)
                scale_sum += tl.sum(kn_grad_charge * kn_dots, 1)
                tl.store(kn_tag_part_ptr + (part * K + k)[:, None] * D + d[None, :],
                         _real_dot(tl.trans(kn_grad_charge * scale), probe, SPLIT))
                tl.store(kn_off_part_ptr + part * K + k, tl.sum(kn_grad_charge, 0))
                kn_fired = tl.where(rows[:, None], tl.floor(kn_rate + tl.load(kn_phase_ptr + h * K + k)[None, :]),
                                    0.0)
                tl.store(kn_pay_part_ptr + (part * K + k)[:, None] * P + p[None, :],
                         _dot_real(tl.trans(kn_fired), grad_packet, SPLIT))
        tl.store(grad_probe_ptr + (bh * T + t)[:, None] * D + d[None, :], grad_probe, mask=rows[:, None])
        tl.store(scale_part_ptr + bh * NT + tb, tl.sum(scale_sum, 0))

    @triton.jit
    def _back_cells_kernel(probe_ptr, tag_ptr, payload_ptr, offset_ptr, scale_ptr, phase_ptr, peak_ptr, total_ptr,
                           share_ptr, grad_packet_ptr, grad_total_ptr, budget, grad_tag_ptr, grad_payload_ptr,
                           H, T, cells, kept, R,
                           D: tl.constexpr, P: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
                           EXACT: tl.constexpr, HALF: tl.constexpr, SPLIT: tl.constexpr):
        """The backward pass for a block of cells: the gradients of their tags and payloads, from every byte that
        reaches them."""
        bh = tl.program_id(0)
        h = bh % H
        c0 = tl.program_id(1) * BC
        c = c0 + tl.arange(0, BC)
        cols = c < cells
        d = tl.arange(0, D)
        p = tl.arange(0, P)
        tag = tl.load(tag_ptr + (bh * cells + c)[:, None] * D + d[None, :], mask=cols[:, None], other=0.0)
        payload = tl.load(payload_ptr + (bh * cells + c)[:, None] * P + p[None, :], mask=cols[:, None], other=0.0)
        scale = tl.load(scale_ptr + h)
        phase = tl.load(phase_ptr + h * cells + c, mask=cols, other=0.0)
        first = tl.maximum(c0 - kept, 0)
        last = tl.minimum(c0 + BC - kept + R - 1, T)
        grad_tag = tl.zeros([BC, D], tl.float32)
        grad_payload = tl.zeros([BC, P], tl.float32)
        for t0 in range(first, last, BT):
            t = t0 + tl.arange(0, BT)
            rows = t < T
            probe = tl.load(probe_ptr + (bh * T + t)[:, None] * D + d[None, :], mask=rows[:, None], other=0.0)
            dots = _pulse_dot(probe, tl.trans(tag), HALF)
            age = kept + t[:, None] - c[None, :]
            valid = (age >= 0) & (age < R) & rows[:, None] & cols[None, :]
            offset = tl.load(offset_ptr + h * R + age, mask=valid, other=0.0)
            charge = tl.where(valid, dots * scale + offset, float("-inf"))
            peak = tl.load(peak_ptr + bh * T + t, mask=rows, other=0.0)
            total = tl.load(total_ptr + bh * T + t, mask=rows, other=1.0)
            rate = tl.math.div_rn(tl.where(valid, _grow(charge - peak[:, None], EXACT), 0.0), total[:, None]) * budget
            fired = tl.where(valid, tl.floor(rate + phase[None, :]), 0.0)
            grad_packet = tl.load(grad_packet_ptr + (bh * T + t)[:, None] * P + p[None, :], mask=rows[:, None],
                                  other=0.0)
            grad_fired = (_real_dot(grad_packet, tl.trans(payload), SPLIT)
                          + tl.load(grad_total_ptr + bh * T + t, mask=rows, other=0.0)[:, None])
            share = tl.load(share_ptr + bh * T + t, mask=rows, other=0.0)
            grad_charge = tl.where(valid, rate * (grad_fired - share[:, None]), 0.0)
            grad_tag += _real_dot(tl.trans(grad_charge * scale), probe, SPLIT)
            grad_payload += _dot_real(tl.trans(fired), grad_packet, SPLIT)
        tl.store(grad_tag_ptr + (bh * cells + c)[:, None] * D + d[None, :], grad_tag, mask=cols[:, None])
        tl.store(grad_payload_ptr + (bh * cells + c)[:, None] * P + p[None, :], grad_payload, mask=cols[:, None])


def _launch(kernel, name, grid, *args, **meta):
    """Launch a kernel over its grid with its tuned blocks and warps; with fewer stages where its tiles do not fit the
    GPU's shared memory (STAGES then holds what it runs with)."""
    block_bytes, block_cells, warps, stages = TUNING[name]
    meta = dict(meta, BT=block_bytes, BC=block_cells)
    if _INTERPRET:
        kernel[grid](*args, **meta)
        return
    key = (name, tuple(sorted(meta.items())), warps)
    stages = STAGES.get(key, stages)
    while True:
        try:
            kernel[grid](*args, **meta, enable_fp_fusion=False, num_warps=warps, num_stages=stages)
            break
        except OutOfResources:
            if stages == 1:
                raise
            stages -= 1
    STAGES[key] = stages


# The stages each kernel has run with, by its name, shape and warps.
STAGES = {}


def _by_age(probe, steps, kept, reach):
    """A table of every byte's cells by age [batch, heads, bytes, reach] for a kernel to fill. The kernels write every
    age that holds a cell; only a stream's first bytes have ages before its first cell, and those bytes start at
    zero (at full size a table is 268 MB, and filling it whole took as long as a forward kernel)."""
    table = probe.new_empty(probe.shape[0], probe.shape[1], steps, reach)
    table[:, :, :max(0, reach - 1 - kept)].zero_()
    return table


class FusedRace(torch.autograd.Function):
    """A stream read's budgeted race in fused kernels: from the probes [batch, heads, bytes, channels], the tags and
    payloads of the tape's and the window's cells [batch, heads, cells, channels], each head's scale [heads], the
    offsets of the ages [heads, reach], the knowledge cells' tags, payloads and offsets (or None), and the phases of
    the cells [heads, cells] and of the knowledge cells [heads, knowledge]; to the payload pulses each byte's spikes
    pass on [batch, heads, bytes, payload channels], each byte's number of spikes [batch, heads, bytes] and the
    knowledge cells' spikes [batch, heads, bytes, knowledge] (or None)."""

    @staticmethod
    def forward(ctx, probe, tag, payload, scale, offset, kn_tag, kn_payload, kn_off, phase, kn_phase, kept, reach,
                budget):
        probe, tag, payload = probe.contiguous(), tag.contiguous(), payload.contiguous()
        batch, heads, steps, channels = probe.shape
        cells, payload_channels = tag.shape[2], payload.shape[3]
        knowledge = 0 if kn_tag is None else kn_tag.shape[1]
        scale_shape = scale.shape
        scale, offset, phase = scale.reshape(heads).contiguous(), offset.contiguous(), phase.contiguous()
        none = probe.new_zeros(1)
        kn = ((kn_tag.contiguous(), kn_payload.contiguous(), kn_off.contiguous(),
               kn_phase.reshape(heads, -1).contiguous()) if knowledge else (none, none, none, none))
        grown = _by_age(probe, steps, kept, reach)
        kn_grown = probe.new_empty(batch, heads, steps, knowledge) if knowledge else none
        peak = probe.new_empty(batch, heads, steps)
        shape = dict(K=max(knowledge, 16), KNOW=bool(knowledge), HALF=not _INTERPRET)
        _launch(_grown_kernel, "grown", (batch * heads, triton.cdiv(steps, TUNING["grown"][0])),
                probe, tag, offset, scale, kn[0], kn[2], grown, kn_grown, peak,
                heads, steps, cells, kept, reach, D=channels, EXACT=not _INTERPRET, **shape)
        # The race's total runs over the cells in the read by age's order and layout, then the knowledge cells.
        total = grown.sum(-1, keepdim=True)
        if knowledge:
            total = total + kn_grown.sum(-1, keepdim=True)
        total = total.reshape(batch, heads, steps)
        packet = probe.new_empty(batch, heads, steps, payload_channels)
        fired_total = probe.new_empty(batch, heads, steps)
        kn_fired = probe.new_empty(batch, heads, steps, knowledge) if knowledge else none
        _launch(_fire_kernel, "fire", (batch * heads, triton.cdiv(steps, TUNING["fire"][0])),
                grown, total, payload, phase, kn_grown, kn[1], kn[3], packet, fired_total,
                kn_fired, float(budget), heads, steps, cells, kept, reach, P=payload_channels, **shape)
        ctx.save_for_backward(probe, tag, payload, scale, offset, *kn, phase, peak, total)
        ctx.dims = (kept, reach, float(budget), knowledge, scale_shape)
        return packet, fired_total, (kn_fired if knowledge else None)

    @staticmethod
    def backward(ctx, grad_packet, grad_fired_total, grad_kn_fired):
        probe, tag, payload, scale, offset, kn_tag, kn_payload, kn_off, kn_phase, phase, peak, total = \
            ctx.saved_tensors
        kept, reach, budget, knowledge, scale_shape = ctx.dims
        batch, heads, steps, channels = probe.shape
        cells, payload_channels = tag.shape[2], payload.shape[3]
        grad_packet = (torch.zeros_like(probe[..., :1]).expand(-1, -1, -1, payload_channels) if grad_packet is None
                       else grad_packet).contiguous()
        grad_fired_total = (torch.zeros_like(total) if grad_fired_total is None else grad_fired_total).contiguous()
        with_kn_grad = knowledge and grad_kn_fired is not None
        grad_kn_fired = grad_kn_fired.contiguous() if with_kn_grad else probe.new_zeros(1)
        blocks = triton.cdiv(steps, TUNING["back_rows"][0])
        grad_probe = torch.empty_like(probe)
        grad_charge = _by_age(probe, steps, kept, reach)
        share = probe.new_empty(batch, heads, steps)
        scale_part = probe.new_empty(batch * heads * blocks)
        kn_parts = ((probe.new_empty(batch * blocks * heads, knowledge, channels),
                     probe.new_empty(batch * blocks * heads, knowledge),
                     probe.new_empty(batch * blocks * heads, knowledge, payload_channels))
                    if knowledge else (probe.new_zeros(1),) * 3)
        shape = dict(D=channels, P=payload_channels, EXACT=not _INTERPRET, HALF=not _INTERPRET)
        _launch(_back_rows_kernel, "back_rows", (batch * heads, blocks), probe, tag, payload, offset, scale, kn_tag, kn_off,
                kn_payload, kn_phase, peak, total, grad_packet, grad_fired_total, grad_kn_fired, budget,
                grad_probe, grad_charge, share, scale_part, *kn_parts,
                heads, steps, cells, kept, reach, blocks, K=max(knowledge, 16),
                BK=min(max(knowledge, 16), TUNING["back_rows"][1]), KNOW=bool(knowledge),
                GRAD_KN_FIRED=bool(with_kn_grad), SPLIT=SPLIT["back_rows"], **shape)
        grad_tag, grad_payload = torch.empty_like(tag), torch.empty_like(payload)
        _launch(_back_cells_kernel, "back_cells", (batch * heads, triton.cdiv(cells, TUNING["back_cells"][1])),
                probe, tag, payload, offset, scale,
                phase, peak, total, share, grad_packet, grad_fired_total, budget, grad_tag, grad_payload,
                heads, steps, cells, kept, reach, SPLIT=SPLIT["back_cells"], **shape)
        grad_scale = scale_part.view(batch, heads, blocks).sum((0, 2)).reshape(scale_shape)
        grad_offset = grad_charge.sum((0, 2))
        grad_kn = (None, None, None)
        if knowledge:
            grad_kn = (kn_parts[0].view(-1, heads, knowledge, channels).sum(0),
                       kn_parts[2].view(-1, heads, knowledge, payload_channels).sum(0),
                       kn_parts[1].view(-1, heads, knowledge).sum(0))
        return (grad_probe, grad_tag, grad_payload, grad_scale, grad_offset, *grad_kn, None, None, None, None, None)


def race(probe, tag, payload, scale, offset, kn_tag, kn_payload, kn_off, phase, kn_phase, kept, reach, budget):
    """FusedRace.apply with its arguments named: see FusedRace."""
    return FusedRace.apply(probe, tag, payload, scale, offset, kn_tag, kn_payload, kn_off, phase, kn_phase, kept,
                           reach, budget)
