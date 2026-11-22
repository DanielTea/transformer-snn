"""The split of a paired gap by what the scored bytes are (experiments/gap_by_piece.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from gap_by_piece import CONTEXT, GRAM, WHAT, byte_kinds, context_kinds, fit, piece_bits  # noqa: E402


def test_byte_kinds():
    data = np.frombuffer(b"aZ5 \n[.\xc3", np.uint8)
    assert [WHAT[k] for k in byte_kinds(data)] == ["letter", "letter", "digit", "space", "line break", "markup",
                                                   "punctuation", "multi-byte"]


def test_context_kinds_copy_near_far_and_new():
    rng = np.random.default_rng(0)
    phrase = rng.integers(0, 256, GRAM + 1).astype(np.uint8)
    filler = lambda n: rng.integers(0, 256, n).astype(np.uint8)  # noqa: E731
    # The phrase, then again 30 bytes later (near), then again 300 bytes after that (far only for near=100).
    data = np.concatenate((phrase, filler(30), phrase, filler(300), phrase, filler(5)))
    kinds = context_kinds(data, near=100, far=1000)
    first, second, third = GRAM, GRAM + 1 + 30 + GRAM, GRAM + 1 + 30 + GRAM + 1 + 300 + GRAM
    assert CONTEXT[kinds[first]] == "new"
    assert CONTEXT[kinds[second]] == "copyable near"
    assert CONTEXT[kinds[third]] == "copyable far only"
    # The same eight bytes followed by another byte: seen, other follower.
    changed = data.copy()
    changed[second] = (int(data[second]) + 1) % 256
    assert CONTEXT[context_kinds(changed, near=100, far=1000)[second]] == "seen, other follower"
    # Beyond the far reach it is new again.
    assert CONTEXT[context_kinds(data, near=10, far=20)[second]] == "new"


def test_piece_bits_sums_small_pieces_into_common_ones():
    bits = np.arange(2 * 9, dtype=float).reshape(2, 9)
    summed = piece_bits(bits, 16, 64)
    assert summed.shape == (2, 3)
    assert np.allclose(summed[:, 0], bits[:, :4].sum(1))
    assert np.allclose(summed[:, 2], bits[:, 8])


def test_fit_recovers_the_gap_of_every_kind():
    rng = np.random.default_rng(1)
    counts = rng.integers(0, 50, size=(400, 3)).astype(float)
    truth = np.array([0.05, -0.3, 0.0])
    gaps = counts @ truth + rng.normal(0, 0.05, 400)
    solution, low, high = fit(counts, gaps, rng.integers(400, size=(50, 400)))
    assert np.allclose(solution, truth, atol=0.01)
    assert np.all(low <= solution) and np.all(solution <= high)
