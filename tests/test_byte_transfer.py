import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F


@pytest.fixture
def evaluator(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import eval_byte_transfer
    return eval_byte_transfer


def test_matched_windows_cover_each_target_once(evaluator):
    windows = list(evaluator.scoring_windows(b"0123456789", context_bytes=6, targets_per_window=3))
    assert windows == [(1, 1, b"0123"), (4, 3, b"123456"), (7, 3, b"456789")]
    assert b"".join(sequence[prefix:] for _, prefix, sequence in windows) == b"123456789"


def test_word_scores_mask_prefix_and_padding_and_check_the_true_end(evaluator):
    class BytePredictions(torch.nn.Module):
        def forward(self, tokens):
            following = torch.zeros(256, dtype=torch.long)
            for byte, prediction in (("b", "c"), ("c", "d"), ("d", " "), ("z", "y"), ("y", "x")):
                following[ord(byte)] = ord(prediction)
            return F.one_hot(following[tokens], num_classes=256).double() * 12

    windows = [evaluator.Window(b"abcd", prefix_bytes=2), evaluator.Window(b"zy", 1, overlap_hit=True)]
    result = evaluator.score_windows(BytePredictions(), windows, 2, torch.device("cpu"),
                                    word_accuracy=True, overlap=True)
    assert result["target_bytes"] == 3
    assert result["overlap_flagged_targets"] == 1
    assert result["without_overlap_hits"]["target_bytes"] == 2
    assert result["bpb"] == pytest.approx(math.log2(1 + 255 * math.exp(-12)), abs=2e-7)
    assert result["exact_matches"] == 2
    assert result["boundary_exact_matches"] == 1
