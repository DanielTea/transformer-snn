"""The leaderboard generator (experiments/plot_event_lm_study.py) carries hand-written release sections over."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from plot_event_lm_study import KEPT_END, KEPT_START, kept_sections  # noqa: E402


def test_hand_written_sections_are_kept_in_order(tmp_path):
    board = tmp_path / "LEADERBOARD.md"
    board.write_text(f"# Title\n\nintro\n\n{KEPT_START}\n## v19\n\ntable\n{KEPT_END}\n\n{KEPT_START}\n## v18\n{KEPT_END}\n\n"
                     "## Generated\n")
    assert kept_sections(board) == [f"{KEPT_START}\n## v19\n\ntable\n{KEPT_END}", f"{KEPT_START}\n## v18\n{KEPT_END}"]
    assert kept_sections(tmp_path / "missing.md") == []
    board.write_text(f"{KEPT_START}\n## open\n")
    with pytest.raises(ValueError, match="no end marker"):
        kept_sections(board)


def test_the_repository_leaderboard_keeps_its_release_sections():
    board = Path(__file__).resolve().parents[1] / "docs" / "LEADERBOARD.md"
    titles = [section.split("\n")[1] for section in kept_sections(board)]
    assert titles[:3] == ["## The looped stack at full length · 368,640,000 training targets · seed 20001",
                          "## Each model at its best rate · 368,640,000 training targets · seed 20001",
                          "## Word heads, trained on streams · 368,640,000 training targets · seed 20001"]
