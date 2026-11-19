"""Where the later letters of words differ between checkpoints: by what the window showed of the word before.

usage: later_letters.py CHECKPOINT... [--windows 512] [--averaged] [--threads 8] [--json OUT]
Run from the repository root; the first checkpoint is the reference.

The windows and the scoring are those of words_in_view.py. The rows hold the letters after a word's first that
cannot be copied (no earlier match of four or more bytes whose follower is the target). They are split by what
the window showed before the target:

  the letters of the word so far never began an earlier word of the window,
  they began one, never followed by the target letter,
  they began one and the target letter followed them there,

then by whether the whole word occurred before, by the number of words that started in the window before the
target, and by the letter's place in its word. A letter whose beginning was never seen has to come from what
the network knows about the language; the words in view can only tell it which word is likely. The intervals
resample whole windows.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "experiments/read_clocks")
from words_in_view import (arguments, earlier_matches, paired_bits, paired_rows, validation_windows,  # noqa: E402
                           word_places, write_rows)

NEVER, WITHOUT, WITH = 0, 1, 2


def earlier_beginnings(tokens, inside):
    """For every target inside a word, what the window showed of the letters before it, and of the whole word.

    The first array holds NEVER where the letters of the word so far never began an earlier word of the window,
    WITHOUT where they did but the target letter never followed them, and WITH where it did. The second marks
    the targets of words that occurred as a whole before. A word that the window cuts off counts as a beginning
    only.
    """
    known = np.zeros((tokens.shape[0], tokens.shape[1] - 1), dtype=np.int8)
    whole = np.zeros(known.shape, dtype=bool)
    for row, line in enumerate(tokens):
        followers, words = {}, set()
        column = 0
        while column < len(line):
            if not inside[row, column]:
                column += 1
                continue
            end = column
            while end < len(line) and inside[row, end]:
                end += 1
            letters = bytes(line[column:end].astype(np.uint8).tolist())
            complete = end < len(line)
            # The target at index column + offset - 1 is the letter at column + offset.
            for offset in range(1, end - column):
                beginning, target = letters[:offset], letters[offset]
                if beginning in followers:
                    known[row, column + offset - 1] = WITH if target in followers[beginning] else WITHOUT
                if complete and letters in words:
                    whole[row, column + offset - 1] = True
            for offset in range(1, end - column):
                followers.setdefault(letters[:offset], set()).add(letters[offset])
            if complete:
                words.add(letters)
            column = end
    return known, whole


def main():
    args = arguments(__doc__)
    torch.set_num_threads(args.threads)
    tokens = validation_windows(args.windows)
    length, copyable = earlier_matches(tokens, 512)
    copyable = (length >= 4) & copyable
    bits, names = paired_bits(args.checkpoints, tokens, args.averaged, torch.device(args.device))
    inside, start, place = word_places(tokens)
    seen = start.cumsum(1)[:, :-1]
    known, whole = earlier_beginnings(tokens, inside)
    later = (place[:, 1:] >= 2) & ~copyable
    row, rows = paired_rows(bits, width=58)
    row("later letters, not copyable", later)
    row("  beginning never began an earlier word", later & (known == NEVER))
    row("  beginning seen, never with this letter", later & (known == WITHOUT))
    row("  beginning seen with this letter", later & (known == WITH))
    row("  whole word seen before", later & whole)
    row("  whole word not seen before", later & ~whole)
    for low, high, label in ((0, 3, "0-3"), (4, 9999, "4+")):
        view = (seen >= low) & (seen <= high)
        row(f"{label} words in view: beginning never seen", later & view & (known == NEVER))
        row(f"{label} words in view: seen, never with this letter", later & view & (known == WITHOUT))
        row(f"{label} words in view: seen with this letter", later & view & (known == WITH))
    for low, high, label in ((2, 2, "2nd"), (3, 3, "3rd"), (4, 5, "4th-5th"), (6, 99, "6th+")):
        spot = (place[:, 1:] >= low) & (place[:, 1:] <= high) & (seen >= 4)
        row(f"{label} letter, 4+ words in view, beginning never seen", later & spot & (known == NEVER))
        row(f"{label} letter, 4+ words in view, seen with this letter", later & spot & (known == WITH))
    write_rows(args.json, names, args, rows)


if __name__ == "__main__":
    main()
