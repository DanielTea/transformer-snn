# Diagnostics of the word-time study

Scripts written and run during the study of 18 to 19 November
2026. Each runs from the repository root, reads `data/enwik8`
and prints its result; `--json` also writes the rows to a
file. They measure trained checkpoints. None is part of the
trainer or the evaluators.
[Study page](../../docs/READ_CLOCKS.md) ·
[Records](../../results/event_lm/read_clocks)

`--averaged` takes the running weight average out of a
snapshot that holds both sets of weights. A released
`model.pt.averaged` is already the average and is passed
without it. `--device cuda` or `--device mps` moves a script
to a GPU.

| Script | Asks |
|---|---|
| `head_use.py` | Where the heads of a network look, in bytes back and in words back |
| `read_addresses.py` | How far the read heads have moved their addresses: the offsets by age in bytes and in beats, and the gains |
| `read_ablate.py` | What the reads are worth, and what the match of probe and tag is worth in them |
| `read_match_stats.py` | How fine the probe and tag pulses of the reads are |
| `words_in_view.py` | Where one checkpoint is ahead of another: by copyable targets, by the place in a word and by the words in view |
| `later_letters.py` | The same for the letters after a word's first, by what the window showed of the word before |
| `word_probe.py` | How much a network's stream knows about the word in progress and the five words before it |
| `cell_use.py` | How the race cells share their spikes: the cells that fire at a byte and the cells in use over a text |

`words_in_view.py`, `later_letters.py` and `word_probe.py` take
Transformer checkpoints of `train_event_lm.py` as well as
decoders, so that the two can be set side by side on the same
windows.

The linear readouts of `word_probe.py` are ordinary real-valued
layers that are fitted on a frozen network. They measure the
network and are no part of any model.
