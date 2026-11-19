# ShuntFormer v14 research handoff

The release restores six checkpoints of the read-clock study.
Two are the released decoder, with clock cells and the address
rate, as its last and its averaged weights. Two are the decoder
with the address rate alone, and two are the Transformer of the
standard shape. The repository carries no v14 weights. The
bundled model that runs from a clone is still
[v11](../models/spiking-byte-lm-v11/README.md).
[Model card](../models/spiking-byte-lm-v14/README.md) ·
[Measurements, circuit and limits](READ_CLOCKS.md) ·
[All 49 arms](READ_CLOCKS_ARMS.md)

The decoder of this release has no stream form. It is scored in
512-byte windows. The streaming evaluator, the native integer
decoder, the pulse audit and the streaming generator refuse it.
A window generator that is new in this release writes text with
it.
[Limits](READ_CLOCKS.md#limits)

## Restore the checkpoints

Use Python 3.12 and install `requirements.txt`. For dataset
preparation install `requirements-benchmarks.txt` as well.

The decoders and all but one of the pilots were trained on two
NVIDIA RTX PRO 6000 Blackwell GPUs with Python 3.12 and the CUDA
build of PyTorch 2.15.0, and the decoders were scored there. One
pilot and the second run of the decoder with the clock at word
ends ran on the GPU of an Apple M3 Max. The Transformer of the
standard shape ran its first 15,000 updates on the first machine
and the other 75,000 on the second, where it was scored. Two
CUDA runs with one seed are not bit-identical, so a retrained
arm will not reproduce a checkpoint hash.

The checkpoints are in three archives. Download and verify the
ones you need from the repository root.

```bash
(
  set -e
  mkdir -p checkpoints/downloads
  release=https://github.com/DanielTea/transformer-snn/releases/download/spiking-byte-lm-v14
  curl -fL --retry 3 "$release/spiking-byte-lm-v14-checkpoints.sha256" \
    -o checkpoints/downloads/spiking-byte-lm-v14-checkpoints.sha256
  for part in clocks rate transformer; do
    asset="spiking-byte-lm-v14-checkpoints-$part.tar.gz"
    curl -fL --retry 3 "$release/$asset" -o "checkpoints/downloads/$asset"
    (cd checkpoints/downloads && grep " $asset\$" spiking-byte-lm-v14-checkpoints.sha256 | shasum -a 256 -c -)
    tar -xzf "checkpoints/downloads/$asset"
    shasum -a 256 -c "checkpoints/SHA256SUMS-v14-$part"
  done
)
```

| Archive | Size | Contents of `checkpoints/read_clocks` |
|---|---:|---|
| `clocks` | 259 MB | `tag8-a8`: clock cells and the address rate, the model of this release |
| `rate` | 259 MB | `rate8`: the address rate alone |
| `transformer` | 259 MB | `transformer-std`: the Transformer of the standard shape under AdamW |

A folder is named after its arm in the study. `model.pt` holds
the weights of the last update and `model.pt.averaged` the
running weight average, which is what the study scores.

The test table also scores three checkpoints of earlier
releases. The AdamW Transformer
`scratch-study/transformer_s20001.pt` is the reference of every
difference and is in the `scratch` archive of v12. The v13
decoder `state_reads/pilot-race64-d8-c350-dl4` and the
Transformer under the rotated optimizer
`state_reads/full-transformer-rot` are in the `cells` and
`transformer` archives of v13.
[v12 restore](V12_HANDOFF.md#restore-the-checkpoints) ·
[v13 restore](V13_HANDOFF.md#restore-the-checkpoints)

The training machines wrote their own absolute paths into the
checkpoints' arguments. The released files were written again
with repository-relative paths. Every tensor is bit-identical,
and the record lists both hashes of every file.
[Resaved checkpoints](../results/event_lm/read_clocks/resaved_checkpoints.json) ·
[Release assets](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v14) ·
[Exact checkpoint manifest](../models/spiking-byte-lm-v14/manifest.json) ·
[All checksums](../results/event_lm/read_clocks_checkpoints.sha256)

## Rescore

Prepare enwik8 and the PG-19 books as for v11.
[Dataset commands](RESIDUAL_SCALE.md#reproduce-the-data-and-launches)

One evaluator scores any checkpoint of the trainer in warmed
windows. Every window starts from an empty state and holds 256
unscored bytes before its 256 scored ones. The first checkpoint
is the reference: every later one gets its difference to it over
whole windows with a paired 95% interval.

```bash
python experiments/eval_warmed_windows.py --device cuda --enwik8 data/enwik8 \
  --checkpoint checkpoints/scratch-study/transformer_s20001.pt \
    checkpoints/read_clocks/tag8-a8/model.pt.averaged \
    checkpoints/read_clocks/rate8/model.pt.averaged \
    checkpoints/read_clocks/transformer-std/model.pt.averaged \
  --json checkpoints/work/test_read_clocks.json
```

`--device mps` scores on an Apple GPU and `--device cpu` on a
CPU. The races are computed in floating point, so a decoder's
score can differ in the fifth decimal between machines.

For PG-19 the evaluator fits one temperature per checkpoint on
the validation books and scores the 16 test books.

```bash
python experiments/eval_warmed_windows.py --device cuda --books-dir data/pg19-test \
  --calibration-books data/pg19-validation \
  --calibration-manifest results/event_lm/correction_pg19_validation_manifest.json \
  --checkpoint checkpoints/scratch-study/transformer_s20001.pt \
    checkpoints/read_clocks/tag8-a8/model.pt.averaged \
  --json checkpoints/work/pg19_clock_cells.json
```

[Evaluator](../experiments/eval_warmed_windows.py) ·
[Test records](../results/event_lm/read_clocks/scores)

## Write text

A decoder with state reads carries nothing from one byte to the
next, so the generator of the earlier releases refuses it. The
window generator runs the last 512 bytes again for every new
byte. That costs a whole pass per byte and works for any model
of the trainer, the Transformers and the v13 decoder included.

```bash
python experiments/generate_windows.py checkpoints/read_clocks/tag8-a8/model.pt.averaged \
  --prompt 'The first chapter' --bytes 256
```

`--temperature 0` writes the most likely byte. The default is
0.8 with `--seed 0`. On the CPU of a laptop the released decoder
writes about eight bytes per second.
[Generator](../experiments/generate_windows.py) ·
[Sample](../models/spiking-byte-lm-v14/README.md#write-text)

## Rerun the diagnostics

The diagnostics of the study are short scripts in
[`experiments/read_clocks`](../experiments/read_clocks). Each
runs from the repository root, reads `data/enwik8` and prints
its result. The folder's README says what each asks.

All of them run on the released files. The study's records hold
their output on the windows named there.

```bash
python experiments/read_clocks/head_use.py checkpoints/read_clocks/tag8-a8/model.pt.averaged
python experiments/read_clocks/read_addresses.py 90000=checkpoints/read_clocks/tag8-a8/model.pt.averaged
python experiments/read_clocks/clock_beats.py checkpoints/read_clocks/tag8-a8/model.pt.averaged
python experiments/read_clocks/cell_use.py checkpoints/read_clocks/tag8-a8/model.pt.averaged
python experiments/read_clocks/spike_levels.py checkpoints/read_clocks/tag8-a8/model.pt.averaged
python experiments/read_clocks/spike_noise.py checkpoints/read_clocks/tag8-a8/model.pt.averaged
python experiments/read_clocks/read_ablate.py checkpoints/read_clocks/tag8-a8/model.pt.averaged
python experiments/read_clocks/read_match_stats.py checkpoints/read_clocks/tag8-a8/model.pt.averaged
```

The splits of the gap and the linear readouts compare networks
on the same windows. The first checkpoint is the reference.

```bash
python experiments/read_clocks/words_in_view.py \
  checkpoints/state_reads/full-transformer-rot/model.pt.averaged \
  checkpoints/state_reads/pilot-race64-d8-c350-dl4/model.pt.averaged \
  checkpoints/read_clocks/tag8-a8/model.pt.averaged \
  checkpoints/read_clocks/rate8/model.pt.averaged
python experiments/read_clocks/later_letters.py \
  checkpoints/state_reads/full-transformer-rot/model.pt.averaged \
  checkpoints/state_reads/pilot-race64-d8-c350-dl4/model.pt.averaged \
  checkpoints/read_clocks/tag8-a8/model.pt.averaged \
  checkpoints/read_clocks/rate8/model.pt.averaged
python experiments/read_clocks/word_probe.py \
  checkpoints/state_reads/full-transformer-rot/model.pt.averaged \
  checkpoints/state_reads/pilot-race64-d8-c350-dl4/model.pt.averaged \
  checkpoints/read_clocks/tag8-a8/model.pt.averaged
python experiments/read_clocks/word_probe.py --at ends \
  checkpoints/state_reads/full-transformer-rot/model.pt.averaged \
  checkpoints/read_clocks/tag8-a8/model.pt.averaged
```

`read_addresses.py` takes any number of `UPDATES=CHECKPOINT`
pairs and prints how far the addresses had moved at each. The
study's curves come from snapshots during training, which are
copies of `model.pt.partial` and are not released.

`--averaged` in a script takes the running average out of a
snapshot that holds both. A released `model.pt.averaged` is
already the average and is passed without it.

The made-up language of the study trains small decoders on a
CPU. Without `--arms` it runs every arm, the beat fibres of v13
included.

```bash
python experiments/probe_word_time.py --seed 0 --steps 6000 --threads 3 \
  --arms 'read, ages in bytes' 'read, clock at word ends' 'read, clock cells' \
  --output checkpoints/work/word_time_reads_s0.json
```

## Regenerate the figures

```bash
python experiments/plot_read_clocks.py
python experiments/plot_read_clocks.py --tables
python experiments/plot_read_clocks.py --arms > docs/READ_CLOCKS_ARMS.md
```

The second command prints the tables of the
[study page](READ_CLOCKS.md) and the third writes the
[arms page](READ_CLOCKS_ARMS.md).
`python experiments/plot_event_lm_study.py` regenerates the
[leaderboard](LEADERBOARD.md).

## Train

The study page has the complete command of the released decoder.
[Command](READ_CLOCKS.md#reproduce)

| Flag | Sets |
|---|---|
| `--read-address-rate 8` | The stored age offsets and gains of a read count eight times, so one update moves an address eight times as far |
| `--read-beats 8` | A read also knows each cell's age in beats of a clock, counted from 0 to 7 with one more value for all older cells |
| `--read-beat-start 1` | Every head starts with one unit of charge for one age in beats of its own |
| `--read-clock end` | The beat falls on the first byte after every word. The clock reads the bytes and is a measurement, no spike circuit |
| `--read-clock learned` | Every head has a clock cell whose spike is the beat |
| `--lr-scale clock_rest=8` | Eight times the learning rate for the resting charge of the clock cells, which sets how often a cell beats |
| `--read-marks` | A beat marks the cell it falls on, and a head has a second offset for the marked cell of each age |
| `--granule-relays 256` | The fibres of a cell layer are the axons of 256 relay neurons that hear the whole stream |
| `--granule-budget 0 --granule-slope` | Threshold cells in place of race cells, which learn while silent |
| `--granule-races 7` | The cells of a cluster race in that many equal groups |
| `--granule-soft` | Cells that answer with real numbers. Diagnosis only: the decoder is then a hybrid |
| `--save-every 1500 --resume` | Writes the whole training state to `<checkpoint>.resume` every 1,500 updates and continues from it when it exists |
| `--stop-at 24000` | Leaves after that update with the state saved |

A saved state continues on another device, and since this
release in another folder: move `model.pt.resume` and pass the
new `--checkpoint`. A test checks that a continued run ends at
the weights of an unbroken one. The released decoder was
continued once, at 19,500 updates.

**Read a pilot between 9,000 and 30,000 updates.** Both
Transformers take their lead there. Leads at 12,000 updates did
not last: the clock cells were 0.007 ahead of the address rate
at 24,000 updates and 0.002 at 90,000. A second run of one
decoder ended 0.006 from the first at 12,000 updates, so smaller
differences mean nothing.

Averaged validation loss of the two released decoders and the
two Transformers, for a pilot to be set against:

| Updates | Clock cells | Address rate | Transformer, 256 wide, rotated | Transformer, standard shape |
|---:|---:|---:|---:|---:|
| 3,000 | 1.7751 | 1.7715 | 1.9608 | 1.8538 |
| 6,000 | 1.6262 | 1.6341 | 1.7597 | 1.6259 |
| 9,000 | 1.5769 | 1.5823 | 1.6148 | 1.5455 |
| 12,000 | 1.5408 | 1.5450 | 1.5304 | 1.4976 |
| 18,000 | 1.4974 | 1.5075 | 1.4587 | 1.4382 |
| 24,000 | 1.4671 | 1.4737 | 1.4172 | 1.3986 |
| 30,000 | 1.4433 | 1.4490 | 1.3844 | 1.3702 |
| 90,000 | 1.3315 | 1.3336 | 1.2783 | 1.2607 |

[All curves](../results/event_lm/read_clocks/arms.json)

## Next pilots

The gap to the Transformer is on targets with nothing to copy,
and it is largest where four or more words are in view. A linear
readout finds the earlier words in the decoder's stream. What
the decoder lacks is the use of them.
[Where the gap is](READ_CLOCKS.md#where-the-gap-is) ·
[The stream knows the words](READ_CLOCKS.md#the-stream-knows-the-words)

**Dense cells on relay neurons, at full length.** A race answers
a byte with about one cell of a cluster. Threshold cells on
relay neurons answer with many and were closing in on the race
cells when the pilots ended: 0.018 behind at 12,000 updates,
0.013 at 18,000 and 0.008 at 24,000. Nobody has seen whether
they pass. The arm is `slope-relay`, the base of the pilots with
these flags in place of its cell flags.

```bash
... --granule-relays 256 --granule-cells 306 --granule-budget 0 --granule-slope
```

The base of the pilots has the clock at word ends, which reads
the bytes. A decoder for a release takes the clock cells or the
address rate alone. With the clock at word ends the arm has
34,816,772 parameters.
[Cells that answer for themselves](READ_CLOCKS.md#cells-that-answer-for-themselves)

**Marks, at full length.** With `--read-marks` on the clock at
word ends the arm led its base by 0.006 to 0.010 from 12,000 to
24,000 updates and was stopped there. With clock cells it was
level at 6,000. The lead of the clock cells themselves was as
large at those updates and did not last, so the marks need a run
of 90,000 updates before they count.

**A second seed.** The two released decoders are 0.0013 bpb
apart on the test. Neither has a second run of full length, and
without one the clock cells cannot be told from the address rate
alone.

**A stream form.** Still open from v13. A read would carry the
tags and payloads of its last 512 bytes and the beats of its
clock cells, and a cell layer the last three bytes of its
channels.
[v13 handoff](V13_HANDOFF.md#next-pilots)

## What not to try again

Each was one arm with one seed, and most were stopped within
12,000 updates. The numbers are on the
[study page](READ_CLOCKS.md#layouts-that-changed-nothing) and
the [arms page](READ_CLOCKS_ARMS.md).

- Eight heads per read, seven reads, and heads that start with
  eight units of charge on one word each.
- An address rate of 64.
- Race cells on fibres that hold the last words. These are the
  beat fibres that the v13 handoff proposed.
- More and smaller clusters, cell layers right after the reads,
  wider fans with fewer cells, thinner blocks with more cells.
- Several races within a cluster, and a softer race where the
  gradient is taken.
- Relay neurons that also bring the stream as it was at the end
  of the last three words.

## Where things are

| Path | Holds |
|---|---|
| `snnformer/event_lm.py` | `_StateRead` with the address rate, the ages in beats, the marks and the clock cells; `_GranuleLayer` with relay neurons, threshold cells and grouped races |
| `experiments/train_event_lm.py` | The trainer with the flags above |
| `experiments/eval_warmed_windows.py` | The window evaluator for enwik8 and PG-19 |
| `experiments/generate_windows.py` | The window generator |
| `experiments/read_clocks/` | The diagnostics of the study |
| `experiments/probe_word_time.py` | The made-up language |
| `experiments/plot_read_clocks.py` | Figures, tables and the arms page |
| `tests/test_state_reads.py`, `tests/test_granule_cells.py`, `tests/test_train_resume.py`, `tests/test_generate_windows.py`, `tests/test_warmed_windows.py`, `tests/test_rotated_adam.py` | 125 tests of the reads, the clocks, the cells, continued runs, the window generator, the window evaluator and the optimizer |
| `results/event_lm/read_clocks/` | Arms, run records, test scores, diagnostic outputs |
