# ShuntFormer v13 research handoff

The release restores seven checkpoints of the state-read study.
Two are the best spike-only decoder, with its last and its
averaged weights. Two are the decoder with spike reads alone,
one is the soft-read hybrid, and two are the Transformer trained
with the rotated optimizer. The repository carries no v13
weights. The bundled model that runs from a clone is still
[v11](../models/spiking-byte-lm-v11/README.md).
[Model card](../models/spiking-byte-lm-v13/README.md) ·
[Measurements, circuit and limits](STATE_READS.md) ·
[All 89 arms](STATE_READS_ARMS.md)

The decoder of this release has no stream form. It is scored in
512-byte windows. The streaming evaluator, the native integer
decoder, the pulse audit and the generator refuse it.
[Limits](STATE_READS.md#limits)

## Restore the checkpoints

Use Python 3.12 and install `requirements.txt`. For dataset
preparation install `requirements-benchmarks.txt` as well.

The training runs used two NVIDIA RTX PRO 6000 Blackwell GPUs
with Python 3.12 and the CUDA build of PyTorch 2.15.0. The seven
test scores of the study were computed there. The PG-19 scores,
a second scoring of the release decoder and the tests of the
beat fibres ran on the CPU of an Apple M3 Max. Two CUDA runs with
one seed are not bit-identical, so a retrained arm will not
reproduce a checkpoint hash.

The checkpoints are in three archives. Download and verify the
ones you need from the repository root.

```bash
(
  set -e
  mkdir -p checkpoints/downloads
  release=https://github.com/DanielTea/transformer-snn/releases/download/spiking-byte-lm-v13
  curl -fL --retry 3 "$release/spiking-byte-lm-v13-checkpoints.sha256" \
    -o checkpoints/downloads/spiking-byte-lm-v13-checkpoints.sha256
  for part in cells reads transformer; do
    asset="spiking-byte-lm-v13-checkpoints-$part.tar.gz"
    curl -fL --retry 3 "$release/$asset" -o "checkpoints/downloads/$asset"
    (cd checkpoints/downloads && grep " $asset\$" spiking-byte-lm-v13-checkpoints.sha256 | shasum -a 256 -c -)
    tar -xzf "checkpoints/downloads/$asset"
    shasum -a 256 -c "checkpoints/SHA256SUMS-v13-$part"
  done
)
```

| Archive | Size | Contents of `checkpoints/state_reads` |
|---|---:|---|
| `cells` | 259 MB | `pilot-race64-d8-c350-dl4`: spike reads and race cells on delayed fibres, the model of this release |
| `reads` | 386 MB | `full-rate-rot`: spike reads alone. `full-soft-d13w256`: the soft-read hybrid under AdamW |
| `transformer` | 259 MB | `full-transformer-rot`: the Transformer under the rotated optimizer |

A folder is named after its arm in the study. `model.pt` holds
the weights of the last update and `model.pt.averaged` the
running weight average, which is what the study scores. The
soft-read run kept no average.

The test table also scores three v12 checkpoints: the AdamW
Transformer `scratch-study/transformer_s20001.pt`, the plain
decoder `scratch-study/clock_s20001.pt` and the head decoder
`v12-scratch/heads_s20001.pt`. They are in the `scratch` archive
of v12.
[v12 restore](V12_HANDOFF.md#restore-the-checkpoints)

The training machine wrote its own absolute paths into the
checkpoints' arguments. The released files were written again
with repository-relative paths. Every tensor is bit-identical,
and the record lists both hashes of every file.
[Resaved checkpoints](../results/event_lm/state_reads/resaved_checkpoints.json) ·
[Release assets](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v13) ·
[Exact checkpoint manifest](../models/spiking-byte-lm-v13/manifest.json) ·
[All checksums](../results/event_lm/state_reads_checkpoints.sha256)

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
    checkpoints/state_reads/pilot-race64-d8-c350-dl4/model.pt \
    checkpoints/state_reads/pilot-race64-d8-c350-dl4/model.pt.averaged \
  --json checkpoints/work/test_spike_reads_and_cells.json
```

On the CPU of a laptop the release decoder scores about 4,300
targets per second, 19 minutes for the test split. The races are
computed in floating point, so the score can differ in the fifth
decimal between machines.

For PG-19 the evaluator fits one temperature per checkpoint on
the validation books and scores the 16 test books.

```bash
python experiments/eval_warmed_windows.py --device cuda --books-dir data/pg19-test \
  --calibration-books data/pg19-validation \
  --calibration-manifest results/event_lm/correction_pg19_validation_manifest.json \
  --checkpoint checkpoints/scratch-study/transformer_s20001.pt \
    checkpoints/state_reads/pilot-race64-d8-c350-dl4/model.pt.averaged \
  --json checkpoints/work/pg19_cells.json
```

[Evaluator](../experiments/eval_warmed_windows.py) ·
[Test records](../results/event_lm/state_reads/scores)

## Rerun the probes

The probes of the study are short scripts in
[`experiments/state_reads`](../experiments/state_reads). Each
runs from the repository root, reads `data/enwik8`, uses a GPU
when it finds one and prints its result. The folder's README
says what each asks and which output it produced.

Most probes compared snapshots during training. Only the final
checkpoints of four arms are released, so those probes need a
retrained arm with `--save-every` and a copy of its partial file
at every step to compare. These run on the released files.

```bash
python experiments/state_reads/tap_use.py checkpoints/state_reads/pilot-race64-d8-c350-dl4/model.pt.averaged
python experiments/state_reads/granule_stats.py checkpoints/state_reads/pilot-race64-d8-c350-dl4/model.pt.averaged
python experiments/state_reads/headroom.py checkpoints/state_reads/pilot-race64-d8-c350-dl4/model.pt.averaged
python experiments/state_reads/match_split.py --windows 512 \
  checkpoints/state_reads/full-transformer-rot/model.pt.averaged \
  checkpoints/state_reads/pilot-race64-d8-c350-dl4/model.pt.averaged
python experiments/state_reads/context_split.py --save checkpoints/work/context_scores.npz \
  checkpoints/state_reads/full-transformer-rot/model.pt.averaged \
  checkpoints/state_reads/pilot-race64-d8-c350-dl4/model.pt.averaged \
  checkpoints/state_reads/full-rate-rot/model.pt.averaged
python experiments/state_reads/expert_mix.py checkpoints/work/context_scores.npz
python experiments/state_reads/quantize_transformer.py checkpoints/state_reads/full-transformer-rot/model.pt
```

`--averaged` in a probe takes the running average out of a
snapshot that holds both. A released `model.pt.averaged` is
already the average and is passed without it.

The tests of the beat fibres and the earlier thought-tape probe
are in `experiments`.

```bash
python experiments/probe_word_time.py --seed 0 --output checkpoints/work/word_time_s0.json
python experiments/probe_clock_cells.py CHECKPOINT
python experiments/probe_state_attention.py --help
python experiments/probe_match_gap.py --help
```

## Regenerate the figures

```bash
python experiments/plot_state_reads.py
python experiments/plot_state_reads.py --tables
python experiments/plot_state_reads.py --arms > docs/STATE_READS_ARMS.md
```

The second command prints the tables of the
[study page](STATE_READS.md) and the third writes the
[arms page](STATE_READS_ARMS.md).
`python experiments/plot_event_lm_study.py` regenerates the
[leaderboard](LEADERBOARD.md).

## Train

The study page has the complete command of the release decoder.
[Command](STATE_READS.md#reproduce)

| Flag | Sets |
|---|---|
| `--read-after 2,4,6` | Blocks after which a state read stands |
| `--read-mode race --read-budget 256` | The spike read: a race of rates for 256 spikes per head |
| `--read-mode soft` | The softmax read of the hybrid reference |
| `--read-width 256 --read-window 512` | Channels of tag, payload and probe, and bytes in reach |
| `--read-resolution 4` | Read pulses per payload pulse |
| `--granule-after 0,1,3,5` | Blocks after which a layer of cells stands |
| `--granule-clusters 16 --granule-cells 350 --granule-fan 256` | Clusters per layer, cells per cluster, channels per cluster |
| `--granule-budget 64` | Race cells with that many spikes per cluster and byte. Without it the cells fire past a threshold |
| `--granule-delays 4` | Fibres of 0 to 3 bytes delay |
| `--granule-beats`, `--granule-clock`, `--granule-hold` | Beat fibres, untested at full size |
| `--rotated-adam 50` | The rotated optimizer with that many updates between basis refreshes |
| `--weight-average 0.999` | A running weight average, saved as `model.pt.averaged` and scored at every evaluation |
| `--save-every 1500` | Writes `<checkpoint>.partial` with the last and the averaged weights every 1,500 updates. The study's snapshots are copies of it |

The decoder checks that every synapse sum of whole spikes stays
in the range where its integer arithmetic is exact. The check
covers the blocks, the cell layers and the spike reads. In the
release decoder the largest sum uses 47% of that range in a
block and 43% in a read. The softmax read of the hybrid is a
floating-point reference and is not checked.

Judge a design at 12,000 updates or later. Two runs of one
configuration differ by 0.018 bpb with their last weights and
0.026 with their averaged weights at 3,000 updates, and the
order of two designs changed more than once between 3,000 and
12,000.

## Next pilots

The gap to the Transformer is on targets with nothing to copy,
and it opens with the second word back.
[Where the gap is](STATE_READS.md#where-the-gap-is)

**Beat fibres at full size.** They are built for exactly that
and have only been tested on a made-up language. Two arms of
12,000 updates each start from the release command, keep
`--granule-delays 4` and add the beats.

```bash
... --granule-beats 4 --granule-clock word
... --granule-beats 4 --granule-clock learned --granule-cells 348
```

The first clock reads the word boundaries from the bytes. It is
a diagnostic that shows what word time is worth, not a spike-only
circuit. The second is the learned clock. Its clock cells add
90,176 parameters, and 348 cells per cluster keep the decoder at
34,844,056, below the Transformer's 34,845,696. Both
configurations passed a dry run of two updates on a CPU. A cell
layer with learned clocks takes 2.6 times as long as one without
on a laptop CPU. Its cost on a GPU is not measured.

Judge both against the release run's averaged validation: 1.7743
at 3,000 updates, 1.6499 at 6,000, 1.6033 at 9,000 and 1.5687
at 12,000. Then look where the gain is with `context_split.py`,
by the words in view, and where the learned clocks beat with
`probe_clock_cells.py`. The better arm gets a run of 90,000
updates and a test score.

**What the made-up language already settled.** Start the clocks
at a pace of 0.25 beats per byte, the default. A start near one
beat per byte stays a delay line (3.04 bits against a bound of
2.53). One clock cell per beat fibre learns in a third of the
updates that one per cluster needs. Hearing the mean over beats
in training and whole beats at scoring does not work (3.51), and
annealing from one to the other gains nothing. Keeping a
stretch's last count scored worst there, but that language has
no context in its stream, so try `--granule-hold last` once at
full size.

**A stream form.** A read would carry the tags and payloads of
its last 512 bytes, and a cell layer the last three bytes of its
channels. Without it the decoder has no streaming score, no
native integer run, no pulse audit and no generator, and its row
on the leaderboard stands on window scores alone.

**Open at 12,000 updates.** Cells after every block and fibres
of 8 delays are level with the release decoder there, 1.5617
and 1.5661 against 1.5687. Neither was run further. The 10-block
shape with delays was level with the 8-block shape at 9,000. A
second seed of the release run does not exist.

## What not to try again

Each was one arm with one seed. The numbers are on the
[study page](STATE_READS.md#what-was-ruled-out) and the
[arms page](STATE_READS_ARMS.md).

- More reads, wider reads, eight heads, reads after other blocks.
- A stream of 704 or 1,024 channels with wider blocks.
- A finer weight grid, `--router-charge-bits 9`.
- Step sizes above 3e-4 under AdamW, and 6e-4 or 1.5e-4 under
  the rotated optimizer.
- A larger pulse budget or none.
- Cells that fire past a threshold, with or without a price on
  their spikes, with or without delays.
- Relay neurons before the fibres, slow fibres that hold a share
  of their charge, many small clusters.
- A larger step size for the cells alone.

## Where things are

| Path | Holds |
|---|---|
| `snnformer/event_lm.py` | `_StateRead`, `_GranuleLayer`, `_RaceFire`, `_ClockFire` and the beat fibres |
| `experiments/rotated_adam.py` | The rotated optimizer |
| `experiments/train_event_lm.py` | The trainer with the flags above |
| `experiments/eval_warmed_windows.py` | The window evaluator for enwik8 and PG-19 |
| `experiments/state_reads/` | The probes of the study |
| `experiments/plot_state_reads.py` | Figures, tables and the arms page |
| `tests/test_state_reads.py`, `tests/test_granule_cells.py`, `tests/test_rotated_adam.py` | 62 tests of the reads, the cells, the beat fibres and the optimizer |
| `results/event_lm/state_reads/` | Arms, run records, test scores, probe outputs |
| `results/event_lm/beat_fibres/` | The tests on the made-up language |
| `results/event_lm/state_attention/` | The thought-tape probe of 17 November |
