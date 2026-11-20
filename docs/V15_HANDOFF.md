# ShuntFormer v15 research handoff

The release restores six checkpoints of the stream-form study.
Two are the released stack, trained on windows of 512 and then
1,024 bytes, as its last and its averaged weights. Two are the
same stack trained on windows of 512 bytes throughout, and two
are the reference Transformer. The repository carries no v15
weights. The bundled model that runs from a clone is still
[v11](../models/spiking-byte-lm-v11/README.md).
[Model card](../models/spiking-byte-lm-v15/README.md) · [Measurements, circuit and limits](STREAM_FORM.md) · [All 55 arms](STREAM_FORM_ARMS.md)

The stack of this release reads a stream. It is scored on
unbroken streams with its state carried from piece to piece, and
it writes text a byte at a time from that state. It has no native
integer decoder and no pulse audit, and the streaming evaluator
and generator of v11 refuse it.
[Limits](STREAM_FORM.md#limits)

## Restore the checkpoints

Use Python 3.12 and install `requirements.txt`. For dataset
preparation install `requirements-benchmarks.txt` as well.

The two stacks, the reference Transformer and 45 of the 55 arms
were trained on two NVIDIA RTX PRO 6000 Blackwell GPUs with
Python 3.12 and the CUDA build of PyTorch 2.15.0. Six early arms
ran on the GPU of an Apple M3 Max, three on a CPU, and one began
on that GPU and ended on a CPU. The stacks were scored on the
CUDA GPUs, and so was the Transformer except for five scorings on
the Apple GPU. Every score record names its device. Two CUDA runs
with one seed are not bit-identical, so a retrained arm will not
reproduce a checkpoint hash.

The checkpoints are in three archives. Download and verify the
ones you need from the repository root.

```bash
(
  set -e
  mkdir -p checkpoints/downloads
  release=https://github.com/DanielTea/transformer-snn/releases/download/spiking-byte-lm-v15
  curl -fL --retry 3 "$release/spiking-byte-lm-v15-checkpoints.sha256" \
    -o checkpoints/downloads/spiking-byte-lm-v15-checkpoints.sha256
  for part in long-windows windows transformer; do
    asset="spiking-byte-lm-v15-checkpoints-$part.tar.gz"
    curl -fL --retry 3 "$release/$asset" -o "checkpoints/downloads/$asset"
    (cd checkpoints/downloads && grep " $asset\$" spiking-byte-lm-v15-checkpoints.sha256 | shasum -a 256 -c -)
    tar -xzf "checkpoints/downloads/$asset"
    shasum -a 256 -c "checkpoints/SHA256SUMS-v15-$part"
  done
)
```

| Archive | Size | Contents of `checkpoints/stream_form` |
|---|---:|---|
| `long-windows` | 258 MB | `long-windows`: the stack on windows of 512 and then 1,024 bytes, the model of this release |
| `windows` | 258 MB | `windows`: the same stack on windows of 512 bytes throughout |
| `transformer` | 259 MB | `transformer`: the reference Transformer under the rotated optimizer |

`model.pt` holds the weights of the last update and
`model.pt.averaged` the running weight average, which is what the
study scores.

A fourth asset, `spiking-byte-lm-v15-stream-bits.tar.gz`, holds
the bits of every piece of every stream of the scorings along
streams, on the test and validation splits and on PG-19, which
the paired intervals are computed from. It unpacks to
`checkpoints/stream_form/bits`.

The training machines wrote their own absolute paths into the
checkpoints' arguments. The released files were written again
with repository-relative paths. Every tensor is bit-identical,
and the record lists both hashes of every file.
[Resaved checkpoints](../results/event_lm/stream_form/resaved_checkpoints.json) · [Release assets](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v15) · [Exact checkpoint manifest](../models/spiking-byte-lm-v15/manifest.json) · [All checksums](../results/event_lm/stream_form_checkpoints.sha256)

## Rescore

Prepare enwik8 as for v11.
[Dataset commands](RESIDUAL_SCALE.md#reproduce-the-data-and-launches)

**The stack as a stream.** `eval_stream.py` cuts a split into 16
parts that run side by side as unbroken streams. The stack takes
them in pieces of 512 bytes and carries its state from piece to
piece. A part after the first hears the 8,192 bytes before it
first, unscored. The score does not depend on the length of a
piece.

```bash
python experiments/eval_stream.py --device cuda --enwik8 data/enwik8 \
  --checkpoint checkpoints/stream_form/long-windows/model.pt.averaged \
  --json checkpoints/work/test_long_windows_stream.json \
  --bits checkpoints/work/test_long_windows_stream.npy
```

`--reach 4096 --reach-levels 5 6 7` lets the reads after levels 5
to 7 reach 4,096 cells back, which is the setting of the released
stack. The window-trained stack takes `--reach 2048
--reach-levels 4 5 6 7`.

`--teach-levels 0 1 2 3 4 5 6 7 8 9 10 --teach-threshold
16777216` adds the fast output synapses from the cells of all
eleven levels. They count in 64-bit numbers, so this scoring runs
on `--device cuda` or `--device cpu` and not on an Apple GPU.

`--split validation` scores the validation split, and `--bytes
1000000` only the first million bytes of a split, which is how
the reach and the threshold were chosen.

**The Transformer in warmed windows.** A window holds `--history`
unscored bytes before its `--context` scored ones.

```bash
python experiments/eval_warmed_windows.py --device cuda --enwik8 data/enwik8 \
  --checkpoint checkpoints/stream_form/transformer/model.pt.averaged \
  --context 16 --history 496 --json checkpoints/work/test_transformer_history496.json
```

**The Transformer along the same streams, with and without fast
output synapses.** `eval_taught_reference.py` cuts the split as
`eval_stream.py` does and keeps the bits of every window, so that
a stack can be compared with it piece by piece. A rate of 0 is
the Transformer as it is, and 0.07 is the rate that is best on
the validation split.

```bash
python experiments/eval_taught_reference.py --device cuda --enwik8 data/enwik8 \
  --checkpoint checkpoints/stream_form/transformer/model.pt.averaged \
  --context 16 --history 496 --rates 0 0.07 \
  --json checkpoints/work/test_transformer_taught_history496.json \
  --bits checkpoints/work/test_transformer_taught_history496.npy
```

**Paired intervals.** Two scorings of the same streams are
compared piece by piece. The row picks a checkpoint of a stream
scoring or a rate of a Transformer's.

```bash
python experiments/compare_stream_scores.py \
  --first checkpoints/work/test_transformer_taught_history496.json \
    checkpoints/work/test_transformer_taught_history496.npy \
  --second checkpoints/work/test_long_windows_stream.json checkpoints/work/test_long_windows_stream.npy \
  --json checkpoints/work/long_windows_over_transformer.json
```

`--device mps` scores on an Apple GPU and `--device cpu` on a
CPU. The races are computed in floating point, so a score can
differ in the fifth decimal between machines.
[Stream scorer](../experiments/eval_stream.py) · [Window scorer](../experiments/eval_warmed_windows.py) · [Transformer along streams](../experiments/eval_taught_reference.py) · [Test records](../results/event_lm/stream_form/scores)

## Write text

The stack writes a byte at a time from the state that the stream
before it left. A new byte costs one step of the stream, however
long the text behind it is.

```bash
python experiments/generate_stream.py checkpoints/stream_form/long-windows/model.pt.averaged \
  --prompt 'The first chapter' --bytes 256 --reach 4096 --reach-levels 5 6 7
```

`--temperature 0` writes the most likely byte. The default is 0.8
with `--seed 0`. `--teach-levels` and `--teach-threshold` add the
fast output synapses, which then learn from the prompt and from
the bytes that the stack writes itself. On the CPU of a laptop
the released stack writes about 15 bytes per second.
[Generator](../experiments/generate_stream.py) · [Sample](../models/spiking-byte-lm-v15/README.md#write-text)

## Rerun the scans

The reach by level and the threshold of the fast synapses were
chosen on the first million validation bytes. A scan is the
stream scorer with other settings:

```bash
python experiments/eval_stream.py --device cuda --enwik8 data/enwik8 --split validation \
  --bytes 1000000 --reach 4096 --reach-levels 5 6 7 \
  --checkpoint checkpoints/stream_form/long-windows/model.pt.averaged \
  --json checkpoints/work/reach_5_7_4096.json
```

The scans of the study ran on weights saved during training, at
21,000 to 31,500 updates. Those snapshots are copies of
`model.pt.partial` and are not released, so a scan of the
released weights gives other numbers than the records.
[Reach by level](../results/event_lm/stream_form/reach_by_level.json) · [Fast synapses](../results/event_lm/stream_form/fast_synapses.json)

The float stand-ins of the study ran on a private build.
`experiments/stack/diagnosis_build.py` patches the package in the
same way: its switches replace whole spikes by real numbers, the
pool by a LayerNorm or an exact division, and lift the bound of a
membrane. A run with any of them is no spiking network.
[Diagnosis build](../experiments/stack/diagnosis_build.py)

## Regenerate the figures

```bash
python experiments/plot_stream_form.py
python experiments/plot_stream_form.py --tables
python experiments/plot_stream_form.py --arms > docs/STREAM_FORM_ARMS.md
```

The second command prints the tables of the [study
page](STREAM_FORM.md) and the third writes the [arms
page](STREAM_FORM_ARMS.md). `python
experiments/plot_event_lm_study.py` regenerates the
[leaderboard](LEADERBOARD.md).

## Train

The study page has the complete commands of the released stack.
[Commands](STREAM_FORM.md#reproduce)

| Flag | Sets |
|---|---|
| `--stack` | A decoder without blocks: every level is a state read and then a cell layer, and `--depth` counts the levels |
| `--pool-fine 16 --pool-rest 4 --pool-rate 32` | The neurons of reads and cell layers count their threshold in quanta sixteen times finer, rest at four of them, and their pool fires once for every 64 pulses of the stream |
| `--output-pool 128` | A pool of the same kind for the output neurons, with 128 quanta to a threshold |
| `--granule-fan 1024 --granule-delays 4` | A cell hears every channel now and once more after one to three bytes |
| `--read-delays 4` | The tag neurons of a read hear every channel again one to three bytes later |
| `--read-reach 1024` | In a stream, and in a window longer than `--read-window`, a read reaches this many cells back. The cells beyond its window take the age offset of the window's oldest cell |
| `--pool-lifts-ceiling` | The pool divides a neuron's charge before its membrane bounds it |
| `--stream 20` | Train on streams: stretches of 20 consecutive windows, with the state handed on and no gradient through it |
| `--stream-groups 15` | Fifteen groups of streams take turns, so that consecutive updates hear different texts |
| `--stream-from 16500` | Windows from an empty state up to that update and streams after it |
| `--granule-gate 4`, `--granule-veto 4`, `--read-turn`, `--read-phases 16` | Gated cells, a veto dendrite, turned probes and pacemaker phases. None of them helped |
| `--feed-forward relu`, `--age-offsets 64`, `--untied-output` | Controls for the Transformer: the stack's plain units, its age offsets and its untied output |
| `--save-every 1500 --resume` | Writes the whole training state to `<checkpoint>.resume` every 1,500 updates and continues from it when it exists |
| `--stop-at 24000` | Leaves after that update with the state saved |

**Fork, do not rerun.** `experiments/stack/fork_state.py`
continues a saved state as a new arm under other settings. A fork
and its parent share everything up to the fork, so their
difference after a few thousand updates is the effect of the
change. Two runs from scratch with different seeds are 0.001 to
0.005 apart, and that hides most of what this study measured. The
parent keeps running beside the fork as its control.

**Score both with one scorer.** The trainer validates a stack on
streams as well, but the bytes it reads depend on `--ctx`. Arms
with other windows are compared with `eval_stream.py --bytes
1000000` on the same weights.

**Read a pilot from 9,000 updates on.** The Transformer takes its
lead over a stack between 6,000 and 15,000 updates, and the
distance changes little after that.

Averaged validation loss in windows of 512 bytes from an empty
state, for a pilot to be set against:

| Averaged validation, windows of 512 bytes | 3,000 | 6,000 | 9,000 | 12,000 | 18,000 | 24,000 | 45,000 | 90,000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Transformer | 1.8644 | 1.5903 | 1.5068 | 1.4574 | 1.4008 | 1.3650 | 1.2957 | 1.2348 |
| Stack on windows of 512 bytes, released as `windows` | 1.7090 | 1.5811 | 1.5246 | 1.4868 | 1.4415 | 1.4079 | 1.3425 | 1.2828 |

Both train at a rate of 0.0003. The stack leads up to 6,000
updates and is behind from 9,000 on. The released stack validates
on windows of 1,024 bytes from 24,000 updates, which is another
measure.

[All curves](../results/event_lm/stream_form/arms.json)

## Next pilots

**Score the control.** A Transformer trained as the released
stack was, on windows of 512 bytes up to 24,000 updates and on
windows of 1,024 bytes from there, was training when this release
was made. Score it along the same streams with 1,008 bytes of
history before 16 scored ones, on both splits and with its fast
output synapses at 0.07. Until then the stack is level with a
Transformer that reads 512 bytes.

**Long windows from the start, and longer ones.** Windows of
1,024 bytes from 24,000 updates gained the stack 0.030 on the
test, and its reads still gain from a reach of 4,096 cells. Train
a stack on windows of 1,024 bytes from the first update, and fork
one to windows of 2,048 bytes in batches of two, so that every
cell that a read reaches at scoring is inside the gradient.

**A second seed of both.** The released stack is ahead on the
test split and behind on validation, by more than two seeds of a
stack differ by. A second seed of the stack and of the
Transformer tells whether the order on each split is the models'
or the text's.

**Fast synapses that forget.** Their gain is larger early in a
stream than late, 0.009 in its first quarter and 0.006 in its
last, and stronger synapses fade faster. A leak or a falling rate
may keep the early gain.

**Weight decay on gains, grains and offsets.** The trainer decays
every parameter with two or more dimensions, which includes the
read gains and grains, the age offsets and the resting thresholds
of the cells. Exempting them by name is hygiene. Test it with a
fork beside its parent.

**A native integer stream.** No integer decoder has run the
stack, and no audit has counted its pulses.

## What not to try again

Each was one arm with one seed unless a second is named. The
numbers are on the [study page](STREAM_FORM.md#what-did-not-help)
and the [arms page](STREAM_FORM_ARMS.md).

- Gated cells, in four variants: with a price on their own
  spikes, without a pulse budget, with one rate for all synapses,
  and on delayed fibres. They lead for 1,500 updates and stall or
  leave the range in which they learn.
- A veto dendrite that closes a cell.
- Turned probes and pacemaker phases in place of age offsets.
- A wider learning slope for the cells.
- A learning rate of 0.001 for a stack whose pool does not lift
  the ceiling. The membranes saturate.
- An address rate of 8 for a stack. Its heads stay spread over
  the whole window.
- Training on streams in plain stretches. Rows that stay in one
  text make updates that resemble each other.
- A reach beyond 4,096 cells, and a longer reach at the lowest or
  the highest levels.

## Where things are

| Path | Holds |
|---|---|
| `snnformer/event_lm.py` | The stack (`stack`), the fine pool, delayed fibres for cells and tags, `EventLM.stream` with `StreamState`, the reach of a read, and `EventLM.teach` with the fast output synapses |
| `snnformer/transformer_reference.py` | The reference Transformer with its controls |
| `experiments/train_event_lm.py` | The trainer with the flags above |
| `experiments/eval_stream.py` | The stream scorer |
| `experiments/eval_taught_reference.py` | The Transformer along the same streams, with fast output synapses as a control |
| `experiments/compare_stream_scores.py` | Paired intervals between two scorings of the same streams |
| `experiments/generate_stream.py` | The stream generator |
| `experiments/stack/fork_state.py` | Continue a saved run as a new arm |
| `experiments/stack/diagnosis_build.py` | The float stand-ins of the study |
| `experiments/plot_stream_form.py` | Figures, tables and the arms page |
| `tests/test_stack.py` and 14 other new test files | 154 tests of the stack, the pool, the fibres, the stream form, the fast synapses, the generator, the scorers and the controls. The whole suite has 865 |
| `results/event_lm/stream_form/` | Arms, run records, test scores, reach scans and fast-synapse probes |
