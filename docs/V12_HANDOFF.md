# ShuntFormer v12 research handoff

The release restores 21 checkpoints of the coasting-head study.
Four are trained from scratch on identical windows, five are the
v11 arms after 16 times the training, three continue the 16×
clock decoder, and nine belong to the follow-up runs. The
repository carries no v12 weights. The bundled model that runs
from a clone is still
[v11](../models/spiking-byte-lm-v11/README.md).
[Model card](../models/spiking-byte-lm-v12/README.md) ·
[Measurements, circuit and limits](COASTING_HEADS.md)

## Restore the checkpoints

Use Python 3.12 and install `requirements.txt`. For dataset
preparation install `requirements-benchmarks.txt` as well. The
head replay compiles
[`snnformer/coasting_heads.c`](../snnformer/coasting_heads.c)
with `cc` or `gcc` on first use. Set `CC` to use another
compiler.

The training runs used two NVIDIA RTX PRO 6000 Blackwell GPUs
with Python 3.12 and the CUDA build of PyTorch 2.15.0. The
fit-only sweeps on the frozen v11 decoder ran on an Apple M3 Max.
Two CUDA runs with one seed are not bit-identical, so a retrained
arm will not reproduce a checkpoint hash.

The checkpoints are in three archives. Download and verify the
ones you need from the repository root.

```bash
(
  set -e
  mkdir -p checkpoints/downloads
  release=https://github.com/DanielTea/transformer-snn/releases/download/spiking-byte-lm-v12
  curl -fL --retry 3 "$release/spiking-byte-lm-v12-checkpoints.sha256" \
    -o checkpoints/downloads/spiking-byte-lm-v12-checkpoints.sha256
  for part in scratch continuations followups; do
    asset="spiking-byte-lm-v12-checkpoints-$part.tar.gz"
    curl -fL --retry 3 "$release/$asset" -o "checkpoints/downloads/$asset"
    (cd checkpoints/downloads && grep " $asset\$" spiking-byte-lm-v12-checkpoints.sha256 | shasum -a 256 -c -)
    tar -xzf "checkpoints/downloads/$asset"
    shasum -a 256 -c "checkpoints/SHA256SUMS-v12-$part"
  done
)
```

| Archive | Size | Contents |
|---|---:|---|
| `scratch` | 518 MB | `scratch-study`: plain, synapse and Transformer arms from scratch. `v12-scratch`: the head arm, the model of this release |
| `continuations` | 1.00 GB | `long16-study`: five v11 arms after 38,400 updates. `v12-long`: plain, synapse and head continuations of the 16× clock decoder |
| `followups` | 1.16 GB | `v12-rep`: the second pair. `v12-round2-same`, `v12-round2-refit`: round 2. `v12-v2pair`, `scratch-v2-study`, `v12-snaps`: the second mixture |

Every archive also holds `donor/clock-synapses-readout.pt`, the
24 synapse strengths of the released v11 clock decoder in a file
of 2 kB. The synapse arms were trained beside it. The v11
checkpoint `surprise-clock-study/clock-synapses_s10007.pt` gives
the trainer the same strengths. The fit-only measurements and the
16× runs also need the v10 and v11 checkpoints.
[v11 restore](V11_HANDOFF.md#restore-the-checkpoints)

The two files in `scratch-v2-study` end in `.pt.partial`. They
are the last saved weights of two runs that were stopped, after
30,000 and 66,000 of 90,000 updates.
[Release assets](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v12) ·
[Exact checkpoint manifest](../models/spiking-byte-lm-v12/manifest.json) ·
[All checksums](../results/event_lm/coasting_heads_checkpoints.sha256)

## Replay and rescore

Prepare enwik8 and the v1 training catalogue as for v11.
[Dataset commands](RESIDUAL_SCALE.md#reproduce-the-data-and-launches)

The held-out documents and the second mixture come from
FineWeb-Edu, FineMath and Stack-Edu. The second catalogue goes to
`data/training-v2/prepared` and its held-out documents to
`data/training-v2/prepared/heldout`.
[Sources](../results/event_lm/training_mixture/sources_v2.json) ·
[Catalogue](../results/event_lm/training_mixture/prepared_v2.json) ·
[Held-out documents](../results/event_lm/training_mixture/heldout_v2.json)

```bash
python experiments/prepare_byte_training_mixture.py \
  --source-manifest results/event_lm/training_mixture/sources_v2.json \
  --sources data/training-v2/sources --output data/training-v2/prepared
```

One launcher runs the study in order. It keeps every checkpoint
and record that already exists, so with restored checkpoints it
only rescores.

```bash
DEVICE=cuda bash experiments/run_coasting_heads.sh checkpoints
```

The launcher was assembled from the job scripts of the recorded
runs after they finished. Its control flow was checked in a dry
run, and one head continuation was started from it for 120
updates. It has not been run end to end.

A single head decoder is rescored by replay. The probe records
the decoder over enwik8 validation and test, fits the strengths
of the v11 synapses and of the heads on validation, and replays
both over the test split.

```bash
python experiments/probe_coasting_heads.py --enwik8 data/enwik8 --device cuda \
  --checkpoint checkpoints/v12-scratch/heads_s20001.pt \
  --cache-dir checkpoints/work/cache-heads --work-dir checkpoints/work/build \
  --synapses alone --score-test --setting 'orders=1,2,3,4,5,6,7,8;tape=65536' \
  --json checkpoints/work/recall_scratch-heads.json
```

The recorded outputs take several GB per decoder. Delete the
cache directory after the fit.

A head decoder is trained beside fixed strengths.
`--head-readout` names the strength file and `--load-weights`
starts from another decoder's weights with a fresh optimizer.
The launcher has the complete commands.

```bash
python experiments/train_event_lm.py --head-readout results/event_lm/coasting_heads/readout/heads.json \
  --load-weights checkpoints/long16-study/clock_s10007.pt ...
```

`forward_chunk` and `step` raise for a decoder with heads. The
streaming evaluator, the native integer decoder and the generator
therefore do not accept a head checkpoint yet. Windows that
start from an empty stream work, which is how the history split
and the held-out documents are scored.

## Regenerate the figures

```bash
python experiments/plot_coasting_heads.py
python experiments/plot_coasting_heads.py --tables
```

The second command prints the tables of the
[study page](COASTING_HEADS.md). The
[leaderboard](LEADERBOARD.md) is unchanged, because a head
decoder has no streaming score to rank.
