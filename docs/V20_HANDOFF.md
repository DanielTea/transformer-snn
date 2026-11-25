# Volley v20 research handoff

The release restores one checkpoint, as its running average of weights: the best spike-only stack so far, v19's
stack with its levels 2 to 8 run twice through the same synapses, trained for 90,000 updates at 6e-4 beside v19's
pair (`fsp-loop`, the model of this release). The pair it is read against, the plain stack `fsp-st` and the
Transformer `fsp-tf`, are assets of [v19](V19_HANDOFF.md).

The repository carries no v20 weights. The bundled model that runs from a clone is still [v11](../models/spiking-byte-lm-v11/README.md).
[Model card](../models/spiking-byte-lm-v20/README.md) · [Scaling-law study](SCALING_LAW.md) · [Roadmap](ROADMAP.md)

v20 is a release about depth by recurrence.
- **The loop holds at full length**: 1.1039 on test, 0.0090 ahead of the plain stack, and 0.025 behind the Transformer where the plain stack is 0.034 behind.
- **Its gain is knowledge**: nine tenths of it is on new contexts, where the distance to the Transformer falls from 0.090 to 0.078.
- **The loop's form**: the pool that lifts the ceiling helps it (0.0095 in a full-size pilot), the old pulse budget does not suffice, and skip fibres add nothing to it on the test bed.
- **The gated cells**: a homeostatic gate is the first spiking gate that catches up with data, a signed gate falls behind, and in floats no bounded gate falls behind with data.
- **Sleep**: a stack that replays every finished window of its stream and rewrites its reads' cells, built and on test.

## Restore the checkpoint

The archive unpacks under `checkpoints/`:

| Archive | File | SHA-256 |
|---|---|---|
| `spiking-byte-lm-v20-checkpoints-fsp-loop.tar.gz` | `full_size_loop/fsp-loop/model.pt.averaged` | `8ae6acc5eade13e62115e1702f0c2e1bd77f13b6b3786c136fddd3bddd81ae3d` |

```bash
tar -xzf spiking-byte-lm-v20-checkpoints-fsp-loop.tar.gz -C checkpoints
(cd checkpoints && shasum -a 256 -c ../results/event_lm/knowledge_gap/v20_checkpoints.sha256)
```

The file is the scored file with the training arguments that name folders rewritten to repository paths; the
weights are identical. Its state dictionary names every pass of the loop, so the synapses of levels 2 to 8 are stored
twice, as identical copies. The bits archive (`spiking-byte-lm-v20-bits.tar.gz`) holds the per-piece bits of every
scoring recorded since v19: the looped stack's, the loop variants', the test bed's screens' (energy controls, looped
skip fibres, homeostatic, proportional and signed gates, the coarse output, the first sleep runs) and the float
diagnostic of bounded gates, so that every paired gap in the records can be recomputed. Unpack it at the
repository's root.

## Rescore

The looped stack at its best reading (reach 8,192, fast output synapses on the cells of all 18 passes at 2^24):

```bash
python experiments/eval_stream.py --device cuda --enwik8 data/enwik8 --split test \
  --checkpoint checkpoints/full_size_loop/fsp-loop/model.pt.averaged --reach 8192 --piece 256 \
  --teach-levels 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 --teach-threshold 16777216 \
  --json loop.json --bits loop.npy
```

The paired gaps to v19's plain stack and Transformer (their scorings are in the v19 bits archive) and the split by
what each byte's context offers:

```bash
python experiments/compare_stream_scores.py --first st.json st.npy --second loop.json loop.npy --json loop-st.json
python experiments/compare_stream_scores.py --first tf.json tf.npy --first-row 0 --second loop.json loop.npy --json loop-tf.json
python experiments/gap_by_piece.py --enwik8 data/enwik8 --split test --near 2032 --far 8192 \
  --first tf.json tf.npy --first-row 0 --second loop.json loop.npy --json gap.json
```

The share of membrane entries at their bound:

```bash
python experiments/membrane_saturation.py --checkpoint checkpoints/full_size_loop/fsp-loop/model.pt.averaged \
  --enwik8 data/enwik8 --split validation --bytes 65536 --json saturation.json
```

## Train

The looped stack is the pair's stack recipe (`full_size_best_rate_pair.json`) with `--stack-loop 2,8,2
--pulse-budget 507000` in both stages: windows of 512 bytes in batches of 8 for 24,000 updates, then unbroken streams
of windows of 2,048 bytes for 66,000 more, with `--read-knowledge 128 --read-reach 2048`, at `--lr 6e-4`. The full
flags are in the [manifest](../models/spiking-byte-lm-v20/manifest.json). New since v19: the homeostatic gate is
`--granule-gate-home SHARE` on a ramp gate, the signed gate `--granule-gate-signed` on an inner gate, and sleep
`--stack-sleep WINDOW`.

## Open

- The looped stack's next full-length run, in the form that two full-size pilots pick: a third pass
  (`full_size_loop_passes.json`) and the Transformer's rate under the lifting pool (`full_size_loop_variants.json`).
- Sleep on the test bed at 24,000 updates, alone and on the loop (`sleep_with_data.json`).
- The homeostatic gate at 96,000 updates (`homeostatic_gate_at_length.json`).
- The unit's price in floats at full size (`full_size_units_in_floats.json`).
- The slope far below the threshold on the server's seeds and at full size (`subthreshold_slope_with_data.json`).
