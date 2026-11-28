# Volley v23 research handoff

The release restores one checkpoint, as its running average of weights: the second seed (20002) of v22's recipe,
the lifted loop with power-law cells, trained for 90,000 updates at 6e-4 on the fast path (`fsp-lift2-s2`, the model
of this release). The models it is read against are assets of earlier releases: its first seed `fsp-lift2` of
[v22](V22_HANDOFF.md), the lifted loop `fsp-lift` of [v21](V21_HANDOFF.md), and the Transformer `fsp-tf` of
[v19](V19_HANDOFF.md).

The repository carries no v23 weights. The bundled model that runs from a clone is still [v11](../models/spiking-byte-lm-v11/README.md).
[Model card](../models/spiking-byte-lm-v23/README.md) · [Knowledge-gap study](KNOWLEDGE_GAP.md) · [Roadmap](ROADMAP.md)

v23 is a release about a measure fixed in advance, and about how far two seeds lie apart.
- **Ahead on test, level on validation, not a win by the rule**: without fast output synapses the second seed scores
  1.0751 on test and 1.1051 on validation, against the Transformer's 1.0790 and 1.1048: 0.0039 ahead (interval
  0.0029 to 0.0048) and 0.0003 behind (0.0007 ahead to 0.0013 behind). The record
  (`results/event_lm/knowledge_gap/full_size_power_cells_second_seed.json`) asked for both intervals below zero.
- **Two seeds 0.0003 apart**: the first measure of the spread of one recipe at full size. The lead on test is the
  recipe's, not one seed's luck.
- **Fast output synapses cost power-law stacks 0.01**, with both seeds; they are read without them from here on, and
  a scan of their threshold is registered (`fast_synapse_threshold_power.json`).
- **Practice**: the Transformer's halves of `practice_within_reach.json` and `practice_with_rehearsal.json`, and the
  stack's half of `practice_that_stands.json` (learning from attempts below their task's mean ends the filler falls
  and costs reach; four of six runs hold, so it does not stand by its rule).

## Restore the checkpoint

The archive unpacks under `checkpoints/`:

| Archive | File | SHA-256 |
|---|---|---|
| `spiking-byte-lm-v23-checkpoints-fsp-lift2-s2.tar.gz` | `full_size_lifted_loop/fsp-lift2-s2/model.pt.averaged` | `57638a54462bf9c9717b3241505aa622dc1bd525c33d5a2a52b47354ae9b0017` |

```bash
tar -xzf spiking-byte-lm-v23-checkpoints-fsp-lift2-s2.tar.gz -C checkpoints
(cd checkpoints && shasum -a 256 -c ../results/event_lm/knowledge_gap/v23_checkpoints.sha256)
```

The file is the scored file with the training arguments that name folders rewritten to repository paths; the
weights are identical. The bits archive (`spiking-byte-lm-v23-bits.tar.gz`) holds the per-piece bits of every
scoring of this checkpoint (with and without fast output synapses, as trained, and the reach scans), so that every
paired gap in its record can be recomputed with the v22 bits of the first seed and the v19 bits of the Transformer.
Unpack it at the repository's root.

## Rescore

Without fast output synapses at the reach of 8,192, the registered reading:

```bash
python experiments/eval_stream.py --device cuda --enwik8 data/enwik8 --split test \
  --checkpoint checkpoints/full_size_lifted_loop/fsp-lift2-s2/model.pt.averaged --reach 8192 --piece 256 \
  --json s2.json --bits s2.npy
```

The paired gap to the Transformer, the two seeds' spread, and the split by what each byte's context offers:

```bash
python experiments/compare_stream_scores.py --first tf.json tf.npy --first-row 0 --second s2.json s2.npy --json s2-tf.json
python experiments/compare_stream_scores.py --first lift2.json lift2.npy --second s2.json s2.npy --json s2-lift2.json
python experiments/gap_by_piece.py --enwik8 data/enwik8 --split test --near 2032 --far 8192 \
  --first tf.json tf.npy --first-row 0 --second s2.json s2.npy --json gap.json
```

The two seeds' mean, which reading (2) uses, is the mean of their bits piece by piece (same streams, lead and
pieces), written as a scoring of its own and paired the same way.

## Train

fsp-lift2-s2 is fsp-lift2's recipe with seed 20002: `--stack-loop 2,8,2 --pulse-budget 507000 --pool-lifts-ceiling
--granule-power 8`, windows of 512 bytes in batches of 8 for 24,000 updates, then unbroken streams of windows of
2,048 bytes for 66,000 more with `--read-knowledge 128 --read-reach 2048`, at `--lr 6e-4`, one cosine over 90,000
updates. On the fast path: `--fused-optimizer --exact-products` for the windows and `--fused-reads --fused-optimizer
--exact-products` for the streams, on code 452f365. Before it trained, the same code scored fsp-lift2 on the first
million validation bytes to the same bits as the code that trained it (1.1327126674205783). Training took 4,228
seconds for the windows and about 13,720 for the streams. The full flags are in the
[manifest](../models/spiking-byte-lm-v23/manifest.json).

## Open

- The fast output synapses' threshold for power-law cells, on both seeds (`fast_synapse_threshold_power.json`).
- The gap's slope with data at full size: the power-law stack and the Transformer at 45,000, 90,000 and 180,000
  updates (`full_size_data_slope.json`), running this weekend.
- The Transformer's half of practice that stands; line heads in practice (`line_heads_in_practice.json`).
- What the stack lacks on new contexts, 0.025 to 0.030 a byte on three quarters of the bytes: the next full-size
  candidate comes from the bed, as the record says.
- A second seed of the Transformer, which every full-size claim still lacks.
- Cubic cells at full size (`fs-loop6-lift-pc3`).
