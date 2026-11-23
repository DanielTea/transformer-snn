# Volley v18 research handoff

The release restores two checkpoints, each as its running average of weights:

- the best spike-only stack so far, v17's with one word head in every read (`l2048-word-st`, the model of this release);
- the full-size stack of the learning-rate pilots with the pool that lifts the ceiling, at peak rate 1.2e-3 (`fs-st12-lift`, 24,000 updates).

The repository carries no v18 weights. The bundled model that runs from a clone is still [v11](../models/spiking-byte-lm-v11/README.md).
[Model card](../models/spiking-byte-lm-v18/README.md) · [Scaling-law study](SCALING_LAW.md) · [Roadmap](ROADMAP.md)

v18 is a release about learning rates and what caps them.
- **The test bed** wanted higher rates than 3e-4, but its flattening with data is not an artefact of the rate (two seeds).
- **At full size** the stack's membranes saturate at higher rates, which is why the Transformer gains more from them.
- **The pool that lifts the ceiling** removes the saturation at the same energy; the gap that remains is knowledge.
- **Word heads** at full size give the best stack so far, by less than their own mark.

## Restore the checkpoints

Each archive unpacks under `checkpoints/`:

| Archive | File | SHA-256 |
|---|---|---|
| `spiking-byte-lm-v18-checkpoints-l2048-word-st.tar.gz` | `knowledge_gap/l2048-word-st/model.pt.averaged` | `935ed2197c7ee1250efdbf2f14271a8c6565f0a9efd2c05628281ed51d44ba57` |
| `spiking-byte-lm-v18-checkpoints-fs-st12-lift.tar.gz` | `full_size_pilots/fs-st12-lift/model.pt.averaged` | `d867c519d85a34879dbcc08069dc0693c355c408417ce334e53315e7b71ebcc8` |

```bash
for name in l2048-word-st fs-st12-lift; do
  tar -xzf spiking-byte-lm-v18-checkpoints-$name.tar.gz -C checkpoints
done
(cd checkpoints && shasum -a 256 -c ../results/event_lm/knowledge_gap/v18_checkpoints.sha256)
```

Each file is the scored file with the training arguments that name folders rewritten to repository paths; the
weights are identical. The bits archive (`spiking-byte-lm-v18-bits.tar.gz`) holds the per-piece bits of every
scoring in this release: the word heads' and the full-size pilots'.

## Rescore

The stack at its best reading (reach 8,192, fast output synapses on all levels at 2^24):

```bash
python experiments/eval_stream.py --device cuda --enwik8 data/enwik8 --split test \
  --checkpoint checkpoints/knowledge_gap/l2048-word-st/model.pt.averaged --reach 8192 --piece 256 \
  --teach-levels 0 1 2 3 4 5 6 7 8 9 10 --teach-threshold 16777216 --json word.json --bits word.npy
```

A full-size pilot as the pilots were scored (the first 2,000,000 validation bytes as streams):

```bash
python experiments/eval_stream.py --device cuda --enwik8 data/enwik8 --split validation --bytes 2000000 \
  --checkpoint checkpoints/full_size_pilots/fs-st12-lift/model.pt.averaged --json lift.json --bits lift.npy
```

How often membranes sit at their bound (or, under a lifting pool, counts at their cap), level by level:

```bash
python experiments/membrane_saturation.py --enwik8 data/enwik8 --bytes 32768 \
  --checkpoint checkpoints/full_size_pilots/fs-st12-lift/model.pt.averaged --json saturation.json
```

## Train

The pool that lifts the ceiling is `--pool-lifts-ceiling`; noisy thresholds are `--cell-noise 0.5`; gated cells are
`--granule-gate 4 --granule-shunt` with fewer cells for equal parameters. The full-size pilot recipe is the stack of
`full_size_rate_pilot.json` (v16's stack with `--read-forget`, windows of 512 bytes, batch 8, 24,000 updates,
1,000 warmup updates, cosine schedule) at the peak rate given.

## Open

- The full-size pair at each model's best rate (`full_size_best_rate_pair.json`, training).
- The data-scaling curves at each model's best rate (`data_scaling_long.json`).
- Quantal release and stochastic firing (`weight_precision_with_data.json`, `cell_dither_with_data.json`,
  `randomness_at_best_rate.json`).
- The full size's cell shape at four seeds (`cell_shape_with_data.json`).
- The test bed's screens of gated cells and noisy thresholds by their slope, and the full-size screen's third arm.
