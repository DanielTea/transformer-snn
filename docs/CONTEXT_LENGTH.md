# Context length in the Transformer comparison

The matched full-attention references were trained on 512-byte enwik8
windows. They use rotary positions and can run a longer sequence
with the same weights. This evaluation tests extrapolation on enwik8
and the 16 complete PG-19 books. Each long window scores 1,024 targets
after up to 1,024 warmup bytes. Both checkpoints use the same
temperature grid on eight separate validation books.
[Transformer implementation](../snnformer/transformer_reference.py) ·
[Evaluation runner](../experiments/eval_event_lm.py)

| Transformer | enwik8 at 512 bytes | enwik8 at 2,048 bytes | PG-19 at 512 bytes | PG-19 at 2,048 bytes | Key and value state at 2,048 bytes |
|---|---:|---:|---:|---:|---:|
| 1,630,848 parameters | 1.81449 | 5.40489 | 3.11725 | 4.88472 | 12 MiB |
| 3,256,832 parameters | 1.86093 | 5.45663 | 3.05929 | 4.73640 | 16 MiB |

The scores are in bits per byte. Both contexts cover the same 4,999,999
enwik8 targets and 8,404,368 PG-19 targets. PG-19 loss rose by 1.76747
and 1.67711 bits per byte with the original weights. These long-window
scores measure extrapolation from 512-byte training.
[Frozen-weight enwik8 records](../results/event_lm/context_extrapolation/enwik8_transformers_context2048.json) ·
[Frozen-weight PG-19 records](../results/event_lm/context_extrapolation/pg19_transformers_context2048.json) ·
[512-byte comparison](SCALE_3260K.md)

On four 2,048-byte enwik8 validation windows, scoring the last 1,024
bytes differed between CPU and MPS by at most 0.000012 bits per byte.
On MPS, the first 512 output positions were identical when processed
alone and as a prefix of the 2,048-byte input.
[Long-window verification](../results/event_lm/context_extrapolation/long_window_parity.json)

## Train the longer context

Both models continued their 1.63M-parameter seed-0 parents on two
2,048-byte enwik8 windows per batch, with 6,553,600 additional
targets and seed 711. The SNN used chronological held-charge
gradients and a 310,000-pulse training target. Full attention saw
up to 2,048 bytes per scoring window. The comparison below also
includes the same model families and training exposure at 512 bytes.

| Model | Trained window | enwik8 bpb | PG-19 bpb | PG-19 raw bpb | State |
|---|---:|---:|---:|---:|---:|
| Held charge with write credit | 512 | 1.98691 | 2.99097 | 3.10893 | 21,320 B |
| Full attention | 512 | 1.79625 | 3.11887 | 3.35801 | 3 MiB |
| Held charge with write credit | 2,048 | 1.98884 | 2.96180 | 3.07240 | 21,320 B |
| Full attention | 2,048 | 1.81712 | 3.11395 | 3.33558 | 12 MiB |

With longer training windows, the SNN's PG-19 loss fell by
0.02916 bits per byte and its enwik8 loss rose by 0.00193. At
2,048 bytes it led on all 16 PG-19 books, with an aggregate lead
of 0.15215 calibrated bits per byte and 0.26317 at raw scale.
Full attention led on enwik8 by 0.17172 at 2,048 bytes and reached
its lowest enwik8 loss with the 512-byte continuation. The matched
comparison uses one continuation seed.
[2,048-byte training, per-book scores, and audit](../results/event_lm/long_context_continuation) ·
[512-byte training and scores](../results/event_lm/trace_write_credit)

The `spiking` array and corpus setup are in the release handoff. Run
both from the repository root in Bash, using a new output directory.
[Training setup](HANDOFF.md#train-and-continue)

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
continuation=(--data data/enwik8 --device mps --ctx 2048 --bs 2
  --steps 1600 --lr 0.0004 --warmup 64 --eval-every 200 --eval-bs 2
  --eval-batches 8 --final-eval-batches 48 --validation-only
  --seed 711 --tag ctx2048_continuation)
run=checkpoints/work/long-context-continuation

python experiments/train_event_lm.py "${continuation[@]}" "${spiking[@]}" \
  --local-trace --trace-write-credit --initialize-from models/spiking-byte-lm-v2/weights.pt \
  --results-dir "$run/results" --checkpoint "$run/trace_ctx2048_s711.pt"
python experiments/train_event_lm.py "${continuation[@]}" \
  --variant transformer --dim 192 --depth 4 --heads 4 --mlp-ratio 3.3515625 \
  --initialize-from checkpoints/scale-1630k-study/transformer_s0.pt \
  --results-dir "$run/results" --checkpoint "$run/transformer_ctx2048_s711.pt"
```

## Repeat at 3.26M parameters

Both model families continued their released seed-0 checkpoints
on 6,553,600 more enwik8 targets with seed 711. The 512-byte arms
use eight windows per batch. The 2,048-byte arms use two. Each
model has seen 32,768,000 training targets at evaluation. The
SNN uses chronological held-charge gradients and a 310,000-pulse
training target.

| Model | Trained window | enwik8 bpb | PG-19 bpb | PG-19 raw bpb | State |
|---|---:|---:|---:|---:|---:|
| Seven-block held charge | 512 | 1.96430 | 2.88668 | 2.98308 | 31,692 B |
| Full attention | 512 | 1.84280 | 3.06325 | 3.26568 | 4 MiB |
| Seven-block held charge | 2,048 | 1.97126 | 2.87693 | 2.96339 | 31,692 B |
| Full attention | 2,048 | 1.85459 | 3.05282 | 3.24905 | 16 MiB |

The SNN scored lower on each of the 16 PG-19 books at both lengths.
At 2,048 bytes its aggregate lead was 0.17589 calibrated bits per
byte and 0.28566 at raw scale. Full attention led enwik8 by 0.12151
bits per byte at 512 bytes and by 0.11667 at 2,048. The shorter
windows gave lower enwik8 loss within each model family. Native
CPU and MPS counts matched on 4,096 enwik8 test bytes in both
SNN audits. The 512- and 2,048-byte models emitted 308,067 and
306,441 pulses per byte, respectively.
[Training, corpus scores, and native audit](../results/event_lm/scale_3260k_context) ·
[Released parents](SCALE_3260K.md)
