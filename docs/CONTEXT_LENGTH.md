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

Both architectures completed optimizer steps on two real 2,048-byte
enwik8 windows per batch with MPS fallback disabled. The comparison
continues the 1.63M-parameter seed-0 models using those windows. Both
arms see 6,553,600 additional targets with seed 711. The SNN uses
the chronological held-charge gradient and its usual 310,000-pulse
training target. The Transformer will be evaluated with up to
2,048 bytes of attention context.
[SNN training smoke](../results/event_lm/context_extrapolation/event_correction_code_s711_long_context_smoke.json) ·
[Transformer training smoke](../results/event_lm/context_extrapolation/transformer_s711_long_context_smoke.json)

The `spiking` array and corpus setup are in the release handoff. Run
both from the repository root in Bash.
[Training setup](HANDOFF.md#train-and-continue)

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
continuation=(--data data/enwik8 --device mps --ctx 2048 --bs 2
  --steps 1600 --lr 0.0004 --warmup 64 --eval-every 200 --eval-bs 2
  --eval-batches 8 --final-eval-batches 48 --validation-only
  --seed 711 --tag ctx2048_continuation)
run=checkpoints/long-context-continuation

python experiments/train_event_lm.py "${continuation[@]}" "${spiking[@]}" \
  --local-trace --trace-write-credit --initialize-from models/spiking-byte-lm-v2/weights.pt \
  --results-dir "$run/results" --checkpoint "$run/trace_ctx2048_s711.pt"
python experiments/train_event_lm.py "${continuation[@]}" \
  --variant transformer --dim 192 --depth 4 --heads 4 --mlp-ratio 3.3515625 \
  --initialize-from checkpoints/scale-1630k-study/transformer_s0.pt \
  --results-dir "$run/results" --checkpoint "$run/transformer_ctx2048_s711.pt"
```
