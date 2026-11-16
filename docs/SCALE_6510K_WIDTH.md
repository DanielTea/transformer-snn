# Doubling residual width at fixed stream state

This study doubles the seven-block spike decoder's residual width
from 384 to 768. Its 187 expansion channels, axon delays, local
held-charge cells, four-head error bank, and 16× firing resolution
stay at the seven-block geometry. These expansion and memory
populations determine the persistent stream state. The wider
residual neurons reset their firing phases each byte. They add
synaptic capacity without adding a recurrent tensor.

| Model | Parameters | Stream state |
|---|---:|---:|
| Seven-block SNN at width 384 with compact axons | 3,257,088 | 19,536 B |
| Seven-block SNN at width 768 with compact axons | 6,514,176 | 19,536 B |
| Four-block Transformer at width 384 | 6,514,944 | 6 MiB at 512 bytes |

The wider Transformer uses six 64-channel heads and a SwiGLU
hidden width of 879. The SNN footprint was measured with the
native integer decoder. Both wider arms passed three training
steps on full enwik8 batches of eight 512-byte windows on
Apple Metal, with CPU fallback disabled. The resulting SNN
matched native CPU, native chunked and Metal output counts on
4,096 enwik8 validation bytes. The single-byte path matched on
the first 513.
[SNN training smoke](../results/event_lm/scale_6510k_width/event_correction_code_s0_scale_6510k_width_smoke.json) ·
[Transformer training smoke](../results/event_lm/scale_6510k_width/transformer_s0_scale_6510k_width_smoke.json) ·
[Native integer audit](../results/event_lm/scale_6510k_width/native_width_smoke.json)

## Full comparison

Both arms train from initialization and use seed 0, 6,400 updates,
26,214,400 enwik8 targets, a 160-step warmup and a peak learning
rate of 0.003. The SNN uses chronological held-charge gradients
and the 310,000-pulse target. It retains the same firing rule and
bank geometry as the released seven-block model. Chronological
write credit changes how gradients reach the trace during training.
The separate 3.26M run used the original trace gradient.
[3.26M control](SCALE_3260K.md) · [Chronological gradient](TRACE_WRITE_CREDIT.md)

Run from the repository root in Bash with the `spiking` array
defined in the handoff.
[Training setup](HANDOFF.md#train-and-continue)

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
options=(--data data/enwik8 --device mps --ctx 512 --bs 8 --steps 6400
  --lr 0.003 --warmup 160 --eval-every 400 --eval-bs 8 --eval-batches 8
  --final-eval-batches 48 --validation-only --seed 0 --tag scale_6510k_width)
run=checkpoints/work/scale-6510k-width
python experiments/train_event_lm.py "${options[@]}" "${spiking[@]}" \
  --dim 768 --hidden 187 --depth 7 --local-trace --trace-write-credit --compact-axon-state \
  --results-dir "$run/results" --checkpoint "$run/trace_s0.pt"
python experiments/train_event_lm.py "${options[@]}" --variant transformer \
  --dim 384 --depth 4 --heads 6 --mlp-ratio 3.43359375 \
  --results-dir "$run/results" --checkpoint "$run/transformer_s0.pt"
```
