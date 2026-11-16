# Growing residual width from a trained spiking decoder

The seven-block decoder can add residual neurons while keeping
its fixed axons, local held-charge cells, and verified error
bank. The wider initialization copies every existing synapse.
Synapses that produce charge in a new residual neuron start at
zero, and the synapses that receive from it start at their normal
random values. The new channels are silent at initialization.
Their outgoing synapses let gradients reach the new neurons
as soon as training starts.

The inhibitory threshold is sized from the two expansion
populations that it regulates. This maintains a 2,048-tick
threshold as the residual stream grows from 384 to 768 channels.
The wider stream may deliver more charge to the pool as its new
neurons learn to fire. The pool then raises the firing threshold
of the content and steering neurons that receive those signals.
The extra residual neurons reset their firing phases every byte.
[Growth and inhibition implementation](../snnformer/event_lm.py)

Function-preserving widening has prior work in Net2Net. Here the
added spiking neurons start silent and use the same inhibitory
pool and bounded recurrent state as the trained circuit.
[Net2Net](https://arxiv.org/abs/1511.05641)

## Measured initialization

Source and grown SNN matched every one of the 1,048,576 native
output counts on the first 4,096 enwik8 validation bytes.
Each emitted a total of 1,292,767,233 pulses. Single-stream CPU
chunked and Metal batch output also matched. Compact axons
brought persistent state to 19,536 bytes. A next-byte training
pass on 512 real enwik8 targets produced nonzero gradients for
the added neurons in the input and all seven blocks.
[Native parity, pulse counts, and gradient check](../results/event_lm/grown_6510k_residual/warm_growth.json)

The matched Transformer grows from its trained 3.26M checkpoint.
The SwiGLU intermediate width expands from 696 to 1,756 and
starts the new units with zero output weights. Its residual size,
attention, and context stay at the original reference geometry.
The Metal logits matched exactly across eight enwik8 validation
windows containing 4,096 targets. The added outputs received
gradients in every block.
[Transformer growth implementation](../snnformer/transformer_reference.py) ·
[Matched initialization check](../results/event_lm/grown_6510k_residual/warm_growth_transformer.json)

## Matched continuation

| Model | Parameters | Persistent stream state |
|---|---:|---:|
| SNN with wider residual stream | 6,514,176 | 19,536 B |
| Transformer with wider SwiGLU | 6,513,152 | 4 MiB at 512 bytes |

Both continue from the released 3.26M pair with seed 711, 1,600
AdamW steps and 6,553,600 additional enwik8 targets. The
continuations use 512-byte windows, a peak learning rate of
0.0004, and 64 warmup steps. The SNN uses the original 310,000
pulse target and chronological held-charge gradients. An
intervention silences the added residual neurons after training
to measure how their activity affects validation loss.
[Earlier matched 3.26M continuation](CONTEXT_LENGTH.md#repeat-at-326m-parameters) ·
[SNN Metal smoke](../results/event_lm/grown_6510k_residual/event_correction_code_s711_grown_residual_smoke.json) ·
[Transformer Metal smoke](../results/event_lm/grown_6510k_residual/transformer_s711_grown_ffn_smoke.json)

Run from the repository root in Bash with the `spiking` array
from the handoff. Both released 3.26M parents are required.
[Training and checkpoint setup](HANDOFF.md#train-and-continue)

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
warm=(--data data/enwik8 --device mps --ctx 512 --bs 8 --steps 1600
  --warmup 64 --lr 0.0004 --seed 711 --eval-every 200 --eval-bs 8
  --eval-batches 8 --final-eval-batches 48 --validation-only)
run=checkpoints/work/grown-6510k-residual

python experiments/train_event_lm.py "${warm[@]}" "${spiking[@]}" \
  --dim 768 --hidden 187 --depth 7 --local-trace --trace-write-credit \
  --compact-axon-state --inhibition-by-expansion --grow-residual \
  --initialize-from checkpoints/scale-3260k-study/trace_s0.pt \
  --tag scale_6510k_grown_residual --results-dir "$run/results" --checkpoint "$run/grown_s711.pt"
python experiments/train_event_lm.py "${warm[@]}" --variant transformer \
  --dim 256 --depth 4 --heads 4 --mlp-ratio 10.2890625 --grow-feed-forward \
  --initialize-from checkpoints/scale-3260k-study/transformer_s0.pt \
  --tag scale_6510k_grown_ffn --results-dir "$run/results" --checkpoint "$run/transformer_s711.pt"
```

## Finer synaptic charge at the same firing resolution

A second continuation grows the SNN and changes the routing
blocks from six to eight fractional charge bits. The initializer
places the original synapses exactly on their trained quantized
grid. In the finer representation, both the old synaptic
currents and their firing thresholds use four times as many
ticks. The signed pulse counts therefore match at initialization.
Training can move the router synapses in quarter-size steps
while using the same 16× firing resolution and pulse target.

On the 4,096-byte validation sample, all output counts matched
the parent on CPU and Metal. Both decoders emitted 1,292,767,233
pulses in total. The finer, wider model uses 19,536 bytes of
state. Its paired training run adds `--router-charge-bits 8`
to the grown SNN command and uses a separate output path.
[Finer-grid initialization audit](../results/event_lm/grown_6510k_residual/warm_fine_charge.json)
