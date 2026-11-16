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

| Continuation | enwik8 bpb | PG-19 bpb | PG-19 raw bpb |
|---|---:|---:|---:|
| Grown 6.51M SNN | 1.95646 | 2.86904 | 2.95290 |
| Grown 6.51M Transformer | 1.84266 | 3.06318 | 3.26549 |

The wider SNN reduced enwik8 loss by 0.00784 bits per byte
and PG-19 loss by 0.01764 relative to the 512-byte 3.26M
SNN continuation. The grown SNN led the matched Transformer
by 0.19414 bits per byte on PG-19 after calibration and
0.31259 in raw loss. The calibrated loss was lower on all
16 complete test books. The Transformer led enwik8 by 0.11380
bits per byte. First occurrences of eight-byte prefixes and
repeats at least 512 bytes away account for 80.6% of that gap.
[SNN book scores](../results/event_lm/grown_6510k_residual/pg19_grown_s711.json) ·
[Transformer book scores](../results/event_lm/grown_6510k_residual/pg19_transformer_s711.json) ·
[Prefix-age chart](../results/figures/event_lm_grown_prefix_age.png)

On a separate 262,144-target validation probe, silencing
the added residual neurons raised loss from 1.87218 to
1.88876. Native decoding matched Metal and chunked output
on 4,096 enwik8 test bytes. The native audit counted
306,923 pulses per byte and carries activity for the
added neurons in each block.
[Added-neuron intervention](../results/event_lm/grown_6510k_residual/probe_growth_262k.json) ·
[Native audit](../results/event_lm/grown_6510k_residual/native_grown_s711.json)

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
[Finer-grid initialization audit](../results/event_lm/grown_fine_charge/warm_fine_charge.json)

The finer-charge continuation scored **1.95110** on enwik8 and
**2.85083** on calibrated PG-19. The gains over the six-bit
continuation are 0.00536 and 0.01821 bits per byte. Raw PG-19 loss
was 2.93341. The native pulse count rose by 0.84% to 309,505 per
byte. State stayed at 19,536 bytes. The finer model also had the
lower training-time validation loss and is the v4 reference.
[Fine-charge training and evaluation](../results/event_lm/grown_fine_charge) ·
[v4 model card](../models/spiking-byte-lm-v4/README.md)

## Extend training exposure

The data-scale comparison continues the two trained grown
checkpoints on 16,384,000 more enwik8 targets each. It uses
4,000 steps at 512 bytes and batch size eight, seed 1109,
128 warmup steps, and a peak learning rate of 0.0003.
Each model reaches 49,152,000 cumulative targets. The SNN
uses the same six-bit router charge and 310,000-pulse target
as the first grown continuation.

| Model at 49,152,000 training targets | enwik8 bpb | PG-19 bpb | Raw PG-19 bpb | Stream state |
|---|---:|---:|---:|---:|
| Six-bit grown SNN | 1.89440 | 2.82683 | 2.91901 | 19,536 B |
| Grown full-attention Transformer | 1.80629 | 3.03772 | 3.24647 | 4 MiB |

The additional exposure lowered enwik8 loss by 0.06206 bits per
byte for the SNN and 0.03637 for full attention. The gap narrowed
from 0.11380 to 0.08811, a 22.6% reduction over this interval.
First occurrences of eight-byte prefixes and repeats at least
512 bytes away contribute 80.1% of the remaining gap. The
calibrated PG-19 loss is lower for the SNN on all 16 books.
The SNN emits 309,688 pulses per byte in the native audit.
[Matched scores and audit](../results/event_lm/scale_6510k_data) ·
[Exposure chart](../results/figures/event_lm_6510k_data.png) ·
[Continuation commands](HANDOFF.md#earlier-exposure-stages)

On 65,536 enwik8 validation targets, resetting the local charge
every 512 bytes raised loss from 1.83258 to 1.84087. Muting the
bank's correction projection raised it to 1.83320. On a
separate 16,384-byte validation activity sample, 87.9% of
held-charge read pulses came from cells changed in the preceding
seven bytes. When a cell held charge and its steering count was
zero, 96.8% of those channel steps rewrote charge.
[State intervention](../results/event_lm/scale_6510k_data/probe_trace_s1109.json) ·
[Charge age profile](../results/event_lm/scale_6510k_data/trace_ages_s1109.json) ·
[Expansion experiment](EXPANSION_GROWTH.md)

### 65,536,000 targets

The second continuation adds 4,000 steps and 16,384,000 targets
to each of the 49.15M checkpoints. It uses seed 2503 and retains
the window length, batch size, learning rate, and warmup.

| Model at 65,536,000 training targets | enwik8 bpb | PG-19 bpb | Raw PG-19 bpb | Stream state |
|---|---:|---:|---:|---:|
| Six-bit grown SNN | 1.86525 | 2.81231 | 2.90664 | 19,536 B |
| Grown full-attention Transformer | 1.77728 | 2.98071 | 3.18604 | 4 MiB |

The SNN gained 0.02914 bits per byte on enwik8 and full attention
gained 0.02901. The gap is 0.08798, close to 0.08811 at 49.15M.
First occurrences of eight-byte prefixes and repeats at least 512
bytes apart account for 78.6% of the gap. On PG-19 the calibrated
SNN loss is lower on all 16 books. The SNN emits 308,230 pulses per
byte in the native audit. Both continuations use one seed per stage.
[Paired held-out scores and audits](../results/event_lm/scale_6510k_next_data) ·
[Prefix age chart](../results/figures/event_lm_65536_prefix_age.png)

On the same 65,536-target validation sample used at 49.15M, replacing
input-dependent inhibition with per-block constants raised loss from
1.79955 to 1.84712. Resetting the held charge every 512 bytes raised
it to 1.80872, and muting bank correction spikes raised it to 1.79993.
On 16,384 validation bytes, 87.5% of held-charge read pulses came from
cells that had changed within seven bytes.
[SNN state interventions](../results/event_lm/scale_6510k_next_data/probe_trace_s2503.json) ·
[Held-charge activity](../results/event_lm/scale_6510k_next_data/trace_ages_s2503.json)

```bash
PYTHON=.venv/bin/python bash experiments/run_v4_continuation.sh scale-data-snn
PYTHON=.venv/bin/python bash experiments/run_v4_continuation.sh scale-data-transformer
```

Run the two validation diagnostics against the archived 65.54M SNN.

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
mkdir -p checkpoints/work
snn=checkpoints/scale-6510k-next-data-study/trace_s2503.pt
python experiments/probe_event_state.py --enwik8 data/enwik8 --checkpoint "$snn" \
  --device mps --chunk-size 512 --targets 65536 --json checkpoints/work/probe_trace_s2503.json
python experiments/profile_charge_trace.py --enwik8 data/enwik8 --checkpoint "$snn" \
  --split validation --bytes 16384 --json checkpoints/work/trace_ages_s2503.json
```

## Double the residual stream at fixed persistent state

The next width comparison grows the SNN's residual population
from 768 to 1,536. Its signed content and steering populations
retain their existing widths and delay allocations. The SNN
has 13,028,352 trainable parameters and uses 19,536 stream
bytes. On the 49.15M checkpoint, the initializer preserved
every output count and all 1,293,067,588 emitted pulses over
4,096 validation bytes. The widened model also completed three
training batches on Metal.
[SNN warm start and training smoke](../results/event_lm/residual_growth_13030k)

The Transformer grows its feed-forward width from 1,756 to
3,877, for 13,028,864 parameters. Its warm logits matched on
eight 512-byte Metal validation windows, and the added output
weights received gradients in all four blocks.
[Transformer warm start](../results/event_lm/residual_growth_13030k/warm_transformer.json)

The full comparison starts from the completed 65.54M pair
and continues both original-width controls. Each arm uses
seed 3307, 1,600 steps, 512-byte windows, and 6,553,600
additional targets, for 72,089,600 targets per model. The v5
archive contains both starting checkpoints. The launcher defaults
to Metal.

Widening the archived SNN preserved every output count and all
1,288,006,522 emitted pulses on 4,096 validation bytes. All seven
blocks received gradients in the new residual projections. The
archived Transformer also preserved its logits and gave the new
feed-forward outputs gradients in all four blocks.
[65.54M SNN warm start](../results/event_lm/residual_growth_13030k/warm_residual_65m.json) ·
[65.54M Transformer warm start](../results/event_lm/residual_growth_13030k/warm_transformer_65m.json)

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
mkdir -p checkpoints/work
python experiments/audit_residual_growth.py --enwik8 data/enwik8 --device mps \
  --parent checkpoints/scale-6510k-next-data-study/trace_s2503.pt --dim 1536 \
  --json checkpoints/work/warm_residual_65m.json
python experiments/audit_growth_gradients.py transformer --enwik8 data/enwik8 --device mps \
  --checkpoint checkpoints/scale-6510k-next-data-study/transformer_s2503.pt \
  --mlp-ratio 22.716796875 --seed 3307 --json checkpoints/work/warm_transformer_65m.json
```

```bash
PYTHON=.venv/bin/python bash experiments/run_v4_continuation.sh residual-growth-snn
PYTHON=.venv/bin/python bash experiments/run_v4_continuation.sh residual-growth-transformer
PYTHON=.venv/bin/python bash experiments/run_v4_continuation.sh residual-growth-control
PYTHON=.venv/bin/python bash experiments/run_v4_continuation.sh residual-growth-control-transformer
```

| Model at 72,089,600 targets | Parameters | enwik8 bpb | PG-19 bpb | PG-19 raw bpb | State |
|---|---:|---:|---:|---:|---:|
| Original-width SNN | 6,514,176 | 1.87259 | 2.81227 | 2.90738 | 19,536 B |
| Original-width full-attention Transformer | 6,513,152 | 1.77292 | 2.95995 | 3.16079 | 4 MiB |
| Wider residual SNN | 13,028,352 | 1.87451 | 2.79672 | 2.88959 | 19,536 B |
| Wider full-attention Transformer | 13,028,864 | 1.77285 | 2.95990 | 3.16073 | 4 MiB |

Doubling residual width increased enwik8 loss by 0.00192 bits per
byte and reduced calibrated PG-19 loss by 0.01555 relative to the
original-width continuation. The wider SNN led both controls on
all 16 PG-19 books. The widened Transformer led on enwik8 by
0.10166. The SNN native audit counted 308,111 pulses per byte with
exact CPU and Metal parity. Silencing the added residual outputs
after training increased loss from 1.80283 to 1.81011 on 65,536
enwik8 validation targets. The matched comparison uses one seed.
The two Transformer widths scored within 0.00007 bits per byte
on each corpus at this exposure.
[Paired scores, audits, and intervention](../results/event_lm/residual_growth_13030k) ·
[Original-width Transformer scores](../results/event_lm/scale_6510k_short_control)

### Refine router charge at fixed width

The v4 finer-charge run changed both the residual width and the
charge grid relative to its starting weights. This continuation
starts at width 768 and uses the same 6,514,176 parameters and
19,536-byte stream state as the v5 parent. It sets the loaded router
weights to their previous six-bit physical values, then trains on
an eight-bit charge grid. Every old router charge and threshold
uses four times as many ticks on the finer grid. Firing resolution
is 16×, with the original-width control's pulse target, seed 3307,
and 6,553,600 additional training targets.

On 4,096 enwik8 validation bytes, CPU step, CPU chunk and Metal
counts agreed. The unchanged-width models emitted the same output
counts and 1,288,006,522 total pulses on that sample. All seven
expansion populations had nonzero gradients. The trainable router
can now move in quarter-size charge increments from that starting
function.
[Warm-start record](../results/event_lm/finer_router_charge/warm_finer_router_charge.json)

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
mkdir -p checkpoints/work
python experiments/audit_residual_growth.py --enwik8 data/enwik8 --device mps \
  --parent checkpoints/scale-6510k-next-data-study/trace_s2503.pt --refine-router-charge 8 \
  --json checkpoints/work/warm_finer_router_charge.json
bash experiments/run_v4_continuation.sh finer-router-charge
```

| Model at 72,089,600 targets | enwik8 bpb | PG-19 bpb | PG-19 raw bpb | Native pulses per byte |
|---|---:|---:|---:|---:|
| Six-bit router | 1.87259 | 2.81227 | 2.90738 | 309,048 |
| Eight-bit router | 1.86770 | 2.82082 | 2.92020 | 309,180 |
| Original-width full-attention Transformer | 1.77292 | 2.95995 | 3.16079 | — |

The finer router lowered enwik8 loss by 0.00489 bits per byte
relative to the matched six-bit continuation. Calibrated PG-19
loss increased by 0.00855. On the 65,536-target enwik8 validation
probe, returning the trained router to six-bit charge raised loss
from 1.79341 to 1.79744. The eight-bit model stayed at 19,536
stream bytes and emitted 0.043% more pulses on the native sample.
CPU single-byte, CPU chunked, and Metal counts matched on the
audited prefixes. The held-out comparison uses one seed.
[Fine-charge held-out scores, parity, and intervention](../results/event_lm/finer_router_charge)
