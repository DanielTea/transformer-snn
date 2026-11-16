# Let stored tags compete for the read

The v5 SNN proposes an address in each forecast-error head. The
selected cell's tag can inhibit the returned payload. On the native
enwik8 audit, this removed 408,583 of 528,278 payload pulses, or 77.3%.
The new circuit lets this information change which cell gets read.
[v5 native audit](../results/event_lm/scale_6510k_next_data/native_trace_s2503.json)

For each slot with positive address drive $a_j$, compare its stored
tag to the current forecast tag using the same tolerance interval
as the payload veto. Let $v_j$ be the largest excess across its tag
coordinates. The excess drives an inhibitory spike train through a
learned conductance $g_h$ shared by the slots of head $h$. The
read competition uses $a_j - \operatorname{fire}(g_h v_j)$. An
eligible head emits one read address. The winning tag still verifies
the returned payload, using the full excess to cancel coincident
payload spikes.

The next write retains the original competition over $a_j$ and
uses the previous write address. All tags and payloads read by the
new competition come from the charge entering the byte. A single
tolerance rule now helps choose the read address, bound the write,
and filter the returned error. The bank adds four learned synaptic
conductances and uses 19,536 persistent bytes across the full model.
[Circuit](../snnformer/event_lm.py) · [Causal state update](../snnformer/correction_memory.py) ·
[Spiking associative memory](https://proceedings.mlr.press/v282/podlaski26a.html)

## Warm start and matched training

At zero conductance, CPU stepping, CPU chunks, and Metal agreed on
4,096 enwik8 validation bytes. The extended model and its parent
emitted identical vocabulary counts and 1,288,006,522 total pulses.
All four head conductances received nonzero gradients on 512 enwik8
training targets. The continuation uses 1,600 updates at 512 bytes
and batch size eight, seed 3307, and a 310,000-pulse training target.
It shares the 65.54M parent and all 6,553,600 additional training
targets with the original-width SNN control. The native audit
records rerouted heads and inhibition pulses.
[Warm-start audit](../results/event_lm/tag_read_competition/warm_tag_reads.json) ·
[Original-width control](GROWN_RESIDUAL.md#double-the-residual-stream-at-fixed-persistent-state)

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
mkdir -p checkpoints/work
python experiments/audit_residual_growth.py --enwik8 data/enwik8 --device mps \
  --parent checkpoints/scale-6510k-next-data-study/trace_s2503.pt --tag-read-competition \
  --json checkpoints/work/warm_tag_reads.json
bash experiments/run_v4_continuation.sh tag-read-competition
```

### Zero-initialized continuation

| Model at 72,089,600 targets | enwik8 bpb | PG-19 bpb | Raw PG-19 bpb | Native pulses per byte |
|---|---:|---:|---:|---:|
| Original-width SNN | 1.87259 | 2.81227 | 2.90738 | 309,048 |
| Tag read competition | 1.87077 | 2.81844 | 2.91860 | 309,523 |

All four trained conductances quantized to zero. The native run
recorded zero rerouted reads and zero inhibition pulses from the
new circuit. Zeroing its gains left the validation loss exactly
1.79985497. The score differences reflect the one-seed training
comparison, and the exported read circuit followed the original
address competition.
[Native counts, ablation, and held-out scores](../results/event_lm/tag_read_competition)

### Force active gains on the parent

With the v5 weights fixed, set each head to the same gain and
measure 65,536 enwik8 validation targets. Age, rerouted read
heads, and payload pulse counts come from a native 4,096-byte
validation prefix. Mean age weights the nonzero payload
coordinates before the veto.

| Conductance | Validation bpb | Rerouted read heads | Mean payload age in bytes | Passed payload spikes per byte |
|---|---:|---:|---:|---:|
| 0.00 | 1.79955 | 0.00% | 11.36 | 21.05 |
| 0.25 | 1.80043 | 68.77% | 194.11 | 61.81 |
| 0.50 | 1.80043 | 69.68% | 199.95 | 62.25 |
| 0.75 | 1.80039 | 69.91% | 200.65 | 62.42 |
| 1.00 | 1.80040 | 70.14% | 201.64 | 62.65 |

Stronger verification routed reads to older cells and increased
surviving payload. Loss increased by 0.00083 to 0.00088 bits per
byte. The four active settings have identical write counts and
correction pulses to the zero-gain setting.
[Validation intervention](../results/event_lm/tag_read_competition/gain_sweep_parent.json)

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python experiments/probe_event_state.py \
  --enwik8 data/enwik8 --device mps --targets 65536 --chunk-size 512 \
  --checkpoint checkpoints/scale-6510k-next-data-study/trace_s2503.pt \
  --tag-read-gains 0 0.25 0.5 0.75 1 --json checkpoints/work/tag_read_gain_sweep.json
```

### Train tag agreement against observed forecast error

The next-byte observation is available during training. At each
position, score each eligible cell's pre-write payload by how well
its signed correction would reconstruct that observation in the
forecast code. A lower smooth L1 error gives a higher target
probability. The read preference $a_j-\operatorname{fire}(g_hv_j)$
learns to rank the same candidates using cross-entropy. The auxiliary
gradient reaches the read gain and learned tags through the inhibition
path. The observed next code, stored payload, and base address drive
are detached within this objective. Inference uses the spike circuit
at the top of this page.

Both follow-up arms initialize the head conductances to 0.25 and
use the same seed, optimizer, 6,553,600 targets, and pulse target as
the first continuation. One uses the language and code objectives.
The other adds the ranking objective with weight 0.10.

A three-update Metal smoke of the ranking arm produced a finite
objective. Its saved checkpoint rerouted 10,734 read heads over 4,096
native test bytes, and its vocabulary counts matched Metal.
[Training smoke and native audit](../results/event_lm/tag_read_competition)

```bash
bash experiments/run_v4_continuation.sh tag-read-active-start
bash experiments/run_v4_continuation.sh tag-read-supervision
```
