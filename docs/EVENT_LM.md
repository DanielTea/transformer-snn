# Coincidence-routed spiking byte decoders

These models train from scratch on enwik8 and predict the next byte. The CPU
export implements the input, hidden layers, recurrence, and vocabulary output
with integer tensors. The GPU training path uses floating-point master weights,
integer-valued forward currents and spike counts, and straight-through
gradients. Its benchmark loss uses normalized weights \(2^{c/8}\), where \(c\)
is the signed output spike count. This scoring transform is external to the
decoder. Greedy autoregressive generation chooses the byte whose output count
is largest, resolving ties by lowest byte ID.

[Follow-up experiments](EVENT_ROUTING_V2.md) cover delay feedback, spike
timing, and signed steering on contiguous text.

## Spike pathway

```text
One active byte input synapse
  → integrate-and-fire input neurons
  → three blocks of integrate-and-fire neurons and delay lines
  → 256 pairs of positive and negative output neurons
  → winning byte by signed spike count
```

One byte addresses an afferent in a 256-way input population. The integer
weight on its synapse is the charge delivered by its event. Each between-block
signal is the difference of the counts emitted by two sign rails. Neurons
perform subtractive reset at their threshold. A variant can carry the
remaining integer membrane phase across bytes or reset it before the next
byte. Delayed events travel through fixed-size buffers. The streaming export
uses dense integer arithmetic in PyTorch. It measures language prediction and
execution parity. Sparse asynchronous throughput and hardware energy are
outside this experiment.

## Routing experiments

In the **self-coincidence** model, each expanded neuron sends its current
packet and retains a delayed copy. A packet of magnitude \(n\) occupies the
first \(n\) slots on its sign rail. The current packet and its delayed copy
then use the same clock. Delayed events that coincide with the current packet
on the same rail go to one projection. The remaining events go to a second
projection. Their signed counts sum to the original delayed count. The
**fixed routing** control sends alternate packet slots to each projection.

In the **learned steering** model, independent integrate-and-fire steering
neurons emit pulses that split both current and delayed content. Each packet
still sends every event to exactly one destination. The **current-only
steering** control routes current events with the same learned pulses and
splits delayed events by alternating slots. It retains the number of
parameters, the delay schedule, and the current-event routing. The
**convergent input** variant adds two separate synapses for byte input events
from one and two positions ago, with a narrower hidden expansion to match the
parameter budget.

The candidate contribution is the count-conserving split of present and
delayed spike packets into independently trained pathways, with the same
learned spike train steering both in the second model. The burst slots are
canonical encodings of counts. The decoder has no learned within-burst
latencies. Prior work has independently used delays, spike coincidence and
spike-count language models. The address-and-value latency design discussed
earlier remains a separate proposal. [Delay learning in SNNs](https://arxiv.org/abs/2306.17670)
· [Spike-driven Transformer](https://arxiv.org/abs/2307.01694)
· [SpikingBrain](https://arxiv.org/abs/2509.05276)
· [Source](../snnformer/event_lm.py)

## Held-out results

Means over seeds 0, 1 and 2. All models trained on 3,072,000 enwik8 target
bytes per seed. The enwik8 test scored 196,608 sampled targets per model
and seed. PG-19 scored 2,097,136 targets from 16 real, checksum-verified
test excerpts. Both evaluations reset state at their 256-byte window
boundaries. Lower bits per byte is better.

| Model | Parameters | enwik8 | PG-19 |
| --- | ---: | ---: | ---: |
| Fixed routing | 380,928 | 3.4549 | 4.2234 |
| Self-coincidence | 380,928 | 2.9572 | 3.8083 |
| Current-only steering | 380,928 | 2.6446 | 3.5979 |
| Learned steering | 380,928 | 2.5644 | 3.5535 |
| Convergent input and learned steering | 381,696 | 2.5637 | 3.5754 |
| Full-context Transformer | 381,024 | 2.4645 | 3.4010 |
| Local Transformer | 381,024 | **2.4178** | **3.3678** |

Self-coincidence beat fixed routing across all seeds on both corpora.
Learning to steer delayed events reduced loss relative to current-only
steering in all three seeds on each corpus by a mean of 0.0802 on enwik8
and 0.0444 on PG-19. The fixed split can send more mass to its first
pathway on short packets, so these comparisons match architecture and
dimensions without forcing equal route occupancy. The local Transformer
had the best mean on both datasets. The evaluations cover short-context
language prediction. The direct receptive field of the learned steering
model at this depth is 39 bytes, including the current byte, and it resets
its membrane phase each byte. Long-distance retrieval remains unverified.
[enwik8 training curves and final scores](../results/event_lm)
· [PG-19 seed 0](../results/event_lm/pg19_delay_v1_s0.json)
· [Seed 1](../results/event_lm/pg19_delay_v1_s1.json)
· [Seed 2](../results/event_lm/pg19_delay_v1_s2.json)

At four times the training budget, two fresh seed-0 runs used 12,288,000
training targets each. Learned steering scored 2.3108 bits per byte on
enwik8, and the local Transformer scored 2.0864. The local Transformer
continued to lead. [Learned steering](../results/event_lm/event_steered_s0_delay_12m.json)
· [Local Transformer](../results/event_lm/transformer_local_s0_delay_12m.json)

## Integer inference verification

All five trained seed-0 spiking variants emitted identical counts in CPU
integer streaming, CPU integer batches, and CPU and MPS training-format
batches on a 1,024-byte enwik8 test excerpt. A PyTorch dispatch check
observed zero floating-point tensor operations inside the native decoder
during those integer passes and 64 further greedy generation steps. The
learned steering model used 29,184 bytes of persistent state tensors for
one stream. The complete set of sampled seed-0 enwik8 test windows was also
rescored with the CPU integer learned-steering and convergent-input models.
Both scores matched the corresponding MPS scores within \(10^{-6}\) bits
per byte. [Parity and generation records](../results/event_lm/integer_verification.json)
· [Integer enwik8 scores](../results/event_lm/integer_enwik8_s0.json)

## Reproduce

All models use 96 hidden channels, three blocks, 256-byte windows and 12 input
windows per training step. enwik8 is checksum-verified and uses the standard
90M/5M/5M split. The sampled validation and test targets are disjoint within
each split. Transformer comparisons use the same byte draws and training
schedule and within 0.2% of the event-model parameter counts. The wider of the
two Transformer windows is the full 256-byte training context. The local
model uses 32 positions per layer.

```bash
export ENWIK8_PATH=/absolute/path/to/enwik8
mkdir -p checkpoints
for seed in 0 1 2; do
  for model in event event_tonic event_steered event_steered_local event_convergent transformer_local transformer; do
    args=()
    case "$model" in
      event_convergent) args=(--hidden 164 --no-carry) ;;
      event_steered|event_steered_local) args=(--hidden 192 --no-carry) ;;
    esac
    PYTORCH_ENABLE_MPS_FALLBACK=0 python experiments/train_event_lm.py \
      --variant "$model" --device mps --ctx 256 --bs 12 --steps 1000 --warmup 100 \
      --eval-every 100 --eval-bs 8 --eval-batches 8 --final-eval-batches 96 \
      --mlp-ratio 4.421875 --seed "$seed" --tag delay_v1 \
      --checkpoint "checkpoints/${model}_delay_v1_s${seed}.pt" "${args[@]}"
  done
done

python experiments/prepare_pg19.py \
  --manifest results/lm_scaling/pg19_test16_manifest.json \
  --output data/pg19-test16-scaling
for seed in 0 1 2; do
  PYTORCH_ENABLE_MPS_FALLBACK=0 python experiments/eval_event_lm.py \
    --books-dir data/pg19-test16-scaling --device mps --context 256 \
    --checkpoint checkpoints/*_delay_v1_s"$seed".pt \
    --json "results/event_lm/pg19_delay_v1_s${seed}.json"
done

python experiments/generate_event_lm.py checkpoints/event_steered_delay_v1_s0.pt \
  --prompt 'The history of ' --bytes 64
```

For the two 12,288,000-target runs, use seed 0 with the same model-specific
arguments and change `--steps` to 4000, `--eval-every` to 400, and `--tag` and
the checkpoint filename to `delay_12m`.
