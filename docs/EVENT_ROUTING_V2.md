# Feedback and timing in the spike router

The fixed-point [byte decoder](EVENT_LM.md) routes signed spike packets into
two learned pathways. This study tests two changes to that circuit on real
text. Each experiment changes one routing rule and uses the same training
bytes, neuron widths, and optimizer settings as its control.

## Delay feedback

The original learned steering decoder has one fixed tap per channel. The
selectable-delay variants can read from the usual taps and one longer tap
per block. Each channel learns a prior over the taps. The static control
uses the available tap with the largest prior. The feedback model sums
evidence from the preceding eight bytes before it picks a tap for the
current byte.

For signed count (p_t) and a candidate delay (d), the evidence at byte
(t) is (|p_t|-|p_t-p_{t-d}|). This is the reduction in unmatched pulses
relative to silence. The same ring of spike counts supplies both the
candidate packets and the evidence. The experiment with a margin rewards
a challenger only for credit that exceeds the prior winner's credit by
more than eight spikes over the window. Every variant commits the current
packet after its read. The integer export uses a hard winner. Training uses
a surrogate gradient for the prior.

Delay learning and predictive spike plasticity have precedents. The
proposal tested here is to use unmatched pulses in this decoder to change
which fixed delay supplies the next packet. [Learned delays](https://arxiv.org/abs/2306.17670)
· [Online delay learning](https://arxiv.org/abs/2601.00668)
· [Predictive spike plasticity](https://www.nature.com/articles/s41467-023-40651-w)
· [Delay selector](../snnformer/event_lm.py)

## Timing and two-way steering

The timed variant uses the fraction of the steering charge left over
after a firing threshold to choose one of four start ticks. The count
still gives the number of steering spikes. The shift changes the
intersection of the steering and content bursts. Every content pulse
still goes to exactly one destination. The count-only control is the
original learned steering model, with the same parameter shapes and count.

With both bursts aligned, their overlap is already maximal. The first
refinement used one timing bit to exchange the destinations of the two
groups after computing the aligned overlap. A second refinement used the
bit to choose a neighboring delay while leaving routing of current packets
unchanged. These trials motivate testing signed steering directly. The
signed variant starts from an even split, with one positive steering
spike moving one available pulse toward the first pathway and one negative
spike moving it toward the second. It uses the original parameter count,
fixed delay line, and streaming state size.

Residual membrane potential and spike retiming are established ideas.
These variants test specific ways to use the charge remainder or both
steering rails in the existing packet split. [Subtractive reset and residual potential](https://openaccess.thecvf.com/content_CVPR_2020/papers/Han_RMP-SNN_Residual_Membrane_Potential_Neuron_for_Enabling_Deeper_High-Accuracy_and_CVPR_2020_paper.pdf)
· [Count-preserving spike retiming](https://arxiv.org/abs/2602.03284)
· [Spike router](../snnformer/event_lm.py)

## Real-text protocol

Each run trained from scratch for 800 steps on enwik8 with eight sampled
512-byte windows per step, for 3,276,800 training targets. Each seed uses
the same windows across arms. The 192-wide steering models have 380,928
parameters. The selectable-delay models have 383,616 and the 32-byte
local Transformer has 381,888. The phase tests and their control allocate
the same state, as do the feedback tests and the static delay control.
The Transformer comparison uses the same training bytes and a parameter
count within 0.46% of each spike model. The streaming state budgets
differ across the two model families.

The validation pilot uses seed 41. The fixed comparisons use seeds 0, 1,
and 2, with 196,608 sampled enwik8 test bytes per run. The continuous
scoring protocol scores each target once. It reconstructs the state with
256 preceding bytes at chunk boundaries. This covers the tested spiking
models' maximum dependence of 232 preceding bytes. The local
Transformer also sees its entire sliding window, with RoPE rebased at
each chunk. PG-19 uses the 16 checksum-verified test excerpts from the
prior study.

The timing intervention rotates start ticks across channels at the same
byte in every layer. It preserves the burst counts at each local
intervention. Activations downstream can change. The delay intervention
disallows taps longer than 32 bytes per block using the same checkpoint.
Results by prefix distance refer to the nearest prior occurrence of the
same eight input bytes in the real text. These bins describe repetition
and help interpret the tap interventions.

## Results

Three-seed held-out means in bits per byte. Lower is better. Each PG-19
run scored 2,097,136 targets on contiguous excerpts. The enwik8 column
uses the sampled test windows from the training harness.

| Model | Parameters | enwik8 | PG-19 |
| --- | ---: | ---: | ---: |
| Local Transformer | 381,888 | **2.4838** | **3.4050** |
| Original count steering | 380,928 | 2.5996 | 3.5432 |
| Signed steering | 380,928 | 2.5982 | 3.5303 |
| Original steering with a start tick | 380,928 | 2.8044 | 3.6612 |
| Timing swaps the two routes | 380,928 | 3.5678 | 4.4349 |
| Learned static delays | 383,616 | 2.6275 | 3.5633 |
| Delay feedback | 383,616 | 2.6585 | 3.5734 |
| Delay feedback with margin | 383,616 | 2.6431 | 3.5545 |

**Timing.** In the seed-41 pilot, shifting steering bursts retained 61.9%
of the coincidences available when the same packets were aligned. This
comparison held both counts fixed at the routing point. On PG-19,
rotating the learned start ticks increased held-out
loss by 0.0583 bits per byte on average, with an increase in all three
seeds. The trained model used timing and the independently trained
count-only model still scored better. Swapping entire groups also
scored worse. Using the bit to choose a neighboring delay scored 3.2008
on validation in seed 41, versus 2.6300 for count steering. That
configuration stopped at the pilot.

**Delay feedback.** On enwik8, feedback lost 0.0310 bits per byte
against the matched static control. The margin recovered 0.0154.
On PG-19 the margin beat unconstrained feedback in three seeds and
beat the static control in two. Limiting per-block delay to 32 bytes
increased the feedback model's mean PG-19 loss by 0.0006 bits per byte
and the margin model's by 0.0011. This gives little evidence that the
extra delay range improved language prediction.

| Age of the preceding matching eight-byte prefix | Targets per seed | PG-19 loss increase after capping feedback taps at 32 |
| --- | ---: | ---: |
| Unseen | 1,494,667 | 0.00048 |
| 1–31 bytes | 7,934 | 0.00174 |
| 32–127 bytes | 17,951 | 0.00268 |
| 128–511 bytes | 43,104 | 0.00139 |
| 512 or more bytes | 533,480 | 0.00085 |

**Signed steering.** Mean PG-19 loss fell by 0.0129 bits per byte and
the enwik8 mean changed by 0.0014. Each dataset had two wins out of
three seeds. On sampled enwik8 test data, suppressing negative
steering spikes in the trained models raised loss by 1.24–1.35 bits
per byte. The models depend on the two-way split, while the gain over
the original router is small and varies by seed.
[Mechanism checks](../results/event_lm/routing_v2_mechanism_checks.json)
· [PG-19 seed 0](../results/event_lm/pg19_routing_v2_s0.json)
· [Signed steering seed 0](../results/event_lm/pg19_routing_v2_balanced_s0.json)
· [Seed 1](../results/event_lm/pg19_routing_v2_s1.json)
· [Seed 2](../results/event_lm/pg19_routing_v2_s2.json)

On the entire enwik8 test split, seed 0 scored 2.5872 for signed
steering, 2.5889 for the original router, and 2.4736 for the local
Transformer across 4,999,999 targets. [Continuous enwik8 results](../results/event_lm/enwik8_routing_v2_contiguous_s0.json)

The larger comparison uses one seed. The first three models below
trained on the same 12,288,000 enwik8 targets. The fourth checkpoint is
the 3,276,800-target local Transformer from the main experiment.
The two PG-19 groups have disjoint book IDs.

| Model | enwik8 sample at 256 context | PG-19 group of 16 | PG-19 group of 8 |
| --- | ---: | ---: | ---: |
| Signed steering, larger run | 2.3186 | 3.4270 | 3.5188 |
| Original steering, larger run | 2.3108 | 3.4171 | 3.5128 |
| Local Transformer, larger run | **2.0864** | 3.5433 | 3.5990 |
| Local Transformer, shorter run | 2.4820† | **3.4060** | **3.4750** |

† The shorter run uses its 512-byte enwik8 evaluation from the main experiment.

Both larger spike checkpoints beat the larger Transformer on both book
groups under the fixed scoring transform. The shorter Transformer
checkpoint beat both on both groups. Signed steering also lost to
the original spike router at this training budget. [16-book results](../results/event_lm/pg19_routing_v2_12m_s0.json)
· [Eight-book results](../results/event_lm/pg19_routing_v2_12m_disjoint_s0.json)
· [Larger signed training run](../results/event_lm/event_balanced_s0_delay_12m.json)

The signed model kept the original router's 29,184 bytes of
streaming state. The static and feedback controls each used
178,176 bytes, including the matched delay buffers. The Transformer
used 73,728 bytes. All seven spiking configurations tested at seed
0 agreed exactly across batched and streaming CPU integer inference
and the MPS batch path on 1,537 held-out enwik8 bytes. A dispatch check
observed zero floating-point tensor operations in the native
decoder on those paths. Float operations remain part of training and
benchmark scoring. RoPE rebasing changed the Transformer logits by
a maximum of 0.00018 on the verification excerpt and left the mean
bits per byte equal at the recorded precision. [Parity and state record](../results/event_lm/routing_v2_integer_verification.json)

## Reproduce

```bash
export ENWIK8_PATH=/absolute/path/to/enwik8
export PYTORCH_ENABLE_MPS_FALLBACK=0
mkdir -p checkpoints
for seed in 0 1 2; do
  for variant in event_steered event_timed event_phase_swap event_static event_feedback event_margin event_balanced transformer_local; do
    python experiments/train_event_lm.py --variant "$variant" --device mps \
      --hidden 192 --no-carry --ctx 512 --bs 8 --steps 800 --warmup 80 \
      --eval-every 200 --eval-bs 8 --eval-batches 8 --final-eval-batches 48 \
      --mlp-ratio 4.4375 --seed "$seed" --tag routing_v2 \
      --checkpoint "checkpoints/${variant}_routing_v2_s${seed}.pt"
  done
done

python experiments/prepare_pg19.py \
  --manifest results/lm_scaling/pg19_test16_manifest.json \
  --output data/pg19-test16-scaling
for seed in 0 1 2; do
  python experiments/eval_event_lm.py --books-dir data/pg19-test16-scaling \
    --device mps --context 2048 --history 256 --batch-size 6 \
    --delay-limit 32 --shuffle-phase --checkpoint checkpoints/*_routing_v2_s"$seed".pt \
    --json "results/event_lm/pg19_routing_v2_s${seed}.json"
done

python experiments/eval_event_lm.py --enwik8 "$ENWIK8_PATH" \
  --device mps --context 2048 --history 256 --batch-size 6 \
  --checkpoint checkpoints/*_routing_v2_s0.pt \
  --json results/event_lm/enwik8_routing_v2_contiguous_s0.json

python experiments/generate_event_lm.py checkpoints/event_balanced_routing_v2_s0.pt \
  --prompt 'The history of ' --bytes 64
```

The timing-controlled neighboring-tap variant is
`--variant event_phase_delay`. Use seed 41, `--tag routing_pilot`, and
`--validation-only` to reproduce its validation result. For the larger
signed-steering run, use seed 0, context 256, batch size 12, 4,000 steps,
warmup 100, evaluation every 400 steps, 96 final evaluation batches,
MLP ratio 4.421875, and tag `delay_12m`. The existing `delay_12m`
learned-steering and Transformer checkpoints used those settings.
