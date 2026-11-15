# Dendrite: spike-addressed memory and hybrid successor modeling

See the [benchmark report](BENCHMARKS.md) for charts and the
[reproduction guide](REPRODUCE_DENDRITE.md) for a source map and commands.

The spiking-backbone models use address memory for recall and byte
prediction. The separate successor language model uses a conventional
Transformer, an exact suffix cache and a spike readout. Its lower
enwik8 loss measures the hybrid. The spiking event language model
scored **4.469 bits per byte** on its sampled test, versus **2.431**
for the local Transformer control. These experiments do not establish
a spiking-backbone language modeling advantage or frontier performance.

Dendrite also has an [event memory](#event-memory-for-variable-delays)
that learns when to capture an address and when to write a value. It can
carry a pending address across intervening tokens without a fixed lag.

Dendrite continues ShuntFormer's use of inhibition as normalization. It replaces
ShuntFormer's pairwise coincidence scores with a table of spike-addressed
memories. A separate short delay line can keep recent spike patterns available
even when the matching address has a long, noisy history.

## The mixer

Each head projects the current block input spikes to a spike value vector
`v[t]`, an address word `a[t]` of `b` binary detectors, and a binary write
permission `g[t]`. A detector has complementary outputs: its ON event
selects bit 1 and its OFF event selects bit 0. The conjunction of those
events activates exactly one of `2**b` addresses. The GPU/CPU training
implementation forms that one-hot code with products of complementary
bits, with surrogate gradients through the threshold detectors.

```text
write_address = a[t - delay]        # omit the write before enough history exists
S[write_address] += v[t]            # only if g[t] fires
C[write_address] += 1               # only if g[t] fires
read = S[a[t]] / max(1, C[a[t]])     # read after write
relay = 0.5 * v[t - relay_delay]    # optional; zero before the sequence starts
output = output_spikes(gain * (read + relay))
```

Even heads write under the current address; odd heads write under the
previous one by default. A head with a delay of one can therefore associate
a token with the value spikes that arrive immediately after it. Dendrite
with relay uses a one-token tap on even heads and a two-token tap on odd
heads. These fixed taps add neither learned parameters nor a sequence-length
dependent cache. The input and output projections, residual stream and CD-GLU
remain in the ShuntFormer scaffold.

Every attended bucket stores a sum and a write count. The read is bounded by
the value spike range, even as the individual accumulators increase. An empty
bucket reads zero. The mean also loses identities when distinct keys map to
the same address. Increasing the number of bits increases table size
exponentially, and address drift can cause misses across layers or contexts.

## Training and streaming

- Batched causal training computes a bucket table for each fixed-size block,
  prefix-sums the tables, and handles the inclusive causal reads within each
  block with a small masked product. At fixed block width and address count
  it scales linearly with sequence length. The vectorized reference uses
  dense PyTorch kernels.
- `SpikingLM.step` reads and optionally updates one bucket per head per token.
  The only sequence state is the table, delay rings and token index. Disabling
  absolute positions removes the learned context-length cap. Each table has
  `heads * 2**b * (head_width + 1)` scalar entries per layer, sample and
  simulation timestep, plus the short delay rings. Accumulators still have
  finite numerical precision.
- `--strict-causal` uses lagged BatchNorm statistics. The current batch is
  normalized with statistics from completed training batches. It updates
  the statistics for the next batch after computing its normalized values.
  Standard sequence BatchNorm in the historical experiments pooled across
  positions in training mode. Those results are kept for provenance and
  aren't used as matched causal baselines.
- Spike projections take bounded spike counts. The PyTorch read performs a
  division; for a single simulation step, a positively scaled output
  threshold can instead be compared to the accumulated spike sum with the
  threshold scaled by the write count. With the relay, the comparison also
  needs the local contribution. No hardware kernel, energy measurements,
  or spiking equivalence across physical timesteps is claimed.

## Prior work

[RACE Attention](https://arxiv.org/html/2510.04008#S6.SS4) already uses
causal bucket sums and counts with soft sign-code assignments.
[PCAF](https://arxiv.org/html/2606.10435#S3.SS3) already binds prior contexts
to successor tokens in a sparse memory that stores candidate records.
[BitBrain](https://pmc.ncbi.nlm.nih.gov/articles/PMC10071999/) uses
coincidence-based spiking address decoders. Dendrite's experimental contribution
is the combination of hard spike addresses, gated delayed binding, normalized
fixed-size bucket state, and optional local value relays in a directly trained
spiking Transformer. The individual ideas have precedents.

## enwik8 results

On the small position-free comparison (2 layers, width 64, 4 heads,
context 96, 1.38M training bytes), Dendrite beat ShuntFormer-4 under
lagged BatchNorm in all three runs. The entries below are held-out test bits
per byte on 76,800 sampled bytes; lower is better.

| Model | Parameters | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---:|---:|---:|---:|---:|
| ShuntFormer-4 | 135,710 | 5.676 | 5.791 | 5.879 | 5.782 |
| Dendrite (4 bits, no relay) | 121,430 | 5.178 | 5.656 | 5.435 | 5.411 |

The gain did not carry over to the larger run (3 layers, width 96, 4 heads,
context 128, 3.69M training bytes). All rows here use the same lagged
normalization and a single seed, with 409,600 sampled bytes per held-out split.

| Model | Absolute positions | Parameters | Validation bpb | Test bpb |
|---|---|---:|---:|---:|
| ShuntFormer-4 | No | 390,631 | 4.230 | 4.281 |
| ShuntFormer-4 | Yes | 402,919 | 4.351 | 4.403 |
| ANN softmax | Yes | 401,743 | 4.188 | 4.242 |
| Dendrite (4 bits) | No | 340,027 | 4.310 | 4.379 |
| Dendrite + relay (4 bits) | No | 340,027 | 4.293 | 4.365 |
| Dendrite with zero write delay (4 bits) | No | 340,027 | 4.203 | 4.267 |

The zero-delay ablation scored best among the spiking models in this
single larger run. Its 0.014 bpb edge over position-free ShuntFormer-4
is too small to establish a consistent advantage. The relay checkpoint
was reloaded and gave identical CPU logits in batched and streaming
inference on two 1,024-token test streams. The streaming state stayed at
21,600 bytes per stream, including bucket tables and delay rings.
On a separate set of 65,536 identical test targets, its loss was 4.328 bpb
with 128 tokens of context, 4.270 at 512, and 4.273 at 1,024. That is one
checkpoint and a small sampled evaluation, rather than a long-context
language benchmark.

These are within-repository comparisons at small budgets. They do not
establish a public leaderboard result; e.g., the much larger recurrent
[AstroSNN](https://arxiv.org/html/2312.07625#S3.SS2) reports substantially
lower loss on enwik8 under a different protocol.

## Associative recall

Train on eight bindings and eight queries with a pool of 32 keys and 32
values. Then test on up to 32 bindings with the query count and vocabulary
held fixed. Each held-out result uses 4,096 generated sequences with a
separate seed. Every model uses the same two-layer, width-64, four-head
scaffold, 3,800 training steps, and lagged normalization. Dendrite uses
fewer parameters than ShuntFormer-4 at this width. The historical 99%
ShuntFormer-4 result used a different training normalization and setup.

| Model | Seeds | 8 bindings | 16 bindings | 32 bindings |
|---|---:|---:|---:|---:|
| ShuntFormer-4, position-free | 3 | 14.62% | 10.57% | 7.81% |
| ShuntFormer-4, absolute positions | 1 | 16.17% | — | — |
| ANN softmax, absolute positions | 1 | 14.20% | — | — |
| Dendrite + relay, 6 bits | 3 | 93.84% | 87.36% | 74.42% |
| Dendrite + relay, 4 bits | 1 | 63.11% | 49.19% | 31.35% |
| Dendrite + relay, 6 bits, zero write delay | 1 | 12.95% | 8.36% | 5.98% |
| **Dendrite, 6 bits** | **3** | **97.83%** | **96.88%** | **95.26%** |

The six-bit model without the local relay was best for recall. Its
32-binding scores were 93.35%, 95.42% and 97.00%. The relay ablations
show that write timing and address capacity matter. In the seed 1 relay
checkpoint's first layer, both delayed heads fired the write gate for
every value symbol and only four of the 32 key symbols. The one-token
write delay is a direct inductive bias for adjacent key/value pairs.
The controls struggled under this training protocol, so these results
do not rank Dendrite against stronger causal Transformer implementations
or against the original Zoology benchmark.

## Reproduce

Install the dependencies from `requirements.txt` and extract the canonical
raw [enwik8 corpus](https://www.mattmahoney.net/dc/textdata.html). For example:

```bash
python experiments/train_lm.py --data /path/to/enwik8 --variant dendrite_relay \
  --strict-causal --no-position --dim 96 --depth 3 --heads 4 --ctx 128 --bs 24 \
  --steps 1200 --warmup 100 --lr 0.003 --eval-every 300 --eval-batches 12 \
  --final-eval-batches 100 --threads 1 --seed 0 --tag reproduce \
  --checkpoint checkpoints/dendrite-relay-enwik8-s0.pt
```

Use the same command and flags with `ours_k4`, `dendrite`, and
`dendrite_nodelay` for the dense control and architecture ablations; pick a
separate checkpoint path or omit it. Runs with the same result name are
skipped unless `--force` is supplied. Set `--device auto` to use an available
accelerator. Model weights and constructor settings can be loaded with:

```python
import torch
from snnformer import SNNConfig, SpikingLM

saved = torch.load("checkpoints/dendrite-relay-enwik8-s0.pt",
                   map_location="cpu", weights_only=True)
model = SpikingLM(SNNConfig(**saved["cfg"]), **saved["model_kwargs"])
model.load_state_dict(saved["state_dict"])
model.eval()

logits, state = model.step(torch.tensor([ord("T")]))
logits, state = model.step(torch.tensor([ord("h")]), state)
```

To check recall at longer lengths, the position-free runner supports
`--eval-pairs`; this holds the number of queries fixed and increases the number
of preceding bindings. Its generator is the repository's simplified form of
[Zoology MQAR](https://arxiv.org/html/2312.04927). Runs report seed, sample
counts, curves and training settings in the result JSON.

```bash
python experiments/recall.py --variant dendrite --address-bits 6 \
  --strict-causal --no-position --pairs 8 --keys 32 --queries 8 \
  --dim 64 --heads 4 --steps 3800 --bs 64 --lr 0.003 --threads 1 \
  --eval-every 500 --val-samples 512 --eval-pairs 16 32 --seed 0 \
  --tag reproduce --checkpoint checkpoints/dendrite-recall-b6-s0.pt
```

## Event memory for variable delays

`dendrite_event` replaces the fixed-delay ring with one pending address
per head. A learned capture spike replaces that address. A separate
commit spike writes the current value under the pending address and then
clears it. Capture takes precedence when both gates fire. When neither
gate fires, the address stays intact for any number of tokens. The bucket
sum and count retain the completed writes, as in the fixed-delay model.

Training computes the pending address with a prefix scan over the most
recent capture or commit. Its backward pass accumulates state gradients
up to the next such event, matching the hard recurrence's derivative
before the spike surrogate is applied. Streaming keeps only the pending
address and the fixed-size bucket table. Each head can hold one unfinished
binding. Other captures or commits in the gap can interfere, and the
completed writes still share the finite address table.

Earlier SpiNNaker work implements a spiking latch controlled by a store
signal. Work on polychronous chains trains synapses with heterogeneous
delays to replay spike motifs. Dendrite learns a discrete address and
two separate decisions from the tokens. One event holds the address;
a later event binds the current value to it. The held address has no
decay when neither gate fires, and reads normalize the stored values.
[SpiNNaker spiking memory](https://arxiv.org/html/2206.03957#S4.SS5) ·
[Polychronous chains](https://arxiv.org/html/2604.14096)

### Variable-gap recall

Each sequence presents four key–value bindings and four queries. Keys,
values and distractors come from separate groups of token IDs. The model
learns their roles from the tokens and next-value supervision. Bindings
arrive sequentially, with one pending at a time. Training uses up to
four distractors between key and value and 28 tokens per example. At
test time, each exact gap uses the same sampled keys, values and queries
at a total length of 524. Packing the bindings in this test can also
change value-to-query distances.

| Model | Parameters | Seeds | Train-range test | Gap 0 | Gap 4 | Gap 16 | Gap 64 | Gap 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed-delay Dendrite | 99,782 | 1 | 5.96% | 8.69% | 3.37% | 3.47% | 3.56% | 3.52% |
| Full RoPE Transformer | 104,192 | 3 | 26.19% | 6.09% | 6.09% | 5.99% | 5.88% | 6.14% |
| Dendrite event | 100,310 | 3 | 99.97% | 99.97% | 99.97% | 99.95% | 99.97% | 99.95% |

All three models get the same examples, width, depth and 1,200 optimization
steps. The full Transformer uses standard token-local pre-LayerNorm,
SwiGLU and RoPE and could be evaluated beyond its training length. The
typed token groups and sequential bindings are part of this controlled
task. The results measure duration between events under that protocol.

A separate seed-0 checkpoint probe fixed the value and query positions,
as well as the total length, at 8,204 tokens. Moving the keys to give
gaps of 0, 4, 128, 512 or 2,048 distractors gave 512 of 512 correct
queries at every gap. The same checkpoint's streaming logits agreed
with batched inference within `2.9e-6` on two held-out sequences.
The per-stream state used 34,880 bytes from step 1 through step 524.
An additional 8,000-step RoPE control scored 24.80% in the training
range and 3.37% at gap 128. The controlled task strongly favors the
explicit capture and commit rule.

In the first layer of the seed-0 Dendrite checkpoint, every head
captured all 32 key symbols. Three heads ignored all 16 distractor
symbols, and the fourth could commit on one of them. These detector
measurements describe a model trained on a known distractor vocabulary.

```bash
python experiments/recall_distance.py --variant dendrite_event --address-bits 6 \
  --keys 32 --pairs 4 --queries 4 --train-max-gap 4 --eval-gaps 0 4 16 64 128 \
  --eval-context-length 524 --eval-bs 16 --eval-samples 512 --val-samples 256 \
  --threads 1 --lr 0.003 --steps 1200 --eval-every 300 --seed 0 --tag reproduce \
  --checkpoint checkpoints/dendrite-event-recall-s0.pt
```

Use the same arguments with `--variant transformer` to reproduce the
full attention control. Leave the position setting at its default to
enable RoPE for that model. Use a separate checkpoint name for each run.
For the anchored seed-0 probe, set `--eval-context-length 8204`,
`--eval-gaps 0 4 128 512 2048`, `--eval-samples 128`, `--eval-bs 8` and
`--anchor-eval-values`.

### Event memory in language models

The language comparisons use the same depth, width, number of heads,
training bytes and optimizer schedule. The conventional controls use
full or 32-token sliding attention with RoPE. The hybrid retains the
same sliding attention and adds a learned spike-memory residual in
each block. The compact hybrid projects each residual to width 24
before it enters the event memory and uses a smaller feed-forward
expansion to match the controls' parameter count. The hybrid has
analog attention and spike memory.

| enwik8 model | Parameters | Seeds | Test bits per byte |
|---|---:|---:|---:|
| Full RoPE Transformer | 357,696 | 3 | 2.473 |
| Local RoPE Transformer | 357,696 | 3 | 2.431 |
| Compact local Transformer + event memory | 357,486 | 3 | 2.516 |
| Full-width local Transformer + event memory | 358,134 | 1 | 2.834 |
| Dendrite event, integer-spike backbone (one neuron step per byte) | 341,203 | 1 | 4.469 |

All rows train on 3.072M sampled byte targets at context 256 and
evaluate on the same 307,200 held-out bytes. In the event model, a
single neuron step per byte produces bounded spike counts. Address
memory persists across bytes. The recall result measures binding
across distractors, while the language modeling run measures next-byte
prediction. Reducing the memory
branch's width and initial residual scale improved the hybrid. Both
hybrids still scored above the conventional baseline's loss. For
published scale, LongVQ reports 0.97 bpb on enwik8 at 47M parameters.
[LongVQ enwik8 results](https://arxiv.org/html/2404.11163#S7)

In a separate paired probe scoring 8,192 test bytes with 256 preceding
bytes, silencing the trained compact hybrid's event path raised loss
from 2.600 to 2.622 bpb. The independently trained local-only
Transformer scored 2.477 bpb on those same targets. Extending history
to 1,024 bytes did not improve the compact hybrid on that probe.

Training directly at context 1,024 also widened the gap. At seed 0
with 4.096M training bytes and 262,144 sampled test bytes, the compact
hybrid scored **3.289 bpb**, the full Transformer **2.627**, and the
local Transformer **2.419**. This longer-context run used four sequences
per batch in place of the 12 used in the context-256 runs.

```bash
python experiments/train_lm.py --variant transformer_event --event-width 24 \
  --mlp-ratio 3.64 --data /path/to/enwik8 --dim 96 --depth 3 --heads 4 \
  --ctx 256 --bs 12 --steps 1000 --warmup 100 --eval-every 250 \
  --eval-bs 12 --eval-batches 10 --final-eval-batches 100 --lr 0.003 \
  --threads 1 --tag reproduce --checkpoint checkpoints/hybrid-reproduce.pt
```

Use `transformer_local` or `transformer` with the default MLP ratio
of 4 for the controls. `experiments/probe_lm_context.py` scores the
same target bytes after different history lengths and can mute the
trained hybrid's memory residual for a paired evaluation.

## Predicting bytes from verified successor memory

The hybrid `transformer_successor` variant adds memory at the output of
a conventional local Transformer with sliding-window softmax attention
and a continuous feed-forward network. A write stores the observed byte
under an exact suffix of earlier bytes. Each suffix length has 2,048 slots. The hash
chooses a slot, and a read compares the entire packed suffix. A hash
collision can evict a record. A slot for an unrelated suffix cannot
vote for a prediction. When several matching suffixes predict the
same byte, their votes add in probability space.

A readout with 241 learned parameters uses match length, age, the
backbone's probability for the candidate, agreement among suffix
orders, and past prediction outcomes. Only outcomes already observed
in the stream enter its running counters. The gate quantizes each vote
to a count of up to 16 spikes. Scaling the increment by suffix length
lets short matches contribute less mass without rounding them straight
to zero. The byte Transformer remains analog.

Neural caches and PCAF already mix a successor prediction with the
language model's probability. PCAF retrieves bounded candidates from
hash buckets and can rank them by continuous key similarity and recency.
StateSMix already stores a recent successor with a confidence count in
a hash table, and exact n-gram lookup has other precedents. Dendrite
verifies the full suffix at several lengths and uses observed prediction
outcomes and cross-order agreement to allocate 0–16 spikes per match.
Each spike's contribution to the probability mixture scales with the
squared ratio of suffix length to maximum suffix length. We compare the
readout with a tuned fixed mixture and a learned continuous gate.
The experiments establish differences from these implementations;
they do not establish an exhaustive novelty claim.
[Neural cache](https://arxiv.org/abs/1612.04426) ·
[PCAF memory and gate](https://arxiv.org/html/2606.10435#S3.SS3) ·
[StateSMix successor predictor](https://arxiv.org/html/2605.02904#S4.SS6) ·
[Exact and fuzzy n-gram matching](https://arxiv.org/html/2411.00066#S3) ·
[Direct-mapped n-gram cache](https://aclanthology.org/P11-1027/)

### Next-byte results

The baseline is the same 357,696-parameter local Transformer trained
with context 1,024. It stays frozen when fitting the readout on enwik8
training bytes. The three-seed comparison uses 524,288 held-out bytes
at context 4,096. Seeds vary the readout training and keep the backbone
fixed. The fixed mixture fits one validation coefficient for each
longest matching suffix length. All cache variants use the same
five suffix lengths and table size.

| Readout | Test bits per byte |
|---|---:|
| Unchanged local Transformer | 2.429 |
| Fixed mixture by longest match | 2.119 |
| Learned spikes, uniform increments | 2.122 |
| Learned spikes, increments scaled by suffix length | **2.097** |
| Learned continuous gate, with scaled maximum weights | 2.094 |

The scaled spike readout improved on the fixed mixture with this
backbone. With an independently trained local checkpoint from the
context-256 run, it reduced loss from 2.387 to 2.094 bpb, while the
tuned fixed mixture scored 2.090.

| Continuous evaluation, seed 0 | Target bytes | Local Transformer | Scaled spike readout | Votes aged 94 bytes or more |
|---|---:|---:|---:|---:|
| enwik8 test | 4,999,999 | 2.457 | **2.074** | 2.212 |
| Eight held-out PG-19 excerpts | 1,048,568 | 3.416 | **3.052** | 3.088 |

The continuous scorer keeps memory across processing chunks and resets
between independent streams. The last column removes votes whose stored
successor falls within the local Transformer's 94-byte receptive field.
The PG-19 figures use the enwik8-trained gate without fitting on the
books. The excerpt manifest records the selection rule, offsets and
checksums. These results compare compact byte models on next-byte loss.
[PG-19 source](https://github.com/google-deepmind/pg19)

The checkpoint's streaming state used 319,616 bytes per stream at both
1 and 1,024 consumed bytes. That includes 240 KiB of tagged memory,
72 KiB of local attention state and a small number of counters. Batched
and token-by-token log probabilities agreed within `1.2e-5` on two
held-out 1,024-byte inputs.

```bash
python experiments/train_successor.py \
  --base checkpoints/transformer-local-enwik8-long-s0.pt --data /path/to/enwik8 \
  --scaled-votes --spike-warmup 80 --steps 320 --ctx 2048 --bs 4 \
  --eval-context 4096 --eval-windows 128 --tag reproduce --seed 0 \
  --checkpoint checkpoints/dendrite-successor-reproduce.pt

python experiments/eval_successor.py \
  --checkpoint checkpoints/dendrite-successor-reproduce.pt --data /path/to/enwik8
```

The base checkpoint uses `transformer_local` with width 96, depth 3,
four heads, context 1,024, batch size 4 and 1,000 steps. Its training
configuration is recorded in the `event_long` result. Pass
`--raw /path/to/book1 /path/to/book2` to also score independent raw
byte streams. The `transformer_successor` variant is also available
in `train_lm.py` for training the complete model from scratch.

With the same seed, dimensions, optimizer schedule and 4.096M
training bytes as that local control, the successor variant trained
from scratch scored **2.104 bpb** at context 1,024, versus **2.419**
for the local Transformer. Both scored the same 262,144 test bytes.
On the continuous evaluation, that successor checkpoint scored
**2.025 bpb** against **2.457** for the independently trained local
control on enwik8, and **2.965** against **3.416** on the same eight
PG-19 excerpts. This comparison uses one training seed. The model
adds 241 learned parameters and fits the entire network in one training
run using the same byte budget.
