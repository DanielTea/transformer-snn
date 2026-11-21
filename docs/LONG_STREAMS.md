# Long streams

v15 left the spiking stack level with a Transformer: ahead on
the enwik8 test split, behind on validation. That Transformer
read 512 bytes, while the stack trained on windows of 1,024
bytes and read up to 4,096 cells back. This release trains
Transformers on the stack's own window lengths. **At equal
training context the stack is behind**: by 0.035 bits per byte
on the test split at 1,024 bytes and by 0.032 at 2,048, and by
0.042 and 0.038 on validation. Longer windows help both models
alike, and the gap narrows by less than 0.006 a doubling.

![The stack and the Transformer at training windows of 512, 1,024 and 2,048 bytes, test and validation](../results/figures/event_lm_long_streams.png)

<!-- SUMMARY OF THE REST: slopes, books, cells, scaling, plasticity -->

## Limits

- Every run is one seed, except the second seeds named in the
  text. The paired intervals cover the choice of the scored
  pieces, not the training run.
- All runs share one schedule: 90,000 updates of 4,096 targets,
  of which the first 24,000 use windows of 512 bytes. The longer
  windows begin from those saved states.
- The stack is scored as unbroken streams and the Transformer in
  windows of its training length, 16 scored bytes after the rest
  as history. Both read every byte of a split once.
- Settings chosen on validation bytes (reach, forgetting slope,
  thresholds) were chosen on the first million of them, and the
  remaining four million are also reported alone.

## The comparison at equal context

| Training window | Stack, as a stream | Transformer, in windows | Stack minus Transformer | Paired 95% interval |
|---|---:|---:|---:|---:|
| 512 bytes, test | 1.1852 | 1.1439 | +0.0413 | +0.0405 to +0.0422 |
| 512 bytes, validation | 1.2093 | 1.1629 | +0.0464 | +0.0455 to +0.0473 |
| 1,024 bytes, test | 1.1536 | 1.1182 | +0.0354 | +0.0344 to +0.0363 |
| 1,024 bytes, validation | 1.1828 | 1.1410 | +0.0417 | +0.0408 to +0.0426 |
| 2,048 bytes, test | 1.1390 | 1.1067 | +0.0323 | +0.0314 to +0.0333 |
| 2,048 bytes, validation | 1.1687 | 1.1303 | +0.0384 | +0.0374 to +0.0394 |

The stack reads as far as it trained: 512, 1,024 and 2,048
cells back. The rows at 512 bytes are v15's stack and
Transformer; at 1,024, v15's released stack and a Transformer
trained like it; at 2,048, `l2048-flat` and `tf2048`, both new.
With fast output synapses on both models at 2,048 bytes (the
stack's at the reach chosen on the first million validation
bytes, 8,192 cells at levels 4 to 7), the stack scores 1.1347
and 1.1666 and the Transformer 1.1032 and 1.1290: the stack is
behind by 0.0315 [0.0305, 0.0325] and 0.0376 [0.0366, 0.0386].

A stream lets the stack read further than it trained at no cost,
which a Transformer's window does not. At 1,024 bytes that gave
v15 its level score: a reach of 4,096 cells after levels 5 to 7
scored 1.1406 on test, and with a fixed forgetting slope (below)
a reach of 16,384 scores 1.1324. At 2,048 bytes it no longer
helps: the best longer reach on the first million validation
bytes scores 1.1404 on test, against 1.1390 as trained.

<!-- l2048-forget: the learned slopes at equal context, and paired against l2048-flat -->

## Forgetting by a power of the age

A read is a softmax in whole spikes. Every cell within reach has
a charge, the match of the present probe with the cell's tag
plus an offset for the cell's age, and the cells share a budget
of 256 spikes in proportion to e to the power of their charges.
Each head learns one offset for each of the last 512 ages. A cell
older than that took the offset of the oldest age in v15,
whatever its age.

Here a cell beyond the window takes that offset plus a slope
times the number of doublings of its age beyond the window:

    offset(age) = offset(511) + s · log2(age / 511)

for ages from 512 to the reach. The share of a cell then falls
as its age to the power s / ln 2, a power law of forgetting.
Memory in people forgets by a power of the time that has passed,
and the chance that an item is needed again falls by a power of
the time since it was last needed (Anderson and Schooler, 1991).
Text has the same property: a word that has appeared is likely
to appear again soon, and less likely the longer it has not.

A slope below −0.69 charge units a doubling (−ln 2) keeps what
all far cells together can take of a read bounded at any reach,
since the shares then fall faster than the inverse of the age.
Above it, the far past takes a larger share the further a read
reaches. Each of the 88 heads of the eleven reads learns its
slope, from zero at the start (`--read-forget`), and a control
holds every slope at zero (`--read-forget-rate 0`).

The closest position bias in Transformers is the logarithmic
kernel of KERPLE (Chi et al., 2022), which subtracts a learned
multiple of the log of the distance from every attention score.
ALiBi (Press et al., 2022) subtracts a multiple of the distance
itself, which forgets exponentially. Here the slope applies only
beyond the window of learned offsets, and the offsets within it
stay free.

**One slope for every head, set at scoring.** On the released
stack of v15, with no training at all, a slope chosen on the
first million validation bytes:

| Reach | Slope 0 | −0.5 | −1 | −2 |
|---|---:|---:|---:|---:|
| 4,096 after levels 5 to 7 (v15's setting) | 1.19405 | | 1.19323 | |
| 8,192 | 1.23214 | 1.19016 | 1.18721 | 1.20088 |
| 16,384 | | | **1.18671** | 1.20060 |

Without forgetting, a reach of 8,192 is 0.045 worse than with a
slope of −1. The best slope, −1, makes a far cell's share fall
as its age to the power −1.44, just past the bound. With it, at
a reach of 16,384, the released stack scores 1.1324 on the test
split and 1.1618 on validation: better than v15's own setting
by 0.0082 on test [0.0078, 0.0085] and by 0.0076 on validation
[0.0072, 0.0079], and 0.0115 ahead of the Transformer that reads
512 bytes on test. It is still 0.0142 behind the Transformer that
trained on 1,024 bytes.

<!-- learned slopes: every head's exponent, the share of read spikes beyond the window, l2048-forget against l2048-flat -->

## Learning while reading a book

The released stack of v15 has fast output synapses that every
arriving byte teaches, by a local rule in whole numbers, after
the byte has been predicted. They start at zero in every stream.
Two scorings of the same streams, with and without them, keep
their bits for every piece of 512 bytes. Pooling their
difference over the pieces that follow the same amount of
reading gives the gain after that much of a stream
(`experiments/reading_curve.py`).

![The gain of the fast synapses against how much of a stream they have read](../results/figures/event_lm_long_streams_reading.png)

On the 16 PG-19 test books, read in parts of 256 KB, the gain
grows as the square root of the text read: 0.0042 bits per byte
in the first 8 KB, 0.0096 after 16 to 32 KB, 0.0185 after 64 to
128 KB and 0.0278 after 128 to 256 KB. A power law through the
six stretches has an exponent of 0.52. The reads of the stack
reach at most 4,096 bytes back, so all of this rise comes from
the fast synapses. On the enwik8 test split the gain stays
between 0.004 and 0.010 wherever it is measured: a stretch of
Wikipedia changes its subject every few kilobytes, and a book
does not.

The number of distinct words in a text grows about as the square
root of its length (Heaps' law). That the gain grows the same way
suggests that the synapses learn the book's own names and words
as they recur. It is a reading of the curve, not a test of it.

<!-- whole books as single streams: released stack taught and not, the Transformer taught along the same books -->

## The next cell

A plain threshold cell fires one spike for every threshold of
charge above its rest, so its count rises in a line from
silence, as a rectifier does. v15's walk from the Transformer to
the stack found that plain units in place of gated ones cost
most, 0.025 bits per byte. Two cells that multiply charges were
piloted on the server, each priced like every other spike, to
24,000 updates under rules set before the weekend:

- **Squared cells** (`--granule-gate 4 --granule-self-gate
  --granule-price 1`): a gate dendrite that hears the fibres
  through the cell's own synapses lets the same charge in, so k
  thresholds of charge fire k² spikes. They collapsed at once:
  5.26 bits per byte at 1,500 updates.
- **Gated cells** (`--granule-gate 4 --granule-price 1`, 713
  cells for the same parameters): 1.9253 at 1,500 updates, then
  2.55 at 3,000, with 396,000 pulses a byte by 4,500.

Both were stopped as unstable under the pre-registered rule.
Products of charges are unstable at this learning rate, priced
or not. The price on plain cells alone (`plain-price`) runs level
with the plain stack: 1.4064 against 1.4076 at 24,000 updates.

In the Transformer, squared rectifiers (`tf-relu2`) recover what
plain units lose: 1.5087 at 9,000 updates, against GELU's 1.5094
and plain rectifiers' 1.5249 at the walk's rate. The spike form
of a squared unit is the open question; this weekend's form of
it does not train.

## How the two models scale

The weekend's runs give the first view of the two models along
more than one axis.

![Validation loss of the stack and the Transformer along their training, at depths 11 and 22](../results/figures/event_lm_long_streams_scaling.png)

- **Context.** Each doubling of the training window gives both
  models about half the gain of the previous one, close to a
  loss of L∞ + A/C. The stack gains a little more each time, so
  the gap shrinks from 0.041 to 0.035 to 0.032 on test. Three
  points fix the three numbers of each curve exactly, with no
  error left to judge them by; taken at face value they leave a
  gap of 0.029 at any window.
- **Depth.** Doubling the depth to 22 levels (69.4M parameters)
  gains the Transformer 0.043 bits per byte at 24,000 updates
  and the stack 0.020, in windows from an empty state; on
  windows of 2,048 bytes, 0.037 and 0.021 at 31,500. The gap
  widens with size, from 0.035 to 0.057 at 24,000 updates. The
  deeper stack kept the budget of 310,000 pulses a byte, which
  counts the spikes of the whole stack, so each of its levels
  fired about half as much: this is depth at equal energy, and a
  test-bed arm with the budget doubled measures how much of the
  shortfall the budget is.
- **Training.** The stack is ahead for about the first 7,500
  updates (5,800 at depth 22), by up to 0.25 bits per byte, and
  then falls behind. Over 3,000 to 24,000 updates, a power law
  loss = E + A t^−α fits the stacks with α of 0.23 to 0.31 and
  the Transformers with 0.74 to 0.85, over two seeds and both
  depths.

On every axis we can measure, the Transformer's curve is at
least as steep as the stack's: scaling this design does not
close the gap. What the stack has that the Transformer has not
is a stream, and with it learning while reading that grows with
the text.

## Trained plasticity

v15's fast synapses were given to a stack that never saw them.
A study on a small test bed trained four rules in the loop and
asked whether reading-time learning gets clearly stronger. No
rule trained in the loop beats v15's rule given after training
when each is scored with its own rule. But a stack trained with
a decaying rule in the loop has cells that v15's rule uses
better: with that rule while reading it beats the plain stack by
0.0130 bits per byte [0.0122, 0.0138] along the whole stream and
on a long one, which passed the study's go/no-go by its second
route. Its full-size sibling of `l2048-flat` is training. The
whole study, its criteria and every arm are on [its own
page](TRAINED_PLASTICITY.md).

![The trained-plasticity test bed: each arm with its own rule and its cells with v15's rule](../results/figures/event_lm_trained_plasticity.png)

## Reproduce

<!-- commands -->
