# Every training arm of the read-clock study

The study of 18 to 19 November 2026 trained 49 arms. Each row is
one arm with one seed. Validation is 16 batches of 8 windows of
512 enwik8 validation bytes and scores the running weight
average. Every decoder arm is the release command of v13 with
flags added; the flags are given as they differ from those of
`end8-a8`, which are `--read-beats 8 --read-beat-start 1
--read-address-rate 8 --read-clock end`. The last column is an
arm's loss minus that of `end8-a8` at the same update, and for a
Transformer minus that of the reference Transformer.

Most pilots were told to stop at 12,000 updates and were stopped
by hand before that when they fell behind. The study later found
that the first 6,000 updates are the wrong place to judge an arm:
there every decoder leads the Transformer, and the Transformer's
lead forms after that. So an arm that was stopped early was
judged on its early learning alone. 10 arms were stopped before
their first evaluation at 1,500 updates and have no loss.

The flags do not record the code. Flags were added during the
study, and every one of them is in the code of this release. Two
more pilots ran on the laptop before the first arm of this list,
with a clock cell that heard the whole stream. The second left
the range of exact sums near 1,500 updates and neither wrote a
loss, and the clock cell of the release hears its read's tags
through synapses that saturate instead.
[Study page](READ_CLOCKS.md) ·
[All flags and curves](../results/event_lm/read_clocks/arms.json)

## The three decoders

The address rate alone, with the clock at word ends, and with a
learned clock cell for every head. `end8-a8-laptop` is the second
of these trained again on another machine to 12,000 updates.

| Arm | Started, UTC | Parameters | Differs in | Updates | Averaged | Against the base |
|---|---|---:|---|---:|---:|---:|
| `rate8` | 18 16:27 | 34,819,544 | no `--read-beats`, no `--read-beat-start`, no `--read-clock` | 90,000 | 1.3336 | −0.0023 at 30,000 |
| `end8-a8` | 18 16:27 | 34,819,652 | the base of this page | 30,000 | 1.4513 |  |
| `tag8-a8` | 18 16:27 | 34,822,736 | `--read-clock learned`, `--lr-scale clock_rest=8` | 90,000 | 1.3315 | −0.0080 at 30,000 |
| `end8-a8-laptop` | 18 15:21 | 34,819,652 | the base of this page | 12,000 | 1.5388 | −0.0060 |

## Read heads and clocks

More heads, more reads, marked cells, a faster clock cell, heads
that start sharp on one word each, and an address rate of 64 in
place of 8.

| Arm | Started, UTC | Parameters | Differs in | Updates | Averaged | Against the base |
|---|---|---:|---|---:|---:|---:|
| `end8-a8-h8` | 18 16:27 | 34,825,928 | `--read-heads 8` | 12,000 | 1.5517 | +0.0069 |
| `r7h8` | 18 16:43 | 34,831,400 | `--read-after 0,1,2,3,4,5,6`, `--read-heads 8`, `--granule-cells 174` | 12,000 | 1.5425 | −0.0023 |
| `end8-a8-m` | 18 17:00 | 34,819,760 | `--read-marks` | 24,000 | 1.4678 | −0.0102 |
| `tag8-a8-m` | 18 17:00 | 34,822,844 | `--read-marks`, `--read-clock learned`, `--lr-scale clock_rest=8` | 6,000 | 1.6230 | −0.0008 |
| `tag8-a8-c8` | 18 16:43 | 34,822,736 | `--read-clock learned`, `--lr-scale clock=8` | 7,500 | 1.5978 | −0.0014 |
| `end8-a8-s8` | 18 18:37 | 34,819,652 | `--read-beat-start 8` | 3,000 | 1.8922 | +0.1221 |
| `rate64` | 18 21:10 | 34,819,544 | `--read-address-rate 64`, no `--read-beats`, no `--read-beat-start`, no `--read-clock` | | | |
| `end64-m` | 18 21:10 | 34,819,760 | `--read-address-rate 64`, `--read-marks` | 24,000 | 1.4773 | −0.0007 |

## Beat fibres for the race cells

The beat fibres of v13 at full size: cells that hear the stream
of the last words, with the clock at word starts and at word
ends.

| Arm | Started, UTC | Parameters | Differs in | Updates | Averaged | Against the base |
|---|---|---:|---|---:|---:|---:|
| `word-sum` | 18 12:51 | 34,819,544 | `--granule-beats 4`, `--granule-clock word`, `--granule-hold sum`, no `--read-beats`, no `--read-beat-start`, no `--read-address-rate`, no `--read-clock` | 6,000 | 1.6755 | +0.0517 |
| `gend3-sum` | 18 17:24 | 34,795,780 | `--granule-fan 384`, `--granule-cells 233`, `--granule-beats 3`, `--granule-clock end`, `--granule-hold sum` | 3,000 | 1.7966 | +0.0265 |
| `gend3-mean` | 18 17:24 | 34,795,780 | `--granule-fan 384`, `--granule-cells 233`, `--granule-beats 3`, `--granule-clock end`, `--granule-hold mean` | 6,000 | 1.6367 | +0.0129 |

## Clusters

More and smaller clusters, each on fibres of its own, cell layers
right after the reads, and fibres that reach eight bytes back.

| Arm | Started, UTC | Parameters | Differs in | Updates | Averaged | Against the base |
|---|---|---:|---|---:|---:|---:|
| `c32n175` | 18 17:49 | 34,819,716 | `--granule-clusters 32`, `--granule-cells 175`, `--granule-budget 32` | 6,000 | 1.6311 | +0.0073 |
| `c64n87` | 18 17:57 | 34,754,180 | `--granule-clusters 64`, `--granule-cells 87`, `--granule-budget 16` | 4,500 | 1.6847 | +0.0123 |
| `c32f128` | 18 17:52 | 34,809,220 | `--granule-clusters 32`, `--granule-cells 349`, `--granule-fan 128`, `--granule-budget 32` | 4,500 | 1.6893 | +0.0169 |
| `c64f64` | 18 17:49 | 34,820,996 | `--granule-clusters 64`, `--granule-cells 348`, `--granule-fan 64`, `--granule-budget 16` | 3,000 | 1.7975 | +0.0274 |
| `c350n16` | 18 17:55 | 34,820,988 | `--granule-clusters 350`, `--granule-cells 16`, `--granule-budget 4` | | | |
| `c700n8` | 18 18:10 | 34,822,388 | `--granule-clusters 700`, `--granule-cells 8`, `--granule-budget 2` | | | |
| `g0246` | 18 18:26 | 34,819,652 | `--granule-after 0,2,4,6` | 6,000 | 1.6302 | +0.0064 |
| `dl8` | 18 19:18 | 34,819,652 | `--granule-delays 8` | | | |

## Races within a cluster

The cells of a cluster race in equal groups, so that more cells
answer at a byte. The last arm raises the pulse budget by the
spikes that its races add.

| Arm | Started, UTC | Parameters | Differs in | Updates | Averaged | Against the base |
|---|---|---:|---|---:|---:|---:|
| `r7b9` | 18 18:44 | 34,819,652 | `--granule-races 7`, `--granule-budget 9` | 4,500 | 1.6895 | +0.0171 |
| `r25b3` | 18 18:44 | 34,819,652 | `--granule-races 25`, `--granule-budget 3` | 3,000 | 1.7881 | +0.0180 |
| `r50b2` | 18 18:44 | 34,819,652 | `--granule-races 50`, `--granule-budget 2` | 3,000 | 1.8018 | +0.0317 |
| `r175b1` | 18 18:44 | 34,819,652 | `--granule-races 175`, `--granule-budget 1` | 3,000 | 1.7954 | +0.0253 |
| `r50b8` | 18 19:29 | 34,819,652 | `--granule-races 50`, `--granule-budget 8` | | | |
| `r50b8p` | 18 19:31 | 34,819,652 | `--granule-races 50`, `--granule-budget 8`, `--pulse-budget 331500` | 12,000 | 1.5630 | +0.0182 |

## Blocks against cells

Thinner blocks with more cells at the same size, and more cells
at a larger size.

| Arm | Started, UTC | Parameters | Differs in | Updates | Averaged | Against the base |
|---|---|---:|---|---:|---:|---:|
| `h122c624` | 18 19:07 | 34,804,420 | `--hidden 122`, `--delay-group-widths 38,38,23,23`, `--granule-cells 624` | 1,500 | 2.2660 | +0.1402 |
| `h61c762` | 18 19:07 | 34,784,580 | `--hidden 61`, `--delay-group-widths 19,19,12,11`, `--granule-cells 762` | 1,500 | 2.4680 | +0.3422 |
| `big-n700` | 18 19:32 | 46,310,852 | `--granule-cells 700` | 1,500 | 2.1100 | −0.0158 |
| `big-c32` | 18 19:32 | 46,310,916 | `--granule-clusters 32` | | | |

## What a cluster hears and reaches

Wider fans with fewer cells at the same size, and clusters that
hear more channels than their axons reach or the reverse.

| Arm | Started, UTC | Parameters | Differs in | Updates | Averaged | Against the base |
|---|---|---:|---|---:|---:|---:|
| `f512c175` | 18 19:23 | 34,808,452 | `--granule-fan 512`, `--granule-cells 175` | 3,000 | 1.7697 | −0.0004 |
| `f1024c88` | 18 19:23 | 34,868,420 | `--granule-fan 1024`, `--granule-cells 88` | 4,500 | 1.6839 | +0.0115 |
| `f1408c63` | 18 19:59 | 34,686,596 | `--granule-fan 1408`, `--granule-cells 63` | | | |
| `f1408k32` | 18 19:59 | 34,506,372 | `--granule-fan 1408`, `--granule-clusters 32`, `--granule-cells 31` | | | |
| `h1408r256` | 18 20:03 | 34,836,932 | `--granule-fan 1408`, `--granule-reach 256`, `--granule-cells 108` | | | |
| `h256r1408` | 18 20:03 | 34,836,932 | `--granule-fan 256`, `--granule-reach 1408`, `--granule-cells 108` | | | |

## How cells learn and answer

Racing cells that learn from a softer race than they fire in,
cells that answer with a smooth function of their own charge and
do not spike, and threshold cells that learn below their
threshold. The last four arms give the cells relay neurons that
hear the whole stream, so that a cell weighs a learned view of
all of it. In two of them the cells also hear the relays as they
were at the last letter of each of the last three words.

| Arm | Started, UTC | Parameters | Differs in | Updates | Averaged | Against the base |
|---|---|---:|---|---:|---:|---:|
| `soft2` | 18 19:36 | 34,819,652 | `--granule-soften 2` | 1,500 | 2.1541 | +0.0283 |
| `soft4` | 18 19:36 | 34,819,652 | `--granule-soften 4` | 3,000 | 1.7903 | +0.0202 |
| `softcells` | 18 20:11 | 34,819,652 | `--granule-soft`, `--pulse-budget 305904` | 12,000 | 1.5728 | +0.0280 |
| `slope` | 18 20:17 | 34,819,588 | `--granule-budget 0`, `--granule-slope` | 12,000 | 1.5717 | +0.0269 |
| `slope-relay` | 18 21:18 | 34,816,772 | `--granule-budget 0`, `--granule-slope`, `--granule-relays 256`, `--granule-cells 306` | 24,000 | 1.4863 | +0.0083 |
| `soft-relay` | 18 21:19 | 34,816,836 | `--granule-soft`, `--pulse-budget 305904`, `--granule-relays 256`, `--granule-cells 306` | 24,000 | 1.4819 | +0.0039 |
| `slope-relay-last` | 18 21:25 | 34,816,772 | `--granule-budget 0`, `--granule-slope`, `--granule-relays 256`, `--granule-cells 306`, `--granule-beats 3`, `--granule-clock end`, `--granule-hold last` | 24,000 | 1.4935 | +0.0155 |
| `relay-last` | 18 21:30 | 34,816,836 | `--granule-relays 256`, `--granule-cells 306`, `--granule-beats 3`, `--granule-clock end`, `--granule-hold last` | 24,000 | 1.4888 | +0.0108 |

## Transformers of the standard shape

The reference Transformer is 256 wide with 6 layers and an MLP of
42 times its width. These have the usual shape at the same size:
512 wide, 11 layers, 8 heads, an MLP of 4 times the width.

| Arm | Started, UTC | Parameters | Differs in | Updates | Averaged | Against the base |
|---|---|---:|---|---:|---:|---:|
| `transformer-std` | 18 16:27 | 34,752,000 | a Transformer under AdamW | 90,000 | 1.2607 | −0.0176 |
| `transformer-std-rot` | 18 16:27 | 34,752,000 | a Transformer under the rotated optimizer | 7,500 | 1.5448 | −0.1685 at 6,000 |

