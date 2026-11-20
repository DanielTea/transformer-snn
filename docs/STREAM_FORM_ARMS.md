# Every training arm of the stream-form study

The study of 19 to 20 November 2026 trained 55 arms at full size.
Each row is one arm with one seed. Validation is 16 batches of 8
windows of 512 enwik8 validation bytes from an empty state and
scores the running weight average. A stack arm is the stack
command below with what its row names added, where a flag given
again replaces the value of the command. A Transformer arm is the
Transformer command with its additions. The last column is an
arm's loss minus that of the reference Transformer `tf-rot-3e-4`
at the same update.

Most arms were pilots that were stopped when they had answered
their question, so their last update says where they were stopped
and not where they would have ended. An arm that continues
another took over its saved state at the update named, and its
curve starts there. A fork and its parent share everything up to
that update, which makes their difference more reliable than a
difference between runs. One arm was stopped before its first
evaluation and has no loss.

The flags do not record the code. Flags were added while the
study ran, and every one of them is in the code of this release.
Arms that name a build ran on a private build whose switches
`experiments/stack/diagnosis_build.py` reproduces. The fine pool
is `--pool-fine 16 --pool-rest 4 --pool-rate 32 --output-pool
128`, and the arms that started before the pool was in the
package ran it as a build.

The stack command:

```text
--variant event_steered --no-carry --stack --dim 512 --hidden 4 --depth 11
--spike-resolution 16 --router-charge-bits 8 --pulse-budget 310000
--read-after 0,1,2,3,4,5,6,7,8,9,10 --read-width 512 --read-heads 8
--read-window 512 --read-mode race --read-budget 256 --read-resolution 4
--read-address-rate 64 --granule-after 0,1,2,3,4,5,6,7,8,9,10
--granule-clusters 1 --granule-fan 512 --granule-cells 2038 --granule-slope
--granule-cap 128 --read-cap 128 --rotated-adam 50 --rotate-stacks
--weight-average 0.999
```

The Transformer command:

```text
--variant transformer --dim 512 --depth 11 --heads 8 --mlp-ratio 4
--weight-average 0.999
```

Both end with:

```text
--data data/enwik8 --training-mixture
data/training-v1/prepared/catalogue.json --ctx 512 --bs 8 --steps 90000
--warmup 1000 --lr 3e-4 --wd 0.05 --seed 20001 --eval-every 1500 --eval-bs 8
--eval-batches 16 --final-eval-batches 64 --validation-only --save-every
1500
```

[Study page](STREAM_FORM.md) ·
[All flags and curves](../results/event_lm/stream_form/arms.json)

## The stack in the Transformer's layout

The first stacks: a read and a layer of threshold cells at each
of eleven levels, with the coarse pool of the decoders.
`stack-r64` is the stack command as it stands, and the others
change the address rate, the phases of a read or the learning
rate, or give the cells delayed fibres.

| Arm | Started, UTC | On | Parameters | Adds | Updates | Averaged | Above the Transformer |
|---|---|---|---:|---|---:|---:|---:|
| `stack-a` | 19 08:46 | laptop GPU | 34,820,162 | `--read-address-rate 8` | 4,500 | 1.7380 | +0.0592 |
| `stack-r64` | 19 09:12 | laptop GPU | 34,820,162 | the command as it stands | 6,000 | 1.6226 | +0.0323 |
| `stack-d4` | 19 09:03 | server CPU | 34,818,325 | `--read-address-rate 8 --granule-fan 1024 --granule-delays 4 --granule-cells 1359` | 1,500 | 2.1833 | +0.0559 |
| `stack-d4-r64` | 19 09:15 | server CPU | 34,818,325 | `--granule-fan 1024 --granule-delays 4 --granule-cells 1359` | 3,000 | 1.7632 | −0.1012 |
| `stack-p16-r64` | 19 10:24 | laptop GPU, then server CPU | 34,819,810 | `--read-phases 16 --granule-cells 2006` | | | |
| `stack-lr1e3` | 19 14:14 | laptop GPU | 34,820,162 | `--lr 1e-3 --eval-every 750` | 6,000 | 1.6289 | +0.0386 |
| `stack-3e-4` | 19 16:22 | server GPU | 34,820,162 | the command as it stands | 18,000 | 1.4644 | +0.0636 |
| `stack-1e-3` | 19 16:22 | server GPU | 34,820,162 | `--lr 1e-3` | 12,000 | 1.5585 | +0.1011 |

## Reference Transformers

The Transformer of the standard shape under the rotated optimizer
at two learning rates. The run at 0.0003 is the reference of this
release. `transformer-std-rot` is the same configuration, begun
in the study of v14 and continued here from its saved state at
7,500 updates on another machine: the two runs are 0.002 apart at
9,000 updates.

| Arm | Started, UTC | On | Parameters | Adds | Updates | Averaged | Above the Transformer |
|---|---|---|---:|---|---:|---:|---:|
| `tf-rot-3e-4` | 19 16:22 | server GPU | 34,752,000 | `--rotated-adam 50` | 90,000 | 1.2348 |  |
| `tf-rot-1e-3` | 19 16:22 | server GPU | 34,752,000 | `--rotated-adam 50 --lr 1e-3` | 45,000 | 1.2925 | −0.0032 |
| `transformer-std-rot` | 19 11:24 | server CPU | 34,752,000 | `--rotated-adam 50` | 10,500 | 1.4812 | +0.0024 |

## Transformers with parts of the stack

Controls in floats: Transformers that take over the stack's plain
units, its age offsets or its untied output, one at a time and
together, and two with other units. None is a spiking network.

| Arm | Started, UTC | On | Parameters | Adds | Updates | Averaged | Above the Transformer |
|---|---|---|---:|---|---:|---:|---:|
| `tf-twin` | 19 11:33 | laptop GPU | 34,802,688 | `--feed-forward relu --age-offsets 64` | 12,000 | 1.5162 | +0.0588 |
| `tf-relu-rot-1e-3` | 19 17:10 | server GPU | 34,757,632 | `--rotated-adam 50 --lr 1e-3 --feed-forward relu` | 12,000 | 1.4821 | +0.0247 |
| `tf-gelu-rot-1e-3` | 19 18:29 | server GPU | 34,757,632 | `--rotated-adam 50 --lr 1e-3 --feed-forward relu`, build `SNN_TF_ACT=gelu` | 12,000 | 1.4653 | +0.0079 |
| `tf-rglu-rot-1e-3` | 19 18:29 | server GPU | 34,752,000 | `--rotated-adam 50 --lr 1e-3`, build `SNN_TF_GATE=relu` | 12,000 | 1.4698 | +0.0124 |
| `tf-age-rot-1e-3` | 19 18:01 | server GPU | 34,797,056 | `--rotated-adam 50 --lr 1e-3 --age-offsets 64` | 12,000 | 1.4643 | +0.0069 |
| `twin-rot-1e-3` | 19 18:01 | server GPU | 34,802,688 | `--rotated-adam 50 --lr 1e-3 --feed-forward relu --age-offsets 64` | 12,000 | 1.4806 | +0.0232 |
| `twin-untied-rot-1e-3` | 19 19:54 | server GPU | 34,933,760 | `--rotated-adam 50 --lr 1e-3 --feed-forward relu --age-offsets 64`, build `SNN_TF_UNTIE=1` | 9,000 | 1.5276 | +0.0208 |

## Float stand-ins of the stack

Stacks that hear a LayerNorm or an exact division of the stream,
count in real numbers, or have no pulse budget. They place the
distance to the Transformer and are no spike mechanisms. Their
switches are those of `experiments/stack/diagnosis_build.py`.

| Arm | Started, UTC | On | Parameters | Adds | Updates | Averaged | Above the Transformer |
|---|---|---|---:|---|---:|---:|---:|
| `ln-1e-3` | 19 16:20 | server GPU | 34,843,714 | `--lr 1e-3`, build `SNN_LN=1 SNN_LN_OUT=1` | 12,000 | 1.5078 | +0.0504 |
| `nl1-1e-3` | 19 16:20 | server GPU | 34,820,162 | `--lr 1e-3`, build `SNN_LN=1 SNN_LN_OUT=1 SNN_NORM=l1` | 15,000 | 1.4784 | +0.0574 |
| `ln-2e-3` | 19 16:22 | server GPU | 34,843,714 | `--lr 2e-3`, build `SNN_LN=1 SNN_LN_OUT=1` | 4,500 | 1.6659 | −0.0129 |
| `ln-3e-4` | 19 16:22 | server GPU | 34,843,714 | build `SNN_LN=1 SNN_LN_OUT=1` | 12,000 | 1.5083 | +0.0509 |
| `ln-free-1e-3` | 19 17:10 | server GPU | 34,843,714 | `--lr 1e-3 --pulse-budget 100000000`, build `SNN_LN=1 SNN_LN_OUT=1` | 9,000 | 1.5337 | +0.0269 |
| `real-ln-1e-3` | 19 17:10 | server GPU | 34,843,714 | `--lr 1e-3`, build `SNN_LN=1 SNN_LN_OUT=1 SNN_REAL=1` | 9,000 | 1.5379 | +0.0311 |
| `real-ln-free-1e-3` | 19 17:12 | server GPU | 34,843,714 | `--lr 1e-3 --pulse-budget 100000000`, build `SNN_LN=1 SNN_LN_OUT=1 SNN_REAL=1` | 15,000 | 1.4641 | +0.0431 |
| `real-ln-free-noceil-1e-3` | 19 20:08 | server GPU | 34,843,714 | `--lr 1e-3 --pulse-budget 100000000`, build `SNN_LN=1 SNN_LN_OUT=1 SNN_NOCEIL=1 SNN_REAL=1`, continues `real-ln-free-1e-3` from 9,000 | 12,000 | 1.4947 | +0.0373 |
| `fp16-lnout-1e-3` | 19 18:59 | server GPU | 34,821,186 | `--lr 1e-3`, build `SNN_FINE=16 SNN_LN_OUT=1 SNN_POOL=4,32` | 9,000 | 1.5500 | +0.0432 |

## The fine pool

The pool of inhibition that divides: thresholds in quanta sixteen
times finer and a pool cell that fires once for every 64 pulses
of the stream. Then turned probes, a wider learning slope for the
cells, and a ceiling that the pool lifts, as a run of its own and
as a fork beside its unchanged parent.

| Arm | Started, UTC | On | Parameters | Adds | Updates | Averaged | Above the Transformer |
|---|---|---|---:|---|---:|---:|---:|
| `fp16-1e-3` | 19 17:01 | server GPU | 34,820,162 | the fine pool, `--lr 1e-3` | 19,500 | 1.4810 | +0.0885 |
| `fp16-3e-4` | 19 17:03 | server GPU | 34,820,162 | the fine pool | 25,500 | 1.4086 | +0.0472 |
| `fp16-1e-3-s2` | 19 17:12 | server GPU | 34,820,162 | the fine pool, `--lr 1e-3 --seed 20002` | 12,000 | 1.5222 | +0.0648 |
| `fp16-turn-1e-3` | 19 17:12 | server GPU | 34,820,162 | the fine pool, `--lr 1e-3 --read-turn` | 12,000 | 1.5335 | +0.0761 |
| `fp16-wide0-1e-3` | 19 20:23 | server GPU | 34,820,162 | the fine pool, `--lr 1e-3`, build `SNN_SLOPE=16` | 6,000 | 1.6048 | +0.0145 |
| `fp16-wide-1e-3` | 19 20:22 | server GPU | 34,820,162 | the fine pool, `--lr 1e-3`, build `SNN_SLOPE=16`, continues `fp16-1e-3` from 9,000 | 15,000 | 1.4947 | +0.0737 |
| `fp16-lift-1e-3` | 19 21:40 | server GPU | 34,820,162 | the fine pool, `--lr 1e-3 --pool-lifts-ceiling` | 6,000 | 1.5933 | +0.0030 |
| `fp16-lift-fork-1e-3` | 19 21:44 | server GPU | 34,820,162 | the fine pool, `--lr 1e-3`, build `SNN_LIFT=1`, continues `fp16-1e-3` from 15,000 | 19,500 | 1.4623 | +0.0698 |

## Gated cells and a veto

Cells with a gate dendrite, which is the spike form of the
Transformer's gated unit, in several variants, and cells with a
dendrite that closes them. None learned as well as plain cells.

| Arm | Started, UTC | On | Parameters | Adds | Updates | Averaged | Above the Transformer |
|---|---|---|---:|---|---:|---:|---:|
| `stack-tg` | 19 12:46 | laptop GPU | 34,818,325 | `--read-turn --granule-cells 1359 --granule-gate 4` | 1,500 | 3.3164 | +1.1890 |
| `stack-tgp` | 19 13:14 | laptop GPU | 34,818,325 | `--read-turn --granule-cells 1359 --granule-gate 4 --granule-price 1 --eval-every 300` | 3,000 | 1.9860 | +0.1216 |
| `fpg-1e-3` | 19 18:33 | server GPU | 34,818,325 | the fine pool, `--lr 1e-3 --granule-gate 4 --granule-cells 1359 --lr-scale .gate=0.3,.receive=0.3 --eval-every 500 --granule-price 1` | 6,000 | 1.7352 | +0.1449 |
| `fpg-free-1e-3` | 19 18:33 | server GPU | 34,818,325 | the fine pool, `--lr 1e-3 --granule-gate 4 --granule-cells 1359 --lr-scale .gate=0.3,.receive=0.3 --eval-every 500 --pulse-budget 100000000` | 4,500 | 1.7550 | +0.0762 |
| `fpg-3e-4` | 19 19:13 | server GPU | 34,818,325 | the fine pool, `--lr 3e-4 --granule-gate 4 --granule-cells 1359 --granule-price 1 --eval-every 500` | 1,500 | 3.8263 | +1.6989 |
| `fpg-d4-1e-3` | 19 19:15 | server GPU | 34,801,077 | the fine pool, `--lr 1e-3 --granule-gate 4 --granule-fan 1024 --granule-delays 4 --granule-cells 815 --granule-price 1 --lr-scale .gate=0.2,.receive=0.2 --eval-every 500` | 6,500 | 3.8958 | +0.1945 at 6,000 |
| `fpg-1e-3-s2` | 19 19:45 | server GPU | 34,818,325 | the fine pool, `--lr 1e-3 --granule-gate 4 --granule-cells 1359 --granule-price 1 --eval-every 500 --lr-scale .gate=0.3,.receive=0.3 --seed 20002` | 2,000 | 1.7717 | −0.2698 at 1,500 |
| `fpg-u-1e-3` | 19 19:48 | server GPU | 34,818,325 | the fine pool, `--lr 1e-3 --granule-gate 4 --granule-cells 1359 --granule-price 1 --eval-every 500` | 1,000 | 6.5953 |  |
| `fpg-wide-1e-3` | 19 20:24 | server GPU | 34,818,325 | the fine pool, `--lr 1e-3 --granule-gate 4 --granule-cells 1359 --lr-scale .gate=0.3,.receive=0.3 --eval-every 500 --granule-price 1`, build `SNN_SLOPE=16`, continues `fpg-1e-3` from 4,500 | 5,000 | 1.6945 |  |
| `fp16-veto-1e-3` | 19 21:43 | server GPU | 34,818,325 | the fine pool, `--lr 1e-3 --granule-veto 4 --granule-cells 1359 --lr-scale .veto=0.167` | 6,000 | 1.6161 | +0.0258 |

## Delayed fibres

Cells that hear every channel again after one to three bytes, and
tags that hear the bytes after their own. `fp16-d4-rd4-3e-4` is
the stack that the stream form starts from. It trained on windows
of 512 bytes to the end.

| Arm | Started, UTC | On | Parameters | Adds | Updates | Averaged | Above the Transformer |
|---|---|---|---:|---|---:|---:|---:|
| `fp16-d4-1e-3` | 19 19:46 | server GPU | 34,818,325 | the fine pool, `--lr 1e-3 --granule-fan 1024 --granule-delays 4 --granule-cells 1359` | 6,000 | 1.5993 | +0.0090 |
| `fp16-d4-3e-4` | 19 20:17 | server GPU | 34,818,325 | the fine pool, `--lr 3e-4 --granule-fan 1024 --granule-delays 4 --granule-cells 1359` | 12,000 | 1.4914 | +0.0340 |
| `fp16-d4-rd4-1e-3` | 19 20:33 | server GPU | 34,810,812 | the fine pool, `--lr 1e-3 --granule-fan 1024 --granule-delays 4 --granule-cells 1188 --read-delays 4` | 3,000 | 1.6786 | −0.1858 |
| `fp16-d4-rd4p-1e-3` | 19 20:33 | server GPU | 34,820,206 | the fine pool, `--lr 1e-3 --granule-fan 1024 --granule-delays 4 --granule-cells 1018 --read-delays 4 --read-delay-probes` | 3,000 | 1.6706 | −0.1938 |
| `fp16-d4-rd4-3e-4` | 19 21:06 | server GPU | 34,810,812 | the fine pool, `--lr 3e-4 --granule-fan 1024 --granule-delays 4 --granule-cells 1188 --read-delays 4` | 90,000 | 1.2828 | +0.0480 |

## Streams and long windows

Forks of `fp16-d4-rd4-3e-4`: training on streams in stretches of
20 and 60 windows, in groups of streams that take turns, the same
with a longer reach, and training on windows of 1,024 bytes,
which is the released stack. Their window validation reads
windows of their own length, so the last column is left empty
where the windows are not those of the Transformer.

| Arm | Started, UTC | On | Parameters | Adds | Updates | Averaged | Above the Transformer |
|---|---|---|---:|---|---:|---:|---:|
| `fp16-d4-rd4-s20-3e-4` | 19 22:50 | server GPU | 34,810,812 | the fine pool, `--lr 3e-4 --granule-fan 1024 --granule-delays 4 --granule-cells 1188 --read-delays 4 --stream 20 --stream-from 16500`, continues `fp16-d4-rd4-3e-4` from 16,500 | 28,500 | 1.4122 | +0.0679 |
| `fp16-d4-rd4-s60-3e-4` | 19 22:54 | server GPU | 34,810,812 | the fine pool, `--lr 3e-4 --granule-fan 1024 --granule-delays 4 --granule-cells 1188 --read-delays 4 --stream 60 --stream-from 16500`, continues `fp16-d4-rd4-3e-4` from 16,500 | 19,500 | 1.4514 | +0.0589 |
| `fp16-d4-rd4-g15-3e-4` | 19 23:26 | server GPU | 34,810,812 | the fine pool, `--lr 3e-4 --granule-fan 1024 --granule-delays 4 --granule-cells 1188 --read-delays 4 --stream 20 --stream-from 21000 --stream-groups 15`, continues `fp16-d4-rd4-3e-4` from 21,000 | 24,000 | 1.4221 | +0.0571 |
| `fp16-d4-rd4-g15-r1024-3e-4` | 19 23:26 | server GPU | 34,810,812 | the fine pool, `--lr 3e-4 --granule-fan 1024 --granule-delays 4 --granule-cells 1188 --read-delays 4 --stream 20 --stream-from 16500 --stream-groups 15 --read-reach 1024`, continues `fp16-d4-rd4-s20-3e-4` from 28,500 | 42,000 | 1.3807 | +0.0763 |
| `fp16-d4-rd4-w1024-3e-4` | 19 23:58 | server GPU | 34,810,812 | the fine pool, `--lr 3e-4 --granule-fan 1024 --granule-delays 4 --granule-cells 1188 --read-delays 4 --ctx 1024 --bs 4 --read-reach 1024`, continues `fp16-d4-rd4-3e-4` from 24,000 | 90,000 | 1.2258 |  |

