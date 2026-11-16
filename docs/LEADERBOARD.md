# Spiking byte model leaderboards

Lower bits per byte rank higher. Every row in a comparison uses the same enwik8 training exposure and scores 4,999,999 enwik8 targets and 8,404,368 targets from 16 complete PG-19 books.

State is persistent tensor storage for one stream. Full attention reports FP32 key and value storage at the context length in each comparison. Pulse counts use a native 4,096-byte enwik8 test sample. Each PG-19 temperature is fitted on eight separate validation books. The teacher-guided SNN uses additional training compute from a frozen Transformer.

## Driven held-charge reads · 116,326,400 training targets · seed 8237

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | PG-19 raw | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original held-charge read | 12,799,488 | 28,584 B | 1.79577 | 3 | **2.75495** | 1 | 2.84136 | 309,354 |
| Write input, six-bit reset | 12,802,161 | 28,584 B | 1.79582 | 4 | 2.75706 | 2 | 2.84654 | 309,554 |
| Write input, eight-bit carry | 12,802,161 | 31,257 B | 1.79533 | 2 | 2.75724 | 3 | 2.84850 | 309,331 |
| Full attention | 12,801,024 | 6 MiB | **1.71363** | 1 | 2.88478 | 4 | 3.08081 | — |

![Held-out next-byte loss after training the driven read circuits](../results/figures/event_lm_trace_read_drive.png)

[Circuit, native activity, and matched controls](TRACE_READ_DRIVE.md)

## Packed width and steering transfers · 111,411,200 training targets · seed 7907

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | PG-19 raw | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 187-channel SNN | 9,960,960 | 29,674 B | 1.80153 | 4 | 2.77445 | 5 | 2.86922 | 309,773 |
| 243-channel SNN | 12,799,488 | 32,616 B | 1.80080 | 3 | 2.76563 | 3 | 2.85588 | 308,415 |
| Original full attention | 9,962,496 | 6 MiB | 1.71789 | 2 | 2.88474 | 6 | 3.08009 | — |
| Wider full attention | 12,801,024 | 6 MiB | **1.71786** | 1 | 2.88475 | 7 | 3.08010 | — |
| Delayed steering transfers | 12,879,678 | 32,616 B | 1.80187 | 5 | 2.76241 | 2 | 2.85741 | 308,085 |
| Growth in shortest groups | 12,799,488 | 28,584 B | 1.80256 | 6 | **2.76153** | 1 | 2.85249 | 309,698 |
| Present steering transfers | 12,879,678 | 32,616 B | 1.80400 | 7 | 2.77171 | 4 | 2.86885 | 310,478 |

![Held-out loss at the original and wider sizes](../results/figures/event_lm_packed_width.png)

![Held-out loss changes from channel placement and steering sources](../results/figures/event_lm_packed_circuit_changes.png)

![Validation loss gap as the available history grows](../results/figures/event_lm_packed_width_history.png)

[Width, state budget, and steering circuit](PACKED_WIDTH.md)

## 9.96M axon tap reads · 101,580,800 training targets · seed 6899

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | PG-19 raw | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Continued SNN | 9,960,960 | 29,674 B | 1.81319 | 2 | **2.77756** | 1 | 2.87524 | 309,354 |
| Shortest eligible tap | 9,962,312 | 29,674 B | 1.81655 | 4 | 2.78455 | 2 | 2.88665 | 308,210 |
| Held spike race | 9,962,312 | 29,674 B | 1.81473 | 3 | 2.78521 | 3 | 2.88687 | 309,582 |
| 6-block full attention | 9,962,496 | 6 MiB | **1.73380** | 1 | 2.89702 | 4 | 3.09548 | — |

![Held-out loss for the spike race, shortest tap, and matched controls](../results/figures/event_lm_trace_tap_read.png)

![Validation loss and pulse counts during the tap comparison](../results/figures/event_lm_trace_tap_training.png)

[Timing circuit and interventions](TRACE_TAP_READ.md)

## 9.96M trace inputs and group thresholds · 91,750,400 training targets · seed 5801

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | PG-19 raw | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Continued v6 SNN | 9,960,960 | 29,674 B | 1.82999 | 5 | **2.78446** | 1 | 2.88327 | 307,913 |
| Learned trace input spikes | 9,965,074 | 29,674 B | 1.82956 | 4 | 2.78972 | 3 | 2.89110 | 309,390 |
| Coarse group thresholds | 9,961,004 | 29,674 B | 1.82918 | 2 | 2.78697 | 2 | 2.88461 | 308,616 |
| One-tick group thresholds | 9,961,004 | 29,674 B | 1.82929 | 3 | 2.79350 | 4 | 2.89740 | 308,741 |
| 6-block full attention | 9,962,496 | 6 MiB | **1.74475** | 1 | 2.90840 | 5 | 3.10229 | — |

![Held-out loss for trace inputs, group thresholds, and matched controls](../results/figures/event_lm_trace_integration.png)

![Native resting-threshold offsets learned by each delay group](../results/figures/event_lm_fine_group_thresholds.png)

![Gap in held-out loss by available history at the v6 weights](../results/figures/event_lm_validation_history.png)

[Circuit and interventions](TRACE_INTEGRATION.md)

## 9.96M depth and width growth · 81,920,000 training targets · seed 4603

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | PG-19 raw | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 11-block SNN | 9,960,960 | 29,674 B | 1.84288 | 2 | **2.79854** | 1 | 2.90068 | 309,059 |
| 7-block wider SNN | 9,957,868 | 19,536 B | 1.84496 | 3 | 2.80299 | 2 | 2.90845 | 307,543 |
| 6-block full attention | 9,962,496 | 6 MiB | **1.75713** | 1 | 2.93008 | 3 | 3.12422 | — |

![Held-out loss for deeper and wider SNNs and full attention](../results/figures/event_lm_growth_9960k.png)

![enwik8 loss gap by prefix age for the 11-block SNN](../results/figures/event_lm_growth_9960k_prefix_age.png)

## 6.51M signed residual output · 81,920,000 training targets · seed 4603

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | PG-19 raw | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7-block SNN, ordinary output | 6,514,176 | 19,536 B | 1.84466 | 3 | **2.80932** | 1 | 2.91443 | 309,030 |
| 7-block SNN, correction output | 6,514,176 | 19,536 B | 1.84464 | 2 | 2.86794 | 2 | 2.99541 | 305,045 |
| 4-block full attention | 6,513,152 | 4 MiB | **1.75773** | 1 | 2.93346 | 3 | 3.12986 | — |

![Held-out loss with ordinary and correction output](../results/figures/event_lm_residual_output.png)

![Generated pulses with identical output counts within each pair](../results/figures/event_lm_residual_emissions.png)

## 13.03M residual growth and original-width controls · 72,089,600 training targets · seed 3307

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | PG-19 raw | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Wider residual SNN | 13,028,352 | 19,536 B | 1.87451 | 4 | **2.79672** | 1 | 2.88959 | 308,111 |
| Wider full-attention Transformer | 13,028,864 | 4 MiB | **1.77285** | 1 | 2.95990 | 3 | 3.16073 | — |
| Original-width held-charge SNN | 6,514,176 | 19,536 B | 1.87259 | 3 | 2.81227 | 2 | 2.90738 | 309,048 |
| Original-width full-attention Transformer | 6,513,152 | 4 MiB | 1.77292 | 2 | 2.95995 | 4 | 3.16079 | — |

## 6.51M spike circuit and charge comparisons · 72,089,600 training targets · seed 3307

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | PG-19 raw | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original-width held-charge SNN | 6,514,176 | 19,536 B | 1.87259 | 4 | **2.81227** | 1 | 2.90738 | 309,048 |
| Eight-bit router charge | 6,514,176 | 19,536 B | 1.86770 | 2 | 2.82082 | 5 | 2.92020 | 309,180 |
| Tag read competition, zero start | 6,514,180 | 19,536 B | 1.87077 | 3 | 2.81844 | 3 | 2.91860 | 309,523 |
| Tag read competition, active start | 6,514,180 | 19,536 B | 1.87409 | 6 | 2.81635 | 2 | 2.91291 | 309,599 |
| Tag reads with forecast-error ranking | 6,514,180 | 19,536 B | 1.87620 | 7 | 2.83214 | 6 | 2.93471 | 309,582 |
| Late steering | 6,514,204 | 19,536 B | 1.87323 | 5 | 2.82081 | 4 | 2.92053 | 308,536 |
| Original-width full-attention Transformer | 6,513,152 | 4 MiB | **1.77292** | 1 | 2.95995 | 7 | 3.16079 | — |

## 6.51M at 65.54M observed targets · 65,536,000 training targets · seed 2503

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | PG-19 raw | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Six-bit grown SNN | 6,514,176 | 19,536 B | 1.86525 | 2 | **2.81231** | 1 | 2.90664 | 308,230 |
| Grown full-attention Transformer | 6,513,152 | 4 MiB | **1.77728** | 1 | 2.98071 | 2 | 3.18604 | — |

## 8.74M expansion and original-width control · 55,705,600 training targets · seed 1801

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | PG-19 raw | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Expanded held-charge SNN | 8,739,840 | 26,080 B | 1.88812 | 2 | 2.83753 | 2 | 2.94075 | 309,457 |
| Expanded SNN with active steering | 8,739,840 | 26,080 B | 1.89312 | 4 | **2.82986** | 1 | 2.92630 | 308,645 |
| Grown full-attention Transformer | 8,740,352 | 4 MiB | **1.79731** | 1 | 3.01533 | 4 | 3.22392 | — |
| Original-width held-charge SNN | 6,514,176 | 19,536 B | 1.89057 | 3 | 2.84072 | 3 | 2.94263 | 309,545 |

## 6.52M redistribution of inhibition · 55,705,600 training targets · seed 1801

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | PG-19 raw | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original-width held-charge SNN | 6,514,176 | 19,536 B | 1.89057 | 2 | 2.84072 | 2 | 2.94263 | 309,545 |
| Redistribution of inhibition | 6,524,928 | 19,536 B | **1.88938** | 1 | **2.83881** | 1 | 2.94020 | 308,919 |

## 6.51M frozen teacher guidance and unguided control · 55,705,600 training targets · seed 1801

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | PG-19 raw | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original-width held-charge SNN | 6,514,176 | 19,536 B | **1.89057** | 1 | **2.84072** | 1 | 2.94263 | 309,545 |
| Held-charge SNN with frozen teacher | 6,514,176 | 19,536 B | 1.89537 | 2 | 2.84787 | 2 | 2.96148 | 308,918 |

## 6.51M delayed write confirmation · 55,705,600 training targets · seed 1801

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | PG-19 raw | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original-width held-charge SNN | 6,514,176 | 19,536 B | **1.89057** | 1 | 2.84072 | 2 | 2.94263 | 309,545 |
| Delayed write confirmation | 6,514,204 | 19,536 B | 1.89069 | 2 | **2.83683** | 1 | 2.93664 | 309,026 |

## 6.51M with more training data · 49,152,000 training targets · seed 1109

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | PG-19 raw | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Six-bit grown SNN | 6,514,176 | 19,536 B | 1.89440 | 2 | **2.82683** | 1 | 2.91901 | 309,688 |
| Grown full-attention Transformer | 6,513,152 | 4 MiB | **1.80629** | 1 | 3.03772 | 2 | 3.24647 | — |

## 6.51M growth · 512-byte continuations · 32,768,000 training targets · seed 711

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | PG-19 raw | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Grown SNN, finer charge | 6,514,176 | 19,536 B | 1.95110 | 2 | **2.85083** | 1 | 2.93341 | 309,505 |
| Grown SNN, six-bit charge | 6,514,176 | 19,536 B | 1.95646 | 3 | 2.86904 | 2 | 2.95290 | 306,923 |
| Grown full-attention Transformer | 6,513,152 | 4 MiB | **1.84266** | 1 | 3.06318 | 3 | 3.26549 | — |

## 6.51M trace coincidence · 32,768,000 training targets · seed 711

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | PG-19 raw | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Grown SNN, six-bit charge | 6,514,176 | 19,536 B | 1.95646 | 2 | **2.86904** | 1 | 2.95290 | 306,923 |
| Three-way coincidence | 6,514,176 | 19,536 B | 1.95911 | 3 | 2.87132 | 2 | 2.95721 | 307,297 |
| Grown full-attention Transformer | 6,513,152 | 4 MiB | **1.84266** | 1 | 3.06318 | 3 | 3.26549 | — |

## 3.26M · 512-byte continuations · 32,768,000 training targets · seed 711

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | PG-19 raw | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Held charge | 3,257,088 | 31,692 B | 1.96430 | 2 | **2.88668** | 1 | 2.98308 | 308,067 |
| Signed residual membrane | 3,257,088 | 22,224 B | 1.96628 | 3 | 2.89838 | 2 | 2.99574 | 309,629 |
| Four address phases | 3,257,088 | 25,584 B | 1.99725 | 4 | 2.90730 | 3 | 2.99722 | 311,699 |
| Four phases, corrected forecast loss | 3,257,088 | 25,584 B | 2.00335 | 5 | 2.90802 | 4 | 2.99318 | 310,105 |
| Full-attention Transformer | 3,256,832 | 4 MiB | **1.84280** | 1 | 3.06325 | 5 | 3.26568 | — |

## 3.26M · 2,048-byte continuations · 32,768,000 training targets · seed 711

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | PG-19 raw | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Held charge | 3,257,088 | 31,692 B | 1.97126 | 2 | **2.87693** | 1 | 2.96339 | 306,441 |
| Full-attention Transformer | 3,256,832 | 16 MiB | **1.85459** | 1 | 3.05282 | 2 | 3.24905 | — |

## 6.51M trained from scratch · 26,214,400 training targets · seed 0

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | PG-19 raw | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Wide held-charge SNN | 6,514,176 | 19,536 B | 2.06640 | 2 | **2.98527** | 1 | 3.09220 | 313,057 |
| Full-attention Transformer | 6,514,944 | 6 MiB | **2.03855** | 1 | 3.28561 | 2 | 3.57094 | — |

## 512-byte continuations · 32,768,000 training targets · seed 711

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | PG-19 raw | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Held-charge control | 1,630,992 | 21,320 B | 1.98564 | 4 | 2.98490 | 3 | 3.10338 | 307,730 |
| Opponent steering | 1,631,917 | 21,320 B | 1.98533 | 3 | **2.97309** | 1 | 3.09454 | 308,471 |
| Chronological write credit | 1,630,992 | 21,320 B | 1.98691 | 6 | 2.99097 | 5 | 3.10893 | 307,270 |
| Trace read veto | 1,631,012 | 21,320 B | 1.98405 | 2 | 2.98763 | 4 | 3.11449 | 307,412 |
| Spread coincidence | 1,631,012 | 21,320 B | 1.98644 | 5 | 2.98129 | 2 | 3.10337 | 306,624 |
| Full-attention Transformer | 1,630,848 | 3 MiB | **1.79625** | 1 | 3.11887 | 6 | 3.35801 | — |

## 2,048-byte continuations · 32,768,000 training targets · seed 711

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | PG-19 raw | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Held charge with write credit | 1,630,992 | 21,320 B | 1.98884 | 2 | **2.96180** | 1 | 3.07240 | 304,447 |
| Full-attention Transformer | 1,630,848 | 12 MiB | **1.81712** | 1 | 3.11395 | 2 | 3.33558 | — |

## 3.26M parameter comparison · 26,214,400 training targets · seed 0

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | PG-19 raw | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Seven-block held charge | 3,257,088 | 31,692 B | 1.99402 | 2 | **2.89822** | 1 | 2.98142 | 308,389 |
| Full-attention Transformer | 3,256,832 | 4 MiB | **1.86093** | 1 | 3.05929 | 2 | 3.25745 | — |

## Held-charge replication · 26,214,400 training targets per seed · seeds 0–2

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | PG-19 raw | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Held-charge SNN | 1,630,992 | 21,320 B | 2.00633 | 2 | **2.98979** | 1 | 3.10037 | 307,013 |
| Full-attention Transformer | 1,630,848 | 3 MiB | **1.82812** | 1 | 3.09559 | 2 | 3.29465 | — |

![Held-out loss after doubling residual width](../results/figures/event_lm_residual_growth_13030k.png)

![Held-out loss after changing the spike circuits at 6.51M parameters](../results/figures/event_lm_6510k_circuits.png)

![Held-out loss as enwik8 training exposure grows at 6.51M parameters](../results/figures/event_lm_6510k_data.png)

![enwik8 loss gap by prefix age after 65.54M targets](../results/figures/event_lm_65536_prefix_age.png)

![Held-out loss after expanding the delay populations](../results/figures/event_lm_expansion_8740k.png)

![Held-out next-byte loss for the grown 6.51M models](../results/figures/event_lm_grown_6510k.png)

![enwik8 loss gap for the grown pair by prefix age](../results/figures/event_lm_grown_prefix_age.png)

[Growth study](GROWN_RESIDUAL.md) · [Width from scratch](SCALE_6510K_WIDTH.md) · [Current model card](../models/spiking-byte-lm-v7/README.md) · [Depth growth study](DEPTH_GROWTH.md)

![Held-out next-byte loss for the 512-byte continuations](../results/figures/event_lm_continuation_test.png)

![Held-out loss at both trained context lengths](../results/figures/event_lm_trained_context.png)

[Continuation and scale source records](../results/event_lm/continuation_leaderboard.json) · [Held-charge replication](HELD_CHARGE_REPLICATION.md)

## Depth and activity inhibition · 26,214,400 training targets · seed 0

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|
| Three-block SNN, fixed threshold | 1,630,464 | 14,556 B | 2.24386 | 7 | 3.19752 | 7 | 305,600 |
| Three-block SNN, activity inhibition | 1,630,464 | 14,556 B | 2.16389 | 5 | 3.11971 | 4 | 304,525 |
| Five-block SNN, fixed threshold | 1,630,992 | 17,620 B | 2.19441 | 6 | 3.13879 | 5 | 307,566 |
| Five-block SNN, activity inhibition | 1,630,992 | 17,620 B | 2.07523 | 4 | 3.02885 | 2 | 303,469 |
| Five-block SNN, inhibition + held charge | 1,630,992 | 21,320 B | 2.00577 | 3 | **2.98022** | 1 | 308,178 |
| Local Transformer | 1,630,848 | 64,512 B | 1.89056 | 2 | 3.17881 | 6 | — |
| Full-attention Transformer | 1,630,848 | 3 MiB | **1.81449** | 1 | 3.11725 | 3 | — |

![Held-out loss for matched depth and inhibition models](../results/figures/depth_inhibition_test.png)

![Validation curves and residual spike counts](../results/figures/depth_inhibition.png)

[Evaluation records and ranks](../results/event_lm/depth_inhibition_leaderboard.json) · [Depth study and ablations](DEPTH_INHIBITION.md) · [Replication](DEPTH_INHIBITION.md#full-attention-replication) · [v2 model card](../models/spiking-byte-lm-v2/README.md)

## Earlier context-tag and threshold study

Each score below is a mean over three seeds. Each group has its own training exposure.

### Small comparison · 6,553,600 training targets per seed

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|
| Always-read spiking bank | 407,616 | 32,696 B | 2.39550 | 3 | 3.36437 | 3 | 298,515 |
| Forecast-keyed spiking bank | 407,616 | 32,696 B | 2.43130 | 6 | 3.37224 | 5 | 294,613 |
| Context tags, fixed thresholds | 407,616 | 32,696 B | 2.43025 | 5 | 3.36729 | 4 | 296,319 |
| Context tags, learned thresholds | 407,628 | 32,696 B | 2.42716 | 4 | 3.38277 | 6 | 303,374 |
| Local Transformer | 407,808 | 31,488 B | **2.24107** | 1 | 3.34604 | 2 | — |
| Full-attention Transformer | 407,808 | 1.125 MiB | 2.28026 | 2 | **3.29995** | 1 | — |

### Large comparison · 13,107,200 training targets per seed

| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | Pulses/byte |
|---|---:|---:|---:|---:|---:|---:|---:|
| Context tags, fixed thresholds | 815,232 | 32,696 B | 2.24015 | 3 | **3.25043** | 1 | 303,952 |
| Context tags, learned thresholds | 815,244 | 32,696 B | 2.25351 | 4 | 3.25233 | 2 | 303,493 |
| Local Transformer | 815,232 | 32,768 B | 2.04457 | 2 | 3.26187 | 3 | — |
| Full-attention Transformer | 815,232 | 1.5 MiB | **2.01841** | 1 | 3.29072 | 4 | — |

![Held-out loss by model size and corpus](../results/figures/event_lm_loss.png)

![Loss versus native spike activity for the spiking banks](../results/figures/event_lm_pulse_tradeoff.png)

[Per-seed scores and source paths](../results/event_lm/threshold_scale_leaderboard.json) · [Interventions and research notes](SPIKE_CODE_FORECAST.md#learned-context-keys-and-firing-thresholds) · [Model and reproduction guide](../models/spiking-byte-lm/README.md)

Regenerate this page and its figures with `python experiments/plot_event_lm_study.py`.
