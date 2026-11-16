# Experiment records

| Directory | Contents |
|---|---|
| [recall](recall) | ShuntFormer MQAR archive and successive Dendrite recall variants |
| [recall_distance](recall_distance) | Typed variable-gap recall, RoPE and fixed-delay controls, anchored probe |
| [lm](lm) | enwik8 training curves and final measurements |
| [lm_context](lm_context) | Paired history-length and memory ablation probe |
| [pulse_recall](pulse_recall) | Learned paired-pulse recall, destructive reads and pre-query state clearing |
| [pulse_lm](pulse_lm) | Paired enwik8 long-history scores and bank event counts |
| [pulse_iteration](pulse_iteration) | Matched write-protection and timing runs on held-out enwik8, with old-phase interventions |
| [pulse_global](pulse_global) | Paired enwik8 evaluation of learned thresholds shared by all queries to a cell |
| [pulse_credit](pulse_credit) | Single-write interventions and paired-credit training controls |
| [charge_text](charge_text) | PG-19 and paired enwik8 scores for conserved and independent charge writes |
| [event_lm](event_lm) | Integer spike decoder training, routing interventions, warmed enwik8 and PG-19 scores, and export verification |
| [lm_successor](lm_successor) | Calibration trials, continuous enwik8 and PG-19 scores, excerpt manifest |
| [lm_scaling](lm_scaling) | Matched three-seed scaling evaluation, decode timing and a 16-book PG-19 excerpt manifest |
| [vision](vision) | Historical CIFAR-10 runs |
| [figures](figures) | Plots generated from these records |

The recall figure uses `recall_distance/*_gap_v1_*.json` and
`recall_distance/dendrite_event_s0_anchored_extrapolation.json`.
The successor loss figure uses `lm_successor/scaled_v2_continuous.json`
and `lm_successor/end_to_end_continuous.json`.
The calibration figure uses the three `successor_s*_scaled_v2.json`
runs and the three `successor_s*_verified_v1.json` runs.
Pilot and ablation runs use their own tags.
The scaling figure uses `lm_scaling/seed0.json`, `seed1.json` and
`seed2.json`. The corresponding training runs in `lm/` use the
`scale3m` tag.
The temporal enwik8 figure uses `lm/*_temporal_3m.json`. The matched
longer-budget runs use the `temporal_12m` tag.

The [spike correction study](../docs/SPIKE_CORRECTION.md) uses the
`equal_lookback`, `correction_v1`, and `correction_32k` training tags in
`event_lm/`, with separate `_2x` tags for the longer runs.
The `enwik8_correction_*` and
`pg19_correction_*` records contain per-corpus and per-seed scores.
The `correction_pg19_*_manifest.json` files pin the scored books.
The [count sweep](../docs/SPIKE_CORRECTION.md#spike-count-scaling) uses
`count*_memory_pilot` and `count16_correction_2x` training tags.
Its `spike_resolution_*.json` records contain pulse counts, integer
parity checks, and inference-only scores.
The [prediction error experiment](../docs/PREDICTION_ERROR_MEMORY.md)
uses the `error_long` training tags and evaluates all 16 complete
test books.
The [direct spike forecast experiment](../docs/SPIKE_CODE_FORECAST.md) uses
the `forecast_pilot` tags with seeds 100–102. Its records include
enwik8, PG-19 excerpts, complete books, and the native activity audit.
The [forecast-key comparison](../docs/SPIKE_CODE_FORECAST.md#forecast-keyed-reads)
uses the `forecast_key` tags for the matched narrow banks and for the
shuffled-key and read-age interventions.
The [context tags and firing thresholds comparison](../docs/SPIKE_CODE_FORECAST.md#learned-context-keys-and-firing-thresholds)
uses the `threshold_scale` tags for two matched model sizes, a paired
plain router test, and the local and full-attention Transformer scores.
The [depth and inhibition study](../docs/DEPTH_INHIBITION.md) uses
`event_lm/depth_inhibition` and the matched controls in
`event_lm/scale_1630k`. The `depth_inhibition_replication` directory
holds the additional seeds, and `delay_filter` holds the paired
warm-start experiment. Their checkpoint catalogs match the paths in
the [v2 release manifest](../models/spiking-byte-lm-v2/manifest.json).

The [research handoff](../docs/HANDOFF.md) links the native audits,
full-corpus scores, and exact training records.

| Study | Files |
|---|---|
| [Grown SNN and Transformer](event_lm/grown_6510k_residual) | Warm-start parity, training curves, per-book loss, added-neuron intervention |
| [Finer-charge reference](event_lm/grown_fine_charge) | Exact training record, both held-out scores, native audit and ablation |
| [From-scratch width study](event_lm/scale_6510k_width) | The paired 6.51M training and evaluation records |
| [3.26M context study](event_lm/scale_3260k_context) | Both checkpoints at 512 and 2,048 training bytes per window |
| [49.15M-target continuation](event_lm/scale_6510k_data) | Paired scores, native activity, and enwik8 validation memory interventions |
| [65.54M-target continuation](event_lm/scale_6510k_next_data) | Both parents for the next width study, their held-out scores, and the SNN's native audit and memory interventions |
| [Three-way coincidence](event_lm/trace_coincidence) | Warm-start parity, full held-out scores and routing ablation |
| [8.74M expansion](event_lm/expansion_8740k) | Warm-start checks and four matched arms, including the active-steering rerun |
| [Delayed write confirmation](event_lm/trace_write_confirm) | Warm checks, both held-out scores, write ages and the learned-interval ablation |
| [Redistribution of inhibition](event_lm/inhibition_redistribution) | Warm-start checks, both held-out corpora, native transfer pulses and the zero-transfer intervention |
| [Frozen teacher guidance](event_lm/teacher_guidance) | Student training metadata, teacher digest, held-out scores, and native audits |
| [13.03M residual growth](event_lm/residual_growth_13030k) | Warm starts, matched held-out results, original-width control and added-neuron intervention |
| [6.51M Transformer at 72.09M targets](event_lm/scale_6510k_short_control) | Continuation control for the width and circuit comparisons |
| [Tag read competition](event_lm/tag_read_competition) | Zero-start, active-start and ranked-read training arms, gain sweep on fixed weights, and native integer audits |
| [Late steering](event_lm/late_steering) | Warm identity, held-out scores, native feedback pulses, and zero-gain intervention |
| [Finer router charge at fixed width](event_lm/finer_router_charge) | Warm-start parity, full-corpus scores, eight-bit native pulse audit, and six-bit intervention |
| [Depth growth with signed residual corrections](event_lm/depth_growth) | Six matched endpoints at 81.92M targets, both held-out corpora, warm starts, native parity, interventions and CPU timing |
| [Held-charge inputs and group thresholds](event_lm/trace_integration) | Five matched endpoints at 91.75M targets, paired history and axon tap probes, fine-threshold offsets, and native parity |
| [Axon tap reads](event_lm/trace_tap_read) | Four matched endpoints at 101.58M targets, warm checks, full-corpus scores, per-group winning delays, frozen-weight interventions, and packed-width parity |
| [Packed width and steering transfers](event_lm/packed_width) | Matched 9.96M and 12.8M scores, paired history and routing probes, integer warm starts, CPU timing and generation samples |
| [Residual width on expanded training data](event_lm/residual_scale) | Matched width and corpus controls, full held-out loss, paired prediction gaps, residual ablations and native integer audits |
| [Surprise clock and fast synapses](event_lm/surprise_clock) | Frozen-decoder surprise measurements, six matched continuations, fitted fast synapses, clock and synapse interventions, paired history gaps, warm starts and native integer audits |
| [Training mixture](event_lm/training_mixture) | Pinned sources, prepared corpus inventory, exact sampled-window receipt and PG-19 overlap audit |
| [Fixed-arrival coincidence](event_lm/trace_arrival) | Four continuations, full enwik8 and PG-19, paired early predictions, muted projections and native integer audits |
| [Byte transfer](event_lm/byte_transfer) | Prepared-window catalogue, source hashes, overlap flags and matched scores on four text corpora and both LAMBADA prompts |
| [Held-charge reads driven by write spikes](event_lm/trace_read_drive) | Matched held-out scores, fractional-charge and pulse audits, steering and polarity interventions, and 25.6M growth checks |

`event_lm/continuation_leaderboard.json` groups ranks by training
exposure, geometry, and seed. The v11 archive contains the eight
surprise-study endpoints and their two v10 parents. The v10 archive
contains the five width-study endpoints and their two v9 parents. The v9 archive holds
the four arrival endpoints and their two v8 parents. The v8 archive uses the packed-width,
driven-read and supporting checkpoint catalogs. The v7 archive uses
`event_lm/v7_checkpoints.sha256` and `event_lm/trace_integration_checkpoints.sha256`.
The v6 archive uses `event_lm/v6_checkpoints.sha256` for its studies and
their parent chains.
[v11 archive manifest](../models/spiking-byte-lm-v11/manifest.json) ·
[Surprise-study checkpoint digests](event_lm/surprise_clock_checkpoints.sha256) ·
[v10 archive manifest](../models/spiking-byte-lm-v10/manifest.json) ·
[Width-study checkpoint digests](event_lm/residual_scale_checkpoints.sha256) ·
[v9 archive manifest](../models/spiking-byte-lm-v9/manifest.json) ·
[Arrival checkpoint digests](event_lm/trace_arrival_checkpoints.sha256) ·
[Transfer protocol](../docs/BYTE_TRANSFER.md) ·
[v8 archive manifest](../models/spiking-byte-lm-v8/manifest.json) ·
[v7 archive manifest](../models/spiking-byte-lm-v7/manifest.json) ·
[v6 archive manifest](../models/spiking-byte-lm-v6/manifest.json) ·
[Packed-width checkpoint digests](event_lm/packed_width_checkpoints.sha256) ·
[Driven read checkpoint digests](event_lm/trace_read_drive_checkpoints.sha256) ·
[v8 supporting checkpoint digests](event_lm/v8_supporting_checkpoints.sha256)

Regenerate the [leaderboards and current figures](../docs/LEADERBOARD.md)
with `python experiments/plot_event_lm_study.py` and
`python experiments/plot_depth_inhibition.py`.

Paths in result records identify the input corpus or excerpt. Supply
your local paths at run time. The [reproduction guide](../docs/REPRODUCE_DENDRITE.md)
shows how to recreate the datasets, checkpoints and charts.
