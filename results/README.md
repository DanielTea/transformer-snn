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

Regenerate the [leaderboards and current figures](../docs/LEADERBOARD.md)
with `python experiments/plot_event_lm_study.py` and
`python experiments/plot_depth_inhibition.py`.

Paths in result records identify the input corpus or excerpt. Supply
your local paths at run time. The [reproduction guide](../docs/REPRODUCE_DENDRITE.md)
shows how to recreate the datasets, checkpoints and charts.
