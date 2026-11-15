# Experiment records

| Directory | Contents |
|---|---|
| [recall](recall) | ShuntFormer MQAR archive and successive Dendrite recall variants |
| [recall_distance](recall_distance) | Typed variable-gap recall, RoPE and fixed-delay controls, anchored probe |
| [lm](lm) | enwik8 training curves and final measurements |
| [lm_context](lm_context) | Paired history-length and memory ablation probe |
| [lm_successor](lm_successor) | Calibration trials, continuous enwik8 and PG-19 scores, excerpt manifest |
| [vision](vision) | Historical CIFAR-10 runs |
| [figures](figures) | Plots generated from these records |

The recall figure uses `recall_distance/*_gap_v1_*.json` and
`recall_distance/dendrite_event_s0_anchored_extrapolation.json`.
The successor loss figure uses `lm_successor/scaled_v2_continuous.json`
and `lm_successor/end_to_end_continuous.json`.
The calibration figure uses the three `successor_s*_scaled_v2.json`
runs and the three `successor_s*_verified_v1.json` runs.
Pilot and ablation runs use their own tags.

Paths in result records identify the input corpus or excerpt. Supply
your local paths at run time. The [reproduction guide](../docs/REPRODUCE_DENDRITE.md)
shows how to recreate the datasets, checkpoints and charts.
