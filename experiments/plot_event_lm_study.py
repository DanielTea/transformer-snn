"""Regenerate the matched spiking byte model leaderboards and figures."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

from package_event_lm import residual_scale_mixture
from plot_byte_transfer import plot_transfer, read_transfer, transfer_tables
from plot_dendrite import BLUE, GRAY, MUTED, ORANGE, RESULTS, TEXT, VIOLET, heading, plt, read_result, save, style


ROOT = RESULTS.parent
LABELS = {
    "always": ("Always-read spiking bank", "Always read"),
    "forecast": ("Forecast-keyed spiking bank", "Forecast key"),
    "context": ("Context tags, fixed thresholds", "Fixed thresholds"),
    "learned": ("Context tags, learned thresholds", "Learned thresholds"),
    "transformer_local": ("Local Transformer", "Local Transformer"),
    "full": ("Full-attention Transformer", "Full attention"),
}
ARMS = {
    "small": ("always", "forecast", "context", "learned", "transformer_local", "full"),
    "large": ("context", "learned", "transformer_local", "full"),
}
COLORS = {
    "always": "#138372", "forecast": "#7a6a50", "context": BLUE, "learned": VIOLET,
    "transformer_local": ORANGE, "full": GRAY,
}
CORPORA = (("enwik8", "enwik8", 4_999_999), ("pg19_full", "16 complete PG-19 books", 8_404_368))
READ_ME_START = "<!-- event-lm-leaderboard:start -->"
READ_ME_END = "<!-- event-lm-leaderboard:end -->"
DEPTH = "event_lm/depth_inhibition"
WIDTH = "event_lm/scale_1630k"
OPPONENT = "event_lm/opponent_steering"
CREDIT = "event_lm/trace_write_credit"
VETO = "event_lm/trace_read_veto"
SPREAD = "event_lm/spread_coincidence"
LONG = "event_lm/long_context_continuation"
SCALE = "event_lm/scale_3260k"
SCALE_CONTEXT = "event_lm/scale_3260k_context"
GROWN = "event_lm/grown_6510k_residual"
WIDTH_6510K = "event_lm/scale_6510k_width"
FINE_CHARGE = "event_lm/grown_fine_charge"
SCALE_DATA = "event_lm/scale_6510k_data"
SCALE_NEXT_DATA = "event_lm/scale_6510k_next_data"
SCALE_SHORT_CONTROL = "event_lm/scale_6510k_short_control"
RESIDUAL_GROWTH = "event_lm/residual_growth_13030k"
TAG_READS = "event_lm/tag_read_competition"
LATE_STEERING = "event_lm/late_steering"
FINER_ROUTER = "event_lm/finer_router_charge"
DEPTH_GROWTH = "event_lm/depth_growth"
TRACE_INTEGRATION = "event_lm/trace_integration"
TRACE_TAP_READ = "event_lm/trace_tap_read"
PACKED_WIDTH = "event_lm/packed_width"
TRACE_READ_DRIVE = "event_lm/trace_read_drive"
TRACE_ARRIVAL = "event_lm/trace_arrival"
ARRIVAL_GROUP = "trace_arrival_121241k"
ARRIVAL_TRANSFER = "event_lm/byte_transfer/trace_arrival_s9349.json"
RESIDUAL_SCALE = "event_lm/residual_scale"
RESIDUAL_SCALE_GROUP = "residual_scale_131072k"
RESIDUAL_SCALE_CHECKPOINTS = "event_lm/residual_scale_checkpoints.sha256"
SURPRISE_CLOCK = "event_lm/surprise_clock"
SURPRISE_CLOCK_GROUP = "surprise_clock_140902k"
SURPRISE_CLOCK_CHECKPOINTS = "event_lm/surprise_clock_checkpoints.sha256"
COASTING_HEADS = "event_lm/coasting_heads"
COASTING_HEADS_GROUP = "coasting_heads_368640k"
COASTING_HEADS_CHECKPOINTS = "event_lm/coasting_heads_checkpoints.sha256"
STATE_READS = "event_lm/state_reads"
STATE_READS_CHECKPOINTS = "event_lm/state_reads_checkpoints.sha256"
READ_CLOCKS = "event_lm/read_clocks"
READ_CLOCKS_CHECKPOINTS = "event_lm/read_clocks_checkpoints.sha256"
STREAM_FORM = "event_lm/stream_form"
STREAM_FORM_CHECKPOINTS = "event_lm/stream_form_checkpoints.sha256"
TRACE_COINCIDENCE = "event_lm/trace_coincidence"
WRITE_CONFIRM = "event_lm/trace_write_confirm"
EXPANSION = "event_lm/expansion_8740k"
INHIBITION = "event_lm/inhibition_redistribution"
GUIDANCE = "event_lm/teacher_guidance"
RESIDUAL_MEMBRANE = "event_lm/residual_membrane"
TAG_PHASE = "event_lm/tag_phase_address"
CORRECTED = "event_lm/corrected_forecast"
DEPTH_ARMS = (
    ("Three-block SNN, fixed threshold", WIDTH, "fixed", f"{DEPTH}/native_wide_fixed.json", GRAY),
    ("Three-block SNN, activity inhibition", DEPTH, "wide_inhibited", f"{DEPTH}/native_wide_inhibited.json", BLUE),
    ("Five-block SNN, fixed threshold", DEPTH, "deep_fixed", f"{DEPTH}/native_deep_fixed.json", GRAY),
    ("Five-block SNN, activity inhibition", DEPTH, "deep_inhibited", f"{DEPTH}/native_deep_inhibited.json", BLUE),
    ("Local Transformer", WIDTH, "transformer_local", None, ORANGE),
    ("Full-attention Transformer", WIDTH, "transformer", None, ORANGE),
)
TRACE = ("Five-block SNN, inhibition + held charge", DEPTH, "deep_trace", f"{DEPTH}/native_deep_trace.json", VIOLET)
CONTINUATION_512 = (
    ("Held-charge control", OPPONENT, "control_s711", f"{OPPONENT}/native_control_s711.json", GRAY),
    ("Opponent steering", OPPONENT, "opponent_s711", f"{OPPONENT}/native_opponent_s711.json", VIOLET),
    ("Chronological write credit", CREDIT, "write_credit_s711", f"{CREDIT}/native_write_credit_s711.json", BLUE),
    ("Trace read veto", VETO, "read_veto_s711", f"{VETO}/native_read_veto_s711.json", VIOLET),
    ("Spread coincidence", SPREAD, "spread_s711", f"{SPREAD}/native_spread_s711.json", BLUE),
    ("Full-attention Transformer", CREDIT, "transformer_s711", None, ORANGE),
)
CONTINUATION_2048 = (
    ("Held charge with write credit", LONG, "trace_ctx2048_s711", f"{LONG}/native_trace_ctx2048_s711.json", BLUE),
    ("Full-attention Transformer", LONG, "transformer_ctx2048_s711", None, ORANGE),
)
SCALE_3260K = (
    ("Seven-block held charge", SCALE, "trace_s0", f"{SCALE}/native_trace_s0.json", BLUE),
    ("Full-attention Transformer", SCALE, "transformer_s0", None, ORANGE),
)
SCALE_CONTEXT_512 = (
    ("Held charge", SCALE_CONTEXT, "trace_ctx512_s711", f"{SCALE_CONTEXT}/native_trace_ctx512_s711.json", BLUE),
    ("Signed residual membrane", RESIDUAL_MEMBRANE, "membrane_s711", f"{RESIDUAL_MEMBRANE}/native_membrane_s711.json", VIOLET),
    ("Four address phases", TAG_PHASE, "tag_phase_s711", f"{TAG_PHASE}/native_tag_phase_s711.json", VIOLET),
    ("Four phases, corrected forecast loss", CORRECTED, "corrected_s711", f"{CORRECTED}/native_corrected_s711.json", VIOLET),
    ("Full-attention Transformer", SCALE_CONTEXT, "transformer_ctx512_s711", None, ORANGE),
)
SCALE_CONTEXT_2048 = (
    ("Held charge", SCALE_CONTEXT, "trace_ctx2048_s711", f"{SCALE_CONTEXT}/native_trace_ctx2048_s711.json", BLUE),
    ("Full-attention Transformer", SCALE_CONTEXT, "transformer_ctx2048_s711", None, ORANGE),
)
GROWN_6510K = (
    ("Grown SNN, finer charge", FINE_CHARGE, "fine_s711", f"{FINE_CHARGE}/native_fine_s711.json", VIOLET),
    ("Grown SNN, six-bit charge", GROWN, "grown_s711", f"{GROWN}/native_grown_s711.json", BLUE),
    ("Grown full-attention Transformer", GROWN, "transformer_s711", None, ORANGE),
)
DATA_6510K = (
    ("Six-bit grown SNN", SCALE_DATA, "trace_s1109", f"{SCALE_DATA}/native_trace_s1109.json", BLUE),
    ("Grown full-attention Transformer", SCALE_DATA, "transformer_s1109", None, ORANGE),
)
NEXT_DATA_6510K = (
    ("Six-bit grown SNN", SCALE_NEXT_DATA, "trace_s2503", f"{SCALE_NEXT_DATA}/native_trace_s2503.json", BLUE),
    ("Grown full-attention Transformer", SCALE_NEXT_DATA, "transformer_s2503", None, ORANGE),
)
ORIGINAL_WIDTH_SNN = ("Original-width held-charge SNN", RESIDUAL_GROWTH, "control_s3307",
                      f"{RESIDUAL_GROWTH}/native_control_s3307.json", BLUE)
ORIGINAL_WIDTH_TRANSFORMER = ("Original-width full-attention Transformer", SCALE_SHORT_CONTROL,
                              "control_transformer_s3307", None, ORANGE)
RESIDUAL_GROWTH_13030K = (
    ("Wider residual SNN", RESIDUAL_GROWTH, "residual_s3307", f"{RESIDUAL_GROWTH}/native_residual_s3307.json", VIOLET),
    ("Wider full-attention Transformer", RESIDUAL_GROWTH, "transformer_s3307", None, ORANGE),
    ORIGINAL_WIDTH_SNN,
    ORIGINAL_WIDTH_TRANSFORMER,
)
CIRCUITS_6510K = (
    ORIGINAL_WIDTH_SNN,
    ("Eight-bit router charge", FINER_ROUTER, "fine_charge_s3307", f"{FINER_ROUTER}/native_fine_charge_s3307.json", "#138372"),
    ("Tag read competition, zero start", TAG_READS, "tag_reads_s3307", f"{TAG_READS}/native_tag_reads_s3307.json", GRAY),
    ("Tag read competition, active start", TAG_READS, "tag_active_s3307", f"{TAG_READS}/native_tag_active_s3307.json", BLUE),
    ("Tag reads with forecast-error ranking", TAG_READS, "tag_supervised_s3307",
     f"{TAG_READS}/native_tag_supervised_s3307.json", VIOLET),
    ("Late steering", LATE_STEERING, "late_s3307", f"{LATE_STEERING}/native_late_s3307.json", VIOLET),
    ORIGINAL_WIDTH_TRANSFORMER,
)
GROWTH_81920K = (
    ("11-block SNN", DEPTH_GROWTH, "depth_s4603", f"{DEPTH_GROWTH}/native_depth_s4603.json", VIOLET),
    ("7-block wider SNN", DEPTH_GROWTH, "wide_s4603", f"{DEPTH_GROWTH}/native_wide_s4603.json", BLUE),
    ("6-block full attention", DEPTH_GROWTH, "transformer_s4603", None, ORANGE),
)
RESIDUAL_OUTPUT_81920K = (
    ("7-block SNN, ordinary output", DEPTH_GROWTH, "control_s4603",
     f"{DEPTH_GROWTH}/native_control_s4603.json", BLUE),
    ("7-block SNN, correction output", DEPTH_GROWTH, "corrections_s4603",
     f"{DEPTH_GROWTH}/native_corrections_s4603.json", VIOLET),
    ("4-block full attention", DEPTH_GROWTH, "control_transformer_s4603", None, ORANGE),
)
TRACE_INTEGRATION_91750K = (
    ("Continued v6 SNN", TRACE_INTEGRATION, "control_s5801", f"{TRACE_INTEGRATION}/native_control_s5801.json", BLUE),
    ("Learned trace input spikes", TRACE_INTEGRATION, "transport_s5801", f"{TRACE_INTEGRATION}/native_transport_s5801.json", VIOLET),
    ("Coarse group thresholds", TRACE_INTEGRATION, "thresholds_s5801",
     f"{TRACE_INTEGRATION}/native_thresholds_s5801.json", "#138372"),
    ("One-tick group thresholds", TRACE_INTEGRATION, "fine_thresholds_s5801",
     f"{TRACE_INTEGRATION}/native_fine_thresholds_s5801.json", "#ad4772"),
    ("6-block full attention", TRACE_INTEGRATION, "transformer_s5801", None, ORANGE),
)
TRACE_TAP_101580K = (
    ("Continued SNN", TRACE_TAP_READ, "control_s6899", f"{TRACE_TAP_READ}/native_control_s6899.json", BLUE),
    ("Shortest eligible tap", TRACE_TAP_READ, "fixed_s6899", f"{TRACE_TAP_READ}/native_fixed_s6899.json", "#138372"),
    ("Held spike race", TRACE_TAP_READ, "race_s6899", f"{TRACE_TAP_READ}/native_race_s6899.json", VIOLET),
    ("6-block full attention", TRACE_TAP_READ, "transformer_s6899", None, ORANGE),
)
PACKED_WIDTH_111411K = (
    ("187-channel SNN", PACKED_WIDTH, "control_s7907", f"{PACKED_WIDTH}/native_control_s7907.json", BLUE),
    ("243-channel SNN", PACKED_WIDTH, "wide_s7907", f"{PACKED_WIDTH}/native_wide_s7907.json", VIOLET),
    ("Original full attention", PACKED_WIDTH, "control_transformer_s7907", None, ORANGE),
    ("Wider full attention", PACKED_WIDTH, "transformer_s7907", None, "#a34073"),
    ("Delayed steering transfers", PACKED_WIDTH, "delayed_transfer_s7907",
     f"{PACKED_WIDTH}/native_delayed_transfer_s7907.json", "#138372"),
    ("Growth in shortest groups", PACKED_WIDTH, "short_groups_s7907",
     f"{PACKED_WIDTH}/native_short_groups_s7907.json", "#7a6a50"),
    ("Present steering transfers", PACKED_WIDTH, "present_transfer_s7907",
     f"{PACKED_WIDTH}/native_present_transfer_s7907.json", "#a34073"),
)
TRACE_ARRIVAL_121241K = (
    ("Ordinary SNN", TRACE_ARRIVAL, "control_s9349", f"{TRACE_ARRIVAL}/native_control_s9349.json", GRAY),
    ("Selected arrivals", TRACE_ARRIVAL, "selected_s9349", f"{TRACE_ARRIVAL}/native_selected_s9349.json", BLUE),
    ("Unmatched arrivals", TRACE_ARRIVAL, "unmatched_s9349", f"{TRACE_ARRIVAL}/native_unmatched_s9349.json", VIOLET),
    ("Full attention", TRACE_ARRIVAL, "transformer_s9349", None, ORANGE),
)
RESIDUAL_SCALE_ARMS = (
    ("14.07M SNN", RESIDUAL_SCALE, "snn_s9631", f"{RESIDUAL_SCALE}/native_snn_s9631.json", BLUE),
    ("25.79M SNN", RESIDUAL_SCALE, "wide-snn_s9631", f"{RESIDUAL_SCALE}/native_wide-snn_s9631.json", VIOLET),
    ("25.79M SNN, enwik8 only", RESIDUAL_SCALE, "enwik8-snn_s9631",
     f"{RESIDUAL_SCALE}/native_enwik8-snn_s9631.json", "#187b62"),
    ("14.07M full attention", RESIDUAL_SCALE, "transformer_s9631", None, GRAY),
    ("25.79M full attention", RESIDUAL_SCALE, "wide-transformer_s9631", None, ORANGE),
)
SURPRISE_CLOCK_ARMS = (
    ("25.79M SNN", SURPRISE_CLOCK, "control_s10007", f"{SURPRISE_CLOCK}/native_control_s10007.json", BLUE),
    ("25.79M SNN with fast synapses", SURPRISE_CLOCK, "control-synapses_s10007",
     f"{SURPRISE_CLOCK}/native_control-synapses_s10007.json", "#187b62"),
    ("34.85M SNN, every byte", SURPRISE_CLOCK, "depth_s10007", f"{SURPRISE_CLOCK}/native_depth_s10007.json",
     "#7a6a50"),
    ("34.85M SNN, fixed stride", SURPRISE_CLOCK, "stride_s10007", f"{SURPRISE_CLOCK}/native_stride_s10007.json",
     "#9a6fb0"),
    ("34.85M SNN, surprise clock", SURPRISE_CLOCK, "clock_s10007", f"{SURPRISE_CLOCK}/native_clock_s10007.json",
     VIOLET),
    ("34.85M SNN, surprise clock and fast synapses", SURPRISE_CLOCK, "clock-synapses_s10007",
     f"{SURPRISE_CLOCK}/native_clock-synapses_s10007.json", "#c2410c"),
    ("25.79M full attention", SURPRISE_CLOCK, "control-transformer_s10007", None, GRAY),
    ("34.85M full attention", SURPRISE_CLOCK, "transformer_s10007", None, ORANGE),
)
COASTING_HEADS_ARMS = (
    ("34.85M SNN, surprise clock", f"{COASTING_HEADS}/scratch-study", "clock_s20001",
     f"{COASTING_HEADS}/scratch-study/native_clock_s20001.json", VIOLET),
    ("34.85M SNN, surprise clock and fast synapses", f"{COASTING_HEADS}/scratch-study", "clock-syn_s20001",
     f"{COASTING_HEADS}/scratch-study/native_clock-syn_s20001.json", "#c2410c"),
    ("34.85M SNN, surprise clock and read heads", f"{COASTING_HEADS}/v12-scratch", "heads_s20001",
     f"{COASTING_HEADS}/v12-scratch/native_heads_s20001.json", "#187b62"),
    ("34.85M full attention", f"{COASTING_HEADS}/scratch-study", "transformer_s20001", None, ORANGE),
)
# Model, optimizer, enwik8 record, PG-19 record and checkpoint of every row scored in warmed windows.
STATE_READS_ARMS = (
    ("34.82M SNN, spike reads and race cells", "Rotated, averaged", "test_spike_reads_and_cells",
     "pg19_spike_reads_and_cells", "checkpoints/state_reads/pilot-race64-d8-c350-dl4/model.pt.averaged", "#187b62"),
    ("34.65M SNN, spike reads", "Rotated, averaged", "test_spike_reads", "pg19_rotated",
     "checkpoints/state_reads/full-rate-rot/model.pt.averaged", VIOLET),
    ("34.65M hybrid, soft reads", "AdamW", "test_soft_reads", None,
     "checkpoints/state_reads/full-soft-d13w256/model.pt", "#e0a000"),
    ("34.85M SNN, surprise clock and read heads", "AdamW", "test_v12_decoders", None,
     "checkpoints/v12-scratch/heads_s20001.pt", GRAY),
    ("34.85M SNN, surprise clock", "AdamW", "test_v12_decoders", None,
     "checkpoints/scratch-study/clock_s20001.pt", GRAY),
    ("34.85M full attention", "AdamW", "test_v12_decoders", "pg19_spike_reads_and_cells",
     "checkpoints/scratch-study/transformer_s20001.pt", ORANGE),
    ("34.85M full attention", "Rotated, averaged", "test_transformer_rotated", "pg19_rotated",
     "checkpoints/state_reads/full-transformer-rot/model.pt.averaged", ORANGE),
)
# The same for the read-clock study. A record with a folder is one of an earlier study.
READ_CLOCKS_ARMS = (
    ("34.82M SNN, clock cells and address rate", "Rotated, averaged", "test_clock_cells",
     "pg19_clock_cells_and_address_rate", "checkpoints/read_clocks/tag8-a8/model.pt.averaged", VIOLET),
    ("34.82M SNN, address rate", "Rotated, averaged", "test_address_rate", "pg19_clock_cells_and_address_rate",
     "checkpoints/read_clocks/rate8/model.pt.averaged", "#e0559b"),
    ("34.82M SNN, spike reads and race cells", "Rotated, averaged", f"{STATE_READS}/scores/test_spike_reads_and_cells",
     f"{STATE_READS}/scores/pg19_spike_reads_and_cells",
     "checkpoints/state_reads/pilot-race64-d8-c350-dl4/model.pt.averaged", "#4a5568"),
    ("34.85M full attention, 256 wide", "AdamW", f"{STATE_READS}/scores/test_v12_decoders",
     f"{STATE_READS}/scores/pg19_spike_reads_and_cells", "checkpoints/scratch-study/transformer_s20001.pt", ORANGE),
    ("34.85M full attention, 256 wide", "Rotated, averaged", f"{STATE_READS}/scores/test_transformer_rotated",
     f"{STATE_READS}/scores/pg19_rotated", "checkpoints/state_reads/full-transformer-rot/model.pt.averaged", ORANGE),
    ("34.75M full attention, standard shape", "AdamW, averaged", "test_transformer_standard",
     "pg19_transformer_standard", "checkpoints/read_clocks/transformer-std/model.pt.averaged", ORANGE),
)
# The stream-form study: model, how it reads the test, its enwik8 record, the rate of a Transformer's fast output
# synapses where the record holds several, its PG-19 record, checkpoint and colour.
STREAM_FORM_STACK, STREAM_FORM_CONTROL, STREAM_FORM_REFERENCE = (
    f"checkpoints/stream_form/{folder}/model.pt.averaged" for folder in ("long-windows", "windows", "transformer"))
STREAM_FORM_ARMS = (
    ("34.81M SNN stack, long windows", "Stream, longer reach, taught output synapses", "test_long_windows_taught",
     None, "pg19_long_windows_taught", STREAM_FORM_STACK, VIOLET),
    ("34.81M SNN stack, long windows", "Stream, longer reach", "test_long_windows_reach", None,
     "pg19_long_windows_reach", STREAM_FORM_STACK, VIOLET),
    ("34.81M SNN stack, long windows", "Stream", "test_long_windows_stream", None, "pg19_long_windows_stream",
     STREAM_FORM_STACK, VIOLET),
    ("34.81M SNN stack, windows of 512 bytes", "Stream, longer reach, taught output synapses", "test_windows_taught",
     None, None, STREAM_FORM_CONTROL, "#138372"),
    ("34.81M SNN stack, windows of 512 bytes", "Stream, longer reach", "test_windows_reach", None,
     "pg19_windows_reach", STREAM_FORM_CONTROL, "#138372"),
    ("34.81M SNN stack, windows of 512 bytes", "Stream", "test_windows_stream", None, "pg19_windows_stream",
     STREAM_FORM_CONTROL, "#138372"),
    ("34.75M full attention, standard shape", "Windows with 496 bytes of history, taught output synapses",
     "test_transformer_taught_history496", 0.07, None, STREAM_FORM_REFERENCE, ORANGE),
    ("34.75M full attention, standard shape", "Windows with 496 bytes of history", "test_transformer_history496",
     None, "pg19_transformer_history496", STREAM_FORM_REFERENCE, ORANGE),
    ("34.75M full attention, standard shape", "Windows with 448 bytes of history", "test_transformer_history448",
     None, "pg19_transformer_history448", STREAM_FORM_REFERENCE, ORANGE),
    ("34.75M full attention, standard shape", "Windows with 256 bytes of history", "test_transformer_history256",
     None, None, STREAM_FORM_REFERENCE, ORANGE),
)
# The long-streams study (v16): the same rows at equal training context of 2,048 bytes, from its public records.
LONG_STREAMS = "event_lm/long_streams"
LONG_STREAMS_CHECKPOINTS = "event_lm/long_streams_checkpoints.sha256"
LONG_FORGET, LONG_FLAT, LONG_REFERENCE, LONG_REFERENCE_1024 = (
    f"checkpoints/long_streams/{folder}/model.pt.averaged" for folder in ("l2048-forget", "l2048-flat", "tf2048",
                                                                          "tf1024"))
LONG_STREAMS_ARMS = (
    ("34.81M SNN stack, windows of 2,048 bytes, forgetting slopes", "Stream, chosen reach, taught output synapses",
     "l2048-forget.taught-test", None, "l2048-forget.pg19-taught", LONG_FORGET, VIOLET),
    ("34.81M SNN stack, windows of 2,048 bytes, forgetting slopes", "Stream, chosen reach",
     "l2048-forget.reach-test", None, "l2048-forget.pg19-reach", LONG_FORGET, VIOLET),
    ("34.81M SNN stack, windows of 2,048 bytes, forgetting slopes", "Stream, as trained",
     "l2048-forget.stream-test", None, "l2048-forget.pg19-stream", LONG_FORGET, VIOLET),
    ("34.81M SNN stack, windows of 2,048 bytes", "Stream, chosen reach, taught output synapses",
     "l2048-flat.taught-test", None, "l2048-flat.pg19-taught", LONG_FLAT, "#3b2c9c"),
    ("34.81M SNN stack, windows of 2,048 bytes", "Stream, chosen reach", "l2048-flat.reach-test", None,
     "l2048-flat.pg19-reach", LONG_FLAT, "#3b2c9c"),
    ("34.81M SNN stack, windows of 2,048 bytes", "Stream, as trained", "l2048-flat.stream-test", None, None,
     LONG_FLAT, "#3b2c9c"),
    ("34.75M full attention, windows of 2,048 bytes", "Windows with 2,032 bytes of history, taught output synapses",
     "tf2048.taught-test-h2032", 0.07, None, LONG_REFERENCE, ORANGE),
    ("34.75M full attention, windows of 2,048 bytes", "Windows with 2,032 bytes of history",
     "tf2048.taught-test-h2032", 0.0, "tf2048.pg19-h2032", LONG_REFERENCE, ORANGE),
    ("34.75M full attention, windows of 1,024 bytes", "Windows with 1,008 bytes of history",
     "tf1024.taught-test-h1008", 0.0, "tf1024.pg19-h1008", LONG_REFERENCE_1024, ORANGE),
)
TRACE_READ_DRIVE_116326K = (
    ("Original held-charge read", TRACE_READ_DRIVE, "control_s8237",
     f"{TRACE_READ_DRIVE}/native_control_s8237.json", BLUE),
    ("Write input, six-bit reset", TRACE_READ_DRIVE, "write_s8237",
     f"{TRACE_READ_DRIVE}/native_write_s8237.json", "#7a6a50"),
    ("Present input, six-bit reset", TRACE_READ_DRIVE, "present_s8237",
     f"{TRACE_READ_DRIVE}/native_present_s8237.json", GRAY),
    ("Write input, eight-bit reset", TRACE_READ_DRIVE, "write_fine_s8237",
     f"{TRACE_READ_DRIVE}/native_write_fine_s8237.json", "#138372"),
    ("Write input, eight-bit carry", TRACE_READ_DRIVE, "write_carry_s8237",
     f"{TRACE_READ_DRIVE}/native_write_carry_s8237.json", VIOLET),
    ("Present input, eight-bit carry", TRACE_READ_DRIVE, "present_carry_s8237",
     f"{TRACE_READ_DRIVE}/native_present_carry_s8237.json", "#ad4772"),
    ("Write input, shared steering", TRACE_READ_DRIVE, "write_share_s8237",
     f"{TRACE_READ_DRIVE}/native_write_share_s8237.json", "#5750a1"),
    ("Two write rails, shared steering", TRACE_READ_DRIVE, "write_rails_share_s8237",
     f"{TRACE_READ_DRIVE}/native_write_rails_share_s8237.json", "#138372"),
    ("Present input, shared steering", TRACE_READ_DRIVE, "present_share_s8237",
     f"{TRACE_READ_DRIVE}/native_present_share_s8237.json", "#76611e"),
    ("Full attention", TRACE_READ_DRIVE, "transformer_s8237", None, ORANGE),
)
TRACE_COINCIDENCE_6510K = (
    GROWN_6510K[1],
    ("Three-way coincidence", TRACE_COINCIDENCE, "coincidence_s711",
     f"{TRACE_COINCIDENCE}/native_coincidence_s711.json", VIOLET),
    GROWN_6510K[-1],
)
EXPANSION_8740K = (
    ("Expanded held-charge SNN", EXPANSION, "expanded_s1801", f"{EXPANSION}/native_expanded_s1801.json", BLUE),
    ("Expanded SNN with active steering", EXPANSION, "expanded_steering_s1801",
     f"{EXPANSION}/native_expanded_steering_s1801.json", VIOLET),
    ("Grown full-attention Transformer", EXPANSION, "transformer_s1801", None, ORANGE),
    ("Original-width held-charge SNN", EXPANSION, "control_s1801", f"{EXPANSION}/native_control_s1801.json", GRAY),
)
WRITE_CONFIRM_6510K = (
    EXPANSION_8740K[-1],
    ("Delayed write confirmation", WRITE_CONFIRM, "confirm_s1801", f"{WRITE_CONFIRM}/native_confirm_s1801.json", VIOLET),
)
INHIBITION_6510K = (
    EXPANSION_8740K[-1],
    ("Redistribution of inhibition", INHIBITION, "inhibition_s1801", f"{INHIBITION}/native_inhibition_s1801.json", VIOLET),
)
GUIDANCE_6510K = (
    EXPANSION_8740K[-1],
    ("Held-charge SNN with frozen teacher", GUIDANCE, "guided_s1801", f"{GUIDANCE}/native_guided_s1801.json", VIOLET),
)
SCRATCH_6510K = (
    ("Wide held-charge SNN", WIDTH_6510K, "trace_s0", f"{WIDTH_6510K}/native_trace_s0.json", BLUE),
    ("Full-attention Transformer", WIDTH_6510K, "transformer_s0", None, ORANGE),
)


def summarize(values):
    return {"mean": mean(values), "sd": stdev(values), "by_seed": values}


def read_arm(arm, scale, activity):
    scores = {}
    sources = {}
    metadata = None
    for corpus, _, targets in CORPORA:
        source = f"event_lm/{corpus}_threshold_scale_{arm}_{scale}.json"
        record = read_result(source)
        primary = [row for row in record["models"] if "intervention" not in row]
        if record["target_bytes"] != targets or len(primary) != 3:
            raise ValueError(f"{source} needs three complete test records")
        rows = []
        for row in primary:
            train = read_result(f'event_lm/{row["run_name"]}.json')
            if row["target_bytes"] != targets:
                raise ValueError(f"{source} contains a partial test run")
            rows.append((train["args"]["seed"], train["train_target_bytes"], row))
        rows.sort(key=lambda item: item[0])
        if [seed for seed, _, _ in rows] != [0, 1, 2]:
            raise ValueError(f"{source} needs seeds 0, 1, and 2")
        definitions = {
            (entry["parameters"], entry["streaming_tensor_bytes"], targets_seen) for _, targets_seen, entry in rows
        }
        if len(definitions) != 1:
            raise ValueError(f"{source} mixes parameter, state, or training budgets")
        definition = definitions.pop()
        if metadata is not None and metadata != definition:
            raise ValueError(f"{arm} has inconsistent metadata across corpora")
        metadata = definition
        scores[corpus] = summarize([row["bpb"] for _, _, row in rows])
        sources[corpus] = source
    parameters, state, training_targets = metadata
    pulses = [activity[Path(row["checkpoint"]).name] for _, _, row in rows] if arm not in ("transformer_local", "full") else []
    return {
        "arm": arm, "model": LABELS[arm][0], "parameters": parameters, "state_bytes": state,
        "training_targets": training_targets, "scores": scores, "score_files": sources,
        "pulses_per_byte": summarize(pulses) if pulses else None,
    }


def build():
    leaderboard = {"unit": "bits per byte", "seeds": [0, 1, 2], "groups": {}}
    for scale, arms in ARMS.items():
        audit = read_result(f"event_lm/threshold_scale_{scale}_activity.json")
        activity = {Path(row["checkpoint"]).name: row["total_pulses_per_byte"] for row in audit["models"]}
        rows = [read_arm(arm, scale, activity) for arm in arms]
        exposures = {row["training_targets"] for row in rows}
        if len(exposures) != 1:
            raise ValueError(f"{scale} contains unequal training exposure")
        rank_scores(rows)
        leaderboard["groups"][scale] = rows
    return leaderboard


def rank_scores(rows):
    for corpus, _, _ in CORPORA:
        ordered = sorted(rows, key=lambda row: row["scores"][corpus]["mean"])
        for rank, row in enumerate(ordered, start=1):
            row["scores"][corpus]["rank"] = rank


def build_depth():
    definitions = list(DEPTH_ARMS)
    trace_files = [f"{DEPTH}/{prefix}_deep_trace.json" for prefix in ("enwik8", "pg19", "native")]
    if all((RESULTS / path).is_file() for path in trace_files):
        definitions.insert(4, TRACE)
    rows = read_comparison(definitions, seed=0, context=512, exposure=(0, 26_214_400))
    return {"unit": "bits per byte", "seeds": [0], "groups": {"depth": rows}}


def read_comparison(definitions, *, seed, context, exposure, test_inputs=None):
    rows = []
    if test_inputs is None:
        test_inputs = {}
    for label, directory, suffix, audit_path, color in definitions:
        scores, sources = {}, {}
        scored_models = []
        for corpus, dataset, targets in CORPORA:
            prefix = "pg19" if corpus == "pg19_full" else corpus
            source = f"{directory}/{prefix}_{suffix}.json"
            record = read_result(source)
            (model,) = record["models"]
            books = [(book["book_id"], book["excerpt_sha256"], book["target_bytes"]) for book in model["books"]]
            calibration = record.get("calibration") or {}
            inputs = (books, record.get("data_sha1"), record.get("manifest_sha256"),
                      calibration.get("manifest_sha256"), calibration.get("temperatures"))
            if (record["split"] != "test" or record["target_bytes"] != targets or model["target_bytes"] != targets
                    or inputs != test_inputs.setdefault(corpus, inputs)
                    or model["training_context_bytes"] != context
                    or model.get("full_attention_context_bytes", context) != context):
                raise ValueError(f"{source} does not match the complete {dataset} test inputs")
            scores[corpus] = {"mean": model["bpb"], "by_seed": [model["bpb"]], "raw_bpb": model["raw_bpb"]}
            sources[corpus] = source
            scored_models.append(model)
        if len({(row["run_name"], row["checkpoint"], row["parameters"], row["streaming_tensor_bytes"])
                for row in scored_models}) != 1:
            raise ValueError(f"{label} mixes different checkpoints or state budgets")

        training_path = f'{directory}/{model["run_name"]}.json'
        training = read_result(training_path)
        parent_targets = (training.get("initialization") or {}).get("training_targets", 0)
        if (training["args"]["seed"] != seed or training["args"]["ctx"] != context
                or (parent_targets, training["train_target_bytes"]) != exposure
                or training["args"]["checkpoint"] != model["checkpoint"] or training["params"] != model["parameters"]):
            raise ValueError(f"{training_path} does not match the comparison budget or evaluated checkpoint")
        pulses = None
        if audit_path:
            native = read_result(audit_path)
            (measurement,) = native["models"]
            if (native["input_bytes"] != 4096 or native["split"] != "test"
                    or measurement["checkpoint"] != model["checkpoint"]
                    or measurement["streaming_tensor_bytes"] != model["streaming_tensor_bytes"]):
                raise ValueError(f"{audit_path} does not match the evaluated model and audit input size")
            pulses = {"mean": measurement["total_pulses_per_byte"]}
        rows.append({"model": label, "color": color, "parameters": model["parameters"],
                     "state_bytes": model["streaming_tensor_bytes"], "training_targets": sum(exposure),
                     "scores": scores, "score_files": sources, "training_record": training_path,
                     "native_audit": audit_path, "pulses_per_byte": pulses})
        if training.get("teacher"):
            rows[-1]["teacher"] = training["teacher"]
    rank_scores(rows)
    return rows


def read_warmed_windows(arms=STATE_READS_ARMS, study=STATE_READS):
    """A comparison of a study with every checkpoint scored in windows that start from an empty state."""
    records = {}

    def scored(name, checkpoint, targets):
        if name not in records:
            record = records[name] = read_result(f"{name}.json" if "/" in name else f"{study}/scores/{name}.json")
            protocol = (record["protocol"], record["split"], record["context_bytes"], record["unscored_history_bytes"])
            if protocol != ("causal_warmed_windows", "test", 256, 256):
                raise ValueError(f"{name}: expected test scores in windows of 256 unscored and 256 scored bytes")
        row = next(row for row in records[name]["checkpoints"] if row["checkpoint"] == checkpoint)
        if (row["targets"], row["training_targets"]) != (targets, 368_640_000):
            raise ValueError(f"{name}: {checkpoint} must score {targets:,} targets after 368,640,000 training targets")
        return row

    rows = []
    for model, optimizer, enwik8, pg19, checkpoint, color in arms:
        row = scored(enwik8, checkpoint, CORPORA[0][2])
        scores = {"enwik8": {"mean": row["bpb"]}}
        if pg19 and (RESULTS / (f"{pg19}.json" if "/" in pg19 else f"{study}/scores/{pg19}.json")).is_file():
            books = scored(pg19, checkpoint, CORPORA[1][2])
            scores["pg19_full"] = {"mean": books["calibrated_bpb"], "raw_bpb": books["bpb"],
                                   "temperature": books["temperature"]}
        rows.append({"model": model, "optimizer": optimizer, "parameters": row["parameters"],
                     "checkpoint": checkpoint, "scores": scores, "color": color})
    for corpus, _, _ in CORPORA:
        ordered = sorted((row["scores"][corpus] for row in rows if corpus in row["scores"]),
                         key=lambda score: score["mean"])
        for rank, score in enumerate(ordered, start=1):
            score["rank"] = rank
    return {"unit": "bits per byte", "protocol": "causal_warmed_windows", "context_bytes": 256,
            "unscored_history_bytes": 256, "training_targets": 368_640_000, "seeds": [20001], "rows": rows}


def read_stream_form():
    """The stream-form study: the stacks scored as unbroken streams and the Transformer in warmed windows."""
    def scored(name, rate, checkpoint, targets):
        record = read_result(f"{STREAM_FORM}/scores/{name}.json")
        if rate is None:
            row = next(row for row in record["checkpoints"] if row["checkpoint"] == checkpoint)
        else:
            # A Transformer with fast output synapses: one record holds every rate that was scored.
            if record["checkpoint"] != checkpoint:
                raise ValueError(f"{name}: expected {checkpoint}")
            row = dict(record, **next(row for row in record["fast_output_synapses"] if row["rate"] == rate))
        if (record["split"], row["targets"], row["training_targets"]) != ("test", targets, 368_640_000):
            raise ValueError(f"{name}: {checkpoint} must score {targets:,} test targets after 368,640,000 "
                             "training targets")
        return row

    rows = []
    for model, reads, enwik8, rate, pg19, checkpoint, color in STREAM_FORM_ARMS:
        row = scored(enwik8, rate, checkpoint, CORPORA[0][2])
        scores = {"enwik8": {"mean": row["bpb"]}}
        if pg19 and (RESULTS / f"{STREAM_FORM}/scores/{pg19}.json").is_file():
            scores["pg19_full"] = {"mean": scored(pg19, None, checkpoint, CORPORA[1][2])["bpb"]}
        rows.append({"model": model, "reads_the_test_as": reads, "parameters": row["parameters"],
                     "checkpoint": checkpoint, "scores": scores, "color": color})
    for corpus, _, _ in CORPORA:
        ordered = sorted((row["scores"][corpus] for row in rows if corpus in row["scores"]),
                         key=lambda score: score["mean"])
        for rank, score in enumerate(ordered, start=1):
            score["rank"] = rank
    return {"unit": "bits per byte", "protocol": "unbroken_streams for the stacks, causal_warmed_windows for the "
            "Transformer", "training_targets": 368_640_000, "seeds": [20001], "rows": rows}


def read_long_streams():
    """The long-streams study: the stacks on windows of 2,048 bytes as streams and the Transformers that trained on
    the same windows, in warmed windows of their training length. Rows whose records are not published are left
    out."""
    def scored(name, rate, checkpoint, targets):
        record = read_result(f"{LONG_STREAMS}/scores/{name}.json")
        if rate is None:
            row = next(row for row in record["checkpoints"] if row["checkpoint"] == checkpoint)
        else:
            if record["checkpoint"] != checkpoint:
                raise ValueError(f"{name}: expected {checkpoint}")
            row = dict(record, **next(row for row in record["fast_output_synapses"] if row["rate"] == rate))
        if (row["targets"], row["training_targets"]) != (targets, 368_640_000):
            raise ValueError(f"{name}: {checkpoint} must score {targets:,} targets after 368,640,000 training targets")
        return row

    rows = []
    for model, reads, enwik8, rate, pg19, checkpoint, color in LONG_STREAMS_ARMS:
        if not (RESULTS / f"{LONG_STREAMS}/scores/{enwik8}.json").is_file():
            continue
        row = scored(enwik8, rate, checkpoint, CORPORA[0][2])
        scores = {"enwik8": {"mean": row["bpb"]}}
        if pg19 and (RESULTS / f"{LONG_STREAMS}/scores/{pg19}.json").is_file():
            book = read_result(f"{LONG_STREAMS}/scores/{pg19}.json")
            # A stack's book record holds its checkpoint; a Transformer's names one setting of warmed windows.
            score = (book["checkpoints"][0]["bpb"] if "checkpoints" in book else book.get("bpb"))
            if score is not None:
                scores["pg19_full"] = {"mean": score}
        rows.append({"model": model, "reads_the_test_as": reads, "parameters": row["parameters"],
                     "checkpoint": checkpoint, "scores": scores, "color": color})
    for corpus, _, _ in CORPORA:
        ordered = sorted((row["scores"][corpus] for row in rows if corpus in row["scores"]),
                         key=lambda score: score["mean"])
        for rank, score in enumerate(ordered, start=1):
            score["rank"] = rank
    return {"unit": "bits per byte", "protocol": "unbroken_streams for the stacks, causal_warmed_windows for the "
            "Transformers", "training_targets": 368_640_000, "seeds": [20001], "rows": rows}


def stream_table(rows):
    lines = ["| Model | Parameters | Reads the test as | enwik8 | Rank | PG-19 | Rank |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        enwik8, pg19 = row["scores"]["enwik8"], row["scores"].get("pg19_full")
        books = f'{score_cell(pg19)} | {pg19["rank"]}' if pg19 else "— | —"
        lines.append(f'| {row["model"]} | {row["parameters"]:,} | {row["reads_the_test_as"]} | {score_cell(enwik8)} | '
                     f'{enwik8["rank"]} | {books} |')
    return "\n".join(lines)


def width_gap_by_prefix_age(narrow, wide, narrow_full, wide_full):
    age_scores = [read_result(row["score_files"]["enwik8"])["models"][0]["prefix_age"]
                  for row in (narrow, wide, narrow_full, wide_full)]
    report = {}
    for age in age_scores[0]:
        a, b, c, d = (row[age] for row in age_scores)
        targets = {row["target_bytes"] for row in (a, b, c, d)}
        if len(targets) != 1:
            raise ValueError(f"prefix age {age} must have the same targets at both widths")
        small_gap, large_gap = a["bpb"] - c["bpb"], b["bpb"] - d["bpb"]
        report[age] = {"target_bytes": a["target_bytes"], "narrow_gap_bpb": small_gap, "wide_gap_bpb": large_gap,
                       "change_bpb": large_gap - small_gap}
    return report


def build_research():
    groups = {}
    if (RESULTS / COASTING_HEADS_CHECKPOINTS).is_file():
        rows = read_comparison(COASTING_HEADS_ARMS, seed=20001, context=512, exposure=(0, 368_640_000))
        if [row["parameters"] for row in rows] != [34_845_184, 34_845_184, 34_845_184, 34_845_696]:
            raise ValueError("the coasting-head study must use its four audited sizes")
        groups[COASTING_HEADS_GROUP] = {
            "title": "Coasting read heads, trained from scratch", "seeds": [20001], "rows": rows,
            "training_context_bytes": 512, "training_summary": "368,640,000 training targets · seed 20001",
            "parent_training_targets": 0, "additional_training_targets": 368_640_000,
        }
    if (RESULTS / SURPRISE_CLOCK_CHECKPOINTS).is_file():
        rows = read_comparison(SURPRISE_CLOCK_ARMS, seed=10007, context=512, exposure=(131_072_000, 9_830_400))
        if [row["parameters"] for row in rows] != [25_788_928, 25_788_928, 34_845_184, 34_845_184, 34_845_184,
                                                   34_845_184, 25_790_976, 34_845_696]:
            raise ValueError("the surprise study must use its eight audited sizes")
        groups[SURPRISE_CLOCK_GROUP] = {
            "title": "Surprise clock and fast synapses", "seeds": [10007], "rows": rows,
            "training_context_bytes": 512, "training_summary": "140,902,400 training targets · seed 10007",
            "parent_training_targets": 131_072_000, "additional_training_targets": 9_830_400,
            "transfer_record": f"event_lm/byte_transfer/{SURPRISE_CLOCK_GROUP}.json",
        }
    if (RESULTS / RESIDUAL_SCALE_CHECKPOINTS).is_file():
        rows = read_comparison(RESIDUAL_SCALE_ARMS, seed=9631, context=512, exposure=(121_241_600, 9_830_400))
        mixture = residual_scale_mixture(RESULTS / row["training_record"] for row in rows)
        narrow, wide, enwik8_only, narrow_full, wide_full = rows
        enwik8 = [row["scores"]["enwik8"]["mean"] for row in rows]
        narrow_gap, wide_gap = enwik8[0] - enwik8[3], enwik8[1] - enwik8[4]
        if ([row["parameters"] for row in rows] != [14_066_688, 25_788_928, 25_788_928, 14_068_224, 25_790_976]
                or any(row["state_bytes"] != 28_584 for row in (narrow, wide, enwik8_only))):
            raise ValueError("the residual scale study must use the two audited widths and stream budget")
        groups[RESIDUAL_SCALE_GROUP] = {
            "title": "Residual width on expanded training data", "seeds": [9631], "rows": rows,
            "training_context_bytes": 512, "training_summary": "131,072,000 training targets · seed 9631",
            "parent_training_targets": 121_241_600, "additional_training_targets": 9_830_400,
            "training_mixture": mixture, "transfer_record": f"event_lm/byte_transfer/{RESIDUAL_SCALE_GROUP}.json",
            "early_context_records": {split: f"{RESIDUAL_SCALE}/early_context_{split}.json" for split in ("val", "test")},
            "enwik8_width_gap": {"narrow_bpb": narrow_gap, "wide_bpb": wide_gap,
                                 "change_bpb": wide_gap - narrow_gap, "mixture_minus_enwik8_bpb": enwik8[1] - enwik8[2]},
            "enwik8_width_gap_by_prefix_age": width_gap_by_prefix_age(narrow, wide, narrow_full, wide_full),
        }
    continuation = (26_214_400, 6_553_600)
    for name, title, arms, seed, context, exposure in (
        (ARRIVAL_GROUP, "Scheduled spike arrivals", TRACE_ARRIVAL_121241K, 9349, 512, (116_326_400, 4_915_200)),
        ("trace_read_drive_116326k", "Driven held-charge reads",
         TRACE_READ_DRIVE_116326K, 8237, 512, (111_411_200, 4_915_200)),
        ("packed_width_111411k", "Packed width and steering transfers",
         PACKED_WIDTH_111411K, 7907, 512, (101_580_800, 9_830_400)),
        ("trace_tap_101580k", "9.96M axon tap reads",
         TRACE_TAP_101580K, 6899, 512, (91_750_400, 9_830_400)),
        ("trace_integration_91750k", "9.96M trace inputs and group thresholds",
         TRACE_INTEGRATION_91750K, 5801, 512, (81_920_000, 9_830_400)),
        ("growth_9960k", "9.96M depth and width growth", GROWTH_81920K, 4603, 512, (72_089_600, 9_830_400)),
        ("residual_output_6510k", "6.51M signed residual output", RESIDUAL_OUTPUT_81920K, 4603, 512,
         (72_089_600, 9_830_400)),
        ("residual_growth_13030k", "13.03M residual growth and original-width controls", RESIDUAL_GROWTH_13030K,
         3307, 512, (65_536_000, 6_553_600)),
        ("circuits_6510k", "6.51M spike circuit and charge comparisons", CIRCUITS_6510K,
         3307, 512, (65_536_000, 6_553_600)),
        ("data_6510k_65536", "6.51M at 65.54M observed targets", NEXT_DATA_6510K, 2503, 512, (49_152_000, 16_384_000)),
        ("expansion_8740k", "8.74M expansion and original-width control", EXPANSION_8740K, 1801, 512, (49_152_000, 6_553_600)),
        ("inhibition_6510k", "6.52M redistribution of inhibition", INHIBITION_6510K, 1801, 512, (49_152_000, 6_553_600)),
        ("teacher_guidance_6510k", "6.51M frozen teacher guidance and unguided control", GUIDANCE_6510K, 1801, 512,
         (49_152_000, 6_553_600)),
        ("write_confirm_6510k", "6.51M delayed write confirmation", WRITE_CONFIRM_6510K, 1801, 512, (49_152_000, 6_553_600)),
        ("data_6510k", "6.51M with more training data", DATA_6510K, 1109, 512, (32_768_000, 16_384_000)),
        ("grown_6510k", "6.51M growth · 512-byte continuations", GROWN_6510K, 711, 512, continuation),
        ("trace_coincidence_6510k", "6.51M trace coincidence", TRACE_COINCIDENCE_6510K, 711, 512, continuation),
        ("context_3260k_512", "3.26M · 512-byte continuations", SCALE_CONTEXT_512, 711, 512, continuation),
        ("context_3260k_2048", "3.26M · 2,048-byte continuations", SCALE_CONTEXT_2048, 711, 2048, continuation),
        ("scratch_6510k", "6.51M trained from scratch", SCRATCH_6510K, 0, 512, (0, 26_214_400)),
        ("continuation_512", "512-byte continuations", CONTINUATION_512, 711, 512, continuation),
        ("continuation_2048", "2,048-byte continuations", CONTINUATION_2048, 711, 2048, continuation),
        ("scale_3260k", "3.26M parameter comparison", SCALE_3260K, 0, 512, (0, 26_214_400)),
    ):
        groups[name] = {"title": title, "seeds": [seed], "training_context_bytes": context,
                        "training_summary": f"{sum(exposure):,} training targets · seed {seed}",
                        "parent_training_targets": exposure[0], "additional_training_targets": exposure[1],
                        "rows": read_comparison(arms, seed=seed, context=context, exposure=exposure)}
    groups["held_charge_replication"] = {
        "title": "Held-charge replication",
        "seeds": [0, 1, 2],
        "training_context_bytes": 512,
        "training_summary": "26,214,400 training targets per seed · seeds 0–2",
        "parent_training_targets": 0,
        "additional_training_targets": 26_214_400,
        "rows": read_held_charge_replication(),
    }
    for name, group in groups.items():
        transfer_path = f"event_lm/byte_transfer/{name}.json"
        if (RESULTS / transfer_path).is_file():
            group["transfer_record"] = transfer_path
    groups[ARRIVAL_GROUP].update(
        transfer_record=ARRIVAL_TRANSFER,
        early_context_records={split: f"{TRACE_ARRIVAL}/early_context_{split}.json" for split in ("val", "test")},
    )
    return {"unit": "bits per byte", "groups": groups}


def read_held_charge_replication():
    directory = "event_lm/held_charge_replication"
    controls = "event_lm/depth_inhibition_replication"
    label = "Held-charge SNN"
    per_seed, test_inputs = [], {}
    for seed in (0, 1, 2):
        if seed == 0:
            arms = ((label, *TRACE[1:]), DEPTH_ARMS[-1])
        else:
            arms = (
                (label, directory, f"trace_s{seed}", f"{directory}/native_trace_s{seed}.json", VIOLET),
                ("Full-attention Transformer", controls, f"transformer_s{seed}", None, ORANGE),
            )
        per_seed.append(read_comparison(arms, seed=seed, context=512, exposure=(0, 26_214_400),
                                        test_inputs=test_inputs))

    rows = []
    for runs in zip(*per_seed, strict=True):
        first = runs[0]
        fields = ("model", "color", "parameters", "state_bytes", "training_targets")
        if any(any(run[field] != first[field] for field in fields) for run in runs[1:]):
            raise ValueError(f"{first['model']} changes configuration budgets across replication seeds")
        scores = {}
        for corpus, _, _ in CORPORA:
            measurements = [run["scores"][corpus] for run in runs]
            scores[corpus] = {**summarize([score["mean"] for score in measurements]),
                              "raw_bpb": mean(score["raw_bpb"] for score in measurements)}
        pulses = [run["pulses_per_byte"]["mean"] for run in runs if run["pulses_per_byte"] is not None]
        rows.append({**{field: first[field] for field in fields}, "scores": scores,
                     "pulses_per_byte": summarize(pulses) if pulses else None, "runs": runs})
    rank_scores(rows)
    return rows


def score_cell(stats):
    score = f'{stats["mean"]:.5f}'
    return f'**{score}**' if stats["rank"] == 1 else score


def state_size(state):
    return f"{state / 1024 ** 2:g} MiB" if state >= 1024 ** 2 else f"{state:,} B"


def markdown_table(rows, *, include_ranks, include_raw_pg19=False):
    pg19_label = "PG-19 calibrated" if include_raw_pg19 else "PG-19"
    if include_ranks:
        columns = ["Model", "Parameters", "State", "enwik8", "Rank", pg19_label, "Rank", "Pulses/byte"]
        if include_raw_pg19:
            columns.insert(7, "PG-19 raw")
    else:
        columns = ["Model", "enwik8", pg19_label, "State"]
        if include_raw_pg19:
            columns.insert(3, "PG-19 raw")
    lines = ["| " + " | ".join(columns) + " |", "|---|" + "---:|" * (len(columns) - 1)]
    best_raw_pg19 = min(row["scores"]["pg19_full"]["raw_bpb"] for row in rows) if include_raw_pg19 else None
    for row in rows:
        enwik8, pg19 = (row["scores"][corpus] for corpus, _, _ in CORPORA)
        state = state_size(row["state_bytes"])
        raw = ""
        if include_raw_pg19:
            raw_score = f'{pg19["raw_bpb"]:.5f}'
            if pg19["raw_bpb"] == best_raw_pg19:
                raw_score = f"**{raw_score}**"
            raw = f" {raw_score} |"
        if include_ranks:
            pulses = row["pulses_per_byte"]
            pulse_text = f'{pulses["mean"]:,.0f}' if pulses else "—"
            lines.append(f'| {row["model"]} | {row["parameters"]:,} | {state} | {score_cell(enwik8)} | '
                         f'{enwik8["rank"]} | {score_cell(pg19)} | {pg19["rank"]} |{raw} {pulse_text} |')
        else:
            lines.append(f'| {row["model"]} | {score_cell(enwik8)} | {score_cell(pg19)} |{raw} {state} |')
    return "\n".join(lines)


def warmed_table(rows):
    lines = ["| Model | Parameters | Optimizer | enwik8 | Rank | PG-19 calibrated | Rank | PG-19 raw |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    best_raw = min(row["scores"]["pg19_full"]["raw_bpb"] for row in rows if "pg19_full" in row["scores"])
    for row in rows:
        enwik8, pg19 = row["scores"]["enwik8"], row["scores"].get("pg19_full")
        books = "— | — | —"
        if pg19:
            raw = f'{pg19["raw_bpb"]:.5f}'
            books = f'{score_cell(pg19)} | {pg19["rank"]} | ' + (f"**{raw}**" if pg19["raw_bpb"] == best_raw else raw)
        lines.append(f'| {row["model"]} | {row["parameters"]:,} | {row["optimizer"]} | {score_cell(enwik8)} | '
                     f'{enwik8["rank"]} | {books} |')
    return "\n".join(lines)


def long_streams_paragraphs(long_streams):
    """The section of the long-streams study: the stacks and the Transformers at equal training context."""
    test = {(row["checkpoint"], row["reads_the_test_as"]): row["scores"]["enwik8"]["mean"]
            for row in long_streams["rows"]}
    stack = test[(LONG_FLAT, "Stream, as trained")]
    reference = test[(LONG_REFERENCE, "Windows with 2,032 bytes of history")]
    best = test.get((LONG_FORGET, "Stream, chosen reach, taught output synapses"))
    return [
        "## Long streams at equal training context · 368,640,000 training targets · seed 20001",
        stream_table(long_streams["rows"]),
        "![The stack and the Transformer at training windows of 512, 1,024 and 2,048 bytes, test and validation]"
        "(../results/figures/event_lm_long_streams.png)",
        "![The gain of the fast synapses against how much of a stream they have read]"
        "(../results/figures/event_lm_long_streams_reading.png)",
        "![Validation loss of the stack and the Transformer along their training, at depths 11 and 22]"
        "(../results/figures/event_lm_long_streams_scaling.png)",
        "[Measurements, forgetting slopes and limits](LONG_STREAMS.md) · [Trained plasticity](TRAINED_PLASTICITY.md)",
        "Every model of this comparison trained on windows of 512 bytes for 24,000 updates and on windows of 2,048 "
        "bytes for the remaining 66,000, from random weights, or on windows of 1,024 bytes where named. A stack reads "
        "the test as 16 unbroken streams; a chosen reach is the reach that scored best on the first million "
        "validation bytes. A Transformer reads the same streams in windows of its training length, 16 scored "
        "bytes after the rest as history. Taught output synapses start at zero in every stream and learn from each "
        "byte after it was predicted.",
        f"At equal context, each read as it trained, the stack is {stack - reference:.4f} bpb behind the Transformer "
        "on the enwik8 test." + (
            f" Read at its best, with slopes that learn, a chosen reach and taught output synapses, it is "
            f"{best - reference:.4f} behind." if best is not None else "") + " The level score of the comparison "
        "below came from a stack that trained on longer windows than its Transformer read. Every row is one seed.",
    ]


def knowledge_gap_paragraphs(board):
    """The section of the knowledge-gap study (v17): knowledge cells trained on streams, and the depth-22 pair."""
    test, validation = board["paired"]["knowledge cells against tf2048, test"], board["paired"]["validation"]
    return [
        "## Knowledge cells trained on streams · 368,640,000 training targets · seed 20001",
        stream_table(board["rows"]),
        "![Where the gap is: the stack minus the Transformer by what each byte's context offers]"
        "(../results/figures/event_lm_gap_by_context.png)",
        "[Where the gap is, what was tried, and what moves it](KNOWLEDGE_GAP.md) · [Roadmap](ROADMAP.md)",
        "The knowledge-cell stack is v16's stack with 128 learned, ageless cells in every read head's race. It trained "
        "on windows of 512 bytes for 24,000 updates and then on unbroken streams of windows of 2,048 bytes, the state "
        "handed on from window to window. Read at its best, with a reach of 8,192 cells and taught output synapses, "
        f"it is {test[0]:.4f} bpb behind the Transformer on the enwik8 test (paired 95% interval {test[1][0]:.4f} to "
        f"{test[1][1]:.4f}) and {validation[0]:.4f} on validation. Every row is one seed.",
        "### Depth 22 at the same pulse budget",
        stream_table(board["depth_22_rows"]),
        "Twice the levels at the same pulse budget gain the stack 0.027 bpb on test and the Transformer 0.036: the gap "
        "widens with size, at equal energy for the stack.",
    ]


def write_leaderboards(data, depth, research, transfers, warmed=None, clocks=None, streams=None, long_streams=None,
                       knowledge_gap=None):
    latest = depth["groups"]["depth"]
    paragraphs = [
        "# Spiking byte model leaderboards",
        "Lower bits per byte rank higher. Every row in a comparison has the same number of training targets and scores "
        "4,999,999 enwik8 targets and 8,404,368 targets from 16 complete PG-19 books. A dash marks a score that was "
        "not measured.",
        "State is persistent tensor storage for one stream. Full attention reports FP32 "
        "key and value storage at the context length in each comparison. Pulse counts use "
        "a native 4,096-byte enwik8 test sample. "
        "Each PG-19 temperature is fitted on eight separate validation books. "
        "The teacher-guided SNN uses additional training compute from a frozen Transformer.",
    ]
    study_figures = {
        COASTING_HEADS_GROUP: (
            "![Held-out enwik8 and PG-19 loss of the decoders trained from scratch]"
            "(../results/figures/event_lm_coasting_heads_leaderboard.png)",
            "![Held-out loss against the state of one stream](../results/figures/event_lm_coasting_heads_state.png)",
            "![enwik8 test loss of each decoder alone, with fast synapses and with read heads]"
            "(../results/figures/event_lm_coasting_heads.png)",
            "![Gap to full attention by bytes of history](../results/figures/event_lm_coasting_heads_history.png)",
            "[Head circuit, controls and limits](COASTING_HEADS.md)",
        ),
        SURPRISE_CLOCK_GROUP: (
            "![Held-out loss with the surprise clock and fast synapses](../results/figures/event_lm_surprise_clock.png)",
            "![Where surprise falls inside words and where the trained clock ticks]"
            "(../results/figures/event_lm_surprise_words.png)",
            "![Held-out gain of the fast synapses by state and write rule]"
            "(../results/figures/event_lm_fast_synapse_sweep.png)",
            "![enwik8 loss by distance to the previous occurrence of the context]"
            "(../results/figures/event_lm_surprise_prefix_age.png)",
            "![Gap to the size-matched Transformer by bytes seen since a reset]"
            "(../results/figures/event_lm_surprise_history_gap.png)",
            "[Surprise measurements, circuits, controls and replay](SURPRISE_CLOCK.md)",
        ),
        RESIDUAL_SCALE_GROUP: (
            "![Held-out loss at both residual widths](../results/figures/event_lm_residual_scale.png)",
            "![Change in the enwik8 gap during the first 32 predictions]"
            "(../results/figures/event_lm_residual_scale_early_context.png)",
            "[Width experiment, corpus preparation and replay](RESIDUAL_SCALE.md)",
        ),
        ARRIVAL_GROUP: (
            "![Held-out enwik8 and PG-19 loss for the arrival study](../results/figures/event_lm_trace_arrival.png)",
            "![Paired enwik8 loss gap during the first 32 predictions after reset]"
            "(../results/figures/event_lm_arrival_early_context.png)",
            "[Arrival circuit, ablations and replay](FIXED_ARRIVALS.md)",
        ),
        "trace_read_drive_116326k": (
            "![Held-out next-byte loss after training the driven read circuits]"
            "(../results/figures/event_lm_trace_read_drive.png)",
            "![Spare steering capacity and the effect of write polarity on read counts]"
            "(../results/figures/event_lm_read_steering.png)",
            "[Circuit, native activity, and matched controls](TRACE_READ_DRIVE.md)",
        ),
        "packed_width_111411k": (
            "![Held-out loss at the original and wider sizes](../results/figures/event_lm_packed_width.png)",
            "![Held-out loss changes from channel placement and steering sources]"
            "(../results/figures/event_lm_packed_circuit_changes.png)",
            "![Validation loss gap as the available history grows](../results/figures/event_lm_packed_width_history.png)",
            "[Width, state budget, and steering circuit](PACKED_WIDTH.md)",
        ),
        "trace_tap_101580k": (
            "![Held-out loss for the spike race, shortest tap, and matched controls](../results/figures/event_lm_trace_tap_read.png)",
            "![Validation loss and pulse counts during the tap comparison](../results/figures/event_lm_trace_tap_training.png)",
            "[Timing circuit and interventions](TRACE_TAP_READ.md)",
        ),
        "trace_integration_91750k": (
            "![Held-out loss for trace inputs, group thresholds, and matched controls](../results/figures/event_lm_trace_integration.png)",
            "![Native resting-threshold offsets learned by each delay group](../results/figures/event_lm_fine_group_thresholds.png)",
            "![Gap in held-out loss by available history at the v6 weights](../results/figures/event_lm_validation_history.png)",
            "[Circuit and interventions](TRACE_INTEGRATION.md)",
        ),
        "growth_9960k": (
            "![Held-out loss for deeper and wider SNNs and full attention](../results/figures/event_lm_growth_9960k.png)",
            "![enwik8 loss gap by prefix age for the 11-block SNN](../results/figures/event_lm_growth_9960k_prefix_age.png)",
        ),
        "residual_output_6510k": (
            "![Held-out loss with ordinary and correction output](../results/figures/event_lm_residual_output.png)",
            "![Generated pulses with identical output counts within each pair](../results/figures/event_lm_residual_emissions.png)",
        ),
    }
    if knowledge_gap:
        paragraphs.extend(knowledge_gap_paragraphs(knowledge_gap))
    if long_streams:
        paragraphs.extend(long_streams_paragraphs(long_streams))
    if streams:
        test = {(row["checkpoint"], row["reads_the_test_as"]): row["scores"]["enwik8"]["mean"]
                for row in streams["rows"]}
        stack = test[(STREAM_FORM_STACK, "Stream, longer reach")]
        reference = test[(STREAM_FORM_REFERENCE, "Windows with 496 bytes of history")]
        # The validation split is not ranked here, but it reverses the order of the two, and the table must say so.
        stack_validation = read_result(f"{STREAM_FORM}/scores/validation_long_windows_reach.json")
        stack_validation = next(row["bpb"] for row in stack_validation["checkpoints"]
                                if row["checkpoint"] == STREAM_FORM_STACK)
        reference_validation = read_result(f"{STREAM_FORM}/scores/validation_transformer_taught_history496.json")
        reference_validation = next(row["bpb"] for row in reference_validation["fast_output_synapses"]
                                    if row["rate"] == 0)
        paragraphs.extend([
            "## A stack that reads a stream, trained from scratch · 368,640,000 training targets · seed 20001",
            stream_table(streams["rows"]),
            "![Complete enwik8 test and validation of the two stacks as streams and of the Transformer in warmed "
            "windows](../results/figures/event_lm_stream_form.png)",
            "![The stack as it reads a stream: eleven levels of a state read and a layer of threshold cells]"
            "(../results/figures/event_lm_stream_form_circuit.png)",
            "![Averaged validation loss over training in windows, and the stacks' distance to the Transformer]"
            "(../results/figures/event_lm_stream_form_training.png)",
            "[Measurements, circuit and limits](STREAM_FORM.md) · [Every training arm](STREAM_FORM_ARMS.md)",
            "This comparison reads the test as a stream. A stack carries its state through 16 unbroken parts of "
            "the enwik8 test split and through every PG-19 book. The Transformer carries no state and reads the "
            "same texts in windows that hold the named bytes of history before the scored ones. A longer reach "
            "lets the reads of a stack reach cells beyond its window of training, which adds no synapse. Taught "
            "output synapses start at zero in every stream and learn from each byte after it was predicted, and "
            "the Transformer gets the same rule on its last layer as a control.",
            f"On the enwik8 test the stack on long windows with its longer reach is {reference - stack:.4f} bpb "
            "below the Transformer at its best window. On the validation split it is "
            f"{stack_validation - reference_validation:.4f} above it ({stack_validation:.4f} against "
            f"{reference_validation:.4f}), so the two are level. That stack trained on windows of 1,024 bytes from "
            "24,000 updates, and the Transformer and the other stack on windows of 512 bytes. The Transformer "
            "trained on windows of 1,024 bytes, the control, is in the comparison above: against it the stack is "
            "behind. The three models received identical windows from random weights up to 24,000 updates. Every "
            "row is one seed." if long_streams else
            "24,000 updates, and the Transformer and the other stack on windows of 512 bytes. A Transformer trained "
            "on windows of 1,024 bytes is the control that this comparison does not have yet. The three models "
            "received identical windows from random weights up to 24,000 updates. Every row is one seed.",
        ])
    if clocks:
        scores = {row["checkpoint"].split("/")[2]: row["scores"]["enwik8"]["mean"] for row in clocks["rows"]}
        paragraphs.extend([
            "## Read clocks and the address rate, trained from scratch · 368,640,000 training targets · seed 20001",
            warmed_table(clocks["rows"]),
            "![enwik8 test loss of the two new decoders, the v13 decoder and two Transformers]"
            "(../results/figures/event_lm_read_clocks.png)",
            "![A read head counts time in beats, and the test on a made-up language]"
            "(../results/figures/event_lm_read_clocks_language.png)",
            "![Averaged validation loss over training, and the decoders' distance to the Transformer]"
            "(../results/figures/event_lm_read_clocks_training.png)",
            "[Measurements, pilots and limits](READ_CLOCKS.md) · [Every training arm](READ_CLOCKS_ARMS.md)",
            "This comparison is scored in windows as the one below it: every window starts from an empty state and "
            "holds 256 unscored bytes before its 256 scored ones. The two new decoders are the v13 decoder with an "
            "address rate of 8 for its read heads, and one of them has a clock cell for every head as well. They "
            f"are {scores['rate8'] - scores['tag8-a8']:.4f} bpb apart on enwik8, which is less than a second run "
            "moves a decoder, so the gain over v13 is the address rate's. Neither decoder has a stream form or a "
            "native pulse count, and both write text only by running a whole window for every byte.",
            "The Transformer of the standard shape is 512 wide with 11 layers, 8 heads and an MLP of 4 times its "
            "width. It was trained for this study with AdamW and is scored at the running average of its weights. "
            "The 256-wide Transformer has 6 layers and an MLP of 42 times its width and is the reference of every "
            "earlier comparison. The six models received identical windows from random weights. Every row is one "
            "seed.",
        ])
    if warmed:
        stream = {row["model"]: row["scores"]["enwik8"]["mean"]
                  for row in research["groups"][COASTING_HEADS_GROUP]["rows"]}
        with_books = sum("pg19_full" in row["scores"] for row in warmed["rows"])
        with_books = ("no", "one", "two", "three", "four", "five", "six", "all seven")[with_books]
        paragraphs.extend([
            "## State reads and race cells, trained from scratch · 368,640,000 training targets · seed 20001",
            warmed_table(warmed["rows"]),
            "![enwik8 test loss of the Transformer, the new decoders and the v12 decoders]"
            "(../results/figures/event_lm_state_reads.png)",
            "![The decoder: a stream of 1,408 channels through eight blocks, with race cells after four of them and "
            "state reads after three](../results/figures/event_lm_state_reads_circuit.png)",
            "![Averaged validation loss over training: the decoders lead early, then the Transformer pulls away]"
            "(../results/figures/event_lm_state_reads_training.png)",
            "[Circuit, controls, probes and limits](STATE_READS.md) · [Every training arm](STATE_READS_ARMS.md)",
            "This comparison is scored in windows, not in a stream. Every window starts from an empty state and "
            "holds 256 unscored bytes before its 256 scored ones, which is how full attention is scored in every "
            "comparison. The decoders with state reads and race cells run in windows only. They have no stream "
            "state and no native pulse count, so the table has neither column. The two v12 decoders are the "
            "checkpoints of the comparison below, where they carry their state through one stream and score "
            f"{stream['34.85M SNN, surprise clock and read heads']:.5f} and "
            f"{stream['34.85M SNN, surprise clock']:.5f} on enwik8.",
            "The seven models received identical windows from random weights. Rows marked rotated were trained "
            "with the rotated optimizer and are scored at the running average of their weights. PG-19 was scored "
            f"for {with_books} rows, and its ranks count those rows only. Every row is one seed.",
        ])
    for name, group in research["groups"].items():
        rows = group["rows"]
        paragraphs.extend([
            f'## {group["title"]} · {group["training_summary"]}',
            markdown_table(rows, include_ranks=True, include_raw_pg19=True),
        ])
        paragraphs.extend(study_figures.get(name, ()))
        if name == COASTING_HEADS_GROUP:
            paragraphs.append(
                "The four models received identical windows from random weights. The read heads and the fast "
                "synapses carry one stream of any length. Full attention reads 512 bytes. Each recall circuit "
                "uses the strengths its decoder was trained beside. They were fitted on enwik8 validation with "
                "the released v11 decoder, and no row trains on test text."
            )
        if name == SURPRISE_CLOCK_GROUP:
            paragraphs.append(
                "Fast synapses carry one stream of any length. Full attention rows read 512 bytes. "
                "The synapse strengths are fitted on enwik8 validation and no row trains on test text."
            )
        if name == RESIDUAL_SCALE_GROUP:
            gap = group["enwik8_width_gap"]
            paragraphs.extend([
                f"The enwik8 SNN minus Transformer gap is {gap['narrow_bpb']:+.5f} bpb at 14.07M and "
                f"{gap['wide_bpb']:+.5f} bpb at 25.79M. The change is {gap['change_bpb']:+.5f} bpb.",
                "The enwik8-only control uses the same number of targets and the wide SNN geometry. "
                "The other four arms received identical windows from the new mixture.",
            ])
        if name in transfers:
            transfer, labels, figure = transfers[name]
            corpus, screened, words = transfer_tables(transfer, labels)
            paragraphs.extend([
                "### Byte transfer at the same training exposure",
                "Raw corpus loss in bits per byte uses identical 512-byte windows.",
                corpus,
                "Wikipedia windows with zero hits under the training-overlap screen.",
                screened,
                "LAMBADA scores the same final words under both conditioning lengths. Exact match requires a word "
                "boundary. The gold-length columns score predictions through the last answer byte.",
                words,
                f"![Transfer loss and LAMBADA final-word accuracy](../results/figures/{figure})",
                "[Protocol, provenance, and overlap counts](BYTE_TRANSFER.md) · "
                f"[Exact score records](../results/{group['transfer_record']})",
            ])
    paragraphs.extend([
        "![Held-out loss after doubling residual width](../results/figures/event_lm_residual_growth_13030k.png)",
        "![Held-out loss after changing the spike circuits at 6.51M parameters](../results/figures/event_lm_6510k_circuits.png)",
        "![Held-out loss as enwik8 training exposure grows at 6.51M parameters](../results/figures/event_lm_6510k_data.png)",
        "![enwik8 loss gap by prefix age after 65.54M targets](../results/figures/event_lm_65536_prefix_age.png)",
        "![Held-out loss after expanding the delay populations](../results/figures/event_lm_expansion_8740k.png)",
        "![Held-out next-byte loss for the grown 6.51M models](../results/figures/event_lm_grown_6510k.png)",
        "![enwik8 loss gap for the grown pair by prefix age](../results/figures/event_lm_grown_prefix_age.png)",
        "[Growth study](GROWN_RESIDUAL.md) · [Width from scratch](SCALE_6510K_WIDTH.md) · "
        "[Current model card](../models/spiking-byte-lm-v9/README.md) · [Depth growth study](DEPTH_GROWTH.md)",
        "![Held-out next-byte loss for the 512-byte continuations](../results/figures/event_lm_continuation_test.png)",
        "![Held-out loss at both trained context lengths](../results/figures/event_lm_trained_context.png)",
        "[Continuation and scale source records](../results/event_lm/continuation_leaderboard.json) · "
        "[Held-charge replication](HELD_CHARGE_REPLICATION.md)",
        "## Depth and activity inhibition · 26,214,400 training targets · seed 0",
        markdown_table(latest, include_ranks=True),
        "![Held-out loss for matched depth and inhibition models](../results/figures/depth_inhibition_test.png)",
        "![Validation curves and residual spike counts](../results/figures/depth_inhibition.png)",
        "[Evaluation records and ranks](../results/event_lm/depth_inhibition_leaderboard.json) · "
        "[Depth study and ablations](DEPTH_INHIBITION.md) · [Replication](DEPTH_INHIBITION.md#full-attention-replication) · "
        "[v2 model card](../models/spiking-byte-lm-v2/README.md)",
        "## Earlier context-tag and threshold study",
        "Each score below is a mean over three seeds. Each group has its own training exposure.",
    ])
    for scale, rows in data["groups"].items():
        paragraphs.extend([
            f'### {scale.capitalize()} comparison · {rows[0]["training_targets"]:,} training targets per seed',
            markdown_table(rows, include_ranks=True),
        ])
    paragraphs.extend([
        "![Held-out loss by model size and corpus](../results/figures/event_lm_loss.png)",
        "![Loss versus native spike activity for the spiking banks](../results/figures/event_lm_pulse_tradeoff.png)",
        "[Per-seed scores and source paths](../results/event_lm/threshold_scale_leaderboard.json) · "
        "[Interventions and research notes](SPIKE_CODE_FORECAST.md#learned-context-keys-and-firing-thresholds) · "
        "[Model and reproduction guide](../models/spiking-byte-lm/README.md)",
        "Regenerate this page and its figures with `python experiments/plot_event_lm_study.py`.",
    ])
    (ROOT / "docs/LEADERBOARD.md").write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")
    readme = ROOT / "README.md"
    before, marker, remainder = readme.read_text(encoding="utf-8").partition(READ_ME_START)
    _, end, after = remainder.partition(READ_ME_END)
    if not marker or not end:
        raise ValueError("README.md needs both event model leaderboard markers")
    latest_group = research["groups"].get(
        SURPRISE_CLOCK_GROUP, research["groups"].get(RESIDUAL_SCALE_GROUP, research["groups"][ARRIVAL_GROUP]),
    )
    table = markdown_table(latest_group["rows"],
                           include_ranks=False, include_raw_pg19=True)
    readme.write_text(f"{before}{marker}\n\n{table}\n\n{end}{after}", encoding="utf-8")


def plot_loss(data, output):
    fig, axes = plt.subplots(2, 2, sharey="row", figsize=(11.8, 7.4))
    fig.subplots_adjust(left=0.194, right=0.97, bottom=0.112, top=0.809, hspace=0.49, wspace=0.20)
    heading(fig, "Held-out next-byte loss",
            "Three matched seeds per model · enwik8 and 16 complete PG-19 books")
    for index, rows in enumerate(data["groups"].values()):
        params = f'{rows[0]["parameters"] / 1e3:.0f}k'
        for col, (corpus, name, _) in enumerate(CORPORA):
            ax = axes[index, col]
            style(ax, grid_axis="x")
            ax.set_title(f"About {params} parameters · {name}", fontsize=11, loc="left", color=TEXT)
            for y, row in enumerate(rows):
                metric = row["scores"][corpus]
                color = COLORS[row["arm"]]
                values = metric["by_seed"]
                ax.hlines(y, min(values), max(values), color=color, linewidth=1.8, alpha=0.55)
                ax.scatter(values, [y] * len(values), color=color, s=11, alpha=0.65, zorder=3)
                ax.scatter(metric["mean"], y, color=color, s=41, edgecolors="white", linewidths=0.9, zorder=4)
            ax.set_xlabel("Bits per byte", color=MUTED, labelpad=7)
            ax.set_yticks(range(len(rows)), [LABELS[row["arm"]][1] for row in rows])
            ax.set_ylim(len(rows) - 0.45, -0.55)
            ax.margins(x=0.10)
    fig.text(0.035, 0.033, "Large points are means. Lines span the seeds. PG-19 scores use validation temperatures.",
             color=MUTED, size=9)
    save(fig, output, "event_lm_loss.png")


def plot_pulses(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
    fig.subplots_adjust(left=0.083, right=0.973, bottom=0.24, top=0.745, wspace=0.28)
    heading(fig, "Loss and spike activity",
            "Spiking banks · count audit on 4,096 held-out enwik8 bytes · three-seed means")
    for ax, (corpus, name, _) in zip(axes, CORPORA, strict=True):
        style(ax)
        ax.set_title(name, color=TEXT, size=11, loc="left")
        ax.set_xlabel("Counted pulses per byte, thousands", color=MUTED, labelpad=8)
        ax.set_ylabel("Bits per byte", color=MUTED, labelpad=8)
        ax.axvline(310, color=GRAY, linestyle="--", linewidth=1.2)
        for scale, rows in data["groups"].items():
            for row in rows:
                if row["pulses_per_byte"] is None:
                    continue
                activity = row["pulses_per_byte"]["mean"] / 1000
                loss = row["scores"][corpus]["mean"]
                ax.scatter(activity, loss, color=COLORS[row["arm"]], marker="o" if scale == "small" else "D",
                           s=59, edgecolors="white", linewidths=0.8, zorder=3)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker="o", linestyle="none", color=COLORS[arm], label=LABELS[arm][1])
               for arm in ARMS["small"] if arm not in ("transformer_local", "full")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.065), ncol=4, frameon=False,
               labelcolor=TEXT, fontsize=9)
    fig.text(0.035, 0.032, "Circles mark small models. Diamonds mark large models. The dashed line marks the training budget.",
             color=MUTED, size=9)
    save(fig, output, "event_lm_pulse_tradeoff.png")


def plot_heldout(rows, output, *, title, subtitle, filename, precision=3, height=4.7):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, height), sharey=True)
    fig.subplots_adjust(left=0.32, right=0.973, bottom=0.22, top=0.756, wspace=0.18)
    heading(fig, title, subtitle)
    for ax, (corpus, name, _) in zip(axes, CORPORA, strict=True):
        style(ax, grid_axis="x")
        ax.set_title(name, fontsize=11, loc="left", color=TEXT)
        ax.set_xlabel("Bits per byte", color=MUTED, labelpad=8)
        for y, row in enumerate(rows):
            bpb = row["scores"][corpus]["mean"]
            ax.scatter(bpb, y, color=row["color"], s=44, edgecolors="white", linewidth=0.8, zorder=3)
            ax.annotate(f"{bpb:.{precision}f}", (bpb, y), xytext=(7, 0), textcoords="offset points",
                        va="center", size=8.5, color=TEXT)
        ax.set_yticks(range(len(rows)), [row["model"] for row in rows])
        ax.set_ylim(len(rows) - 0.45, -0.55)
        ax.margins(x=0.25)
    fig.text(0.035, 0.043, "Lower is better. PG-19 uses temperatures fitted on separate validation books.",
             color=MUTED, size=9)
    save(fig, output, filename)


def plot_state(rows, labels, output, *, title, subtitle, filename):
    """Held-out loss against the persistent state of one stream, with one panel per corpus."""
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.3))
    fig.subplots_adjust(left=0.075, right=0.973, bottom=0.215, top=0.775, wspace=0.2)
    heading(fig, title, subtitle)
    ticks = {2 ** 15: "32 KiB", 2 ** 17: "128 KiB", 2 ** 19: "512 KiB", 2 ** 21: "2 MiB", 2 ** 23: "8 MiB"}
    largest = max(row["state_bytes"] for row in rows)
    for ax, (corpus, name, _) in zip(axes, CORPORA, strict=True):
        style(ax, grid_axis="both")
        ax.set_title(name, fontsize=11, loc="left", color=TEXT)
        ax.set_xscale("log")
        ax.set_xlabel("State per stream", color=MUTED, labelpad=8)
        ax.set_ylabel("Bits per byte", color=MUTED, labelpad=8)
        for row, label in zip(rows, labels, strict=True):
            state, bpb = row["state_bytes"], row["scores"][corpus]["mean"]
            last = state == largest
            ax.scatter(state, bpb, color=row["color"], s=62, edgecolors="white", linewidth=0.9, zorder=3)
            ax.annotate(label, (state, bpb), xytext=(-10 if last else 10, 6), textcoords="offset points",
                        ha="right" if last else "left", va="center", size=9.5, color=TEXT)
            ax.annotate(f"{bpb:.5f} · {state_size(state)}", (state, bpb), xytext=(-10 if last else 10, -7),
                        textcoords="offset points", ha="right" if last else "left", va="center", size=8.5,
                        color=MUTED)
        ax.set_xticks(list(ticks), list(ticks.values()))
        ax.minorticks_off()
        ax.set_xlim(2 ** 15 * 0.8, 2 ** 23 * 1.25)
        ax.margins(y=0.14)
    fig.text(0.035, 0.043, "Lower is better. The state axis is logarithmic. Full attention stores FP32 keys and "
             "values for its 512-byte context.", color=MUTED, size=9)
    save(fig, output, filename)


def plot_residual_emissions(output):
    audits = [read_result(f"{DEPTH_GROWTH}/{name}.json") for name in (
        "parent_residual_emissions", "native_depth_s4603", "native_corrections_s4603",
    )]
    if len({(audit["input_sha256"], audit["input_bytes"], audit["split"]) for audit in audits}) != 1:
        raise ValueError("residual emission audits must use the same input")
    parent, depth, corrections = (audit["models"][0] for audit in audits)
    labels = ["Seven-block parent", "Trained 11-block SNN", "Seven-block correction SNN"]
    ordinary = [parent["total_pulses_per_byte"], depth["ordinary_residual_pulses_per_byte"],
                corrections["ordinary_residual_pulses_per_byte"]]
    signed = [parent["all_residual_corrections_pulses_per_byte"], depth["total_pulses_per_byte"],
              corrections["total_pulses_per_byte"]]

    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    fig.subplots_adjust(left=0.25, right=0.96, bottom=0.28, top=0.76)
    heading(fig, "Generated events with the same output counts",
            "Same checkpoint weights within each row · native audit on 4,096 enwik8 test bytes")
    style(ax, grid_axis="x")
    rows = range(len(labels))
    ax.hlines(rows, signed, ordinary, color=GRAY, alpha=0.45)
    for values, name, color in ((ordinary, "Ordinary re-emission", GRAY), (signed, "Signed correction", VIOLET)):
        ax.scatter(values, rows, color=color, s=45, zorder=3, label=name)
        for row, value in zip(rows, values, strict=True):
            ax.annotate(f"{value:,.0f}", (value, row), xytext=(0, 9), textcoords="offset points",
                        ha="center", color=TEXT, fontsize=9)
    ax.set_yticks(rows, labels)
    ax.set_ylim(len(labels) - 0.48, -0.57)
    ax.margins(x=0.14)
    ax.xaxis.set_major_formatter("{x:,.0f}")
    ax.set_xlabel("Generated pulses per byte", color=MUTED, labelpad=7)
    fig.legend(loc="lower center", bbox_to_anchor=(0.6, 0.09), ncol=2, frameon=False, labelcolor=TEXT)
    fig.text(0.036, 0.043, "Forwarded pulses still drive downstream synapses.", color=MUTED, size=9)
    save(fig, output, "event_lm_residual_emissions.png")


def plot_trace_tap_training(rows, output):
    fig, (loss, activity) = plt.subplots(1, 2, figsize=(11.6, 4.65))
    fig.subplots_adjust(left=0.1, right=0.966, bottom=0.27, top=0.755, wspace=0.34)
    heading(fig, "Training the tap read circuits", "32,768 shared enwik8 validation targets per evaluation · seed 6899")
    for ax, title, ylabel in ((loss, "Validation loss", "Bits per byte"),
                              (activity, "Validation activity", "Thousands of pulses per byte")):
        style(ax)
        ax.set_title(title, loc="left", color=TEXT, size=11)
        ax.set_xlabel("Additional training targets, millions", color=MUTED, labelpad=7)
        ax.set_ylabel(ylabel, color=MUTED, labelpad=7)
    for row in rows:
        curve = read_result(row["training_record"])["curve"]
        x = [point["tokens"] / 1_000_000 for point in curve]
        line = dict(color=row["color"], marker="o", markersize=3.5, linewidth=1.45)
        loss.plot(x, [point["val_bpb"] for point in curve], label=row["model"], **line)
        if row["pulses_per_byte"] is not None:
            activity.plot(x, [point["val_pulses_per_byte"] / 1000 for point in curve], **line)
    activity.axhline(310, color=GRAY, linestyle="--", linewidth=1)
    handles, labels = loss.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.075),
               ncol=4, frameon=False, fontsize=9, labelcolor=TEXT)
    fig.text(0.035, 0.034, "The dashed line marks the SNN training target.", color=MUTED, size=9)
    save(fig, output, "event_lm_trace_tap_training.png")


def plot_validation_history(output):
    record = read_result(f"{TRACE_INTEGRATION}/v6_validation_history.json")
    snn, attention = (model["histories"] for model in record["models"])
    histories = [row["history_bytes"] for row in snn]
    if histories != [row["history_bytes"] for row in attention]:
        raise ValueError("history comparisons must use matching warmup lengths")
    snn_loss = [row["bpb"] for row in snn]
    attention_loss = [row["bpb"] for row in attention]

    fig, (score, gap) = plt.subplots(1, 2, figsize=(11.3, 4.6))
    fig.subplots_adjust(left=0.09, right=0.967, bottom=0.21, top=0.76, wspace=0.31)
    heading(fig, "Next-byte loss by available history",
            f'{record["target_bytes"]:,} matched enwik8 validation targets · 16-byte tails')
    for ax in (score, gap):
        style(ax)
        ax.set_xscale("log", base=2)
        ax.set_xticks(histories, [str(history) for history in histories])
        ax.set_xlabel("Warmup bytes before each scored tail", color=MUTED, labelpad=8)
    score.set_title("Held-out loss", loc="left", color=TEXT, fontsize=11)
    score.set_ylabel("Bits per byte", color=MUTED, labelpad=7)
    score.plot(histories, snn_loss, "o-", color=VIOLET, label="11-block SNN")
    score.plot(histories, attention_loss, "o-", color=ORANGE, label="Full attention")
    score.legend(frameon=False, labelcolor=TEXT)
    gap.set_title("SNN minus full attention", loc="left", color=TEXT, fontsize=11)
    gap.set_ylabel("Additional bits per byte", color=MUTED, labelpad=7)
    gap.plot(histories, [a - b for a, b in zip(snn_loss, attention_loss, strict=True)], "o-", color=BLUE)
    gap.set_ylim(0, None)
    save(fig, output, "event_lm_validation_history.png")


def plot_group_thresholds(output):
    audit = read_result(f"{TRACE_INTEGRATION}/native_fine_thresholds_s5801.json")["models"][0]
    base = audit["inhibition_threshold_units_per_base"]
    offsets = [[units - base for units in row] for row in audit["inhibition_threshold_quanta_by_block"]]
    values = [offset for row in offsets for offset in row]
    low, high = min(0, min(values)), max(1, max(values))

    fig, ax = plt.subplots(figsize=(8.5, 5.7))
    fig.subplots_adjust(left=0.13, right=0.94, bottom=0.18, top=0.77)
    heading(fig, "Learned firing thresholds by delay group",
            f"{sum(value != 0 for value in values)} of {len(values)} groups changed at one-tick resolution")
    style(ax, grid_axis="x")
    ax.grid(False)
    heat = ax.imshow(offsets, vmin=low, vmax=high, cmap="Blues", aspect="auto")
    for block, row in enumerate(offsets):
        for group, value in enumerate(row):
            color = "white" if value > low + 0.62 * (high - low) else TEXT
            ax.text(group, block, f"{value:+d}" if value else "0", ha="center", va="center", color=color, fontsize=9)
    ax.set_yticks(range(len(offsets)), range(1, len(offsets) + 1))
    ax.set_xticks(range(len(offsets[0])), range(1, len(offsets[0]) + 1))
    ax.set_xlabel("Delay group", color=MUTED, labelpad=7)
    ax.set_ylabel("Spiking block", color=MUTED, labelpad=7)
    colorbar = fig.colorbar(heat, ax=ax, pad=0.04)
    colorbar.set_label("Offset in charge ticks", color=MUTED)
    fig.text(0.035, 0.045, "Offsets apply before spikes from the shared inhibitory pool arrive.",
             color=MUTED, size=9)
    save(fig, output, "event_lm_fine_group_thresholds.png")


def plot_packed_width_history(output):
    parent, control, wide = (read_result(f"{PACKED_WIDTH}/{name}_validation_history.json")
                             for name in ("parent", "control", "wide"))
    for key in ("data_sha1", "split", "target_bytes", "tail_bytes", "tail_starts"):
        if parent[key] != control[key] or parent[key] != wide[key]:
            raise ValueError("all checkpoints must score the same validation tails")
    fig, ax = plt.subplots(figsize=(9.6, 4.45))
    fig.subplots_adjust(left=0.11, right=0.966, bottom=0.21, top=0.75)
    heading(fig, "The enwik8 gap grows with context",
            f'{parent["target_bytes"]:,} matched validation targets · {parent["tail_bytes"]}-byte scored tails')
    style(ax)
    pairs = ((parent, "9.96M v7 pair · 101.58M targets", GRAY, "^--"),
             (control, "9.96M pair · 111.41M targets", BLUE, "s:"),
             (wide, "12.8M pair · 111.41M targets", VIOLET, "o-"))
    histories = [row["history_bytes"] for row in parent["models"][0]["histories"]]
    for record, name, color, line in pairs:
        snn, attention = (model["histories"] for model in record["models"])
        if any(histories != [row["history_bytes"] for row in model] for model in (snn, attention)):
            raise ValueError("history comparisons must use matching warmup lengths")
        gap = [spike["bpb"] - full["bpb"] for spike, full in zip(snn, attention, strict=True)]
        ax.plot(histories, gap, line, color=color, label=name, markerfacecolor="white")
    ax.set_xscale("log", base=2)
    ax.set_xticks(histories, [str(history) for history in histories])
    ax.set_xlabel("Warmup bytes before each scored tail", color=MUTED, labelpad=8)
    ax.set_ylabel("SNN loss minus full attention, bits per byte", color=MUTED, labelpad=7)
    ax.axhline(0, linewidth=0.9, color=GRAY, alpha=0.7)
    ax.legend(frameon=False, labelcolor=TEXT)
    save(fig, output, "event_lm_packed_width_history.png")


def plot_packed_circuit_changes(baseline, variants, output):
    panels = (("enwik8", "mean", "enwik8"),
              ("pg19_full", "raw_bpb", "PG-19 raw"),
              ("pg19_full", "mean", "PG-19 calibrated"))
    changes = [[row["scores"][corpus][metric] - baseline["scores"][corpus][metric] for row in variants]
               for corpus, metric, _ in panels]
    limit = max(0.001, 1.75 * max(abs(delta) for panel in changes for delta in panel))
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(11.7, 4.25))
    fig.subplots_adjust(left=0.245, right=0.968, bottom=0.215, top=0.745, wspace=0.15)
    heading(fig, "Loss changes from layout and steering",
            "Compared with the even-width SNN · 111.41M enwik8 training targets · seed 7907")
    for ax, (_, _, title), deltas in zip(axes, panels, changes, strict=True):
        style(ax, grid_axis="x")
        ax.set_title(title, loc="left", size=11, color=TEXT)
        ax.axvline(0, color=GRAY, linewidth=1)
        for index, (row, delta) in enumerate(zip(variants, deltas, strict=True)):
            ax.hlines(index, 0, delta, color=row["color"], linewidth=2, alpha=0.52)
            ax.scatter(delta, index, color=row["color"], s=41, edgecolors="white", linewidth=0.8, zorder=3)
            offset, align = (6, "left") if delta >= 0 else (-6, "right")
            ax.annotate(f"{delta:+.5f}", (delta, index), xytext=(offset, 0), textcoords="offset points",
                        ha=align, va="center", size=8.5, color=TEXT)
        ax.set_xlim(-limit, limit)
    axes[0].set_yticks(range(len(variants)), [row["model"] for row in variants])
    axes[0].set_ylim(len(variants) - 0.48, -0.52)
    fig.text(0.035, 0.045, "Values are differences in bits per byte. Negative values indicate lower loss.",
             color=MUTED, size=9)
    save(fig, output, "event_lm_packed_circuit_changes.png")


def plot_read_steering(output):
    headroom = read_result(f"{TRACE_READ_DRIVE}/write_carry_validation_headroom.json")["totals"]
    signed = read_result(f"{TRACE_READ_DRIVE}/native_frozen_write_share_s8237.json")
    equal = read_result(f"{TRACE_READ_DRIVE}/native_frozen_write_equal_rails_share_s8237.json")
    for key in ("input_sha256", "input_bytes", "input_offset", "split"):
        if signed[key] != equal[key]:
            raise ValueError("frozen read steering audits must use the same validation bytes")
    carried, equal_rails = equal["models"]
    reach = [100 * headroom[key] / headroom["blocked_ticks"] for key in (
        "pair_recoverable_ticks", "four_cell_upper_bound_ticks", "delay_group_upper_bound_ticks",
    )]
    changes = [sum(row["trace_read_changed_counts_per_byte_by_block"]) for row in (
        carried, signed["models"][0], equal_rails,
    )]
    panels = (
        ("Spare steering · 16,384 bytes", ("Adjacent pairs", "Blocks of four", "Full delay groups"),
         reach, "Percent of blocked charge", ".1f", "%"),
        ("Frozen read counts · 4,096 bytes", ("Original carry", "Signed sharing", "Equal gains on both rails"),
         changes, "Absolute count change per byte", ".3f", ""),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    fig.subplots_adjust(left=0.17, right=0.972, bottom=0.215, top=0.747, wspace=0.61)
    heading(fig, "Spare steering and write polarity", "Native counts on enwik8 validation text")
    for ax, (title, labels, values, xlabel, precision, unit) in zip(axes, panels, strict=True):
        style(ax, grid_axis="x")
        ax.set_title(title, fontsize=11, loc="left", color=TEXT)
        ax.barh(range(3), values, color=[BLUE, VIOLET, "#138372"], height=0.43)
        for index, value in enumerate(values):
            ax.annotate(f"{value:{precision}}{unit}", (value, index), xytext=(5, 0),
                        textcoords="offset points", va="center", size=9, color=TEXT)
        ax.set_yticks(range(3), labels)
        ax.set_ylim(2.55, -0.55)
        ax.set_xlim(0, 1.24 * max(values))
        ax.set_xlabel(xlabel, color=MUTED, labelpad=7)
    axes[0].set_xticks(range(0, 101, 25))
    fig.text(0.035, 0.045, "Four-cell and delay-group values are capacity bounds. Frozen variants use one trained checkpoint.",
             color=MUTED, size=9)
    save(fig, output, "event_lm_read_steering.png")


def plot_trained_context(research, output):
    short = research["groups"]["continuation_512"]["rows"]
    long = research["groups"]["continuation_2048"]["rows"]
    pairs = (
        ("Held charge with write credit", BLUE, next(row for row in short if row["model"] == "Chronological write credit"), long[0]),
        ("Full-attention Transformer", ORANGE, short[-1], long[-1]),
    )
    plot_paired_change(pairs, output, title="Changing the trained context length",
                       subtitle="1.63M parameters · same starting weights within each model family · seed 711",
                       ticks=("512 bytes", "2,048 bytes"),
                       note="Each arm uses 6.55M additional enwik8 targets. PG-19 uses validation-fitted temperatures.",
                       filename="event_lm_trained_context.png")


def plot_training_exposure(research, output):
    base = research["groups"]["grown_6510k"]["rows"]
    extended = research["groups"]["data_6510k"]["rows"]
    latest = research["groups"]["data_6510k_65536"]["rows"]
    pairs = (
        ("Six-bit grown SNN", BLUE, base[1], extended[0], latest[0]),
        ("Full-attention Transformer", ORANGE, base[-1], extended[-1], latest[-1]),
    )
    gaps = [snn["scores"]["enwik8"]["mean"] - attention["scores"]["enwik8"]["mean"]
            for snn, attention in zip(pairs[0][2:], pairs[1][2:], strict=True)]
    plot_paired_change(pairs, output, title="More training at fixed model size",
                       subtitle="About 6.51M parameters · 512-byte training windows · matched enwik8 exposure",
                       ticks=("32.77M targets", "49.15M targets", "65.54M targets"),
                       note="enwik8 gap in bits per byte at each stage · " + " → ".join(f"{gap:.5f}" for gap in gaps),
                       filename="event_lm_6510k_data.png")


def plot_paired_change(pairs, output, *, title, subtitle, ticks, note, filename):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    fig.subplots_adjust(left=0.08, right=0.965, bottom=0.24, top=0.755, wspace=0.24)
    heading(fig, title, subtitle)
    positions = range(len(ticks))
    for ax, (corpus, name, _) in zip(axes, CORPORA, strict=True):
        style(ax)
        ax.set_title(name, loc="left", size=11, color=TEXT)
        ax.set_ylabel("Bits per byte", color=MUTED, labelpad=7)
        ax.set_xticks(positions, ticks)
        ax.set_xlim(-0.25, len(ticks) - 0.75)
        ax.margins(y=0.22)
        for label, color, *observations in pairs:
            scores = [row["scores"][corpus]["mean"] for row in observations]
            ax.plot(positions, scores, color=color, marker="o", linewidth=1.8, label=label)
            for x, score in zip(positions, scores, strict=True):
                ax.annotate(f"{score:.5f}", (x, score), xytext=(0, 9), textcoords="offset points",
                            ha="center", color=TEXT, size=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.079), ncol=2,
               frameon=False, labelcolor=TEXT, fontsize=9)
    fig.text(0.035, 0.045, note, color=MUTED, size=9)
    save(fig, output, filename)


def plot_prefix_age_gap(spiking_source, attention_source, output, *, subtitle, filename):
    spiking, attention = (read_result(source)["models"][0] for source in (spiking_source, attention_source))
    ages = ("unseen", "1-31", "32-127", "128-511", "512+")
    labels = ("First occurrence", "1–31 bytes", "32–127 bytes", "128–511 bytes", "512+ bytes")
    gaps, contributions = [], []
    for age in ages:
        snn, full = (row["prefix_age"][age] for row in (spiking, attention))
        if snn["target_bytes"] != full["target_bytes"]:
            raise ValueError(f"prefix age {age} contains different test target counts")
        gap = snn["bpb"] - full["bpb"]
        gaps.append(gap)
        contributions.append(gap * snn["target_bytes"] / spiking["target_bytes"])

    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.35), sharey=True)
    fig.subplots_adjust(left=0.20, right=0.96, bottom=0.215, top=0.76, wspace=0.28)
    heading(fig, "Where the enwik8 loss gap accumulates", subtitle)
    for ax, values, title in zip(axes, (gaps, contributions),
                                 ("Gap within prefix group", "Contribution to total gap"), strict=True):
        style(ax, grid_axis="x")
        ax.set_title(title, loc="left", size=11, color=TEXT)
        bars = ax.barh(range(len(ages)), values, color=[BLUE, GRAY, GRAY, GRAY, BLUE], height=0.53)
        ax.bar_label(bars, fmt="%.3f", color=TEXT, size=8.5, padding=4)
        ax.set_yticks(range(len(ages)), labels)
        ax.set_ylim(len(ages) - 0.5, -0.5)
        ax.set_xlim(0, 1.2 * max(values))
        ax.set_xlabel("Additional bits per byte for the SNN", color=MUTED, labelpad=7)
    fraction = (contributions[0] + contributions[-1]) / sum(contributions)
    fig.text(0.035, 0.045, f"First occurrences and repeats at 512+ bytes account for {fraction:.1%} of the total gap.",
             color=MUTED, size=9)
    save(fig, output, filename)


def plot_arrival_early_context(record, run_labels, output):
    reference, *arms = record["models"]
    if record["split"] != "test" or reference["checkpoint"] != record["reference"]:
        raise ValueError("the early-context chart expects a paired enwik8 test record")
    fig, ax = plt.subplots(figsize=(9.4, 4.85))
    fig.subplots_adjust(left=0.11, right=0.96, bottom=0.245, top=0.735)
    heading(fig, "The first 32 predictions after reset",
            f"{record['target_bytes']:,} paired enwik8 test targets · 121.24M training targets · seed 9349")
    style(ax)
    ages = [(1, 4), (5, 8), (9, 16), (17, 24), (25, 32)]
    centers = [(first + last) / 2 for first, last in ages]
    ax.axhline(0, color=MUTED, linewidth=0.85)
    for model, offset, marker in zip(arms, (-0.30, 0, 0.30), ("s", "^", "o"), strict=True):
        label, color, _ = run_labels[model["run_name"]]
        bins = {tuple(row["context_bytes"]): row for row in model["ages"]}
        gaps = [bins[age]["delta_vs_reference_bpb"] for age in ages]
        bounds = [bins[age]["paired_95_bpb"] for age in ages]
        errors = [[max(0, gap - interval[0]) for gap, interval in zip(gaps, bounds, strict=True)],
                  [max(0, interval[1] - gap) for gap, interval in zip(gaps, bounds, strict=True)]]
        positions = [center + offset for center in centers]
        ax.errorbar(positions, gaps, yerr=errors, color=color, marker=marker, markersize=4,
                    linewidth=1.5, capsize=3, label=label)
    ax.set_xticks(centers, [f"{a}–{b}" for a, b in ages])
    ax.set_xlabel("Bytes observed", color=MUTED)
    ax.set_ylabel("Loss relative to full attention (bpb)", color=MUTED)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False, ncols=3, labelcolor=TEXT)
    fig.text(0.035, 0.035, "Whiskers show 95% intervals from resampling paired windows. The reference is at zero.",
             color=MUTED, size=9)
    save(fig, output, "event_lm_arrival_early_context.png")


def labels_for_runs(group, record):
    run_labels = {Path(row["training_record"]).stem: (row["model"], row["color"], row["training_targets"])
                  for row in group["rows"]}
    models = record["models"]
    if (set(run_labels) != {model["run_name"] for model in models}
            or len(models) != len(run_labels)
            or any(run_labels[model["run_name"]][2] != model["training_targets"] for model in models)):
        raise ValueError("the report must score the matched training runs")
    return run_labels


def read_transfers(research):
    reports = {}
    for name, group in research["groups"].items():
        if path := group.get("transfer_record"):
            transfer = read_transfer(RESULTS / path)
            if transfer["models"][0]["attention"] != "transformer":
                raise ValueError(f"{path} must report the full-attention reference first")
            runs = labels_for_runs(group, transfer)
            labels = [runs[model["run_name"]][0] for model in transfer["models"]]
            filename = f"event_lm_byte_transfer_{Path(path).stem}.png"
            reports[name] = (transfer, labels, filename)
    return reports


def read_early_context(group, transfer, *, checkpoint_catalogue, reference):
    catalog = (RESULTS / checkpoint_catalogue).read_text().splitlines()
    checkpoints = dict(line.split("  ", 1)[::-1] for line in catalog)
    for model in transfer["models"]:
        if checkpoints[model["checkpoint"]] != model["checkpoint_sha256"]:
            raise ValueError("the transfer record must score the catalogued checkpoints")
    records = {split: read_result(path) for split, path in group["early_context_records"].items()}
    for split, record in records.items():
        run_labels = labels_for_runs(group, record)
        if (record["split"] != split or run_labels[record["models"][0]["run_name"]][0] != reference
                or any(checkpoints[model["checkpoint"]] != model["sha256"] for model in record["models"])):
            raise ValueError("the early-context record must score the catalogued checkpoints")
    return records["test"], run_labels


def plot_residual_scale(group, early, output):
    narrow, wide, _, narrow_full, wide_full = group["rows"]
    gap = group["enwik8_width_gap"]
    pairs = (("SNN", VIOLET, narrow, wide), ("Full attention", ORANGE, narrow_full, wide_full))
    plot_paired_change(pairs, output, title="Widening both byte models",
                       subtitle="131.07M training targets · identical expanded mixture · seed 9631",
                       ticks=("14.07M parameters", "25.79M parameters"),
                       note=f"enwik8 gap change {gap['change_bpb']:+.5f} bpb. The SNN uses 28,584 B at both widths.",
                       filename="event_lm_residual_scale.png")
    plot_early_width_gap(early, output)


def plot_early_width_gap(early, output):
    rows = early["width_gap"]["ages"][:5]
    values = [row["gap_change_bpb"] for row in rows]
    intervals = [row["paired_95_bpb"] for row in rows]
    lower = [max(0, gap - ci[0]) for gap, ci in zip(values, intervals, strict=True)]
    upper = [max(0, ci[1] - gap) for gap, ci in zip(values, intervals, strict=True)]
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.24, top=0.725)
    heading(fig, "How width changes the enwik8 gap",
            f"{early['target_bytes']:,} paired targets after a reset · seed 9631")
    style(ax)
    ax.axhline(0, color=MUTED, linewidth=0.9)
    ax.errorbar(range(len(rows)), values, yerr=(lower, upper), color=VIOLET, linewidth=1.7, marker="o", capsize=4)
    ax.set_xticks(range(len(rows)), [f"{lo}–{hi}" for lo, hi in (row["context_bytes"] for row in rows)])
    ax.set_xlabel("Bytes observed", color=MUTED)
    ax.set_ylabel("Wide gap minus narrow gap (bpb)", color=MUTED)
    fig.text(0.035, 0.04, "Negative changes close the gap. Whiskers show 95% intervals from paired windows.",
             color=MUTED, size=9)
    save(fig, output, "event_lm_residual_scale_early_context.png")


def main():
    data = build()
    depth = build_depth()
    research = build_research()
    transfers = read_transfers(research)
    early, run_labels = read_early_context(
        research["groups"][ARRIVAL_GROUP], transfers[ARRIVAL_GROUP][0],
        checkpoint_catalogue="event_lm/trace_arrival_checkpoints.sha256", reference="Full attention",
    )
    scale = research["groups"].get(RESIDUAL_SCALE_GROUP)
    if scale:
        scale_early, _ = read_early_context(
            scale, transfers[RESIDUAL_SCALE_GROUP][0],
            checkpoint_catalogue=RESIDUAL_SCALE_CHECKPOINTS, reference="25.79M full attention",
        )
        planned = ("wide-snn_s9631.pt", "wide-transformer_s9631.pt", "snn_s9631.pt", "transformer_s9631.pt")
        keys = ("wide_snn", "wide_transformer", "narrow_snn", "narrow_transformer")
        if tuple(Path(scale_early["width_gap"][key]).name for key in keys) != planned:
            raise ValueError("the enwik8 width gap must use the four mixed training arms")
    summary = RESULTS / "event_lm/threshold_scale_leaderboard.json"
    summary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    summary = RESULTS / "event_lm/depth_inhibition_leaderboard.json"
    summary.write_text(json.dumps(depth, indent=2) + "\n", encoding="utf-8")
    summary = RESULTS / "event_lm/continuation_leaderboard.json"
    summary.write_text(json.dumps(research, indent=2) + "\n", encoding="utf-8")
    warmed = read_warmed_windows() if (RESULTS / STATE_READS_CHECKPOINTS).is_file() else None
    if warmed:
        summary = RESULTS / STATE_READS / "leaderboard.json"
        summary.write_text(json.dumps(warmed, indent=2) + "\n", encoding="utf-8")
    clocks = (read_warmed_windows(READ_CLOCKS_ARMS, READ_CLOCKS)
              if (RESULTS / READ_CLOCKS_CHECKPOINTS).is_file() else None)
    if clocks:
        summary = RESULTS / READ_CLOCKS / "leaderboard.json"
        summary.write_text(json.dumps(clocks, indent=2) + "\n", encoding="utf-8")
    streams = read_stream_form() if (RESULTS / STREAM_FORM_CHECKPOINTS).is_file() else None
    if streams:
        summary = RESULTS / STREAM_FORM / "leaderboard.json"
        summary.write_text(json.dumps(streams, indent=2) + "\n", encoding="utf-8")
    long_streams = read_long_streams() if (RESULTS / LONG_STREAMS_CHECKPOINTS).is_file() else None
    if long_streams:
        summary = RESULTS / LONG_STREAMS / "leaderboard.json"
        summary.write_text(json.dumps(long_streams, indent=2) + "\n", encoding="utf-8")
    knowledge_board = RESULTS / "event_lm/knowledge_gap/leaderboard.json"
    knowledge_gap = json.loads(knowledge_board.read_text()) if knowledge_board.is_file() else None
    write_leaderboards(data, depth, research, transfers, warmed, clocks, streams, long_streams, knowledge_gap)
    output = RESULTS / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plot_loss(data, output)
    plot_pulses(data, output)
    plot_heldout(depth["groups"]["depth"], output, title="Held-out loss after increasing spiking depth",
                 subtitle="1.63M parameters · 26.2M enwik8 training targets · seed 0",
                 filename="depth_inhibition_test.png")
    plot_heldout(research["groups"]["continuation_512"]["rows"], output, title="Held-out loss after 512-byte continuations",
                 subtitle="About 1.63M parameters · 32.77M total enwik8 training targets · seed 711",
                 filename="event_lm_continuation_test.png", precision=5)
    plot_heldout(research["groups"]["grown_6510k"]["rows"], output, title="Held-out loss after growing both models",
                 subtitle="About 6.51M parameters · 32.77M total enwik8 training targets · seed 711",
                 filename="event_lm_grown_6510k.png", precision=5)
    plot_heldout(research["groups"]["expansion_8740k"]["rows"], output, title="Growing the content and steering neurons",
                 subtitle="8.74M expanded models and 6.51M control · 55.71M enwik8 training targets · seed 1801",
                 filename="event_lm_expansion_8740k.png", precision=5)
    plot_heldout(research["groups"]["residual_growth_13030k"]["rows"], output, title="Doubling residual width at fixed stream state",
                 subtitle="13.03M matched pair and 6.51M controls · 72.09M enwik8 training targets · seed 3307",
                 filename="event_lm_residual_growth_13030k.png", precision=5)
    plot_heldout(research["groups"]["circuits_6510k"]["rows"], output, title="Spike circuit and charge comparisons",
                 subtitle="About 6.51M parameters · 72.09M enwik8 training targets · seed 3307",
                 filename="event_lm_6510k_circuits.png", precision=5)
    plot_heldout(research["groups"]["growth_9960k"]["rows"], output, title="Growing spiking depth or width",
                 subtitle="About 9.96M parameters · 81.92M enwik8 training targets · seed 4603",
                 filename="event_lm_growth_9960k.png", precision=5)
    plot_heldout(research["groups"]["residual_output_6510k"]["rows"], output, title="Emitting signed corrections",
                 subtitle="About 6.51M parameters · 81.92M enwik8 training targets · seed 4603",
                 filename="event_lm_residual_output.png", precision=5)
    plot_residual_emissions(output)
    plot_validation_history(output)
    plot_group_thresholds(output)
    plot_heldout(research["groups"]["trace_integration_91750k"]["rows"], output,
                 title="Integrating charge and learning group thresholds",
                 subtitle="About 9.96M parameters · 91.75M enwik8 training targets · seed 5801",
                 filename="event_lm_trace_integration.png", precision=5)
    tap_rows = research["groups"]["trace_tap_101580k"]["rows"]
    plot_heldout(tap_rows, output, title="Reading the axons by spike timing",
                 subtitle="About 9.96M parameters · 101.58M enwik8 training targets · seed 6899",
                 filename="event_lm_trace_tap_read.png", precision=5)
    plot_trace_tap_training(tap_rows, output)
    read_rows = research["groups"]["trace_read_drive_116326k"]["rows"]
    plot_heldout(read_rows, output, title="Held-charge read circuits at matched exposure",
                 subtitle="About 12.8M parameters · 116.33M enwik8 training targets · seed 8237",
                 filename="event_lm_trace_read_drive.png", precision=5,
                 height=4.7 + 0.32 * (len(read_rows) - 4))
    plot_read_steering(output)
    plot_heldout(research["groups"][ARRIVAL_GROUP]["rows"], output, title="Scheduled spike arrivals",
                 subtitle="12.8–14.1M parameters · 121.24M enwik8 training targets · seed 9349",
                 filename="event_lm_trace_arrival.png", precision=5)
    plot_arrival_early_context(early, run_labels, output)
    if scale:
        plot_residual_scale(scale, scale_early, output)
    heads = research["groups"].get(COASTING_HEADS_GROUP)
    if heads:
        plot_heldout(heads["rows"], output, title="Coasting read heads, trained from scratch",
                     subtitle="368.64M training targets · identical windows · seed 20001",
                     filename="event_lm_coasting_heads_leaderboard.png", precision=5)
        plot_state(heads["rows"], ("Plain decoder", "Fast synapses", "Read heads", "Full attention"), output,
                   title="Loss against the state of one stream",
                   subtitle="368.64M training targets · identical windows · seed 20001 · "
                            "state is persistent tensor storage per stream",
                   filename="event_lm_coasting_heads_state.png")
    surprise = research["groups"].get(SURPRISE_CLOCK_GROUP)
    if surprise:
        from plot_surprise_clock import plot_surprise_study

        plot_heldout(surprise["rows"], output, title="Surprise clock and fast synapses",
                     subtitle="140.90M training targets · identical windows · seed 10007",
                     filename="event_lm_surprise_clock.png", precision=5,
                     height=4.7 + 0.32 * (len(surprise["rows"]) - 4))
        plot_surprise_study(surprise, output)
    for transfer, labels, filename in transfers.values():
        plot_transfer(transfer, labels, output, filename)
    packed_rows = research["groups"]["packed_width_111411k"]["rows"]
    narrow_snn, wide_snn, narrow_full, wide_full, delayed, short, present = packed_rows
    width_pairs = (("11-block SNN", BLUE, narrow_snn, wide_snn),
                   ("6-block full attention", ORANGE, narrow_full, wide_full))
    plot_paired_change(width_pairs, output, title="Widening both models at matched exposure",
                       subtitle="111.41M enwik8 training targets · seed 7907",
                       ticks=("9.96M parameters", "12.8M parameters"),
                       note="SNN state grows from 29,674 to 32,616 bytes. Full attention uses 6 MiB at both sizes.",
                       filename="event_lm_packed_width.png")
    plot_packed_circuit_changes(wide_snn, (short, delayed, present), output)
    plot_packed_width_history(output)
    plot_trained_context(research, output)
    plot_training_exposure(research, output)
    plot_prefix_age_gap(f"{OPPONENT}/enwik8_opponent_s711.json", f"{CREDIT}/enwik8_transformer_s711.json", output,
                        subtitle="Opponent steering and 512-byte full attention · 4,999,999 shared targets",
                        filename="event_lm_prefix_age_gap.png")
    plot_prefix_age_gap(f"{GROWN}/enwik8_grown_s711.json", f"{GROWN}/enwik8_transformer_s711.json", output,
                        subtitle="Six-bit grown SNN and 512-byte full attention · 4,999,999 shared targets",
                        filename="event_lm_grown_prefix_age.png")
    plot_prefix_age_gap(f"{SCALE_NEXT_DATA}/enwik8_trace_s2503.json",
                        f"{SCALE_NEXT_DATA}/enwik8_transformer_s2503.json", output,
                        subtitle="65.54M enwik8 training targets · 4,999,999 shared test targets",
                        filename="event_lm_65536_prefix_age.png")
    plot_prefix_age_gap(f"{DEPTH_GROWTH}/enwik8_depth_s4603.json",
                        f"{DEPTH_GROWTH}/enwik8_transformer_s4603.json", output,
                        subtitle="11-block SNN and 512-byte full attention · 4,999,999 shared test targets",
                        filename="event_lm_growth_9960k_prefix_age.png")


if __name__ == "__main__":
    main()
