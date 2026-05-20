from __future__ import annotations

from batpipe.review.band_analysis import compute_spectrogram_db, estimate_band_envelope_db
from batpipe.review.activity_extent_results import (
    build_disconnected_extent,
    build_empty_extent,
    build_extent_from_segments,
    build_single_frame_extent,
)
from batpipe.review.activity_segments import (
    build_mask_activity_segments,
    build_peak_segments,
    merge_activity_segments,
    select_anchor_connected_segments,
)
from batpipe.review.activity_signal import analyze_activity_signal
from batpipe.review.models import (
    ActivityExtent,
    ActivityExtractionConfig,
    ClipWindow,
    DetectionBout,
    SpectrogramConfig,
)


def extract_activity_extent(
    times_s,
    band_envelope_db,
    anchor_start_s: float,
    anchor_end_s: float,
    max_peak_gap_s: float = 0.25,
    max_activity_extension_s: float = 1.0,
    concentration_score=None,
) -> ActivityExtent | None:
    config = ActivityExtractionConfig(
        max_peak_gap_s=max_peak_gap_s,
        max_activity_extension_s=max_activity_extension_s,
    )
    return extract_activity_extent_with_config(
        times_s=times_s,
        band_envelope_db=band_envelope_db,
        concentration_score=concentration_score,
        anchor_start_s=anchor_start_s,
        anchor_end_s=anchor_end_s,
        config=config,
    )


def extract_activity_extent_with_config(
    times_s,
    band_envelope_db,
    anchor_start_s: float,
    anchor_end_s: float,
    config: ActivityExtractionConfig | None = None,
    concentration_score=None,
) -> ActivityExtent | None:
    import numpy as np

    config = config or ActivityExtractionConfig()

    times = np.asarray(times_s, dtype=float)
    envelope = np.asarray(band_envelope_db, dtype=float)
    concentration = np.ones_like(envelope, dtype=float) if concentration_score is None else np.asarray(concentration_score, dtype=float)
    if times.size == 0 or envelope.size == 0 or times.size != envelope.size:
        return None
    if concentration.size != envelope.size:
        return None

    if times.size == 1:
        return build_single_frame_extent(
            peak_time_s=float(times[0]),
            envelope_db=float(envelope[0]),
            concentration_score=float(concentration[0]),
            anchor_start_s=anchor_start_s,
            anchor_end_s=anchor_end_s,
        )

    signal_evidence = analyze_activity_signal(
        times_s=times,
        band_envelope_db=envelope,
        concentration_score=concentration,
        anchor_start_s=anchor_start_s,
        anchor_end_s=anchor_end_s,
        config=config,
    )
    if signal_evidence is None:
        return None

    if len(signal_evidence.peak_times_s) == 0 and not signal_evidence.active_mask.any():
        return build_empty_extent(
            anchor_start_s=anchor_start_s,
            anchor_end_s=anchor_end_s,
            stop_reason="no_activity_peaks",
        )

    segments = build_mask_activity_segments(
        signal_evidence.times_s,
        signal_evidence.active_mask,
        signal_evidence.peak_times_s,
        max_silence_gap_s=config.max_silence_gap_s + signal_evidence.time_step_s,
    )
    if not segments and len(signal_evidence.peak_times_s) > 0:
        half_bin_s = signal_evidence.time_step_s / 2.0
        segments = build_peak_segments(signal_evidence.peak_times_s, half_bin_s, config.max_peak_gap_s)
    segments = select_anchor_connected_segments(
        segments,
        anchor_start_s=anchor_start_s,
        anchor_end_s=anchor_end_s,
        max_activity_extension_s=config.max_activity_extension_s,
        connection_gap_s=signal_evidence.connection_gap_s,
    )
    segments = merge_activity_segments(segments)
    if not segments:
        return build_disconnected_extent(
            anchor_start_s=anchor_start_s,
            anchor_end_s=anchor_end_s,
            peak_times_s=signal_evidence.peak_times_s,
            peak_levels_db=signal_evidence.peak_levels_db,
            peak_concentration_scores=signal_evidence.peak_concentration_scores,
            anchor_level_db=signal_evidence.anchor_level_db,
        )

    return build_extent_from_segments(
        segments=segments,
        peak_times_s=signal_evidence.peak_times_s,
        peak_levels_db=signal_evidence.peak_levels_db,
        peak_concentration_scores=signal_evidence.peak_concentration_scores,
        anchor_start_s=anchor_start_s,
        anchor_end_s=anchor_end_s,
        anchor_level_db=signal_evidence.anchor_level_db,
        time_step_s=signal_evidence.time_step_s,
    )


def extract_bat_activity(
    audio,
    sample_rate_hz: int,
    window: ClipWindow,
    selected_bout: DetectionBout | None,
    max_freq_hz: float,
    activity_extraction_config: ActivityExtractionConfig | None = None,
    spectrogram_config: SpectrogramConfig | None = None,
) -> ActivityExtent | None:
    if selected_bout is None:
        return None
    activity_extraction_config = activity_extraction_config or ActivityExtractionConfig()
    spectrogram_config = spectrogram_config or SpectrogramConfig()

    try:
        spectrogram_analysis = compute_spectrogram_db(audio, sample_rate_hz, spectrogram_config)
    except ValueError:
        return None

    band_analysis = estimate_band_envelope_db(
        spectrum_db=spectrogram_analysis.spectrum_db,
        frequencies_hz=spectrogram_analysis.frequencies_hz,
        selected_bout=selected_bout,
        max_freq_hz=max_freq_hz,
        config=spectrogram_config,
    )
    if band_analysis is None:
        return None

    anchor_start_s = selected_bout.start_time_s - window.start_time_s
    anchor_end_s = selected_bout.end_time_s - window.start_time_s
    return extract_activity_extent_with_config(
        times_s=spectrogram_analysis.times_s,
        band_envelope_db=band_analysis.band_envelope_db,
        concentration_score=band_analysis.concentration_score,
        anchor_start_s=anchor_start_s,
        anchor_end_s=anchor_end_s,
        config=activity_extraction_config,
    )