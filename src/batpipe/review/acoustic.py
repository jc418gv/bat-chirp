from __future__ import annotations

from batpipe.review.band_analysis import compute_spectrogram_db, estimate_band_envelope_db
from batpipe.review.activity_segments import (
    build_boundary_decision,
    build_mask_activity_segments,
    build_peak_evidence,
    build_peak_segments,
    merge_activity_segments,
    select_anchor_connected_segments,
)
from batpipe.review.models import (
    ActivityExtent,
    ActivityExtractionConfig,
    ClipWindow,
    DetectionBout,
    PeakEvidence,
    SpectrogramConfig,
)


def extract_activity_extent(
    times_s,
    band_envelope_db,
    anchor_start_s: float,
    anchor_end_s: float,
    max_peak_gap_s: float = 0.25,
    max_activity_extension_s: float = 1.0,
) -> ActivityExtent | None:
    config = ActivityExtractionConfig(
        max_peak_gap_s=max_peak_gap_s,
        max_activity_extension_s=max_activity_extension_s,
    )
    return extract_activity_extent_with_config(
        times_s=times_s,
        band_envelope_db=band_envelope_db,
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
) -> ActivityExtent | None:
    import numpy as np
    from scipy import signal

    config = config or ActivityExtractionConfig()

    times = np.asarray(times_s, dtype=float)
    envelope = np.asarray(band_envelope_db, dtype=float)
    if times.size == 0 or envelope.size == 0 or times.size != envelope.size:
        return None

    finite_mask = np.isfinite(envelope)
    if not finite_mask.any():
        return None

    envelope = envelope.copy()
    floor = float(np.nanpercentile(envelope[finite_mask], config.floor_percentile))
    envelope[~finite_mask] = floor

    if times.size == 1:
        peak_time_s = float(times[0])
        return ActivityExtent(
            start_time_s=peak_time_s,
            end_time_s=peak_time_s,
            peak_times_s=[peak_time_s],
            segments=[],
            peak_evidence=[
                PeakEvidence(
                    time_s=peak_time_s,
                    envelope_db=float(envelope[0]),
                    relative_level_db=0.0,
                    within_anchor=anchor_start_s <= peak_time_s <= anchor_end_s,
                    included_in_activity=True,
                )
            ],
            left_boundary=build_boundary_decision(
                boundary="left",
                anchor_time_s=anchor_start_s,
                activity_time_s=peak_time_s,
                stop_reason="single_frame_activity",
                included_peak_count=1,
                segment_count=0,
            ),
            right_boundary=build_boundary_decision(
                boundary="right",
                anchor_time_s=anchor_end_s,
                activity_time_s=peak_time_s,
                stop_reason="single_frame_activity",
                included_peak_count=1,
                segment_count=0,
            ),
        )

    time_step_s = float(np.median(np.diff(times))) if times.size > 1 else 0.01
    min_peak_distance = max(1, int(round(config.min_peak_distance_s / max(time_step_s, 1e-6))))

    anchor_mask = (times >= anchor_start_s) & (times <= anchor_end_s)
    anchor_context_mask = (times >= (anchor_start_s - time_step_s)) & (times <= (anchor_end_s + time_step_s))
    if anchor_context_mask.any():
        anchor_level = float(np.nanmax(envelope[anchor_context_mask]))
    elif anchor_mask.any():
        anchor_level = float(np.nanmax(envelope[anchor_mask]))
    else:
        closest_index = int(np.argmin(np.abs(times - ((anchor_start_s + anchor_end_s) / 2.0))))
        anchor_level = float(envelope[closest_index])

    if not np.isfinite(anchor_level):
        return None

    signal_span_db = max(anchor_level - floor, 1e-6)
    threshold = floor + (anchor_level - floor) * config.threshold_ratio
    activity_threshold = floor + signal_span_db * config.activity_threshold_ratio
    prominence = max(signal_span_db * config.prominence_ratio, 1e-6)
    modulation_threshold = signal_span_db * config.activity_modulation_ratio
    peak_indices, _ = signal.find_peaks(
        envelope,
        height=threshold,
        distance=min_peak_distance,
        prominence=prominence,
    )
    if peak_indices.size == 0:
        peak_indices, _ = signal.find_peaks(envelope, height=threshold, distance=min_peak_distance)
    peak_times = times[peak_indices]
    peak_levels = envelope[peak_indices] if peak_indices.size > 0 else np.asarray([], dtype=float)

    local_max = np.maximum.reduce([
        envelope,
        np.roll(envelope, 1),
        np.roll(envelope, -1),
    ])
    local_min = np.minimum.reduce([
        envelope,
        np.roll(envelope, 1),
        np.roll(envelope, -1),
    ])
    local_contrast = local_max - local_min
    if envelope.size > 1:
        local_contrast[0] = abs(float(envelope[1] - envelope[0]))
        local_contrast[-1] = abs(float(envelope[-1] - envelope[-2]))

    peak_support_mask = np.zeros_like(envelope, dtype=bool)
    if peak_indices.size > 0:
        peak_support_radius = max(1, int(round(config.max_peak_gap_s / max(time_step_s * 2.0, 1e-6))))
        for peak_index in peak_indices:
            left_index = max(0, int(peak_index) - peak_support_radius)
            right_index = min(envelope.size, int(peak_index) + peak_support_radius + 1)
            peak_support_mask[left_index:right_index] = True

    active_mask = (envelope >= activity_threshold) & ((local_contrast >= modulation_threshold) | peak_support_mask)
    active_mask |= anchor_mask

    if peak_indices.size == 0 and not active_mask.any():
        return ActivityExtent(
            start_time_s=anchor_start_s,
            end_time_s=anchor_end_s,
            peak_times_s=[],
            segments=[],
            peak_evidence=[],
            left_boundary=build_boundary_decision(
                boundary="left",
                anchor_time_s=anchor_start_s,
                activity_time_s=anchor_start_s,
                stop_reason="no_activity_peaks",
                included_peak_count=0,
                segment_count=0,
            ),
            right_boundary=build_boundary_decision(
                boundary="right",
                anchor_time_s=anchor_end_s,
                activity_time_s=anchor_end_s,
                stop_reason="no_activity_peaks",
                included_peak_count=0,
                segment_count=0,
            ),
        )

    inter_peak_intervals_s = np.diff(peak_times)
    if inter_peak_intervals_s.size > 0:
        connection_gap_s = min(float(np.mean(inter_peak_intervals_s)) * 2.5, config.max_connection_gap_s)
    else:
        connection_gap_s = min(config.max_silence_gap_s * 2.0, config.max_connection_gap_s)

    segments = build_mask_activity_segments(
        times,
        active_mask,
        peak_times,
        max_silence_gap_s=config.max_silence_gap_s + time_step_s,
    )
    if not segments and peak_indices.size > 0:
        half_bin_s = time_step_s / 2.0
        segments = build_peak_segments(peak_times, half_bin_s, config.max_peak_gap_s)
    segments = select_anchor_connected_segments(
        segments,
        anchor_start_s=anchor_start_s,
        anchor_end_s=anchor_end_s,
        max_activity_extension_s=config.max_activity_extension_s,
        connection_gap_s=connection_gap_s,
    )
    segments = merge_activity_segments(segments)
    if not segments:
        return ActivityExtent(
            start_time_s=anchor_start_s,
            end_time_s=anchor_end_s,
            peak_times_s=[],
            segments=[],
            peak_evidence=build_peak_evidence(
                peak_times,
                peak_levels,
                [],
                anchor_start_s=anchor_start_s,
                anchor_end_s=anchor_end_s,
                anchor_level_db=anchor_level,
            ),
            left_boundary=build_boundary_decision(
                boundary="left",
                anchor_time_s=anchor_start_s,
                activity_time_s=anchor_start_s,
                stop_reason="disconnected_activity",
                included_peak_count=0,
                segment_count=0,
            ),
            right_boundary=build_boundary_decision(
                boundary="right",
                anchor_time_s=anchor_end_s,
                activity_time_s=anchor_end_s,
                stop_reason="disconnected_activity",
                included_peak_count=0,
                segment_count=0,
            ),
        )

    start_time_s = min(segment.start_time_s for segment in segments)
    end_time_s = max(segment.end_time_s for segment in segments)
    peak_evidence = build_peak_evidence(
        peak_times,
        peak_levels,
        segments,
        anchor_start_s=anchor_start_s,
        anchor_end_s=anchor_end_s,
        anchor_level_db=anchor_level,
    )
    included_peak_count = sum(1 for item in peak_evidence if item.included_in_activity)
    active_start_center_s = start_time_s + (time_step_s / 2.0)
    active_end_center_s = end_time_s - (time_step_s / 2.0)
    left_extended = active_start_center_s < anchor_start_s
    right_extended = active_end_center_s > anchor_end_s
    left_stop_reason = "activity_dropoff" if left_extended else "anchor_edge"
    right_stop_reason = "activity_dropoff" if right_extended else "anchor_edge"
    return ActivityExtent(
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        peak_times_s=[
            peak_time_s
            for segment in segments
            for peak_time_s in segment.peak_times_s
        ],
        segments=segments,
        peak_evidence=peak_evidence,
        left_boundary=build_boundary_decision(
            boundary="left",
            anchor_time_s=anchor_start_s,
            activity_time_s=start_time_s,
            stop_reason=left_stop_reason,
            included_peak_count=included_peak_count,
            segment_count=len(segments),
        ),
        right_boundary=build_boundary_decision(
            boundary="right",
            anchor_time_s=anchor_end_s,
            activity_time_s=end_time_s,
            stop_reason=right_stop_reason,
            included_peak_count=included_peak_count,
            segment_count=len(segments),
        ),
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
        anchor_start_s=anchor_start_s,
        anchor_end_s=anchor_end_s,
        config=activity_extraction_config,
    )