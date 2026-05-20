from __future__ import annotations

from batpipe.review.models import (
    ActivityBoundaryDecision,
    ActivityExtent,
    ActivityExtractionConfig,
    ActivitySegment,
    ClipWindow,
    DetectionBout,
    PeakEvidence,
    SpectrogramConfig,
)


def compute_spectrogram_db(audio, sample_rate_hz: int, config: SpectrogramConfig):
    import numpy as np
    from scipy import signal

    waveform = np.asarray(audio)
    if waveform.size == 0:
        raise ValueError("Clip audio is empty.")

    if np.issubdtype(waveform.dtype, np.integer):
        info = np.iinfo(waveform.dtype)
        scale = max(abs(info.min), info.max)
        waveform = waveform.astype(np.float32) / float(scale)
    else:
        waveform = waveform.astype(np.float32)

    clip_duration_s = waveform.size / float(sample_rate_hz)
    nperseg = min(config.nperseg, waveform.size)
    noverlap = max(0, int(nperseg * config.noverlap_ratio))
    frequencies_hz, times_s, spectrum = signal.spectrogram(
        waveform,
        fs=sample_rate_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        mode="magnitude",
    )
    spectrum_db = 20.0 * np.log10(np.maximum(spectrum, 1e-12))
    return waveform, clip_duration_s, frequencies_hz, times_s, spectrum_db


def estimate_band_envelope_db(
    spectrum_db,
    frequencies_hz,
    selected_bout: DetectionBout,
    max_freq_hz: float,
    config: SpectrogramConfig,
):
    import numpy as np
    from scipy import ndimage

    band_low_hz = max(0.0, (selected_bout.min_low_freq_hz or 0.0) - config.band_margin_hz)
    band_high_hz = min(max_freq_hz, (selected_bout.max_high_freq_hz or max_freq_hz) + config.band_margin_hz)
    band_mask = (frequencies_hz >= band_low_hz) & (frequencies_hz <= band_high_hz)
    if not band_mask.any():
        return None

    band_spectrum = spectrum_db[band_mask].copy()
    band_spectrum -= np.nanmean(band_spectrum, axis=1, keepdims=True)
    np.maximum(band_spectrum, 0.0, out=band_spectrum)
    band_envelope_db = np.nanpercentile(band_spectrum, config.envelope_percentile, axis=0)
    band_envelope_db = ndimage.gaussian_filter1d(band_envelope_db, sigma=config.gaussian_sigma, mode="nearest")
    return band_envelope_db


def _segment_gap_s(left: ActivitySegment, right: ActivitySegment) -> float:
    return max(0.0, float(right.start_time_s - left.end_time_s))


def _build_peak_segments(
    peak_times_s,
    half_bin_s: float,
    max_peak_gap_s: float,
) -> list[ActivitySegment]:
    import numpy as np

    peak_times = np.asarray(peak_times_s, dtype=float)
    if peak_times.size == 0:
        return []

    segments: list[ActivitySegment] = []
    start_index = 0
    for index in range(1, peak_times.size):
        if (peak_times[index] - peak_times[index - 1]) <= max_peak_gap_s:
            continue

        segment_peak_times = peak_times[start_index:index]
        segments.append(
            ActivitySegment(
                start_time_s=max(0.0, float(segment_peak_times[0] - half_bin_s)),
                end_time_s=float(segment_peak_times[-1] + half_bin_s),
                peak_times_s=[float(value) for value in segment_peak_times],
            )
        )
        start_index = index

    segment_peak_times = peak_times[start_index:]
    segments.append(
        ActivitySegment(
            start_time_s=max(0.0, float(segment_peak_times[0] - half_bin_s)),
            end_time_s=float(segment_peak_times[-1] + half_bin_s),
            peak_times_s=[float(value) for value in segment_peak_times],
        )
    )
    return segments


def _build_mask_activity_segments(
    times_s,
    active_mask,
    peak_times_s,
    *,
    max_silence_gap_s: float,
) -> list[ActivitySegment]:
    import numpy as np

    times = np.asarray(times_s, dtype=float)
    active_indices = np.flatnonzero(np.asarray(active_mask, dtype=bool))
    if active_indices.size == 0:
        return []

    half_bin_s = float(np.median(np.diff(times))) / 2.0 if times.size > 1 else 0.01
    peak_times = np.asarray(peak_times_s, dtype=float)
    segments: list[ActivitySegment] = []
    segment_start_index = int(active_indices[0])
    previous_index = int(active_indices[0])

    def append_segment(start_index: int, end_index: int) -> None:
        segment_start_s = max(0.0, float(times[start_index] - half_bin_s))
        segment_end_s = float(times[end_index] + half_bin_s)
        segment_peak_times = [
            float(peak_time_s)
            for peak_time_s in peak_times
            if segment_start_s <= float(peak_time_s) <= segment_end_s
        ]
        segments.append(
            ActivitySegment(
                start_time_s=segment_start_s,
                end_time_s=segment_end_s,
                peak_times_s=segment_peak_times,
            )
        )

    for current_index in active_indices[1:]:
        current_index = int(current_index)
        if float(times[current_index] - times[previous_index]) <= max_silence_gap_s:
            previous_index = current_index
            continue
        append_segment(segment_start_index, previous_index)
        segment_start_index = current_index
        previous_index = current_index

    append_segment(segment_start_index, previous_index)
    return segments


def _select_anchor_connected_segments(
    segments: list[ActivitySegment],
    anchor_start_s: float,
    anchor_end_s: float,
    max_activity_extension_s: float,
    connection_gap_s: float,
) -> list[ActivitySegment]:
    if not segments:
        return []

    anchor_overlap_start_s = anchor_start_s - max_activity_extension_s
    anchor_overlap_end_s = anchor_end_s + max_activity_extension_s
    seed_indices = [
        index
        for index, segment in enumerate(segments)
        if segment.end_time_s >= anchor_overlap_start_s and segment.start_time_s <= anchor_overlap_end_s
    ]
    if not seed_indices:
        anchor_midpoint_s = (anchor_start_s + anchor_end_s) / 2.0

        def distance_to_anchor(segment: ActivitySegment) -> float:
            if segment.start_time_s <= anchor_midpoint_s <= segment.end_time_s:
                return 0.0
            if anchor_midpoint_s < segment.start_time_s:
                return segment.start_time_s - anchor_midpoint_s
            return anchor_midpoint_s - segment.end_time_s

        seed_indices = [min(range(len(segments)), key=lambda index: distance_to_anchor(segments[index]))]

    left_index = min(seed_indices)
    right_index = max(seed_indices)

    while left_index > 0:
        if _segment_gap_s(segments[left_index - 1], segments[left_index]) > connection_gap_s:
            break
        left_index -= 1

    while right_index < len(segments) - 1:
        if _segment_gap_s(segments[right_index], segments[right_index + 1]) > connection_gap_s:
            break
        right_index += 1

    return segments[left_index : right_index + 1]


def _merge_activity_segments(
    segments: list[ActivitySegment],
) -> list[ActivitySegment]:
    if not segments:
        return []

    return [
        ActivitySegment(
            start_time_s=min(segment.start_time_s for segment in segments),
            end_time_s=max(segment.end_time_s for segment in segments),
            peak_times_s=[
                peak_time_s
                for segment in segments
                for peak_time_s in segment.peak_times_s
            ],
        )
    ]


def _build_peak_evidence(
    peak_times_s,
    peak_levels_db,
    segments: list[ActivitySegment],
    *,
    anchor_start_s: float,
    anchor_end_s: float,
    anchor_level_db: float,
) -> list[PeakEvidence]:
    included_peak_times = {
        round(float(peak_time_s), 9)
        for segment in segments
        for peak_time_s in segment.peak_times_s
    }
    evidence: list[PeakEvidence] = []
    for peak_time_s, peak_level_db in zip(peak_times_s, peak_levels_db):
        peak_time = float(peak_time_s)
        peak_level = float(peak_level_db)
        evidence.append(
            PeakEvidence(
                time_s=peak_time,
                envelope_db=peak_level,
                relative_level_db=peak_level - anchor_level_db,
                within_anchor=anchor_start_s <= peak_time <= anchor_end_s,
                included_in_activity=round(peak_time, 9) in included_peak_times,
            )
        )
    return evidence


def _build_boundary_decision(
    *,
    boundary: str,
    anchor_time_s: float,
    activity_time_s: float,
    stop_reason: str,
    included_peak_count: int,
    segment_count: int,
) -> ActivityBoundaryDecision:
    return ActivityBoundaryDecision(
        boundary=boundary,
        anchor_time_s=anchor_time_s,
        activity_time_s=activity_time_s,
        stop_reason=stop_reason,
        included_peak_count=included_peak_count,
        segment_count=segment_count,
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
            left_boundary=_build_boundary_decision(
                boundary="left",
                anchor_time_s=anchor_start_s,
                activity_time_s=peak_time_s,
                stop_reason="single_frame_activity",
                included_peak_count=1,
                segment_count=0,
            ),
            right_boundary=_build_boundary_decision(
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

    threshold = floor + (anchor_level - floor) * config.threshold_ratio
    activity_threshold = floor + (anchor_level - floor) * config.activity_threshold_ratio
    prominence = max((anchor_level - floor) * config.prominence_ratio, 1e-6)
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

    active_mask = envelope >= activity_threshold
    active_mask |= anchor_mask

    if peak_indices.size == 0 and not active_mask.any():
        return ActivityExtent(
            start_time_s=anchor_start_s,
            end_time_s=anchor_end_s,
            peak_times_s=[],
            segments=[],
            peak_evidence=[],
            left_boundary=_build_boundary_decision(
                boundary="left",
                anchor_time_s=anchor_start_s,
                activity_time_s=anchor_start_s,
                stop_reason="no_activity_peaks",
                included_peak_count=0,
                segment_count=0,
            ),
            right_boundary=_build_boundary_decision(
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

    segments = _build_mask_activity_segments(
        times,
        active_mask,
        peak_times,
        max_silence_gap_s=config.max_silence_gap_s + time_step_s,
    )
    if not segments and peak_indices.size > 0:
        half_bin_s = time_step_s / 2.0
        segments = _build_peak_segments(peak_times, half_bin_s, config.max_peak_gap_s)
    segments = _select_anchor_connected_segments(
        segments,
        anchor_start_s=anchor_start_s,
        anchor_end_s=anchor_end_s,
        max_activity_extension_s=config.max_activity_extension_s,
        connection_gap_s=connection_gap_s,
    )
    segments = _merge_activity_segments(segments)
    if not segments:
        return ActivityExtent(
            start_time_s=anchor_start_s,
            end_time_s=anchor_end_s,
            peak_times_s=[],
            segments=[],
            peak_evidence=_build_peak_evidence(
                peak_times,
                peak_levels,
                [],
                anchor_start_s=anchor_start_s,
                anchor_end_s=anchor_end_s,
                anchor_level_db=anchor_level,
            ),
            left_boundary=_build_boundary_decision(
                boundary="left",
                anchor_time_s=anchor_start_s,
                activity_time_s=anchor_start_s,
                stop_reason="disconnected_activity",
                included_peak_count=0,
                segment_count=0,
            ),
            right_boundary=_build_boundary_decision(
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
    peak_evidence = _build_peak_evidence(
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
        left_boundary=_build_boundary_decision(
            boundary="left",
            anchor_time_s=anchor_start_s,
            activity_time_s=start_time_s,
            stop_reason=left_stop_reason,
            included_peak_count=included_peak_count,
            segment_count=len(segments),
        ),
        right_boundary=_build_boundary_decision(
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
        _, _, frequencies_hz, times_s, spectrum_db = compute_spectrogram_db(audio, sample_rate_hz, spectrogram_config)
    except ValueError:
        return None

    band_envelope_db = estimate_band_envelope_db(
        spectrum_db=spectrum_db,
        frequencies_hz=frequencies_hz,
        selected_bout=selected_bout,
        max_freq_hz=max_freq_hz,
        config=spectrogram_config,
    )
    if band_envelope_db is None:
        return None

    anchor_start_s = selected_bout.start_time_s - window.start_time_s
    anchor_end_s = selected_bout.end_time_s - window.start_time_s
    return extract_activity_extent_with_config(
        times_s=times_s,
        band_envelope_db=band_envelope_db,
        anchor_start_s=anchor_start_s,
        anchor_end_s=anchor_end_s,
        config=activity_extraction_config,
    )