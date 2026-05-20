from __future__ import annotations

from batpipe.review.models import (
    CandidateTrainRange,
    CandidateTrainSegment,
    ClipWindow,
    DetectionBout,
    PeakDetectionConfig,
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


def _build_candidate_train_segments(
    peak_times_s,
    half_bin_s: float,
    max_peak_gap_s: float,
) -> list[CandidateTrainSegment]:
    import numpy as np

    peak_times = np.asarray(peak_times_s, dtype=float)
    if peak_times.size == 0:
        return []

    segments: list[CandidateTrainSegment] = []
    start_index = 0
    for index in range(1, peak_times.size):
        if (peak_times[index] - peak_times[index - 1]) <= max_peak_gap_s:
            continue

        segment_peak_times = peak_times[start_index:index]
        segments.append(
            CandidateTrainSegment(
                start_time_s=max(0.0, float(segment_peak_times[0] - half_bin_s)),
                end_time_s=float(segment_peak_times[-1] + half_bin_s),
                peak_times_s=[float(value) for value in segment_peak_times],
            )
        )
        start_index = index

    segment_peak_times = peak_times[start_index:]
    segments.append(
        CandidateTrainSegment(
            start_time_s=max(0.0, float(segment_peak_times[0] - half_bin_s)),
            end_time_s=float(segment_peak_times[-1] + half_bin_s),
            peak_times_s=[float(value) for value in segment_peak_times],
        )
    )
    return segments


def _segment_peak_gap_s(left: CandidateTrainSegment, right: CandidateTrainSegment) -> float:
    return float(right.peak_times_s[0] - left.peak_times_s[-1])


def _select_anchor_connected_segments(
    segments: list[CandidateTrainSegment],
    anchor_start_s: float,
    anchor_end_s: float,
    max_train_extension_s: float,
    connection_gap_s: float,
) -> list[CandidateTrainSegment]:
    if not segments:
        return []

    anchor_overlap_start_s = anchor_start_s - max_train_extension_s
    anchor_overlap_end_s = anchor_end_s + max_train_extension_s
    seed_indices = [
        index
        for index, segment in enumerate(segments)
        if segment.end_time_s >= anchor_overlap_start_s and segment.start_time_s <= anchor_overlap_end_s
    ]
    if not seed_indices:
        anchor_midpoint_s = (anchor_start_s + anchor_end_s) / 2.0

        def distance_to_anchor(segment: CandidateTrainSegment) -> float:
            if segment.start_time_s <= anchor_midpoint_s <= segment.end_time_s:
                return 0.0
            if anchor_midpoint_s < segment.start_time_s:
                return segment.start_time_s - anchor_midpoint_s
            return anchor_midpoint_s - segment.end_time_s

        seed_indices = [min(range(len(segments)), key=lambda index: distance_to_anchor(segments[index]))]

    left_index = min(seed_indices)
    right_index = max(seed_indices)

    while left_index > 0:
        if _segment_peak_gap_s(segments[left_index - 1], segments[left_index]) > connection_gap_s:
            break
        left_index -= 1

    while right_index < len(segments) - 1:
        if _segment_peak_gap_s(segments[right_index], segments[right_index + 1]) > connection_gap_s:
            break
        right_index += 1

    return segments[left_index : right_index + 1]


def _merge_candidate_train_segments(
    segments: list[CandidateTrainSegment],
) -> list[CandidateTrainSegment]:
    if not segments:
        return []

    return [
        CandidateTrainSegment(
            start_time_s=min(segment.start_time_s for segment in segments),
            end_time_s=max(segment.end_time_s for segment in segments),
            peak_times_s=[
                peak_time_s
                for segment in segments
                for peak_time_s in segment.peak_times_s
            ],
        )
    ]


def estimate_candidate_train_range(
    times_s,
    band_envelope_db,
    anchor_start_s: float,
    anchor_end_s: float,
    max_peak_gap_s: float = 0.25,
    max_train_extension_s: float = 1.0,
) -> CandidateTrainRange | None:
    config = PeakDetectionConfig(
        max_peak_gap_s=max_peak_gap_s,
        max_train_extension_s=max_train_extension_s,
    )
    return estimate_candidate_train_range_with_config(
        times_s=times_s,
        band_envelope_db=band_envelope_db,
        anchor_start_s=anchor_start_s,
        anchor_end_s=anchor_end_s,
        config=config,
    )


def estimate_candidate_train_range_with_config(
    times_s,
    band_envelope_db,
    anchor_start_s: float,
    anchor_end_s: float,
    config: PeakDetectionConfig | None = None,
) -> CandidateTrainRange | None:
    import numpy as np
    from scipy import signal

    config = config or PeakDetectionConfig()

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
        return CandidateTrainRange(float(times[0]), float(times[0]), [float(times[0])], [])

    time_step_s = float(np.median(np.diff(times))) if times.size > 1 else 0.01
    min_peak_distance = max(1, int(round(config.min_peak_distance_s / max(time_step_s, 1e-6))))

    anchor_mask = (times >= anchor_start_s) & (times <= anchor_end_s)
    if anchor_mask.any():
        anchor_level = float(np.nanmax(envelope[anchor_mask]))
    else:
        closest_index = int(np.argmin(np.abs(times - ((anchor_start_s + anchor_end_s) / 2.0))))
        anchor_level = float(envelope[closest_index])

    if not np.isfinite(anchor_level):
        return None

    threshold = floor + (anchor_level - floor) * config.threshold_ratio
    prominence = max((anchor_level - floor) * config.prominence_ratio, 1e-6)
    peak_indices, _ = signal.find_peaks(
        envelope,
        height=threshold,
        distance=min_peak_distance,
        prominence=prominence,
    )
    if peak_indices.size == 0:
        peak_indices, _ = signal.find_peaks(envelope, height=threshold, distance=min_peak_distance)
    if peak_indices.size == 0:
        return CandidateTrainRange(anchor_start_s, anchor_end_s, [], [])

    peak_times = times[peak_indices]
    inter_peak_intervals_s = np.diff(peak_times)
    if inter_peak_intervals_s.size > 0:
        connection_gap_s = min(float(np.mean(inter_peak_intervals_s)) * 2.5, config.max_connection_gap_s)
    else:
        connection_gap_s = min(config.max_peak_gap_s * 2.0, config.max_connection_gap_s)
    half_bin_s = time_step_s / 2.0
    segments = _build_candidate_train_segments(peak_times, half_bin_s, config.max_peak_gap_s)
    segments = _select_anchor_connected_segments(
        segments,
        anchor_start_s=anchor_start_s,
        anchor_end_s=anchor_end_s,
        max_train_extension_s=config.max_train_extension_s,
        connection_gap_s=connection_gap_s,
    )
    segments = _merge_candidate_train_segments(segments)
    if not segments:
        return CandidateTrainRange(anchor_start_s, anchor_end_s, [], [])

    start_time_s = min(segment.start_time_s for segment in segments)
    end_time_s = max(segment.end_time_s for segment in segments)
    return CandidateTrainRange(
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        peak_times_s=[
            peak_time_s
            for segment in segments
            for peak_time_s in segment.peak_times_s
        ],
        segments=segments,
    )


def analyze_candidate_train(
    audio,
    sample_rate_hz: int,
    window: ClipWindow,
    selected_bout: DetectionBout | None,
    max_freq_hz: float,
    peak_detection_config: PeakDetectionConfig | None = None,
    spectrogram_config: SpectrogramConfig | None = None,
) -> CandidateTrainRange | None:
    if selected_bout is None:
        return None
    peak_detection_config = peak_detection_config or PeakDetectionConfig()
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
    return estimate_candidate_train_range_with_config(
        times_s=times_s,
        band_envelope_db=band_envelope_db,
        anchor_start_s=anchor_start_s,
        anchor_end_s=anchor_end_s,
        config=peak_detection_config,
    )