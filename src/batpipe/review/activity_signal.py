from __future__ import annotations

from dataclasses import dataclass

from batpipe.review.model_review import ActivityExtractionConfig


@dataclass(slots=True)
class ActivitySignalEvidence:
    times_s: object
    envelope_db: object
    time_step_s: float
    anchor_level_db: float
    anchor_contrast_db: float
    peak_times_s: object
    peak_levels_db: object
    peak_concentration_scores: object
    active_mask: object
    concentration_score: object
    connection_gap_s: float


def analyze_activity_signal(
    times_s,
    band_envelope_db,
    concentration_score,
    anchor_start_s: float,
    anchor_end_s: float,
    config: ActivityExtractionConfig,
) -> ActivitySignalEvidence | None:
    import numpy as np
    from scipy import signal

    times = np.asarray(times_s, dtype=float)
    envelope = np.asarray(band_envelope_db, dtype=float)
    concentration = np.asarray(concentration_score, dtype=float)
    if times.size == 0 or envelope.size == 0 or times.size != envelope.size:
        return None
    if concentration.size != envelope.size:
        return None

    finite_mask = np.isfinite(envelope)
    if not finite_mask.any():
        return None

    envelope = envelope.copy()
    concentration = concentration.copy()
    floor = float(np.nanpercentile(envelope[finite_mask], config.floor_percentile))
    envelope[~finite_mask] = floor
    concentration[~np.isfinite(concentration)] = 0.0
    concentration = np.clip(concentration, 0.0, 1.0)
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
    anchor_contrast_db = signal_span_db
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
    peak_concentration_scores = concentration[peak_indices] if peak_indices.size > 0 else np.asarray([], dtype=float)

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

    concentration_mask = concentration >= config.concentration_threshold
    temporal_support_mask = (local_contrast >= modulation_threshold) & peak_support_mask
    active_mask = (envelope >= activity_threshold) & (concentration_mask | temporal_support_mask)
    if anchor_contrast_db < config.min_anchor_contrast_db:
        active_mask = anchor_mask.copy()
    active_mask |= anchor_mask

    inter_peak_intervals_s = np.diff(peak_times)
    if inter_peak_intervals_s.size > 0:
        connection_gap_s = min(float(np.mean(inter_peak_intervals_s)) * 2.5, config.max_connection_gap_s)
    else:
        connection_gap_s = min(config.max_silence_gap_s * 2.0, config.max_connection_gap_s)

    return ActivitySignalEvidence(
        times_s=times,
        envelope_db=envelope,
        time_step_s=time_step_s,
        anchor_level_db=anchor_level,
        anchor_contrast_db=anchor_contrast_db,
        peak_times_s=peak_times,
        peak_levels_db=peak_levels,
        peak_concentration_scores=peak_concentration_scores,
        active_mask=active_mask,
        concentration_score=concentration,
        connection_gap_s=connection_gap_s,
    )