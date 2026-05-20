from __future__ import annotations

from batpipe.review.models import ActivityBoundaryDecision, ActivitySegment, PeakEvidence


def build_peak_segments(peak_times_s, half_bin_s: float, max_peak_gap_s: float) -> list[ActivitySegment]:
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


def build_mask_activity_segments(
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


def select_anchor_connected_segments(
    segments: list[ActivitySegment],
    anchor_start_s: float,
    anchor_end_s: float,
    max_activity_extension_s: float,
    connection_gap_s: float,
    adjacent_segment_merge_gap_s: float,
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

    while left_index > 0:
        if _segment_gap_s(segments[left_index - 1], segments[left_index]) > adjacent_segment_merge_gap_s:
            break
        left_index -= 1

    while right_index < len(segments) - 1:
        if _segment_gap_s(segments[right_index], segments[right_index + 1]) > adjacent_segment_merge_gap_s:
            break
        right_index += 1

    return segments[left_index : right_index + 1]


def merge_activity_segments(segments: list[ActivitySegment]) -> list[ActivitySegment]:
    if not segments:
        return []

    return [
        ActivitySegment(
            start_time_s=min(segment.start_time_s for segment in segments),
            end_time_s=max(segment.end_time_s for segment in segments),
            peak_times_s=[peak_time_s for segment in segments for peak_time_s in segment.peak_times_s],
        )
    ]


def build_peak_evidence(
    peak_times_s,
    peak_levels_db,
    peak_concentration_scores,
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
    for peak_time_s, peak_level_db, peak_concentration_score in zip(peak_times_s, peak_levels_db, peak_concentration_scores):
        peak_time = float(peak_time_s)
        peak_level = float(peak_level_db)
        evidence.append(
            PeakEvidence(
                time_s=peak_time,
                envelope_db=peak_level,
                relative_level_db=peak_level - anchor_level_db,
                within_anchor=anchor_start_s <= peak_time <= anchor_end_s,
                included_in_activity=round(peak_time, 9) in included_peak_times,
                concentration_score=float(peak_concentration_score),
            )
        )
    return evidence


def build_boundary_decision(
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


def _segment_gap_s(left: ActivitySegment, right: ActivitySegment) -> float:
    return max(0.0, float(right.start_time_s - left.end_time_s))
