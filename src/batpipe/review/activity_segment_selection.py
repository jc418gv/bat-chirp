from __future__ import annotations

from batpipe.review.model_activity import ActivitySegment


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
        if segment_gap_s(segments[left_index - 1], segments[left_index]) > connection_gap_s:
            break
        left_index -= 1

    while right_index < len(segments) - 1:
        if segment_gap_s(segments[right_index], segments[right_index + 1]) > connection_gap_s:
            break
        right_index += 1

    while left_index > 0:
        if segment_gap_s(segments[left_index - 1], segments[left_index]) > adjacent_segment_merge_gap_s:
            break
        left_index -= 1

    while right_index < len(segments) - 1:
        if segment_gap_s(segments[right_index], segments[right_index + 1]) > adjacent_segment_merge_gap_s:
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


def segment_gap_s(left: ActivitySegment, right: ActivitySegment) -> float:
    return max(0.0, float(right.start_time_s - left.end_time_s))