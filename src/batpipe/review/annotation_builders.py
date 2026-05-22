from __future__ import annotations

from batpipe.review.model_activity import ActivitySegment
from batpipe.review.model_annotations import AuditAnnotation


def _peak_gaps_s(peak_times_s: list[float]) -> list[float]:
    return [
        float(right_peak_time_s - left_peak_time_s)
        for left_peak_time_s, right_peak_time_s in zip(peak_times_s, peak_times_s[1:])
        if right_peak_time_s > left_peak_time_s
    ]


def _estimate_segment_cadence_s(left_segment: ActivitySegment, right_segment: ActivitySegment) -> float | None:
    local_gaps_s = _peak_gaps_s(left_segment.peak_times_s)[-2:] + _peak_gaps_s(right_segment.peak_times_s)[:2]
    if not local_gaps_s:
        return None

    local_gaps_s.sort()
    midpoint = len(local_gaps_s) // 2
    if len(local_gaps_s) % 2 == 1:
        return float(local_gaps_s[midpoint])
    return float((local_gaps_s[midpoint - 1] + local_gaps_s[midpoint]) / 2.0)


def build_detection_gap_annotations(
    segments: list[ActivitySegment],
    min_gap_s: float = 0.8,
    cadence_multiplier: float = 3.0,
) -> list[AuditAnnotation]:
    if len(segments) < 2:
        return []

    annotations: list[AuditAnnotation] = []
    for left_segment, right_segment in zip(segments, segments[1:]):
        gap_start_s = float(left_segment.end_time_s)
        gap_end_s = float(right_segment.start_time_s)
        gap_duration_s = gap_end_s - gap_start_s
        if gap_duration_s <= 0.0 or gap_duration_s < min_gap_s:
            continue

        last_left_peak_s = float(left_segment.peak_times_s[-1]) if left_segment.peak_times_s else gap_start_s
        first_right_peak_s = float(right_segment.peak_times_s[0]) if right_segment.peak_times_s else gap_end_s
        inter_peak_gap_s = max(0.0, first_right_peak_s - last_left_peak_s)
        cadence_s = _estimate_segment_cadence_s(left_segment, right_segment)
        if cadence_s is not None and inter_peak_gap_s < (cadence_s * cadence_multiplier):
            continue

        rationale = "Selected activity spans were merged across a long lull that is much larger than the nearby chirp cadence."
        if cadence_s is None:
            rationale = "Selected activity spans were merged across a long lull that likely contains missed chirps."

        annotations.append(
            AuditAnnotation(
                category="detection_gap",
                start_time_s=gap_start_s,
                end_time_s=gap_end_s,
                source="activity_segment_selection",
                label="Detection gap",
                rationale=rationale,
                related_peak_times_s=[
                    last_left_peak_s,
                    first_right_peak_s,
                ],
            )
        )
    return annotations