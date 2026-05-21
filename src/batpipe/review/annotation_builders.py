from __future__ import annotations

from batpipe.review.model_activity import ActivitySegment
from batpipe.review.model_annotations import AuditAnnotation


def build_detection_gap_annotations(segments: list[ActivitySegment]) -> list[AuditAnnotation]:
    if len(segments) < 2:
        return []

    annotations: list[AuditAnnotation] = []
    for left_segment, right_segment in zip(segments, segments[1:]):
        gap_start_s = float(left_segment.end_time_s)
        gap_end_s = float(right_segment.start_time_s)
        if gap_end_s <= gap_start_s:
            continue
        annotations.append(
            AuditAnnotation(
                category="detection_gap",
                start_time_s=gap_start_s,
                end_time_s=gap_end_s,
                source="activity_segment_selection",
                label="Detection gap",
                rationale="Adjacent selected activity segments were merged into one review event across this lull.",
                related_peak_times_s=[
                    float(left_segment.peak_times_s[-1]) if left_segment.peak_times_s else gap_start_s,
                    float(right_segment.peak_times_s[0]) if right_segment.peak_times_s else gap_end_s,
                ],
            )
        )
    return annotations