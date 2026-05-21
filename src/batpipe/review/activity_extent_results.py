from __future__ import annotations

from batpipe.review.activity_segments import build_boundary_decision, build_peak_evidence
from batpipe.review.model_activity import ActivityExtent, ActivitySegment, PeakEvidence
from batpipe.review.model_annotations import AuditAnnotation


def _positive_peak_gaps_s(peak_times_s: list[float]) -> list[float]:
    return [
        right_peak_time_s - left_peak_time_s
        for left_peak_time_s, right_peak_time_s in zip(peak_times_s, peak_times_s[1:])
        if right_peak_time_s > left_peak_time_s
    ]


def _estimate_recent_left_cadence_s(segments: list[ActivitySegment]) -> float | None:
    if not segments:
        return None

    leftmost_peak_times_s = [float(peak_time_s) for peak_time_s in segments[0].peak_times_s]
    recent_peak_gaps_s = _positive_peak_gaps_s(leftmost_peak_times_s)
    if not recent_peak_gaps_s:
        return None

    return float(max(recent_peak_gaps_s[:2]))


def _estimate_recent_right_cadence_s(segments: list[ActivitySegment]) -> float | None:
    if not segments:
        return None

    rightmost_peak_times_s = [float(peak_time_s) for peak_time_s in segments[-1].peak_times_s]
    recent_peak_gaps_s = _positive_peak_gaps_s(rightmost_peak_times_s)
    if not recent_peak_gaps_s:
        return None

    return float(max(recent_peak_gaps_s[-2:]))


def _looks_like_mid_cadence_start(segments: list[ActivitySegment], time_step_s: float) -> bool:
    if not segments:
        return False

    leftmost_segment = segments[0]
    leftmost_peak_times_s = [float(peak_time_s) for peak_time_s in leftmost_segment.peak_times_s]
    recent_peak_gaps_s = _positive_peak_gaps_s(leftmost_peak_times_s)[:3]
    if len(recent_peak_gaps_s) < 2:
        return False

    smallest_gap_s = min(recent_peak_gaps_s)
    largest_gap_s = max(recent_peak_gaps_s)
    if smallest_gap_s <= 0.0 or largest_gap_s > (smallest_gap_s * 1.6):
        return False

    first_peak_offset_s = leftmost_peak_times_s[0] - float(leftmost_segment.start_time_s)
    return first_peak_offset_s <= max(time_step_s * 1.5, 0.03)


def build_single_frame_extent(
    *,
    peak_time_s: float,
    envelope_db: float,
    concentration_score: float | None,
    anchor_start_s: float,
    anchor_end_s: float,
) -> ActivityExtent:
    return ActivityExtent(
        start_time_s=peak_time_s,
        end_time_s=peak_time_s,
        peak_times_s=[peak_time_s],
        segments=[],
        peak_evidence=[
            PeakEvidence(
                time_s=peak_time_s,
                envelope_db=envelope_db,
                relative_level_db=0.0,
                within_anchor=anchor_start_s <= peak_time_s <= anchor_end_s,
                included_in_activity=True,
                concentration_score=concentration_score,
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


def build_empty_extent(
    *,
    anchor_start_s: float,
    anchor_end_s: float,
    stop_reason: str,
) -> ActivityExtent:
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
            stop_reason=stop_reason,
            included_peak_count=0,
            segment_count=0,
        ),
        right_boundary=build_boundary_decision(
            boundary="right",
            anchor_time_s=anchor_end_s,
            activity_time_s=anchor_end_s,
            stop_reason=stop_reason,
            included_peak_count=0,
            segment_count=0,
        ),
    )


def build_disconnected_extent(
    *,
    anchor_start_s: float,
    anchor_end_s: float,
    peak_times_s,
    peak_levels_db,
    peak_concentration_scores,
    anchor_level_db: float,
) -> ActivityExtent:
    return ActivityExtent(
        start_time_s=anchor_start_s,
        end_time_s=anchor_end_s,
        peak_times_s=[],
        segments=[],
        peak_evidence=build_peak_evidence(
            peak_times_s,
            peak_levels_db,
            peak_concentration_scores,
            [],
            anchor_start_s=anchor_start_s,
            anchor_end_s=anchor_end_s,
            anchor_level_db=anchor_level_db,
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


def build_extent_from_segments(
    *,
    segments,
    peak_times_s,
    peak_levels_db,
    peak_concentration_scores,
    anchor_start_s: float,
    anchor_end_s: float,
    anchor_level_db: float,
    time_step_s: float,
    clip_duration_s: float | None = None,
    selected_segments: list[ActivitySegment] | None = None,
    audit_annotations: list[AuditAnnotation] | None = None,
) -> ActivityExtent:
    start_time_s = min(segment.start_time_s for segment in segments)
    end_time_s = max(segment.end_time_s for segment in segments)
    peak_evidence = build_peak_evidence(
        peak_times_s,
        peak_levels_db,
        peak_concentration_scores,
        segments,
        anchor_start_s=anchor_start_s,
        anchor_end_s=anchor_end_s,
        anchor_level_db=anchor_level_db,
    )
    included_peak_count = sum(1 for item in peak_evidence if item.included_in_activity)
    half_step_s = time_step_s / 2.0
    active_start_center_s = start_time_s + half_step_s
    active_end_center_s = end_time_s - half_step_s
    left_extended = active_start_center_s < anchor_start_s
    right_extended = active_end_center_s > anchor_end_s
    left_stop_reason = "activity_onset" if left_extended else "anchor_edge"
    right_stop_reason = "activity_dropoff" if right_extended else "anchor_edge"

    if clip_duration_s is not None:
        recent_left_cadence_s = _estimate_recent_left_cadence_s(segments)
        leftmost_peak_time_s = min((float(peak_time_s) for segment in segments for peak_time_s in segment.peak_times_s), default=start_time_s)
        recent_right_cadence_s = _estimate_recent_right_cadence_s(segments)
        rightmost_peak_time_s = max((float(peak_time_s) for segment in segments for peak_time_s in segment.peak_times_s), default=end_time_s)
        clip_end_gap_s = max(0.0, float(clip_duration_s - rightmost_peak_time_s))
        if start_time_s <= half_step_s:
            left_stop_reason = "clip_start"
        elif left_extended and recent_left_cadence_s is not None and leftmost_peak_time_s <= recent_left_cadence_s:
            left_stop_reason = "clip_start"
        elif left_extended and _looks_like_mid_cadence_start(segments, time_step_s):
            left_stop_reason = "cadence_gap"
        if end_time_s >= max(0.0, clip_duration_s - half_step_s):
            right_stop_reason = "clip_end"
        elif right_extended and recent_right_cadence_s is not None and clip_end_gap_s <= recent_right_cadence_s:
            right_stop_reason = "clip_end"

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
        selected_segments=list(selected_segments or segments),
        audit_annotations=list(audit_annotations or []),
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