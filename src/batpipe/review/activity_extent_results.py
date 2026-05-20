from __future__ import annotations

from batpipe.review.activity_segments import build_boundary_decision, build_peak_evidence
from batpipe.review.models import ActivityExtent, PeakEvidence


def build_single_frame_extent(
    *,
    peak_time_s: float,
    envelope_db: float,
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
    anchor_start_s: float,
    anchor_end_s: float,
    anchor_level_db: float,
    time_step_s: float,
) -> ActivityExtent:
    start_time_s = min(segment.start_time_s for segment in segments)
    end_time_s = max(segment.end_time_s for segment in segments)
    peak_evidence = build_peak_evidence(
        peak_times_s,
        peak_levels_db,
        segments,
        anchor_start_s=anchor_start_s,
        anchor_end_s=anchor_end_s,
        anchor_level_db=anchor_level_db,
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