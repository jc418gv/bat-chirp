from __future__ import annotations

from batpipe.review.model_activity import ActivityBoundaryDecision, ActivitySegment, PeakEvidence


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