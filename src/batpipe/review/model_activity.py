from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from batpipe.review.model_annotations import AuditAnnotation

BoundaryStopReason = Literal[
    "anchor_edge",
    "activity_onset",
    "activity_dropoff",
    "cadence_gap",
    "clip_start",
    "clip_end",
    "single_frame_activity",
    "disconnected_activity",
    "no_activity_peaks",
]


@dataclass(slots=True)
class ActivitySegment:
    start_time_s: float
    end_time_s: float
    peak_times_s: list[float]

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time_s - self.start_time_s)

    @property
    def peak_count(self) -> int:
        return len(self.peak_times_s)


@dataclass(slots=True)
class PeakEvidence:
    time_s: float
    envelope_db: float
    relative_level_db: float
    within_anchor: bool
    included_in_activity: bool
    concentration_score: float | None = None


@dataclass(slots=True)
class ActivityBoundaryDecision:
    boundary: str
    anchor_time_s: float
    activity_time_s: float
    stop_reason: BoundaryStopReason
    included_peak_count: int
    segment_count: int


@dataclass(slots=True)
class ActivityExtent:
    start_time_s: float
    end_time_s: float
    peak_times_s: list[float]
    segments: list[ActivitySegment]
    peak_evidence: list[PeakEvidence]
    selected_segments: list[ActivitySegment] = field(default_factory=list)
    audit_annotations: list[AuditAnnotation] = field(default_factory=list)
    left_boundary: ActivityBoundaryDecision | None = None
    right_boundary: ActivityBoundaryDecision | None = None

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time_s - self.start_time_s)

    @property
    def segment_count(self) -> int:
        return len(self.segments)