from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


AnnotationCategory = Literal[
    "detection_gap",
    "anchor_only",
    "adjacent_merge",
    "clip_truncation",
    "chirp",
    "behavior",
]


@dataclass(slots=True)
class AuditAnnotation:
    category: AnnotationCategory
    start_time_s: float
    end_time_s: float
    source: str
    label: str
    rationale: str | None = None
    related_peak_times_s: list[float] = field(default_factory=list)
    score: float | None = None

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time_s - self.start_time_s)