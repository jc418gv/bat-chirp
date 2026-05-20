from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CLASSIFICATION_WARNING = (
    "BatDetect2 class labels in this workflow are raw model outputs and not reliable classifications."
)


@dataclass(slots=True)
class ClipDetection:
    start_time_s: float
    end_time_s: float
    det_prob: float | None
    class_prob: float | None
    predicted_class: str
    event: str | None
    low_freq_hz: float | None
    high_freq_hz: float | None


@dataclass(slots=True)
class DetectionBout:
    start_time_s: float
    end_time_s: float
    detections: list[ClipDetection]

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time_s - self.start_time_s)

    @property
    def max_det_prob(self) -> float | None:
        det_probs = [item.det_prob for item in self.detections if item.det_prob is not None]
        if not det_probs:
            return None
        return max(det_probs)

    @property
    def min_low_freq_hz(self) -> float | None:
        low_freqs = [item.low_freq_hz for item in self.detections if item.low_freq_hz is not None]
        if not low_freqs:
            return None
        return min(low_freqs)

    @property
    def max_high_freq_hz(self) -> float | None:
        high_freqs = [item.high_freq_hz for item in self.detections if item.high_freq_hz is not None]
        if not high_freqs:
            return None
        return max(high_freqs)


@dataclass(slots=True)
class ClipWindow:
    start_time_s: float
    end_time_s: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time_s - self.start_time_s)


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


@dataclass(slots=True)
class ActivityBoundaryDecision:
    boundary: str
    anchor_time_s: float
    activity_time_s: float
    stop_reason: str
    included_peak_count: int
    segment_count: int


@dataclass(slots=True)
class ActivityExtent:
    start_time_s: float
    end_time_s: float
    peak_times_s: list[float]
    segments: list[ActivitySegment]
    peak_evidence: list[PeakEvidence]
    left_boundary: ActivityBoundaryDecision | None = None
    right_boundary: ActivityBoundaryDecision | None = None

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time_s - self.start_time_s)

    @property
    def segment_count(self) -> int:
        return len(self.segments)


@dataclass(slots=True)
class ClipSelectionConfig:
    padding_before_s: float = 5.0
    padding_after_s: float = 4.0
    minimum_duration_s: float = 10.0
    bout_gap_s: float = 0.5


@dataclass(slots=True)
class ActivityExtractionConfig:
    max_peak_gap_s: float = 0.25
    max_activity_extension_s: float = 1.0
    floor_percentile: float = 35.0
    activity_threshold_ratio: float = 0.16
    threshold_ratio: float = 0.28
    prominence_ratio: float = 0.12
    min_peak_distance_s: float = 0.03
    max_silence_gap_s: float = 0.12
    max_connection_gap_s: float = 0.5


@dataclass(slots=True)
class SpectrogramConfig:
    nperseg: int = 2048
    noverlap_ratio: float = 0.75
    band_margin_hz: float = 5000.0
    envelope_percentile: float = 85.0
    gaussian_sigma: float = 1.0


@dataclass(slots=True)
class ReviewBatchJob:
    audio_path: Path
    json_path: Path
    output_dir: Path