from __future__ import annotations

from dataclasses import dataclass


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