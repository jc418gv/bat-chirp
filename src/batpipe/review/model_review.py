from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CLASSIFICATION_WARNING = (
    "BatDetect2 class labels in this workflow are raw model outputs and not reliable classifications."
)


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
    adjacent_segment_merge_gap_s: float = 2.0
    detection_gap_min_gap_s: float = 0.8
    detection_gap_cadence_multiplier: float = 3.0
    floor_percentile: float = 35.0
    activity_threshold_ratio: float = 0.16
    activity_modulation_ratio: float = 0.05
    concentration_threshold: float = 0.22
    min_anchor_contrast_db: float = 8.0
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