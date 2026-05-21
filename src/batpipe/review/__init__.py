from __future__ import annotations

from batpipe.review.acoustic import (
    extract_activity_extent,
    extract_activity_extent_with_config,
    extract_bat_activity,
)
from batpipe.review.audio import encode_wav_as_mp3, export_review_clip
from batpipe.review.batch import (
    _resolve_night_output_dir,
    derive_night_token,
    discover_review_jobs,
    export_review_batch,
    write_review_assets_csv,
)
from batpipe.review.clip import build_review_artifact_paths, format_sample_time_token
from batpipe.review.detection import (
    choose_clip_window,
    detections_in_window,
    group_detection_bouts,
    load_clip_detections,
    select_primary_bout,
)
from batpipe.review.models import (
    ActivityBoundaryDecision,
    ActivityExtent,
    ActivityExtractionConfig,
    ActivitySegment,
    AnnotationCategory,
    AuditAnnotation,
    BoundaryStopReason,
    CLASSIFICATION_WARNING,
    ClipDetection,
    ClipSelectionConfig,
    ClipWindow,
    DetectionBout,
    PeakEvidence,
    SpectrogramConfig,
    ReviewBatchJob,
)
from batpipe.review.report import build_review_report
from batpipe.review.spectrogram import render_review_spectrogram

__all__ = [
    "ActivityBoundaryDecision",
    "ActivityExtent",
    "ActivityExtractionConfig",
    "ActivitySegment",
    "AnnotationCategory",
    "AuditAnnotation",
    "BoundaryStopReason",
    "CLASSIFICATION_WARNING",
    "ClipDetection",
    "ClipSelectionConfig",
    "ClipWindow",
    "DetectionBout",
    "PeakEvidence",
    "ReviewBatchJob",
    "SpectrogramConfig",
    "_resolve_night_output_dir",
    "build_review_artifact_paths",
    "build_review_report",
    "choose_clip_window",
    "derive_night_token",
    "detections_in_window",
    "discover_review_jobs",
    "encode_wav_as_mp3",
    "extract_activity_extent",
    "extract_activity_extent_with_config",
    "extract_bat_activity",
    "export_review_batch",
    "export_review_clip",
    "format_sample_time_token",
    "group_detection_bouts",
    "load_clip_detections",
    "render_review_spectrogram",
    "select_primary_bout",
    "write_review_assets_csv",
]