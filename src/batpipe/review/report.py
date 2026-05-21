from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from batpipe.review.models import ActivityExtent, CLASSIFICATION_WARNING, ClipDetection, ClipWindow, DetectionBout


def build_review_report(
    *,
    audio_path: Path,
    json_path: Path,
    payload: dict[str, object],
    sample_local_time: str,
    window: ClipWindow,
    selected_bout: DetectionBout | None,
    activity_extent: ActivityExtent | None,
    sample_rate_hz: int,
    audible_sample_rate_hz: int,
    slowdown_factor: int,
    write_mp3: bool,
    mp3_bitrate: str,
    recording_duration_s: float,
    padding_before_s: float,
    padding_after_s: float,
    bout_gap_s: float,
    clip_start_s: float | None,
    detections_for_clip: list[ClipDetection],
    clip_mp3_path: Path | None,
    audible_mp3_path: Path | None,
) -> dict[str, object]:
    activity_peak_concentrations = [
        item.concentration_score
        for item in (activity_extent.peak_evidence if activity_extent else [])
        if item.concentration_score is not None and item.included_in_activity
    ]
    anchor_peak_concentrations = [
        item.concentration_score
        for item in (activity_extent.peak_evidence if activity_extent else [])
        if item.concentration_score is not None and item.within_anchor
    ]

    def _mean(values: list[float | None]) -> float | None:
        if not values:
            return None
        return float(sum(values) / len(values))

    detection_start_times_recording_s = [float(item.start_time_s) for item in detections_for_clip]
    detection_end_times_recording_s = [float(item.end_time_s) for item in detections_for_clip]
    detection_start_times_clip_s = [
        float(max(0.0, item.start_time_s - window.start_time_s))
        for item in detections_for_clip
    ]
    detection_end_times_clip_s = [
        float(max(0.0, item.end_time_s - window.start_time_s))
        for item in detections_for_clip
    ]
    activity_peak_times_clip_s = [
        float(item.time_s)
        for item in (activity_extent.peak_evidence if activity_extent else [])
        if item.included_in_activity
    ]
    activity_peak_times_recording_s = [
        float(window.start_time_s + item.time_s)
        for item in (activity_extent.peak_evidence if activity_extent else [])
        if item.included_in_activity
    ]

    return {
        "audio_file": str(audio_path),
        "json_file": str(json_path),
        "sample_local_time": sample_local_time,
        "classification_warning": CLASSIFICATION_WARNING,
        "selection_mode": "explicit_window" if clip_start_s is not None else "primary_detection_bout",
        "bout_gap_s": bout_gap_s,
        "leading_context_s": padding_before_s,
        "trailing_context_s": padding_after_s,
        "clip_start_s": window.start_time_s,
        "clip_end_s": window.end_time_s,
        "clip_duration_s": window.duration_s,
        "sample_rate_hz": int(sample_rate_hz),
        "audible_sample_rate_hz": audible_sample_rate_hz,
        "slowdown_factor": slowdown_factor,
        "mp3_enabled": write_mp3,
        "mp3_bitrate": mp3_bitrate if write_mp3 else None,
        "recording_duration_s": recording_duration_s,
        "raw_model_class_label": payload.get("class_name"),
        "selected_bout_start_s": selected_bout.start_time_s if selected_bout else None,
        "selected_bout_end_s": selected_bout.end_time_s if selected_bout else None,
        "selected_bout_duration_s": selected_bout.duration_s if selected_bout else None,
        "selected_bout_detection_count": selected_bout.detection_count if selected_bout else 0,
        "selected_bout_low_freq_hz": selected_bout.min_low_freq_hz if selected_bout else None,
        "selected_bout_high_freq_hz": selected_bout.max_high_freq_hz if selected_bout else None,
        "detection_start_times_recording_s": detection_start_times_recording_s,
        "detection_end_times_recording_s": detection_end_times_recording_s,
        "detection_start_times_clip_s": detection_start_times_clip_s,
        "detection_end_times_clip_s": detection_end_times_clip_s,
        "activity_peak_times_recording_s": activity_peak_times_recording_s,
        "activity_peak_times_clip_s": activity_peak_times_clip_s,
        "activity_start_s": activity_extent.start_time_s + window.start_time_s if activity_extent else None,
        "activity_end_s": activity_extent.end_time_s + window.start_time_s if activity_extent else None,
        "activity_duration_s": activity_extent.duration_s if activity_extent else None,
        "activity_peak_count": len(activity_extent.peak_times_s) if activity_extent else 0,
        "activity_segment_count": activity_extent.segment_count if activity_extent else 0,
        "activity_mean_concentration": _mean(activity_peak_concentrations),
        "activity_min_concentration": min(activity_peak_concentrations) if activity_peak_concentrations else None,
        "anchor_mean_concentration": _mean(anchor_peak_concentrations),
        "activity_segments": [asdict(segment) for segment in activity_extent.segments] if activity_extent else [],
        "activity_peak_evidence": [asdict(item) for item in activity_extent.peak_evidence] if activity_extent else [],
        "activity_left_boundary": asdict(activity_extent.left_boundary) if activity_extent and activity_extent.left_boundary else None,
        "activity_right_boundary": asdict(activity_extent.right_boundary) if activity_extent and activity_extent.right_boundary else None,
        "clip_mp3": str(clip_mp3_path) if clip_mp3_path else None,
        "audible_mp3": str(audible_mp3_path) if audible_mp3_path else None,
        "clip_truncated_at_file_start": window.start_time_s <= 0.0,
        "clip_truncated_at_file_end": window.end_time_s >= recording_duration_s,
        "detections": [asdict(item) for item in detections_for_clip],
    }