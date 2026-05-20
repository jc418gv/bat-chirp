from __future__ import annotations

import json
from pathlib import Path

from batpipe.review.models import ClipDetection, ClipSelectionConfig, ClipWindow, DetectionBout


def _to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def load_clip_detections(json_path: Path) -> tuple[float | None, list[ClipDetection], dict[str, object]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    detections_raw = payload.get("annotation") or payload.get("detections") or []
    detections: list[ClipDetection] = []
    if isinstance(detections_raw, list):
        for item in detections_raw:
            if not isinstance(item, dict):
                continue
            start_time_s = _to_float(item.get("start_time"))
            end_time_s = _to_float(item.get("end_time"))
            if start_time_s is None or end_time_s is None:
                continue
            detections.append(
                ClipDetection(
                    start_time_s=start_time_s,
                    end_time_s=end_time_s,
                    det_prob=_to_float(item.get("det_prob", item.get("detection_score"))),
                    class_prob=_to_float(item.get("class_prob", item.get("class_score"))),
                    predicted_class=str(item.get("class") or item.get("predicted_class") or "unknown"),
                    event=str(item.get("event")) if item.get("event") is not None else None,
                    low_freq_hz=_to_float(item.get("low_freq")),
                    high_freq_hz=_to_float(item.get("high_freq")),
                )
            )
    detections.sort(key=lambda entry: (entry.start_time_s, entry.end_time_s))
    duration_s = _to_float(payload.get("duration"))
    return duration_s, detections, payload


def group_detection_bouts(
    detections: list[ClipDetection],
    max_inter_detection_gap_s: float = 0.5,
) -> list[DetectionBout]:
    if max_inter_detection_gap_s < 0:
        raise ValueError("max_inter_detection_gap_s must be non-negative.")

    if not detections:
        return []

    bouts: list[DetectionBout] = []
    current: list[ClipDetection] = [detections[0]]

    for detection in detections[1:]:
        previous = current[-1]
        gap_s = max(0.0, detection.start_time_s - previous.end_time_s)
        if gap_s <= max_inter_detection_gap_s:
            current.append(detection)
            continue

        bouts.append(
            DetectionBout(
                start_time_s=current[0].start_time_s,
                end_time_s=max(item.end_time_s for item in current),
                detections=current.copy(),
            )
        )
        current = [detection]

    bouts.append(
        DetectionBout(
            start_time_s=current[0].start_time_s,
            end_time_s=max(item.end_time_s for item in current),
            detections=current.copy(),
        )
    )
    return bouts


def select_primary_bout(bouts: list[DetectionBout]) -> DetectionBout | None:
    if not bouts:
        return None

    return max(
        bouts,
        key=lambda bout: (
            bout.detection_count,
            bout.max_det_prob if bout.max_det_prob is not None else -1.0,
            bout.duration_s,
            -bout.start_time_s,
        ),
    )


def choose_clip_window(
    detections: list[ClipDetection],
    recording_duration_s: float,
    clip_start_s: float | None = None,
    clip_duration_s: float | None = None,
    padding_before_s: float = 5.0,
    padding_after_s: float = 4.0,
    minimum_duration_s: float = 10.0,
    bout_gap_s: float = 0.5,
    clip_selection_config: ClipSelectionConfig | None = None,
) -> tuple[ClipWindow, DetectionBout | None]:
    if clip_selection_config is not None:
        padding_before_s = clip_selection_config.padding_before_s
        padding_after_s = clip_selection_config.padding_after_s
        minimum_duration_s = clip_selection_config.minimum_duration_s
        bout_gap_s = clip_selection_config.bout_gap_s

    if recording_duration_s <= 0:
        raise ValueError("Recording duration must be positive.")

    if clip_start_s is not None:
        if clip_duration_s is None or clip_duration_s <= 0:
            raise ValueError("clip_duration_s must be positive when clip_start_s is provided.")
        start_time_s = max(0.0, clip_start_s)
        end_time_s = min(recording_duration_s, start_time_s + clip_duration_s)
        if end_time_s <= start_time_s:
            raise ValueError("Requested clip window is empty.")
        return ClipWindow(start_time_s=start_time_s, end_time_s=end_time_s), None

    if not detections:
        end_time_s = min(recording_duration_s, minimum_duration_s)
        return ClipWindow(start_time_s=0.0, end_time_s=end_time_s), None

    bouts = group_detection_bouts(detections, max_inter_detection_gap_s=bout_gap_s)
    selected_bout = select_primary_bout(bouts)
    if selected_bout is None:
        end_time_s = min(recording_duration_s, minimum_duration_s)
        return ClipWindow(start_time_s=0.0, end_time_s=end_time_s), None

    start_time_s = max(0.0, selected_bout.start_time_s - padding_before_s)
    end_time_s = min(recording_duration_s, selected_bout.end_time_s + padding_after_s)

    if end_time_s - start_time_s >= minimum_duration_s:
        return ClipWindow(start_time_s=start_time_s, end_time_s=end_time_s), selected_bout

    missing_span_s = minimum_duration_s - (end_time_s - start_time_s)
    shift_left_s = min(start_time_s, missing_span_s / 2.0)
    shift_right_s = min(recording_duration_s - end_time_s, missing_span_s - shift_left_s)
    start_time_s -= shift_left_s
    end_time_s += shift_right_s

    if end_time_s - start_time_s < minimum_duration_s:
        remaining_s = minimum_duration_s - (end_time_s - start_time_s)
        if start_time_s <= 0.0:
            end_time_s = min(recording_duration_s, end_time_s + remaining_s)
        else:
            start_time_s = max(0.0, start_time_s - remaining_s)

    return ClipWindow(start_time_s=start_time_s, end_time_s=end_time_s), selected_bout


def detections_in_window(detections: list[ClipDetection], window: ClipWindow) -> list[ClipDetection]:
    return [
        item
        for item in detections
        if item.end_time_s >= window.start_time_s and item.start_time_s <= window.end_time_s
    ]