from __future__ import annotations

"""Local reporting built from upstream BatDetect2 JSON outputs."""

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .audiomoth import normalize_recording_name, parse_audiomoth_timestamp


@dataclass(slots=True)
class DetectionRecord:
    source_file: str
    recording_start: datetime
    detection_start: datetime | None
    detection_end: datetime | None
    start_time_s: float | None
    end_time_s: float | None
    duration_s: float | None
    det_prob: float | None
    class_prob: float | None
    predicted_class: str
    low_freq_hz: float | None
    high_freq_hz: float | None
    json_path: str


def _to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _truncate_to_hour(value: datetime) -> datetime:
    return value.replace(minute=0, second=0, microsecond=0)


def load_detection_records(input_dir: Path) -> tuple[list[DetectionRecord], list[dict[str, object]]]:
    records: list[DetectionRecord] = []
    file_rows: list[dict[str, object]] = []

    json_paths = sorted(
        path
        for path in input_dir.glob("*.json")
        if path.name != "run_manifest.json"
    )

    for json_path in json_paths:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        detections = payload.get("annotation") or payload.get("detections") or []
        if not isinstance(detections, list):
            continue

        source_file = normalize_recording_name(json_path.name)
        recording_start = parse_audiomoth_timestamp(source_file)

        per_file_records: list[DetectionRecord] = []
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            start_time_s = _to_float(detection.get("start_time"))
            end_time_s = _to_float(detection.get("end_time"))
            duration_s = None
            if start_time_s is not None and end_time_s is not None:
                duration_s = max(0.0, end_time_s - start_time_s)
            detection_start = (
                recording_start + timedelta(seconds=start_time_s)
                if start_time_s is not None
                else None
            )
            detection_end = (
                recording_start + timedelta(seconds=end_time_s)
                if end_time_s is not None
                else None
            )

            record = DetectionRecord(
                source_file=source_file,
                recording_start=recording_start,
                detection_start=detection_start,
                detection_end=detection_end,
                start_time_s=start_time_s,
                end_time_s=end_time_s,
                duration_s=duration_s,
                det_prob=_to_float(detection.get("det_prob", detection.get("detection_score"))),
                class_prob=_to_float(detection.get("class_prob", detection.get("class_score"))),
                predicted_class=str(detection.get("class") or detection.get("predicted_class") or "unknown"),
                low_freq_hz=_to_float(detection.get("low_freq")),
                high_freq_hz=_to_float(detection.get("high_freq")),
                json_path=str(json_path),
            )
            per_file_records.append(record)
            records.append(record)

        det_probs = [record.det_prob for record in per_file_records if record.det_prob is not None]
        top_classes = Counter(record.predicted_class for record in per_file_records if record.predicted_class)
        file_rows.append(
            {
                "source_file": source_file,
                "recording_start": recording_start.isoformat(sep=" "),
                "json_path": str(json_path),
                "detection_count": len(per_file_records),
                "max_det_prob": max(det_probs) if det_probs else None,
                "mean_det_prob": (sum(det_probs) / len(det_probs)) if det_probs else None,
                "top_predicted_class": top_classes.most_common(1)[0][0] if top_classes else None,
                "predicted_classes": ";".join(class_name for class_name, _ in top_classes.most_common(5)),
            }
        )

    file_rows.sort(key=lambda row: row["recording_start"])
    return records, file_rows


def write_flat_detections(records: list[DetectionRecord], output_path: Path) -> None:
    fieldnames = [
        "source_file",
        "recording_start",
        "detection_start",
        "detection_end",
        "start_time_s",
        "end_time_s",
        "duration_s",
        "det_prob",
        "class_prob",
        "predicted_class",
        "low_freq_hz",
        "high_freq_hz",
        "json_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "source_file": record.source_file,
                    "recording_start": record.recording_start.isoformat(sep=" "),
                    "detection_start": record.detection_start.isoformat(sep=" ") if record.detection_start else None,
                    "detection_end": record.detection_end.isoformat(sep=" ") if record.detection_end else None,
                    "start_time_s": record.start_time_s,
                    "end_time_s": record.end_time_s,
                    "duration_s": record.duration_s,
                    "det_prob": record.det_prob,
                    "class_prob": record.class_prob,
                    "predicted_class": record.predicted_class,
                    "low_freq_hz": record.low_freq_hz,
                    "high_freq_hz": record.high_freq_hz,
                    "json_path": record.json_path,
                }
            )


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_hourly_summary(
    records: list[DetectionRecord],
    file_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    hour_files: dict[datetime, set[str]] = defaultdict(set)
    for row in file_rows:
        hour_start = _truncate_to_hour(datetime.fromisoformat(str(row["recording_start"])))
        hour_files[hour_start].add(str(row["source_file"]))

    hourly: dict[datetime, dict[str, object]] = {}
    class_counters: dict[datetime, Counter[str]] = defaultdict(Counter)

    for hour_start, files_seen in hour_files.items():
        hourly[hour_start] = {
            "hour_start": hour_start.isoformat(sep=" "),
            "files_seen": len(files_seen),
            "positive_files": 0,
            "total_detections": 0,
            "mean_det_prob": None,
            "max_det_prob": None,
            "top_predicted_class": None,
        }

    per_hour_probs: dict[datetime, list[float]] = defaultdict(list)
    per_hour_positive_files: dict[datetime, set[str]] = defaultdict(set)

    for record in records:
        hour_reference = record.detection_start or record.recording_start
        hour_start = _truncate_to_hour(hour_reference)
        row = hourly.setdefault(
            hour_start,
            {
                "hour_start": hour_start.isoformat(sep=" "),
                "files_seen": 0,
                "positive_files": 0,
                "total_detections": 0,
                "mean_det_prob": None,
                "max_det_prob": None,
                "top_predicted_class": None,
            },
        )
        row["total_detections"] = int(row["total_detections"]) + 1
        if record.det_prob is not None:
            per_hour_probs[hour_start].append(record.det_prob)
            current_max = row["max_det_prob"]
            row["max_det_prob"] = record.det_prob if current_max is None else max(float(current_max), record.det_prob)
        per_hour_positive_files[hour_start].add(record.source_file)
        if record.predicted_class:
            class_counters[hour_start][record.predicted_class] += 1

    for hour_start, row in sorted(hourly.items()):
        probs = per_hour_probs.get(hour_start, [])
        row["positive_files"] = len(per_hour_positive_files.get(hour_start, set()))
        row["mean_det_prob"] = (sum(probs) / len(probs)) if probs else None
        top_classes = class_counters.get(hour_start)
        row["top_predicted_class"] = top_classes.most_common(1)[0][0] if top_classes else None

    return [hourly[key] for key in sorted(hourly)]


def build_nightly_summary(
    records: list[DetectionRecord],
    file_rows: list[dict[str, object]],
    hourly_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    positive_files = sum(1 for row in file_rows if int(row["detection_count"]) > 0)
    det_probs = [record.det_prob for record in records if record.det_prob is not None]
    top_classes = Counter(record.predicted_class for record in records if record.predicted_class)

    first_recording = file_rows[0]["recording_start"] if file_rows else None
    last_recording = file_rows[-1]["recording_start"] if file_rows else None
    active_hours = sum(1 for row in hourly_rows if int(row["total_detections"]) > 0)

    return [
        {
            "first_recording": first_recording,
            "last_recording": last_recording,
            "total_files": len(file_rows),
            "positive_files": positive_files,
            "total_detections": len(records),
            "hours_with_activity": active_hours,
            "mean_detections_per_positive_file": (len(records) / positive_files) if positive_files else None,
            "max_det_prob": max(det_probs) if det_probs else None,
            "top_predicted_class": top_classes.most_common(1)[0][0] if top_classes else None,
        }
    ]


def build_review_queue(file_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    positive_rows = [row for row in file_rows if int(row["detection_count"]) > 0]
    positive_rows.sort(
        key=lambda row: (
            int(row["detection_count"]),
            float(row["max_det_prob"]) if row["max_det_prob"] is not None else -1.0,
        ),
        reverse=True,
    )
    queue = []
    for rank, row in enumerate(positive_rows, start=1):
        queue.append({"rank": rank, **row})
    return queue


def summarize_detection_directory(input_dir: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records, file_rows = load_detection_records(input_dir)
    hourly_rows = build_hourly_summary(records, file_rows)
    nightly_rows = build_nightly_summary(records, file_rows, hourly_rows)
    review_rows = build_review_queue(file_rows)

    outputs = {
        "detections_flat": output_dir / "detections_flat.csv",
        "file_summary": output_dir / "file_summary.csv",
        "review_queue": output_dir / "review_queue.csv",
        "hourly_summary": output_dir / "hourly_summary.csv",
        "nightly_summary": output_dir / "nightly_summary.csv",
    }

    write_flat_detections(records, outputs["detections_flat"])
    write_csv(file_rows, outputs["file_summary"])
    write_csv(review_rows, outputs["review_queue"])
    write_csv(hourly_rows, outputs["hourly_summary"])
    write_csv(nightly_rows, outputs["nightly_summary"])
    return outputs
