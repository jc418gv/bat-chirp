from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import json

from batpipe.audiomoth import is_in_night_window, parse_audiomoth_timestamp
from batpipe.validate import export_validation_clip


@dataclass(slots=True)
class ValidationBatchJob:
    audio_path: Path
    json_path: Path
    output_dir: Path


def write_review_assets_csv(items: list[dict[str, object]], output_path: Path) -> Path:
    fieldnames = [
        "audio_file",
        "json_file",
        "output_dir",
        "sample_local_time",
        "spectrogram_png",
        "clip_wav",
        "clip_mp3",
        "audible_wav",
        "audible_mp3",
        "report_json",
        "clip_start_s",
        "clip_end_s",
        "selected_bout_start_s",
        "selected_bout_end_s",
        "expanded_train_start_s",
        "expanded_train_end_s",
        "expanded_train_segment_count",
        "detections_in_clip",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow({field: item.get(field) for field in fieldnames})
    return output_path


def derive_night_token(audio_paths: list[Path], requested_night_token: str | None = None) -> str:
    if requested_night_token:
        return requested_night_token
    if not audio_paths:
        return "unknown-night"
    earliest_recording = min(parse_audiomoth_timestamp(path) for path in audio_paths)
    return earliest_recording.strftime("%Y%m%d")


def _resolve_night_output_dir(
    output_dir: Path,
    audio_paths: list[Path],
    requested_night_token: str | None = None,
) -> Path:
    night_token = derive_night_token(audio_paths, requested_night_token=requested_night_token)
    if output_dir.name == night_token:
        return output_dir
    return output_dir / night_token


def discover_validation_jobs(
    audio_dir: Path,
    json_dir: Path,
    output_dir: Path,
    name_filters: list[str] | None = None,
    limit: int | None = None,
    requested_night_token: str | None = None,
    night_start_hour: int = 18,
    night_end_hour: int = 12,
) -> tuple[list[ValidationBatchJob], list[Path], int, Path]:
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    active_filters = [value for value in (name_filters or []) if value]
    audio_paths = list(
        dict.fromkeys(
            sorted(audio_dir.glob("*.WAV")) + sorted(audio_dir.glob("*.wav"))
        )
    )
    if active_filters:
        audio_paths = [
            audio_path
            for audio_path in audio_paths
            if all(filter_value in audio_path.name for filter_value in active_filters)
        ]
    if requested_night_token:
        audio_paths = [
            audio_path
            for audio_path in audio_paths
            if is_in_night_window(
                audio_path.name,
                requested_night_token,
                night_start_hour,
                night_end_hour,
            )
        ]

    discovered_count = len(audio_paths)
    if limit is not None:
        audio_paths = audio_paths[:limit]

    night_output_dir = _resolve_night_output_dir(
        output_dir,
        audio_paths,
        requested_night_token=requested_night_token,
    )

    jobs: list[ValidationBatchJob] = []
    missing_json_paths: list[Path] = []
    for audio_path in audio_paths:
        json_path = json_dir / f"{audio_path.name}.json"
        if not json_path.exists():
            missing_json_paths.append(audio_path)
            continue
        jobs.append(
            ValidationBatchJob(
                audio_path=audio_path,
                json_path=json_path,
                output_dir=night_output_dir / audio_path.stem,
            )
        )

    return jobs, missing_json_paths, discovered_count, night_output_dir


def export_validation_batch(
    audio_dir: Path,
    json_dir: Path,
    output_dir: Path,
    clip_start_s: float | None = None,
    clip_duration_s: float | None = None,
    padding_before_s: float = 5.0,
    padding_after_s: float = 4.0,
    minimum_duration_s: float = 10.0,
    bout_gap_s: float = 0.5,
    slowdown_factor: int = 8,
    max_freq_hz: float = 120_000.0,
    name_filters: list[str] | None = None,
    limit: int | None = None,
    write_mp3: bool = True,
    ffmpeg_bin: str = "ffmpeg",
    mp3_bitrate: str = "192k",
    continue_on_error: bool = True,
    requested_night_token: str | None = None,
    night_start_hour: int = 18,
    night_end_hour: int = 12,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs, missing_json_paths, discovered_count, night_output_dir = discover_validation_jobs(
        audio_dir=audio_dir,
        json_dir=json_dir,
        output_dir=output_dir,
        name_filters=name_filters,
        limit=limit,
        requested_night_token=requested_night_token,
        night_start_hour=night_start_hour,
        night_end_hour=night_end_hour,
    )
    night_output_dir.mkdir(parents=True, exist_ok=True)

    exported_items: list[dict[str, object]] = []
    failed_items: list[dict[str, object]] = []
    for job in jobs:
        try:
            result = export_validation_clip(
                audio_path=job.audio_path,
                json_path=job.json_path,
                output_dir=job.output_dir,
                clip_start_s=clip_start_s,
                clip_duration_s=clip_duration_s,
                padding_before_s=padding_before_s,
                padding_after_s=padding_after_s,
                minimum_duration_s=minimum_duration_s,
                bout_gap_s=bout_gap_s,
                slowdown_factor=slowdown_factor,
                max_freq_hz=max_freq_hz,
                write_mp3=write_mp3,
                ffmpeg_bin=ffmpeg_bin,
                mp3_bitrate=mp3_bitrate,
            )
        except Exception as exc:
            failure = {
                "audio_file": str(job.audio_path),
                "json_file": str(job.json_path),
                "output_dir": str(job.output_dir),
                "error": str(exc),
            }
            failed_items.append(failure)
            if not continue_on_error:
                raise
            continue

        exported_items.append(
            {
                "audio_file": str(job.audio_path),
                "json_file": str(job.json_path),
                "output_dir": str(job.output_dir),
                **result,
            }
        )

    summary = {
        "audio_dir": str(audio_dir),
        "json_dir": str(json_dir),
        "output_root_dir": str(output_dir),
        "night_output_dir": str(night_output_dir),
        "night": night_output_dir.name,
        "requested_night_token": requested_night_token,
        "night_start_hour": night_start_hour,
        "night_end_hour": night_end_hour,
        "write_mp3": write_mp3,
        "mp3_bitrate": mp3_bitrate if write_mp3 else None,
        "discovered_audio_files": discovered_count,
        "matched_job_count": len(jobs),
        "missing_json_count": len(missing_json_paths),
        "exported_count": len(exported_items),
        "failed_count": len(failed_items),
        "missing_json_files": [str(path) for path in missing_json_paths],
        "items": exported_items,
        "failures": failed_items,
    }
    summary_path = night_output_dir / "batch_summary.json"
    assets_csv_path = write_review_assets_csv(exported_items, night_output_dir / "review_assets.csv")
    summary["summary_json"] = str(summary_path)
    summary["review_assets_csv"] = str(assets_csv_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary