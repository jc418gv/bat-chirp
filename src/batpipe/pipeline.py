from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from batpipe.aggregate import summarize_detection_directory
from batpipe.detect import (
    build_detection_plan,
    command_as_shell_string,
    discover_audio_files_for_night,
    run_detection_plan,
    write_detection_plan,
)
from batpipe.noise_floor import NoiseReductionConfig, reduce_noise_for_files
from batpipe.review import export_review_batch
from batpipe.review_site import build_review_site
from batpipe.site_config import SiteConfig, resolve_site_path

ProgressCallback = Callable[[str, dict[str, object]], None]


def _ensure_input_directory(path: Path, field_name: str) -> None:
    if not path.exists():
        raise ValueError(f"{field_name} does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"{field_name} is not a directory: {path}")


def _ensure_output_directory(path: Path, field_name: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise ValueError(
            f"{field_name} is not writable: {path}. Choose a writable directory in the site config, "
            f"for example a path under this repo's work/ folder."
        ) from exc
    except FileNotFoundError as exc:
        raise ValueError(f"{field_name} could not be created: {path}") from exc


def _stringify_path_mapping(values: dict[str, object]) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in values.items()
    }


def _default_noise_reduction_output_dir(config: SiteConfig, detection_output_dir: Path) -> Path:
    if config.noise_reduction_output_dir:
        return resolve_site_path(config.noise_reduction_output_dir) or (detection_output_dir.parent / "noise-reduced")
    return detection_output_dir.parent / "noise-reduced"


def _build_noise_reduction_config(config: SiteConfig) -> NoiseReductionConfig:
    return NoiseReductionConfig(
        mode=config.noise_reduction_mode,
        n_fft=config.noise_reduction_n_fft,
        hop=config.noise_reduction_hop,
        noise_floor_percentile=config.noise_reduction_percentile,
        spectral_subtract_oversubtract=config.noise_reduction_spectral_subtract_oversubtract,
        spectral_subtract_floor_ratio=config.noise_reduction_spectral_subtract_floor_ratio,
        spectral_subtract_smoothing_bins=config.noise_reduction_spectral_subtract_smoothing_bins,
        margin_db=config.noise_reduction_margin_db,
        softness_db=config.noise_reduction_softness_db,
        floor_gain=config.noise_reduction_floor_gain,
    )


def run_night_pipeline(
    config: SiteConfig,
    *,
    dry_run: bool = False,
    skip_detection: bool = False,
    skip_summary: bool = False,
    skip_review: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    input_dir = resolve_site_path(config.recording_input_dir)
    detection_output_dir = resolve_site_path(config.detection_output_dir)
    summary_output_dir = resolve_site_path(config.summary_output_dir)
    review_output_dir = resolve_site_path(config.review_output_dir)
    if input_dir is None or detection_output_dir is None:
        raise ValueError("recording_input_dir and detection_output_dir are required.")

    _ensure_input_directory(input_dir, "recording_input_dir")
    _ensure_output_directory(detection_output_dir, "detection_output_dir")

    detection_input_dir = input_dir
    noise_reduction_output_dir: Path | None = None
    if config.noise_reduction_enabled:
        noise_reduction_output_dir = _default_noise_reduction_output_dir(config, detection_output_dir)
        _ensure_output_directory(noise_reduction_output_dir, "noise_reduction_output_dir")
        if not skip_detection and not dry_run:
            selected_audio_files = discover_audio_files_for_night(
                input_dir,
                config.name_contains,
                night_token=config.night_token,
                night_start_hour=config.night_start_hour,
                night_end_hour=config.night_end_hour,
            )
            selected_audio_files = selected_audio_files[: config.subset_limit] if config.subset_limit else selected_audio_files
            if progress_callback is not None:
                progress_callback(
                    "noise_reduction_started",
                    {
                        "selected_audio_files": len(selected_audio_files),
                        "noise_reduction_output_dir": str(noise_reduction_output_dir),
                    },
                )
            reduce_noise_for_files(
                selected_audio_files,
                input_dir=input_dir,
                output_dir=noise_reduction_output_dir,
                config=_build_noise_reduction_config(config),
            )
            if progress_callback is not None:
                progress_callback(
                    "noise_reduction_completed",
                    {
                        "selected_audio_files": len(selected_audio_files),
                        "noise_reduction_output_dir": str(noise_reduction_output_dir),
                    },
                )
            detection_input_dir = noise_reduction_output_dir

    plan = build_detection_plan(
        input_dir=detection_input_dir,
        output_dir=detection_output_dir,
        batdetect2_bin=config.batdetect2_bin,
        model=config.model,
        detection_threshold=config.detection_threshold,
        limit=config.subset_limit,
        name_filters=config.name_contains,
        extra_args=config.extra_args,
        night_token=config.night_token,
        night_start_hour=config.night_start_hour,
        night_end_hour=config.night_end_hour,
    )
    manifest_path = detection_output_dir / "run_manifest.json"
    write_detection_plan(plan, manifest_path)

    result: dict[str, object] = {
        "config": asdict(config),
        "detection_manifest": str(manifest_path),
        "detection_command": command_as_shell_string(plan.batdetect2_command),
        "detection_input_dir": str(detection_input_dir),
        "review_audio_dir": str(input_dir),
        "noise_reduction_output_dir": str(noise_reduction_output_dir) if noise_reduction_output_dir is not None else None,
        "selected_audio_files": plan.selected_file_count,
        "discovered_audio_files": plan.audio_file_count,
        "dry_run": dry_run,
        "skip_detection": skip_detection,
        "skip_summary": skip_summary,
        "skip_review": skip_review,
        "night_token": config.night_token,
        "night_start_hour": config.night_start_hour,
        "night_end_hour": config.night_end_hour,
    }

    if not skip_detection:
        if progress_callback is not None:
            progress_callback(
                "detection_started",
                {
                    "selected_audio_files": plan.selected_file_count,
                    "discovered_audio_files": plan.audio_file_count,
                    "detection_command": result["detection_command"],
                },
            )
        run_detection_plan(plan, dry_run=dry_run)

    if dry_run:
        return result

    if not skip_summary and summary_output_dir is not None:
        _ensure_output_directory(summary_output_dir, "summary_output_dir")
        if progress_callback is not None:
            progress_callback(
                "summary_started",
                {
                    "detection_output_dir": str(detection_output_dir),
                    "summary_output_dir": str(summary_output_dir),
                },
            )
        result["summary_outputs"] = _stringify_path_mapping(
            summarize_detection_directory(detection_output_dir, summary_output_dir)
        )

    if not skip_review and review_output_dir is not None:
        _ensure_output_directory(review_output_dir, "review_output_dir")
        if progress_callback is not None:
            progress_callback(
                "review_started",
                {
                    "review_output_dir": str(review_output_dir),
                    "requested_night_token": config.night_token,
                },
            )
        review_outputs = export_review_batch(
            audio_dir=input_dir,
            json_dir=detection_output_dir,
            output_dir=review_output_dir,
            noise_reduced_audio_dir=noise_reduction_output_dir,
            clip_start_s=config.clip_start_s,
            clip_duration_s=config.clip_duration_s,
            padding_before_s=config.padding_before_s,
            padding_after_s=config.padding_after_s,
            minimum_duration_s=config.minimum_duration_s,
            bout_gap_s=config.bout_gap_s,
            slowdown_factor=config.slowdown_factor,
            max_freq_hz=config.max_freq_hz,
            name_filters=config.name_contains,
            limit=config.subset_limit,
            write_mp3=config.write_mp3,
            ffmpeg_bin=config.ffmpeg_bin,
            mp3_bitrate=config.mp3_bitrate,
            continue_on_error=config.continue_on_error,
            requested_night_token=config.night_token,
            night_start_hour=config.night_start_hour,
            night_end_hour=config.night_end_hour,
            progress_callback=progress_callback,
        )
        if progress_callback is not None:
            progress_callback(
                "review_site_started",
                {
                    "night_output_dir": str(review_outputs["night_output_dir"]),
                    "review_item_count": len(review_outputs.get("items", [])),
                },
            )
        review_outputs.update(
            build_review_site(
                night_output_dir=Path(str(review_outputs["night_output_dir"])),
                review_items=list(review_outputs.get("items", [])),
                summary_outputs=result.get("summary_outputs") if isinstance(result.get("summary_outputs"), dict) else None,
            )
        )
        result["review_outputs"] = review_outputs

    if progress_callback is not None:
        progress_callback(
            "pipeline_completed",
            {
                "night_token": config.night_token,
                "selected_audio_files": plan.selected_file_count,
                "review_output_dir": str(review_output_dir) if review_output_dir is not None else None,
            },
        )

    return result