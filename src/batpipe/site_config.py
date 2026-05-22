from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SiteConfig:
    recording_input_dir: str
    work_root_dir: str | None = None
    night_runs_dir: str | None = None
    detection_output_dir: str | None = None
    summary_output_dir: str | None = None
    review_output_dir: str | None = None
    night_token: str | None = None
    night_start_hour: int = 18
    night_end_hour: int = 12
    batdetect2_bin: str = "batdetect2"
    detection_threshold: float | None = 0.35
    model: str | None = None
    subset_limit: int | None = None
    name_contains: list[str] = field(default_factory=list)
    extra_args: list[str] = field(default_factory=list)
    noise_reduction_enabled: bool = False
    noise_reduction_output_dir: str | None = None
    noise_reduction_mode: str = "spectral_subtract"
    noise_reduction_n_fft: int = 1024
    noise_reduction_hop: int = 128
    noise_reduction_percentile: float = 20.0
    noise_reduction_spectral_subtract_oversubtract: float = 2.5
    noise_reduction_spectral_subtract_floor_ratio: float = 0.01
    noise_reduction_spectral_subtract_smoothing_bins: int = 7
    noise_reduction_margin_db: float = 6.0
    noise_reduction_softness_db: float = 3.0
    noise_reduction_floor_gain: float = 0.05
    clip_start_s: float | None = None
    clip_duration_s: float | None = None
    padding_before_s: float = 5.0
    padding_after_s: float = 4.0
    minimum_duration_s: float = 10.0
    bout_gap_s: float = 0.5
    slowdown_factor: int = 8
    max_freq_hz: float = 120_000.0
    write_mp3: bool = True
    ffmpeg_bin: str = "ffmpeg"
    mp3_bitrate: str = "192k"
    continue_on_error: bool = True


def _resolve_config_path_value(config_path: Path, path_value: object) -> str | None:
    if path_value in (None, ""):
        return None

    candidate = Path(str(path_value)).expanduser()
    if not candidate.is_absolute():
        candidate = (config_path.parent / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return str(candidate)


def _default_child_dir(root_dir: str | None, child_name: str) -> str | None:
    if not root_dir:
        return None
    return str((Path(root_dir) / child_name).resolve())


def load_site_config(path: Path) -> SiteConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Site config must be a JSON object.")

    work_root_dir = _resolve_config_path_value(path, payload.get("work_root_dir"))
    explicit_night_runs_dir = _resolve_config_path_value(path, payload.get("night_runs_dir"))
    detection_output_dir = _resolve_config_path_value(path, payload.get("detection_output_dir"))
    noise_reduction_output_dir = _resolve_config_path_value(path, payload.get("noise_reduction_output_dir"))
    summary_output_dir = _resolve_config_path_value(path, payload.get("summary_output_dir"))
    review_output_value = payload.get("review_output_dir", payload.get("validation_output_dir"))
    review_output_dir = _resolve_config_path_value(path, review_output_value)

    return SiteConfig(
        recording_input_dir=str(_resolve_config_path_value(path, payload["recording_input_dir"])),
        work_root_dir=work_root_dir,
        night_runs_dir=(explicit_night_runs_dir or _default_child_dir(work_root_dir, "night-runs")),
        detection_output_dir=(detection_output_dir or _default_child_dir(work_root_dir, "detections")),
        summary_output_dir=(summary_output_dir or _default_child_dir(work_root_dir, "summary")),
        review_output_dir=(review_output_dir or _default_child_dir(work_root_dir, "review")),
        night_token=(str(payload["night_token"]) if payload.get("night_token") not in (None, "") else None),
        night_start_hour=int(payload.get("night_start_hour", 18)),
        night_end_hour=int(payload.get("night_end_hour", 12)),
        batdetect2_bin=str(payload.get("batdetect2_bin", "batdetect2")),
        detection_threshold=(float(payload["detection_threshold"]) if payload.get("detection_threshold") is not None else None),
        model=(str(payload["model"]) if payload.get("model") is not None else None),
        subset_limit=(int(payload["subset_limit"]) if payload.get("subset_limit") is not None else None),
        name_contains=[str(item) for item in payload.get("name_contains", [])],
        extra_args=[str(item) for item in payload.get("extra_args", [])],
        noise_reduction_enabled=bool(payload.get("noise_reduction_enabled", False)),
        noise_reduction_output_dir=noise_reduction_output_dir,
        noise_reduction_mode=str(payload.get("noise_reduction_mode", "spectral_subtract")),
        noise_reduction_n_fft=int(payload.get("noise_reduction_n_fft", 1024)),
        noise_reduction_hop=int(payload.get("noise_reduction_hop", 128)),
        noise_reduction_percentile=float(payload.get("noise_reduction_percentile", 20.0)),
        noise_reduction_spectral_subtract_oversubtract=float(payload.get("noise_reduction_spectral_subtract_oversubtract", 2.5)),
        noise_reduction_spectral_subtract_floor_ratio=float(payload.get("noise_reduction_spectral_subtract_floor_ratio", 0.01)),
        noise_reduction_spectral_subtract_smoothing_bins=int(payload.get("noise_reduction_spectral_subtract_smoothing_bins", 7)),
        noise_reduction_margin_db=float(payload.get("noise_reduction_margin_db", 6.0)),
        noise_reduction_softness_db=float(payload.get("noise_reduction_softness_db", 3.0)),
        noise_reduction_floor_gain=float(payload.get("noise_reduction_floor_gain", 0.05)),
        clip_start_s=(float(payload["clip_start_s"]) if payload.get("clip_start_s") is not None else None),
        clip_duration_s=(float(payload["clip_duration_s"]) if payload.get("clip_duration_s") is not None else None),
        padding_before_s=float(payload.get("padding_before_s", 5.0)),
        padding_after_s=float(payload.get("padding_after_s", 4.0)),
        minimum_duration_s=float(payload.get("minimum_duration_s", 10.0)),
        bout_gap_s=float(payload.get("bout_gap_s", 0.5)),
        slowdown_factor=int(payload.get("slowdown_factor", 8)),
        max_freq_hz=float(payload.get("max_freq_hz", 120_000.0)),
        write_mp3=bool(payload.get("write_mp3", True)),
        ffmpeg_bin=str(payload.get("ffmpeg_bin", "ffmpeg")),
        mp3_bitrate=str(payload.get("mp3_bitrate", "192k")),
        continue_on_error=bool(payload.get("continue_on_error", True)),
    )


def resolve_site_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    return Path(path_value).expanduser().resolve()