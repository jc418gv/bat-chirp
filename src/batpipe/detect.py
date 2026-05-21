from __future__ import annotations

"""Local orchestration for an external BatDetect2 CLI installation."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import json
import shlex
import subprocess
from typing import Mapping, Sequence

from batpipe.audiomoth import is_in_night_window


AUDIO_SUFFIXES = {".wav", ".wave"}
DEFAULT_BATDETECT2_CUDA_VISIBLE_DEVICES = "0"


@dataclass(slots=True)
class DetectionPlan:
    input_dir: str
    output_dir: str
    audio_file_count: int
    selected_file_count: int
    name_filters: list[str]
    invocation_mode: str
    batdetect2_command: list[str]
    selected_files_manifest: str | None
    detection_threshold: float | None
    model: str | None
    cuda_visible_devices: str | None
    created_at_utc: str


def resolve_batdetect2_cuda_visible_devices(env: Mapping[str, str] | None = None) -> str | None:
    active_env = env or os.environ
    explicit_cuda_visible_devices = active_env.get("CUDA_VISIBLE_DEVICES")
    if explicit_cuda_visible_devices not in (None, ""):
        return explicit_cuda_visible_devices

    requested_override = active_env.get(
        "BATPIPE_BATDETECT2_CUDA_VISIBLE_DEVICES",
        DEFAULT_BATDETECT2_CUDA_VISIBLE_DEVICES,
    )
    if requested_override in (None, "", "all", "ALL", "*"):
        return None
    return requested_override


def discover_audio_files(input_dir: Path, name_filters: Sequence[str] = ()) -> list[Path]:
    lowered_filters = [item.lower() for item in name_filters]
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in AUDIO_SUFFIXES
        and all(token in path.name.lower() for token in lowered_filters)
    )


def discover_audio_files_for_night(
    input_dir: Path,
    name_filters: Sequence[str] = (),
    *,
    night_token: str | None = None,
    night_start_hour: int = 18,
    night_end_hour: int = 12,
) -> list[Path]:
    audio_files = discover_audio_files(input_dir, name_filters)
    if not night_token:
        return audio_files
    return [
        path
        for path in audio_files
        if is_in_night_window(path.name, night_token, night_start_hour, night_end_hour)
    ]


def build_detection_plan(
    *,
    input_dir: Path,
    output_dir: Path,
    batdetect2_bin: str,
    model: str | None,
    detection_threshold: float | None,
    limit: int | None,
    name_filters: Sequence[str],
    extra_args: Sequence[str],
    night_token: str | None = None,
    night_start_hour: int = 18,
    night_end_hour: int = 12,
) -> DetectionPlan:
    audio_files = discover_audio_files_for_night(
        input_dir,
        name_filters,
        night_token=night_token,
        night_start_hour=night_start_hour,
        night_end_hour=night_end_hour,
    )
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {input_dir}")

    selected_files = audio_files[:limit] if limit else audio_files
    selected_files_manifest: Path | None = None
    requires_file_list = bool(limit or name_filters or night_token)

    if requires_file_list:
        invocation_mode = "file_list"
        selected_files_manifest = output_dir / "selected_files.txt"
        selected_files_manifest.parent.mkdir(parents=True, exist_ok=True)
        selected_files_manifest.write_text(
            "\n".join(str(path) for path in selected_files) + "\n",
            encoding="utf-8",
        )
        command = [
            batdetect2_bin,
            "process",
            "file_list",
            str(selected_files_manifest),
            str(output_dir),
        ]
    else:
        invocation_mode = "directory"
        command = [
            batdetect2_bin,
            "process",
            "directory",
            str(input_dir),
            str(output_dir),
        ]

    if model:
        command.extend(["--model", model])
    if detection_threshold is not None:
        command.extend(["--detection-threshold", str(detection_threshold)])
    command.extend(extra_args)

    return DetectionPlan(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        audio_file_count=len(audio_files),
        selected_file_count=len(selected_files),
        name_filters=list(name_filters),
        invocation_mode=invocation_mode,
        batdetect2_command=command,
        selected_files_manifest=(str(selected_files_manifest) if selected_files_manifest else None),
        detection_threshold=detection_threshold,
        model=model,
        cuda_visible_devices=resolve_batdetect2_cuda_visible_devices(),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def write_detection_plan(plan: DetectionPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(plan), indent=2), encoding="utf-8")


def command_as_shell_string(command: Sequence[str]) -> str:
    return shlex.join(command)


def run_detection_plan(plan: DetectionPlan, dry_run: bool = False) -> subprocess.CompletedProcess[str] | None:
    if dry_run:
        return None
    child_env = dict(os.environ)
    if plan.cuda_visible_devices is not None:
        child_env["CUDA_VISIBLE_DEVICES"] = plan.cuda_visible_devices
    return subprocess.run(plan.batdetect2_command, check=True, text=True, env=child_env)
