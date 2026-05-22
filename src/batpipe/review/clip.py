from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from batpipe.audiomoth import parse_audiomoth_timestamp


def format_sample_time_token(audio_path: Path, clip_start_s: float) -> str:
    sample_time = parse_audiomoth_timestamp(audio_path) + timedelta(seconds=max(0.0, clip_start_s))
    return sample_time.strftime("%H%M%S")


def build_review_artifact_paths(
    output_dir: Path,
    audio_path: Path,
    clip_start_s: float,
    slowdown_factor: int,
) -> dict[str, Path | str]:
    sample_local_time = format_sample_time_token(audio_path, clip_start_s)
    return {
        "sample_local_time": sample_local_time,
        "clip_wav": output_dir / f"clip_original_{sample_local_time}.wav",
        "clip_mp3": output_dir / f"clip_original_{sample_local_time}.mp3",
        "audible_wav": output_dir / f"clip_audible_x{slowdown_factor}_{sample_local_time}.wav",
        "audible_mp3": output_dir / f"clip_audible_x{slowdown_factor}_{sample_local_time}.mp3",
        "spectrogram_png": output_dir / f"spectrogram_{sample_local_time}.png",
        "noise_reduced_spectrogram_png": output_dir / f"spectrogram_noise_reduced_{sample_local_time}.png",
        "report_json": output_dir / f"detections_{sample_local_time}.json",
    }