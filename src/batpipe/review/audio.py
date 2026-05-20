from __future__ import annotations

from pathlib import Path
import json

from batpipe.review.acoustic import analyze_candidate_train
from batpipe.review.clip import build_review_artifact_paths
from batpipe.review.detection import choose_clip_window, detections_in_window, load_clip_detections
from batpipe.review.models import ClipSelectionConfig, PeakDetectionConfig, SpectrogramConfig
from batpipe.review.report import build_review_report
from batpipe.review.spectrogram import render_review_spectrogram


def _read_wav_mono(audio_path: Path):
    import warnings
    from scipy.io import wavfile

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Chunk \(non-data\) not understood, skipping it\.")
        sample_rate_hz, audio = wavfile.read(audio_path, mmap=True)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio[:, 0]
    return sample_rate_hz, audio


def encode_wav_as_mp3(
    wav_path: Path,
    mp3_path: Path,
    ffmpeg_bin: str = "ffmpeg",
    bitrate: str = "192k",
    sample_rate_hz: int | None = None,
) -> Path:
    import shutil
    import subprocess

    ffmpeg_path = shutil.which(ffmpeg_bin)
    if ffmpeg_path is None:
        raise FileNotFoundError(f"ffmpeg executable not found: {ffmpeg_bin}")

    command = [
        ffmpeg_path,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(wav_path),
    ]
    if sample_rate_hz is not None:
        command.extend(["-ar", str(sample_rate_hz)])
    command.extend([
        "-codec:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        str(mp3_path),
    ])
    subprocess.run(command, check=True)
    return mp3_path


def export_review_clip(
    audio_path: Path,
    json_path: Path,
    output_dir: Path,
    clip_start_s: float | None = None,
    clip_duration_s: float | None = None,
    padding_before_s: float = 5.0,
    padding_after_s: float = 4.0,
    minimum_duration_s: float = 10.0,
    bout_gap_s: float = 0.5,
    slowdown_factor: int = 8,
    max_freq_hz: float = 120_000.0,
    write_mp3: bool = False,
    ffmpeg_bin: str = "ffmpeg",
    mp3_bitrate: str = "192k",
    clip_selection_config: ClipSelectionConfig | None = None,
    peak_detection_config: PeakDetectionConfig | None = None,
    spectrogram_config: SpectrogramConfig | None = None,
) -> dict[str, object]:
    from scipy.io import wavfile

    sample_rate_hz, audio = _read_wav_mono(audio_path)

    recording_duration_s, detections, payload = load_clip_detections(json_path)
    if recording_duration_s is None:
        recording_duration_s = len(audio) / float(sample_rate_hz)

    window, selected_bout = choose_clip_window(
        detections=detections,
        recording_duration_s=recording_duration_s,
        clip_start_s=clip_start_s,
        clip_duration_s=clip_duration_s,
        padding_before_s=padding_before_s,
        padding_after_s=padding_after_s,
        minimum_duration_s=minimum_duration_s,
        bout_gap_s=bout_gap_s,
        clip_selection_config=clip_selection_config,
    )
    detections_for_clip = detections_in_window(detections, window)

    start_index = max(0, int(window.start_time_s * sample_rate_hz))
    end_index = min(len(audio), int(round(window.end_time_s * sample_rate_hz)))
    clip_audio = audio[start_index:end_index]
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = build_review_artifact_paths(
        output_dir=output_dir,
        audio_path=audio_path,
        clip_start_s=window.start_time_s,
        slowdown_factor=slowdown_factor,
    )
    sample_local_time = str(artifact_paths["sample_local_time"])

    clip_wav_path = artifact_paths["clip_wav"]
    wavfile.write(clip_wav_path, sample_rate_hz, clip_audio)
    clip_mp3_path: Path | None = None
    if write_mp3:
        clip_mp3_path = encode_wav_as_mp3(
            clip_wav_path,
            artifact_paths["clip_mp3"],
            ffmpeg_bin=ffmpeg_bin,
            bitrate=mp3_bitrate,
            sample_rate_hz=min(int(sample_rate_hz), 48_000),
        )

    audible_sample_rate_hz = max(1, int(round(sample_rate_hz / slowdown_factor)))
    audible_wav_path = artifact_paths["audible_wav"]
    wavfile.write(audible_wav_path, audible_sample_rate_hz, clip_audio)
    audible_mp3_path: Path | None = None
    if write_mp3:
        audible_mp3_path = encode_wav_as_mp3(
            audible_wav_path,
            artifact_paths["audible_mp3"],
            ffmpeg_bin=ffmpeg_bin,
            bitrate=mp3_bitrate,
            sample_rate_hz=min(int(audible_sample_rate_hz), 48_000),
        )

    expanded_train = analyze_candidate_train(
        audio=clip_audio,
        sample_rate_hz=sample_rate_hz,
        window=window,
        selected_bout=selected_bout,
        max_freq_hz=max_freq_hz,
        peak_detection_config=peak_detection_config,
        spectrogram_config=spectrogram_config,
    )

    spectrogram_path = artifact_paths["spectrogram_png"]
    render_review_spectrogram(
        audio=clip_audio,
        sample_rate_hz=sample_rate_hz,
        window=window,
        detections=detections_for_clip,
        selected_bout=selected_bout,
        expanded_train=expanded_train,
        output_path=spectrogram_path,
        max_freq_hz=max_freq_hz,
        title=audio_path.name,
        spectrogram_config=spectrogram_config,
    )

    report_path = artifact_paths["report_json"]
    report = build_review_report(
        audio_path=audio_path,
        json_path=json_path,
        payload=payload,
        sample_local_time=sample_local_time,
        window=window,
        selected_bout=selected_bout,
        expanded_train=expanded_train,
        sample_rate_hz=sample_rate_hz,
        audible_sample_rate_hz=audible_sample_rate_hz,
        slowdown_factor=slowdown_factor,
        write_mp3=write_mp3,
        mp3_bitrate=mp3_bitrate,
        recording_duration_s=recording_duration_s,
        padding_before_s=padding_before_s,
        padding_after_s=padding_after_s,
        bout_gap_s=bout_gap_s,
        clip_start_s=clip_start_s,
        detections_for_clip=detections_for_clip,
        clip_mp3_path=clip_mp3_path,
        audible_mp3_path=audible_mp3_path,
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
        "sample_local_time": sample_local_time,
        "clip_wav": str(clip_wav_path),
        "audible_wav": str(audible_wav_path),
        "clip_mp3": str(clip_mp3_path) if clip_mp3_path else None,
        "audible_mp3": str(audible_mp3_path) if audible_mp3_path else None,
        "spectrogram_png": str(spectrogram_path),
        "report_json": str(report_path),
        "clip_start_s": window.start_time_s,
        "clip_end_s": window.end_time_s,
        "selected_bout_start_s": selected_bout.start_time_s if selected_bout else None,
        "selected_bout_end_s": selected_bout.end_time_s if selected_bout else None,
        "expanded_train_start_s": expanded_train.start_time_s + window.start_time_s if expanded_train else None,
        "expanded_train_end_s": expanded_train.end_time_s + window.start_time_s if expanded_train else None,
        "expanded_train_segment_count": expanded_train.segment_count if expanded_train else 0,
        "detections_in_clip": len(detections_for_clip),
    }