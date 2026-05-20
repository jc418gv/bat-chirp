from __future__ import annotations

"""Review export and local post-processing layered on BatDetect2 detections."""

from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
import json

from batpipe.audiomoth import parse_audiomoth_timestamp


def _read_wav_mono(audio_path: Path):
    import warnings
    from scipy.io import wavfile

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Chunk \(non-data\) not understood, skipping it\.")
        sample_rate_hz, audio = wavfile.read(audio_path, mmap=True)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio[:, 0]
    return sample_rate_hz, audio


def _format_sample_time_token(audio_path: Path, clip_start_s: float) -> str:
    sample_time = parse_audiomoth_timestamp(audio_path) + timedelta(seconds=max(0.0, clip_start_s))
    return sample_time.strftime("%H%M%S")


def _build_validation_artifact_paths(
    output_dir: Path,
    audio_path: Path,
    clip_start_s: float,
    slowdown_factor: int,
) -> dict[str, Path | str]:
    sample_local_time = _format_sample_time_token(audio_path, clip_start_s)
    return {
        "sample_local_time": sample_local_time,
        "clip_wav": output_dir / f"clip_original_{sample_local_time}.wav",
        "clip_mp3": output_dir / f"clip_original_{sample_local_time}.mp3",
        "audible_wav": output_dir / f"clip_audible_x{slowdown_factor}_{sample_local_time}.wav",
        "audible_mp3": output_dir / f"clip_audible_x{slowdown_factor}_{sample_local_time}.mp3",
        "spectrogram_png": output_dir / f"spectrogram_{sample_local_time}.png",
        "report_json": output_dir / f"detections_{sample_local_time}.json",
    }


CLASSIFICATION_WARNING = (
    "BatDetect2 class labels in this workflow are raw model outputs and are not trusted as species IDs "
    "for North American review. Treat detections as bat candidates first."
)


@dataclass(slots=True)
class ClipDetection:
    start_time_s: float
    end_time_s: float
    det_prob: float | None
    class_prob: float | None
    predicted_class: str
    event: str | None
    low_freq_hz: float | None
    high_freq_hz: float | None


@dataclass(slots=True)
class DetectionBout:
    start_time_s: float
    end_time_s: float
    detections: list[ClipDetection]

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time_s - self.start_time_s)

    @property
    def max_det_prob(self) -> float | None:
        det_probs = [item.det_prob for item in self.detections if item.det_prob is not None]
        if not det_probs:
            return None
        return max(det_probs)

    @property
    def min_low_freq_hz(self) -> float | None:
        low_freqs = [item.low_freq_hz for item in self.detections if item.low_freq_hz is not None]
        if not low_freqs:
            return None
        return min(low_freqs)

    @property
    def max_high_freq_hz(self) -> float | None:
        high_freqs = [item.high_freq_hz for item in self.detections if item.high_freq_hz is not None]
        if not high_freqs:
            return None
        return max(high_freqs)


@dataclass(slots=True)
class ClipWindow:
    start_time_s: float
    end_time_s: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time_s - self.start_time_s)


@dataclass(slots=True)
class CandidateTrainSegment:
    start_time_s: float
    end_time_s: float
    peak_times_s: list[float]

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time_s - self.start_time_s)


@dataclass(slots=True)
class CandidateTrainRange:
    start_time_s: float
    end_time_s: float
    peak_times_s: list[float]
    segments: list[CandidateTrainSegment]

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time_s - self.start_time_s)

    @property
    def segment_count(self) -> int:
        return len(self.segments)


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
) -> tuple[ClipWindow, DetectionBout | None]:
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


def _build_candidate_train_segments(
    peak_times_s,
    half_bin_s: float,
    max_peak_gap_s: float,
) -> list[CandidateTrainSegment]:
    import numpy as np

    peak_times = np.asarray(peak_times_s, dtype=float)
    if peak_times.size == 0:
        return []

    segments: list[CandidateTrainSegment] = []
    start_index = 0
    for index in range(1, peak_times.size):
        if (peak_times[index] - peak_times[index - 1]) <= max_peak_gap_s:
            continue

        segment_peak_times = peak_times[start_index:index]
        segments.append(
            CandidateTrainSegment(
                start_time_s=max(0.0, float(segment_peak_times[0] - half_bin_s)),
                end_time_s=float(segment_peak_times[-1] + half_bin_s),
                peak_times_s=[float(value) for value in segment_peak_times],
            )
        )
        start_index = index

    segment_peak_times = peak_times[start_index:]
    segments.append(
        CandidateTrainSegment(
            start_time_s=max(0.0, float(segment_peak_times[0] - half_bin_s)),
            end_time_s=float(segment_peak_times[-1] + half_bin_s),
            peak_times_s=[float(value) for value in segment_peak_times],
        )
    )
    return segments


def _segment_peak_gap_s(left: CandidateTrainSegment, right: CandidateTrainSegment) -> float:
    return float(right.peak_times_s[0] - left.peak_times_s[-1])


def _select_anchor_connected_segments(
    segments: list[CandidateTrainSegment],
    anchor_start_s: float,
    anchor_end_s: float,
    max_train_extension_s: float,
    connection_gap_s: float,
) -> list[CandidateTrainSegment]:
    if not segments:
        return []

    anchor_overlap_start_s = anchor_start_s - max_train_extension_s
    anchor_overlap_end_s = anchor_end_s + max_train_extension_s
    seed_indices = [
        index
        for index, segment in enumerate(segments)
        if segment.end_time_s >= anchor_overlap_start_s and segment.start_time_s <= anchor_overlap_end_s
    ]
    if not seed_indices:
        anchor_midpoint_s = (anchor_start_s + anchor_end_s) / 2.0

        def distance_to_anchor(segment: CandidateTrainSegment) -> float:
            if segment.start_time_s <= anchor_midpoint_s <= segment.end_time_s:
                return 0.0
            if anchor_midpoint_s < segment.start_time_s:
                return segment.start_time_s - anchor_midpoint_s
            return anchor_midpoint_s - segment.end_time_s

        seed_indices = [min(range(len(segments)), key=lambda index: distance_to_anchor(segments[index]))]

    left_index = min(seed_indices)
    right_index = max(seed_indices)

    while left_index > 0:
        if _segment_peak_gap_s(segments[left_index - 1], segments[left_index]) > connection_gap_s:
            break
        left_index -= 1

    while right_index < len(segments) - 1:
        if _segment_peak_gap_s(segments[right_index], segments[right_index + 1]) > connection_gap_s:
            break
        right_index += 1

    return segments[left_index : right_index + 1]


def _merge_candidate_train_segments(
    segments: list[CandidateTrainSegment],
) -> list[CandidateTrainSegment]:
    if not segments:
        return []

    return [
        CandidateTrainSegment(
            start_time_s=min(segment.start_time_s for segment in segments),
            end_time_s=max(segment.end_time_s for segment in segments),
            peak_times_s=[
                peak_time_s
                for segment in segments
                for peak_time_s in segment.peak_times_s
            ],
        )
    ]


def estimate_candidate_train_range(
    times_s,
    band_envelope_db,
    anchor_start_s: float,
    anchor_end_s: float,
    max_peak_gap_s: float = 0.25,
    max_train_extension_s: float = 1.0,
) -> CandidateTrainRange | None:
    import numpy as np
    from scipy import signal

    times = np.asarray(times_s, dtype=float)
    envelope = np.asarray(band_envelope_db, dtype=float)
    if times.size == 0 or envelope.size == 0 or times.size != envelope.size:
        return None

    finite_mask = np.isfinite(envelope)
    if not finite_mask.any():
        return None

    envelope = envelope.copy()
    floor = float(np.nanpercentile(envelope[finite_mask], 35))
    envelope[~finite_mask] = floor

    if times.size == 1:
        return CandidateTrainRange(float(times[0]), float(times[0]), [float(times[0])])

    time_step_s = float(np.median(np.diff(times))) if times.size > 1 else 0.01
    min_peak_distance = max(1, int(round(0.03 / max(time_step_s, 1e-6))))

    anchor_mask = (times >= anchor_start_s) & (times <= anchor_end_s)
    if anchor_mask.any():
        anchor_level = float(np.nanmax(envelope[anchor_mask]))
    else:
        closest_index = int(np.argmin(np.abs(times - ((anchor_start_s + anchor_end_s) / 2.0))))
        anchor_level = float(envelope[closest_index])

    if not np.isfinite(anchor_level):
        return None

    threshold = floor + (anchor_level - floor) * 0.28
    prominence = max((anchor_level - floor) * 0.12, 1e-6)
    peak_indices, _ = signal.find_peaks(
        envelope,
        height=threshold,
        distance=min_peak_distance,
        prominence=prominence,
    )
    if peak_indices.size == 0:
        peak_indices, _ = signal.find_peaks(envelope, height=threshold, distance=min_peak_distance)
    if peak_indices.size == 0:
        return CandidateTrainRange(anchor_start_s, anchor_end_s, [], [])

    peak_times = times[peak_indices]
    inter_peak_intervals_s = np.diff(peak_times)
    if inter_peak_intervals_s.size > 0:
        connection_gap_s = float(np.mean(inter_peak_intervals_s)) * 2.5
    else:
        connection_gap_s = max_peak_gap_s * 2.0
    half_bin_s = time_step_s / 2.0
    segments = _build_candidate_train_segments(peak_times, half_bin_s, max_peak_gap_s)
    segments = _select_anchor_connected_segments(
        segments,
        anchor_start_s=anchor_start_s,
        anchor_end_s=anchor_end_s,
        max_train_extension_s=max_train_extension_s,
        connection_gap_s=connection_gap_s,
    )
    segments = _merge_candidate_train_segments(segments)
    if not segments:
        return CandidateTrainRange(anchor_start_s, anchor_end_s, [], [])

    start_time_s = min(segment.start_time_s for segment in segments)
    end_time_s = max(segment.end_time_s for segment in segments)
    return CandidateTrainRange(
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        peak_times_s=[
            peak_time_s
            for segment in segments
            for peak_time_s in segment.peak_times_s
        ],
        segments=segments,
    )


def export_validation_clip(
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
    )
    detections_for_clip = detections_in_window(detections, window)

    start_index = max(0, int(window.start_time_s * sample_rate_hz))
    end_index = min(len(audio), int(round(window.end_time_s * sample_rate_hz)))
    clip_audio = audio[start_index:end_index]
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = _build_validation_artifact_paths(
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
    )

    spectrogram_path = artifact_paths["spectrogram_png"]
    render_validation_spectrogram(
        audio=clip_audio,
        sample_rate_hz=sample_rate_hz,
        window=window,
        detections=detections_for_clip,
        selected_bout=selected_bout,
        expanded_train=expanded_train,
        output_path=spectrogram_path,
        max_freq_hz=max_freq_hz,
        title=audio_path.name,
    )

    report_path = artifact_paths["report_json"]
    report = {
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
        "expanded_train_start_s": expanded_train.start_time_s + window.start_time_s if expanded_train else None,
        "expanded_train_end_s": expanded_train.end_time_s + window.start_time_s if expanded_train else None,
        "expanded_train_duration_s": expanded_train.duration_s if expanded_train else None,
        "expanded_train_peak_count": len(expanded_train.peak_times_s) if expanded_train else 0,
        "expanded_train_segment_count": expanded_train.segment_count if expanded_train else 0,
        "expanded_train_segments": [asdict(segment) for segment in expanded_train.segments] if expanded_train else [],
        "clip_mp3": str(clip_mp3_path) if clip_mp3_path else None,
        "audible_mp3": str(audible_mp3_path) if audible_mp3_path else None,
        "clip_truncated_at_file_start": window.start_time_s <= 0.0,
        "clip_truncated_at_file_end": window.end_time_s >= recording_duration_s,
        "detections": [asdict(item) for item in detections_for_clip],
    }
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


def analyze_candidate_train(
    audio,
    sample_rate_hz: int,
    window: ClipWindow,
    selected_bout: DetectionBout | None,
    max_freq_hz: float,
) -> CandidateTrainRange | None:
    import numpy as np
    from scipy import ndimage, signal

    if selected_bout is None:
        return None

    waveform = np.asarray(audio)
    if waveform.size == 0:
        return None

    if np.issubdtype(waveform.dtype, np.integer):
        info = np.iinfo(waveform.dtype)
        scale = max(abs(info.min), info.max)
        waveform = waveform.astype(np.float32) / float(scale)
    else:
        waveform = waveform.astype(np.float32)

    clip_duration_s = waveform.size / float(sample_rate_hz)

    nperseg = min(2048, waveform.size)
    noverlap = max(0, int(nperseg * 0.75))
    frequencies_hz, times_s, spectrum = signal.spectrogram(
        waveform,
        fs=sample_rate_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        mode="magnitude",
    )
    spectrum_db = 20.0 * np.log10(np.maximum(spectrum, 1e-12))
    band_low_hz = max(0.0, (selected_bout.min_low_freq_hz or 0.0) - 5000.0)
    band_high_hz = min(max_freq_hz, (selected_bout.max_high_freq_hz or max_freq_hz) + 5000.0)
    band_mask = (frequencies_hz >= band_low_hz) & (frequencies_hz <= band_high_hz)
    if not band_mask.any():
        return None

    band_spectrum = spectrum_db[band_mask].copy()
    band_spectrum -= np.nanmean(band_spectrum, axis=1, keepdims=True)
    np.maximum(band_spectrum, 0.0, out=band_spectrum)
    band_envelope_db = np.nanpercentile(band_spectrum, 85.0, axis=0)
    band_envelope_db = ndimage.gaussian_filter1d(band_envelope_db, sigma=1.0, mode="nearest")
    anchor_start_s = selected_bout.start_time_s - window.start_time_s
    anchor_end_s = selected_bout.end_time_s - window.start_time_s
    return estimate_candidate_train_range(times_s, band_envelope_db, anchor_start_s, anchor_end_s)


def render_validation_spectrogram(
    audio,
    sample_rate_hz: int,
    window: ClipWindow,
    detections: list[ClipDetection],
    selected_bout: DetectionBout | None,
    expanded_train: CandidateTrainRange | None,
    output_path: Path,
    max_freq_hz: float,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import signal

    waveform = np.asarray(audio)
    if waveform.size == 0:
        raise ValueError("Clip audio is empty.")

    if np.issubdtype(waveform.dtype, np.integer):
        info = np.iinfo(waveform.dtype)
        scale = max(abs(info.min), info.max)
        waveform = waveform.astype(np.float32) / float(scale)
    else:
        waveform = waveform.astype(np.float32)

    clip_duration_s = waveform.size / float(sample_rate_hz)

    nperseg = min(2048, waveform.size)
    noverlap = max(0, int(nperseg * 0.75))
    frequencies_hz, times_s, spectrum = signal.spectrogram(
        waveform,
        fs=sample_rate_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        mode="magnitude",
    )
    spectrum_db = 20.0 * np.log10(np.maximum(spectrum, 1e-12))
    frequency_mask = frequencies_hz <= max_freq_hz

    figure, (axis, range_axis) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [12, 1.8], "hspace": 0.08},
    )
    mesh = axis.pcolormesh(
        times_s,
        frequencies_hz[frequency_mask] / 1000.0,
        spectrum_db[frequency_mask],
        shading="auto",
        cmap="magma",
    )
    colorbar = figure.colorbar(mesh, ax=[axis, range_axis], fraction=0.04, pad=0.01)
    colorbar.set_label("dB", labelpad=2)
    colorbar.ax.tick_params(labelsize=8)

    range_axis.set_xlim(0, clip_duration_s)
    range_axis.set_ylim(0, 1)
    range_axis.set_yticks([0.75, 0.25])
    range_axis.set_yticklabels(["Detected", "Expanded"])
    range_axis.tick_params(axis="y", length=0)
    range_axis.spines["top"].set_visible(False)
    range_axis.spines["right"].set_visible(False)
    range_axis.spines["left"].set_visible(False)
    range_axis.grid(False)

    if selected_bout is not None and detections:
        detected_start_s = max(0.0, selected_bout.start_time_s - window.start_time_s)
        detected_end_s = min(clip_duration_s, selected_bout.end_time_s - window.start_time_s)
        range_axis.hlines(0.75, detected_start_s, detected_end_s, color="#8bd3dd", linewidth=3.0)
        range_axis.vlines([detected_start_s, detected_end_s], 0.68, 0.82, color="#8bd3dd", linewidth=2.0)

    if expanded_train is not None:
        segments = expanded_train.segments or [
            CandidateTrainSegment(
                start_time_s=expanded_train.start_time_s,
                end_time_s=expanded_train.end_time_s,
                peak_times_s=expanded_train.peak_times_s,
            )
        ]
        for segment in segments:
            expanded_start_s = max(0.0, segment.start_time_s)
            expanded_end_s = min(clip_duration_s, segment.end_time_s)
            range_axis.hlines(0.25, expanded_start_s, expanded_end_s, color="#f4d35e", linewidth=2.2, linestyles="--")
            range_axis.vlines([expanded_start_s, expanded_end_s], 0.18, 0.32, color="#f4d35e", linewidth=1.6, linestyles="--")

    try:
        recording_start_dt = parse_audiomoth_timestamp(title)
    except Exception:
        recording_start_dt = None

    def _wc(sec_from_recording: float) -> str:
        if recording_start_dt is None:
            return f"{sec_from_recording:.1f}s"
        return (recording_start_dt + timedelta(seconds=sec_from_recording)).strftime("%H:%M:%S")

    if recording_start_dt is not None:
        clip_start_wc = (recording_start_dt + timedelta(seconds=window.start_time_s)).strftime("%H:%M:%S")
        clip_end_wc = (recording_start_dt + timedelta(seconds=window.end_time_s)).strftime("%H:%M:%S")
        xlabel = f"Clip time (s)  ·  {clip_start_wc} – {clip_end_wc}"
    else:
        xlabel = f"Time within clip (s)  ·  {window.start_time_s:.1f}s – {window.end_time_s:.1f}s"

    footer_lines: list[str] = []
    if selected_bout is not None:
        footer_lines.append(
            f"Detected: {_wc(selected_bout.start_time_s)} – {_wc(selected_bout.end_time_s)}"
            f"  ({selected_bout.detection_count} detection{'s' if selected_bout.detection_count != 1 else ''})"
        )
    if expanded_train is not None:
        seg_texts = [
            f"{_wc(seg.start_time_s + window.start_time_s)} – {_wc(seg.end_time_s + window.start_time_s)}"
            for seg in (expanded_train.segments or [
                CandidateTrainSegment(
                    start_time_s=expanded_train.start_time_s,
                    end_time_s=expanded_train.end_time_s,
                    peak_times_s=expanded_train.peak_times_s,
                )
            ])[:4]
        ]
        if expanded_train.segment_count > 4:
            seg_texts.append(f"+{expanded_train.segment_count - 4} more")
        footer_lines.append(
            f"Expanded: {', '.join(seg_texts)}"
            f"  ({len(expanded_train.peak_times_s)} peaks, {expanded_train.segment_count}"
            f" segment{'s' if expanded_train.segment_count != 1 else ''})"
        )
    if footer_lines:
        range_axis.text(
            0.0,
            -0.45,
            "\n".join(footer_lines),
            transform=range_axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#222222",
        )

    axis.set_title(f"{title}", fontsize=11, pad=6)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Frequency (kHz)")
    axis.set_ylim(0, max_freq_hz / 1000.0)
    axis.set_xlim(0, clip_duration_s)
    axis.grid(False)
    range_axis.set_xlabel(xlabel)
    figure.subplots_adjust(bottom=0.16, hspace=0.08)
    axis_position = axis.get_position()
    range_position = range_axis.get_position()
    range_axis.set_position([axis_position.x0, range_position.y0, axis_position.width, range_position.height])
    figure.savefig(output_path, dpi=200)
    plt.close(figure)