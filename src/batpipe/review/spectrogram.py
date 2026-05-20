from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from batpipe.audiomoth import parse_audiomoth_timestamp
from batpipe.review.band_analysis import compute_spectrogram_db
from batpipe.review.models import (
    ActivityExtent,
    ActivitySegment,
    ClipDetection,
    ClipWindow,
    DetectionBout,
    SpectrogramConfig,
)


def render_review_spectrogram(
    audio,
    sample_rate_hz: int,
    window: ClipWindow,
    detections: list[ClipDetection],
    selected_bout: DetectionBout | None,
    activity_extent: ActivityExtent | None,
    output_path: Path,
    max_freq_hz: float,
    title: str,
    spectrogram_config: SpectrogramConfig | None = None,
) -> None:
    import matplotlib.pyplot as plt

    spectrogram_config = spectrogram_config or SpectrogramConfig()
    spectrogram_analysis = compute_spectrogram_db(
        audio,
        sample_rate_hz,
        spectrogram_config,
    )
    clip_duration_s = spectrogram_analysis.clip_duration_s
    frequencies_hz = spectrogram_analysis.frequencies_hz
    times_s = spectrogram_analysis.times_s
    spectrum_db = spectrogram_analysis.spectrum_db
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

    range_axis.set_xlim(0, clip_duration_s)
    range_axis.set_ylim(0, 1)
    range_axis.set_yticks([0.75, 0.25])
    range_axis.set_yticklabels(["Detected", "Activity"])
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

    if activity_extent is not None:
        segments = activity_extent.segments or [
            ActivitySegment(
                start_time_s=activity_extent.start_time_s,
                end_time_s=activity_extent.end_time_s,
                peak_times_s=activity_extent.peak_times_s,
            )
        ]
        for segment in segments:
            activity_start_s = max(0.0, segment.start_time_s)
            activity_end_s = min(clip_duration_s, segment.end_time_s)
            range_axis.hlines(0.25, activity_start_s, activity_end_s, color="#f4d35e", linewidth=2.2, linestyles="--")
            range_axis.vlines([activity_start_s, activity_end_s], 0.18, 0.32, color="#f4d35e", linewidth=1.6, linestyles="--")

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
    if activity_extent is not None:
        seg_texts = [
            f"{_wc(seg.start_time_s + window.start_time_s)} – {_wc(seg.end_time_s + window.start_time_s)}"
            for seg in (activity_extent.segments or [
                ActivitySegment(
                    start_time_s=activity_extent.start_time_s,
                    end_time_s=activity_extent.end_time_s,
                    peak_times_s=activity_extent.peak_times_s,
                )
            ])[:4]
        ]
        if activity_extent.segment_count > 4:
            seg_texts.append(f"+{activity_extent.segment_count - 4} more")
        footer_lines.append(
            f"Activity: {', '.join(seg_texts)}"
            f"  ({len(activity_extent.peak_times_s)} peaks, {activity_extent.segment_count}"
            f" segment{'s' if activity_extent.segment_count != 1 else ''})"
        )
        if activity_extent.left_boundary or activity_extent.right_boundary:
            left_reason = activity_extent.left_boundary.stop_reason if activity_extent.left_boundary else "unknown"
            right_reason = activity_extent.right_boundary.stop_reason if activity_extent.right_boundary else "unknown"
            footer_lines.append(f"Edges: left {left_reason} · right {right_reason}")
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