from __future__ import annotations

from batpipe.review.model_activity import ActivitySegment


def build_peak_segments(peak_times_s, half_bin_s: float, max_peak_gap_s: float) -> list[ActivitySegment]:
    import numpy as np

    peak_times = np.asarray(peak_times_s, dtype=float)
    if peak_times.size == 0:
        return []

    segments: list[ActivitySegment] = []
    start_index = 0
    for index in range(1, peak_times.size):
        if (peak_times[index] - peak_times[index - 1]) <= max_peak_gap_s:
            continue

        segment_peak_times = peak_times[start_index:index]
        segments.append(
            ActivitySegment(
                start_time_s=max(0.0, float(segment_peak_times[0] - half_bin_s)),
                end_time_s=float(segment_peak_times[-1] + half_bin_s),
                peak_times_s=[float(value) for value in segment_peak_times],
            )
        )
        start_index = index

    segment_peak_times = peak_times[start_index:]
    segments.append(
        ActivitySegment(
            start_time_s=max(0.0, float(segment_peak_times[0] - half_bin_s)),
            end_time_s=float(segment_peak_times[-1] + half_bin_s),
            peak_times_s=[float(value) for value in segment_peak_times],
        )
    )
    return segments


def build_mask_activity_segments(
    times_s,
    active_mask,
    peak_times_s,
    *,
    max_silence_gap_s: float,
) -> list[ActivitySegment]:
    import numpy as np

    times = np.asarray(times_s, dtype=float)
    active_indices = np.flatnonzero(np.asarray(active_mask, dtype=bool))
    if active_indices.size == 0:
        return []

    half_bin_s = float(np.median(np.diff(times))) / 2.0 if times.size > 1 else 0.01
    peak_times = np.asarray(peak_times_s, dtype=float)
    segments: list[ActivitySegment] = []
    segment_start_index = int(active_indices[0])
    previous_index = int(active_indices[0])

    def append_segment(start_index: int, end_index: int) -> None:
        segment_start_s = max(0.0, float(times[start_index] - half_bin_s))
        segment_end_s = float(times[end_index] + half_bin_s)
        segment_peak_times = [
            float(peak_time_s)
            for peak_time_s in peak_times
            if segment_start_s <= float(peak_time_s) <= segment_end_s
        ]
        segments.append(
            ActivitySegment(
                start_time_s=segment_start_s,
                end_time_s=segment_end_s,
                peak_times_s=segment_peak_times,
            )
        )

    for current_index in active_indices[1:]:
        current_index = int(current_index)
        if float(times[current_index] - times[previous_index]) <= max_silence_gap_s:
            previous_index = current_index
            continue
        append_segment(segment_start_index, previous_index)
        segment_start_index = current_index
        previous_index = current_index

    append_segment(segment_start_index, previous_index)
    return segments