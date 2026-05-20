from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import unittest

import numpy as np

from batpipe.review import ActivityExtent, ActivityExtractionConfig, ActivitySegment, ClipDetection, DetectionBout, ClipWindow, choose_clip_window, detections_in_window, extract_activity_extent, extract_activity_extent_with_config, format_sample_time_token, group_detection_bouts, render_review_spectrogram


class ReviewAcousticTests(unittest.TestCase):
    def test_format_sample_time_token_uses_recording_local_clock(self) -> None:
        sample_token = format_sample_time_token(Path("20260518_020000T.WAV"), 20.9737)

        self.assertEqual(sample_token, "020020")

    def test_choose_clip_window_adds_context_and_minimum_duration(self) -> None:
        detections = [
            ClipDetection(
                start_time_s=54.649,
                end_time_s=54.6656,
                det_prob=0.594,
                class_prob=0.401,
                predicted_class="Nyctalus noctula",
                event="Echolocation",
                low_freq_hz=21171.0,
                high_freq_hz=26714.0,
            ),
            ClipDetection(
                start_time_s=54.94,
                end_time_s=54.957,
                det_prob=0.616,
                class_prob=0.364,
                predicted_class="Nyctalus noctula",
                event="Echolocation",
                low_freq_hz=21171.0,
                high_freq_hz=27105.0,
            ),
        ]

        window, selected_bout = choose_clip_window(detections=detections, recording_duration_s=55.0)

        self.assertAlmostEqual(window.start_time_s, 45.0)
        self.assertAlmostEqual(window.end_time_s, 55.0)
        self.assertAlmostEqual(window.duration_s, 10.0)
        self.assertIsNotNone(selected_bout)
        self.assertEqual(selected_bout.detection_count if selected_bout else 0, 2)

    def test_group_detection_bouts_merges_nearby_detections(self) -> None:
        detections = [
            ClipDetection(10.0, 10.05, 0.4, 0.2, "a", "Echolocation", 20000.0, 25000.0),
            ClipDetection(10.3, 10.35, 0.5, 0.3, "a", "Echolocation", 20000.0, 25000.0),
            ClipDetection(12.0, 12.05, 0.7, 0.4, "b", "Echolocation", 30000.0, 35000.0),
        ]

        bouts = group_detection_bouts(detections, max_inter_detection_gap_s=0.5)

        self.assertEqual(len(bouts), 2)
        self.assertEqual(bouts[0].detection_count, 2)
        self.assertAlmostEqual(bouts[0].start_time_s, 10.0)
        self.assertAlmostEqual(bouts[0].end_time_s, 10.35)

    def test_choose_clip_window_prefers_primary_bout_over_file_wide_span(self) -> None:
        detections = [
            ClipDetection(10.0, 10.04, 0.45, 0.2, "a", "Echolocation", 20000.0, 25000.0),
            ClipDetection(10.2, 10.24, 0.48, 0.25, "a", "Echolocation", 20000.0, 25000.0),
            ClipDetection(48.0, 48.04, 0.9, 0.7, "b", "Echolocation", 30000.0, 35000.0),
        ]

        window, selected_bout = choose_clip_window(
            detections=detections,
            recording_duration_s=60.0,
            padding_before_s=5.0,
            padding_after_s=4.0,
            minimum_duration_s=10.0,
            bout_gap_s=0.5,
        )

        self.assertIsNotNone(selected_bout)
        self.assertAlmostEqual(selected_bout.start_time_s if selected_bout else -1.0, 10.0)
        self.assertAlmostEqual(selected_bout.end_time_s if selected_bout else -1.0, 10.24)
        self.assertAlmostEqual(window.start_time_s, 4.62)
        self.assertAlmostEqual(window.end_time_s, 14.62)

    def test_detections_in_window_filters_to_clip_overlap(self) -> None:
        detections = [
            ClipDetection(1.0, 1.2, 0.6, 0.4, "a", None, 20000.0, 25000.0),
            ClipDetection(4.5, 4.8, 0.5, 0.3, "b", None, 30000.0, 35000.0),
            ClipDetection(9.5, 9.8, 0.4, 0.2, "c", None, 40000.0, 45000.0),
        ]

        window, _ = choose_clip_window(detections=[], recording_duration_s=12.0, clip_start_s=4.0, clip_duration_s=4.0)
        clipped = detections_in_window(detections, window)

        self.assertEqual([item.predicted_class for item in clipped], ["b"])

    def test_extract_activity_extent_extends_to_nearby_peaks(self) -> None:
        times_s = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        band_envelope_db = np.array([-40.0, -40.0, -10.0, -20.0, -11.0, -40.0, -40.0])

        estimated = extract_activity_extent(
            times_s=times_s,
            band_envelope_db=band_envelope_db,
            anchor_start_s=0.38,
            anchor_end_s=0.42,
            max_peak_gap_s=0.25,
            max_activity_extension_s=0.3,
        )

        self.assertIsNotNone(estimated)
        self.assertAlmostEqual(estimated.start_time_s if estimated else -1.0, 0.15)
        self.assertAlmostEqual(estimated.end_time_s if estimated else -1.0, 0.45)
        self.assertEqual(len(estimated.peak_times_s if estimated else []), 2)
        self.assertEqual(estimated.segment_count if estimated else -1, 1)
        self.assertEqual(estimated.left_boundary.stop_reason if estimated and estimated.left_boundary else "", "activity_dropoff")
        self.assertEqual(estimated.right_boundary.stop_reason if estimated and estimated.right_boundary else "", "anchor_edge")

    def test_extract_activity_extent_keeps_multiple_segments_near_anchor(self) -> None:
        times_s = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
        band_envelope_db = np.array([-40.0, -12.0, -40.0, -13.0, -40.0, -40.0, -40.0, -40.0, -40.0, -10.0, -40.0, -11.0, -40.0])

        estimated = extract_activity_extent(
            times_s=times_s,
            band_envelope_db=band_envelope_db,
            anchor_start_s=0.48,
            anchor_end_s=0.52,
            max_peak_gap_s=0.12,
            max_activity_extension_s=0.5,
        )

        self.assertIsNotNone(estimated)
        self.assertAlmostEqual(estimated.start_time_s if estimated else -1.0, 0.025)
        self.assertAlmostEqual(estimated.end_time_s if estimated else -1.0, 0.575)
        self.assertEqual(len(estimated.peak_times_s if estimated else []), 4)
        self.assertEqual(estimated.segment_count if estimated else -1, 1)
        self.assertAlmostEqual(estimated.segments[0].start_time_s if estimated else -1.0, 0.025)
        self.assertAlmostEqual(estimated.segments[0].end_time_s if estimated else -1.0, 0.575)

    def test_extract_activity_extent_bridges_anchor_connected_gaps(self) -> None:
        times_s = np.array([
            0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1,
        ])
        band_envelope_db = np.array([
            -40.0, -40.0, -12.0, -40.0, -40.0, -13.0, -40.0, -40.0, -40.0, -40.0, -40.0, -11.0, -40.0, -40.0, -10.0, -40.0, -40.0, -40.0, -40.0, -40.0, -40.0, -9.0, -40.0,
        ])

        estimated = extract_activity_extent(
            times_s=times_s,
            band_envelope_db=band_envelope_db,
            anchor_start_s=0.56,
            anchor_end_s=0.6,
            max_peak_gap_s=0.2,
            max_activity_extension_s=0.15,
        )

        self.assertIsNotNone(estimated)
        self.assertEqual(estimated.segment_count if estimated else -1, 1)
        self.assertAlmostEqual(estimated.start_time_s if estimated else -1.0, 0.075)
        self.assertAlmostEqual(estimated.end_time_s if estimated else -1.0, 1.075)
        self.assertEqual(len(estimated.peak_times_s if estimated else []), 5)

    def test_extract_activity_extent_bridges_up_to_average_gap_multiplier(self) -> None:
        times_s = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
        band_envelope_db = np.array([
            -40.0, -40.0, -12.0, -40.0, -40.0, -13.0, -40.0, -40.0, -14.0, -40.0, -40.0, -40.0, -40.0, -40.0, -40.0, -40.0, -9.0, -40.0, -40.0,
        ])

        estimated = extract_activity_extent(
            times_s=times_s,
            band_envelope_db=band_envelope_db,
            anchor_start_s=0.79,
            anchor_end_s=0.81,
            max_peak_gap_s=0.18,
            max_activity_extension_s=0.02,
        )

        self.assertIsNotNone(estimated)
        self.assertEqual(estimated.segment_count if estimated else -1, 1)
        self.assertAlmostEqual(estimated.start_time_s if estimated else -1.0, 0.075)
        self.assertAlmostEqual(estimated.end_time_s if estimated else -1.0, 0.825)
        self.assertEqual(len(estimated.peak_times_s if estimated else []), 4)

    def test_extract_activity_extent_with_config_matches_default_behavior(self) -> None:
        times_s = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        band_envelope_db = np.array([-40.0, -40.0, -10.0, -20.0, -11.0, -40.0, -40.0])

        estimated = extract_activity_extent_with_config(
            times_s=times_s,
            band_envelope_db=band_envelope_db,
            anchor_start_s=0.38,
            anchor_end_s=0.42,
            config=ActivityExtractionConfig(max_peak_gap_s=0.25, max_activity_extension_s=0.3),
        )

        self.assertIsNotNone(estimated)
        self.assertAlmostEqual(estimated.start_time_s if estimated else -1.0, 0.15)
        self.assertAlmostEqual(estimated.end_time_s if estimated else -1.0, 0.45)
        self.assertEqual(len(estimated.peak_times_s if estimated else []), 2)

    def test_extract_activity_extent_records_anchor_edge_when_no_extension(self) -> None:
        times_s = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        band_envelope_db = np.array([-40.0, -40.0, -10.0, -40.0, -40.0])

        estimated = extract_activity_extent(
            times_s=times_s,
            band_envelope_db=band_envelope_db,
            anchor_start_s=0.18,
            anchor_end_s=0.22,
            max_peak_gap_s=0.1,
            max_activity_extension_s=0.05,
        )

        self.assertIsNotNone(estimated)
        self.assertEqual(estimated.left_boundary.stop_reason if estimated and estimated.left_boundary else "", "anchor_edge")
        self.assertEqual(estimated.right_boundary.stop_reason if estimated and estimated.right_boundary else "", "anchor_edge")

    def test_extract_activity_extent_tracks_sustained_activity_without_multiple_sharp_peaks(self) -> None:
        times_s = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        band_envelope_db = np.array([-40.0, -35.0, -17.0, -16.0, -15.5, -16.2, -17.0, -35.0, -40.0])

        estimated = extract_activity_extent(
            times_s=times_s,
            band_envelope_db=band_envelope_db,
            anchor_start_s=0.38,
            anchor_end_s=0.52,
            max_peak_gap_s=0.12,
            max_activity_extension_s=0.3,
        )

        self.assertIsNotNone(estimated)
        self.assertAlmostEqual(estimated.start_time_s if estimated else -1.0, 0.15)
        self.assertAlmostEqual(estimated.end_time_s if estimated else -1.0, 0.65)
        self.assertEqual(estimated.segment_count if estimated else -1, 1)
        self.assertGreaterEqual(len(estimated.peak_evidence if estimated else []), 1)
        self.assertEqual(estimated.left_boundary.stop_reason if estimated and estimated.left_boundary else "", "activity_dropoff")
        self.assertEqual(estimated.right_boundary.stop_reason if estimated and estimated.right_boundary else "", "activity_dropoff")

    def test_extract_activity_extent_does_not_extend_flat_noise_plateau(self) -> None:
        times_s = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        band_envelope_db = np.array([-40.0, -16.0, -16.0, -16.0, -10.0, -16.0, -16.0, -16.0, -40.0])

        estimated = extract_activity_extent(
            times_s=times_s,
            band_envelope_db=band_envelope_db,
            anchor_start_s=0.38,
            anchor_end_s=0.42,
            max_peak_gap_s=0.12,
            max_activity_extension_s=0.3,
        )

        self.assertIsNotNone(estimated)
        self.assertAlmostEqual(estimated.start_time_s if estimated else -1.0, 0.35)
        self.assertAlmostEqual(estimated.end_time_s if estimated else -1.0, 0.45)
        self.assertEqual(estimated.segment_count if estimated else -1, 1)
        self.assertEqual(len(estimated.peak_times_s if estimated else []), 1)
        self.assertEqual(estimated.left_boundary.stop_reason if estimated and estimated.left_boundary else "", "anchor_edge")
        self.assertEqual(estimated.right_boundary.stop_reason if estimated and estimated.right_boundary else "", "anchor_edge")

    def test_render_review_spectrogram_aligns_footer_axis_with_spectrogram_axis(self) -> None:
        captured_positions: dict[str, tuple[float, float, float, float] | tuple[float, float]] = {}

        def capture_savefig(figure, *args, **kwargs):
            axis, range_axis, *_ = figure.axes
            captured_positions["spectrogram"] = axis.get_position().bounds
            captured_positions["footer"] = range_axis.get_position().bounds
            captured_positions["spectrogram_xlim"] = axis.get_xlim()
            captured_positions["footer_xlim"] = range_axis.get_xlim()

        window = ClipWindow(start_time_s=25.218, end_time_s=35.218)
        selected_bout = DetectionBout(
            start_time_s=30.528,
            end_time_s=30.909,
            detections=[ClipDetection(30.528, 30.909, 0.667, 0.5, "bat", "Echolocation", 40000.0, 50000.0)],
        )
        activity_extent = ActivityExtent(
            start_time_s=5.207,
            end_time_s=5.908,
            peak_times_s=[5.31, 5.45, 5.62],
            segments=[ActivitySegment(start_time_s=5.207, end_time_s=5.908, peak_times_s=[5.31, 5.45, 5.62])],
            peak_evidence=[],
        )

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "spectrogram.png"
            with patch("matplotlib.figure.Figure.savefig", new=capture_savefig):
                render_review_spectrogram(
                    audio=np.sin(np.linspace(0.0, 50.0, 4096, dtype=np.float32)),
                    sample_rate_hz=256000,
                    window=window,
                    detections=selected_bout.detections,
                    selected_bout=selected_bout,
                    activity_extent=activity_extent,
                    output_path=output_path,
                    max_freq_hz=96000.0,
                    title="20260518_233200T.WAV",
                )

        self.assertIn("spectrogram", captured_positions)
        self.assertIn("footer", captured_positions)
        self.assertAlmostEqual(captured_positions["spectrogram"][0], captured_positions["footer"][0], places=4)
        self.assertAlmostEqual(captured_positions["spectrogram"][2], captured_positions["footer"][2], places=4)
        self.assertEqual(captured_positions["spectrogram_xlim"], captured_positions["footer_xlim"])


if __name__ == "__main__":
    unittest.main()